#!/usr/bin/env python3
"""t-SNE and UMAP parameter-sensitivity analysis of three-condition beta patterns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from fmri_video_analysis.core.config import (
    FIRST_LEVEL_DIR,
    RESULTS_DIR,
    build_four_session_manifest,
)
from fmri_video_analysis.core.data_loading import load_task_network_features


COLORS = {"Simple": "#2E86DE", "Complex": "#E67E22", "Control": "#27AE60"}


def scores(coords: np.ndarray, table: pd.DataFrame) -> dict[str, float]:
    condition = table.condition.to_numpy()
    session = table.session.to_numpy()
    centroids = {name: coords[condition == name].mean(axis=0) for name in COLORS}
    return {
        "condition_silhouette": float(silhouette_score(coords, condition)),
        "session_silhouette": float(silhouette_score(coords, session)),
        "simple_complex_centroid_distance": float(
            np.linalg.norm(centroids["Simple"] - centroids["Complex"])
        ),
        "simple_control_centroid_distance": float(
            np.linalg.norm(centroids["Simple"] - centroids["Control"])
        ),
        "complex_control_centroid_distance": float(
            np.linalg.norm(centroids["Complex"] - centroids["Control"])
        ),
    }


def run(args: argparse.Namespace):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = build_four_session_manifest(Path(args.first_level_dir)).copy()
    table = table.sort_values(["subject_id", "session", "class_id"]).reset_index(
        drop=True
    )
    X_all, _, _, columns = load_task_network_features(table)
    X = X_all[:, columns["visual"]].astype(np.float64)
    # Remove stable between-subject offsets before visualising condition structure.
    for _, indices in table.groupby("subject_id").groups.items():
        X[indices] -= X[indices].mean(axis=0, keepdims=True)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=args.pca_components, random_state=args.seed)
    Z = pca.fit_transform(X)

    configs: list[tuple[str, dict]] = []
    for perplexity in args.tsne_perplexities:
        configs.append(("t-SNE", {"perplexity": perplexity}))
    for neighbors in args.umap_neighbors:
        for min_dist in args.umap_min_dist:
            configs.append(("UMAP", {"n_neighbors": neighbors, "min_dist": min_dist}))

    rows, coordinates = [], []
    for index, (method, params) in enumerate(configs):
        if method == "t-SNE":
            model = TSNE(
                n_components=2,
                perplexity=params["perplexity"],
                init="pca",
                learning_rate="auto",
                max_iter=1500,
                random_state=args.seed,
            )
        else:
            model = umap.UMAP(
                n_components=2,
                n_neighbors=params["n_neighbors"],
                min_dist=params["min_dist"],
                metric="euclidean",
                random_state=args.seed,
            )
        coords = model.fit_transform(Z)
        label = method + " " + ", ".join(f"{k}={v:g}" for k, v in params.items())
        metric = scores(coords, table)
        metric.update(
            {
                "configuration": label,
                "method": method,
                **params,
                "trustworthiness_15": float(trustworthiness(Z, coords, n_neighbors=15)),
            }
        )
        rows.append(metric)
        part = table[["subject_id", "session", "condition", "video_id"]].copy()
        part["configuration"] = label
        part["dim1"] = coords[:, 0]
        part["dim2"] = coords[:, 1]
        coordinates.append(part)
        print(f"Completed {index + 1}/{len(configs)}: {label}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "embedding_parameter_metrics.csv", index=False)
    pd.concat(coordinates, ignore_index=True).to_csv(
        output / "embedding_coordinates.csv", index=False
    )
    (output / "embedding_parameters.json").write_text(
        json.dumps(
            {
                "n_samples": len(table),
                "visual_voxels": X.shape[1],
                "pca_components": args.pca_components,
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "subject_centering": True,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ncols = 3
    nrows = int(np.ceil(len(configs) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(15, 4.3 * nrows))
    for axis, row, part in zip(axes.ravel(), rows, coordinates):
        for condition, color in COLORS.items():
            keep = part.condition == condition
            axis.scatter(
                part.loc[keep, "dim1"],
                part.loc[keep, "dim2"],
                s=9,
                alpha=0.42,
                color=color,
                label=condition,
                edgecolors="none",
            )
        axis.set_title(
            f"{row['configuration']}\ncondition silhouette={row['condition_silhouette']:.3f}; "
            f"T(15)={row['trustworthiness_15']:.3f}",
            fontsize=10,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
    for axis in axes.ravel()[len(configs) :]:
        axis.axis("off")
    axes.ravel()[0].legend(frameon=False, markerscale=2, loc="best")
    figure.suptitle(
        "Three-condition embedding sensitivity (subject-centered visual ROI)",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.975])
    figure.savefig(
        output / "embedding_parameter_sensitivity.png", dpi=300, bbox_inches="tight"
    )
    figure.savefig(output / "embedding_parameter_sensitivity.pdf", bbox_inches="tight")
    plt.close(figure)
    print(metrics.to_string(index=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument("--output-dir", default=str(RESULTS_DIR / "embedding/sensitivity"))
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--tsne-perplexities", type=float, nargs="+", default=[15, 30, 50])
    p.add_argument("--umap-neighbors", type=int, nargs="+", default=[10, 30, 50])
    p.add_argument("--umap-min-dist", type=float, nargs="+", default=[0.05, 0.3, 0.8])
    p.add_argument("--seed", type=int, default=20260801)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
