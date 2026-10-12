#!/usr/bin/env python3
"""Exploratory UMAP visualisation of Simple, Complex, and Control beta maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap

from fmri_video_analysis.core.config import (
    FIRST_LEVEL_DIR,
    ROI_REFERENCE_DIR,
    RESULTS_DIR,
    build_four_session_manifest,
)
from fmri_video_analysis.methods.decoding.cross_video import load_visual_features


COLORS = {"Simple": "#2E86DE", "Complex": "#E67E22", "Control": "#27AE60"}
SESSION_COLORS = {1: "#440154", 2: "#31688e", 3: "#35b779", 4: "#fde725"}


def within_subject_center(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    centered = np.empty_like(X, dtype=np.float32)
    for subject in np.unique(subjects):
        take = subjects == subject
        centered[take] = X[take] - X[take].mean(axis=0, keepdims=True)
    return centered


def prepare_pca(X: np.ndarray, n_components: int, seed: int):
    scaled = StandardScaler().fit_transform(X)
    n_components = min(n_components, scaled.shape[0] - 1, scaled.shape[1])
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    scores = model.fit_transform(scaled)
    return scores.astype(np.float32), model.explained_variance_ratio_


def embed(X: np.ndarray, args: argparse.Namespace, seed: int) -> np.ndarray:
    model = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )
    return model.fit_transform(X).astype(np.float32)


def add_ellipse(axis, points: np.ndarray, color: str):
    if len(points) < 3:
        return
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    # sqrt(chi-square_2(.90)) = sqrt(4.605), for a descriptive 90% ellipse.
    width, height = 2 * np.sqrt(np.maximum(values, 0) * 4.605)
    axis.add_patch(
        Ellipse(
            points.mean(axis=0),
            width,
            height,
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=2,
            alpha=0.9,
        )
    )


def condition_panel(axis, embedding: np.ndarray, table: pd.DataFrame, title: str):
    for condition in ("Simple", "Complex", "Control"):
        take = table.condition.to_numpy() == condition
        points = embedding[take]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=16,
            alpha=0.48,
            color=COLORS[condition],
            edgecolors="none",
            label=condition,
        )
        centroid = points.mean(axis=0)
        axis.scatter(
            centroid[0],
            centroid[1],
            s=125,
            marker="X",
            color=COLORS[condition],
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        add_ellipse(axis, points, COLORS[condition])
    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.legend(frameon=False, ncol=3, loc="best")
    axis.spines[["top", "right"]].set_visible(False)


def session_panel(axis, embedding: np.ndarray, table: pd.DataFrame, title: str):
    for session in (1, 2, 3, 4):
        take = table.session.to_numpy() == session
        points = embedding[take]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=16,
            alpha=0.45,
            color=SESSION_COLORS[session],
            edgecolors="none",
            label=f"Session {session}",
        )
    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.legend(frameon=False, ncol=2, loc="best")
    axis.spines[["top", "right"]].set_visible(False)


def embedding_metrics(
    embedding: np.ndarray, pca_scores: np.ndarray, table: pd.DataFrame
) -> dict:
    condition = table.condition.to_numpy()
    session = table.session.to_numpy()
    subject = table.subject_id.to_numpy()
    return {
        "condition_silhouette_umap": float(silhouette_score(embedding, condition)),
        "session_silhouette_umap": float(silhouette_score(embedding, session)),
        "subject_silhouette_umap": float(silhouette_score(embedding, subject)),
        "condition_silhouette_pca50": float(silhouette_score(pca_scores, condition)),
        "session_silhouette_pca50": float(silhouette_score(pca_scores, session)),
        "subject_silhouette_pca50": float(silhouette_score(pca_scores, subject)),
        "trustworthiness_15": float(
            trustworthiness(pca_scores, embedding, n_neighbors=15)
        ),
    }


def centroid_distances(embedding: np.ndarray, table: pd.DataFrame) -> dict:
    centroids = {
        condition: embedding[table.condition.to_numpy() == condition].mean(axis=0)
        for condition in ("Simple", "Complex", "Control")
    }
    return {
        "simple_complex": float(
            np.linalg.norm(centroids["Simple"] - centroids["Complex"])
        ),
        "simple_control": float(
            np.linalg.norm(centroids["Simple"] - centroids["Control"])
        ),
        "complex_control": float(
            np.linalg.norm(centroids["Complex"] - centroids["Control"])
        ),
    }


def run(args: argparse.Namespace):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = build_four_session_manifest(Path(args.first_level_dir)).copy()
    table = table.sort_values(["subject_id", "session", "condition"]).reset_index(
        drop=True
    )
    mask = ROI_REFERENCE_DIR / "roi_masks/schaefer200_17net_visual_commonmask.nii.gz"
    X, _, indices = load_visual_features(table, mask)
    subjects = table.subject_id.to_numpy()

    global_pca, global_variance = prepare_pca(X, args.pca_components, args.seed)
    centered = within_subject_center(X, subjects)
    centered_pca, centered_variance = prepare_pca(
        centered, args.pca_components, args.seed
    )
    global_embedding = embed(global_pca, args, args.seed)
    centered_embedding = embed(centered_pca, args, args.seed)

    metrics = {
        "n_subjects": int(table.subject_id.nunique()),
        "n_samples": int(len(table)),
        "n_voxels": int(X.shape[1]),
        "conditions": table.condition.value_counts().to_dict(),
        "pca_components": int(centered_pca.shape[1]),
        "global_pca_explained_variance": float(global_variance.sum()),
        "subject_centered_pca_explained_variance": float(centered_variance.sum()),
        "umap_parameters": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "primary_seed": args.seed,
        },
        "global": embedding_metrics(global_embedding, global_pca, table),
        "subject_centered": embedding_metrics(centered_embedding, centered_pca, table),
        "subject_centered_centroid_distances": centroid_distances(
            centered_embedding, table
        ),
    }

    seeds = [args.seed, 7, 42, 101, 2026]
    seed_embeddings = {args.seed: centered_embedding}
    for seed in seeds[1:]:
        seed_embeddings[seed] = embed(centered_pca, args, seed)
    primary_distances = pdist(seed_embeddings[args.seed])
    seed_rows = []
    for seed in seeds:
        current = seed_embeddings[seed]
        seed_rows.append(
            {
                "seed": seed,
                "condition_silhouette": silhouette_score(
                    current, table.condition.to_numpy()
                ),
                "session_silhouette": silhouette_score(
                    current, table.session.to_numpy()
                ),
                "distance_spearman_vs_primary": (
                    1.0
                    if seed == args.seed
                    else spearmanr(primary_distances, pdist(current)).statistic
                ),
            }
        )
    seed_table = pd.DataFrame(seed_rows)
    metrics["seed_stability"] = {
        "seeds": seeds,
        "condition_silhouette_mean": float(seed_table.condition_silhouette.mean()),
        "condition_silhouette_sd": float(seed_table.condition_silhouette.std(ddof=1)),
        "distance_spearman_median_nonprimary": float(
            seed_table.loc[
                seed_table.seed != args.seed, "distance_spearman_vs_primary"
            ].median()
        ),
    }

    embedding_table = table[
        ["subject_id", "session", "condition", "video_id", "beta_path"]
    ].copy()
    embedding_table["global_umap1"] = global_embedding[:, 0]
    embedding_table["global_umap2"] = global_embedding[:, 1]
    embedding_table["subject_centered_umap1"] = centered_embedding[:, 0]
    embedding_table["subject_centered_umap2"] = centered_embedding[:, 1]
    embedding_table.to_csv(output / "umap_embedding.csv", index=False)
    seed_table.to_csv(output / "umap_seed_stability.csv", index=False)
    pd.DataFrame(
        {
            "component": np.arange(1, len(centered_variance) + 1),
            "global_explained_variance_ratio": global_variance,
            "subject_centered_explained_variance_ratio": centered_variance,
        }
    ).to_csv(output / "pca_explained_variance.csv", index=False)
    (output / "umap_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    condition_panel(
        axes[0, 0], global_embedding, table, "A. Global standardisation | condition"
    )
    session_panel(
        axes[0, 1], global_embedding, table, "B. Global standardisation | session"
    )
    condition_panel(
        axes[1, 0],
        centered_embedding,
        table,
        "C. Within-subject centred (primary) | condition",
    )
    session_panel(
        axes[1, 1],
        centered_embedding,
        table,
        "D. Within-subject centred (primary) | session",
    )
    figure.suptitle(
        "UMAP distribution of 1,044 visual-ROI beta maps",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        "UMAP is unsupervised; condition/session labels are used only for colouring. "
        "Ellipses are descriptive 90% covariance ellipses.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=[0, 0.035, 1, 0.975])
    figure.savefig(output / "umap_three_conditions.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "umap_three_conditions.pdf", bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument("--output-dir", default=str(RESULTS_DIR / "embedding/umap"))
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.25)
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--seed", type=int, default=20260731)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
