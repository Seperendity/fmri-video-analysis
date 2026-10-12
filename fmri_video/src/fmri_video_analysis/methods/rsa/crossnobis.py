"""Cross-session diagonal-crossnobis representational geometry in task networks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from fmri_video_analysis.core.config import (
    FIRST_LEVEL_DIR,
    RESULTS_DIR,
    build_four_session_manifest,
)
from fmri_video_analysis.core.data_loading import (
    load_resms_volume,
    load_task_network_features,
)
from fmri_video_analysis.core.environment import runtime_metadata
from fmri_video_analysis.core.provenance import atomic_write_json
from fmri_video_analysis.core.statistics import (
    bh_fdr,
    cohens_dz,
    paired_sign_flip_pvalue,
)
from fmri_video_analysis.methods.decoding.three_class_networks import (
    ALL_NETWORKS,
    PRIMARY_NETWORKS,
)
from fmri_video_analysis.methods.rsa.crossvalidated_distance import (
    CONDITIONS,
    PAIRS,
    pairwise_crossproducts,
    whiten_patterns,
)


def subject_crossnobis(
    patterns: np.ndarray,
    noise_variance: np.ndarray,
    shrinkage: float,
    *,
    spatial_mean_center: bool = False,
) -> dict[str, float]:
    """Return distances using GLM-residual variance external to the beta array."""

    whitened = whiten_patterns(
        patterns,
        noise_variance,
        shrinkage,
        spatial_mean_center=spatial_mean_center,
    )
    return {
        name: float(np.mean(values))
        for name, values in pairwise_crossproducts(whitened).items()
    }


def plot_rsa(subject_table: pd.DataFrame, summary: pd.DataFrame, output: Path):
    order = list(ALL_NETWORKS)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for axis, network in zip(axes.ravel()[:5], order):
        rows = summary[summary.network == network].set_index("distance")
        matrix = np.zeros((3, 3), dtype=float)
        for first, second, name in PAIRS:
            i, j = CONDITIONS.index(first), CONDITIONS.index(second)
            matrix[i, j] = matrix[j, i] = rows.loc[name, "mean_crossnobis"]
        limit = float(np.max(np.abs(matrix))) or 1.0
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
        for i in range(3):
            for j in range(3):
                axis.text(
                    j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9
                )
        axis.set_xticks(range(3), CONDITIONS, rotation=30, ha="right")
        axis.set_yticks(range(3), CONDITIONS)
        axis.set_title(network.replace("_", " "), fontweight="bold")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis = axes.ravel()[5]
    colors = {
        "simple_complex": "#8E44AD",
        "simple_control": "#2E86DE",
        "complex_control": "#E67E22",
    }
    visual = subject_table[subject_table.network == "visual"]
    rng = np.random.default_rng(42)
    for index, (_, _, distance) in enumerate(PAIRS):
        values = visual[visual.distance == distance].crossnobis.to_numpy()
        axis.scatter(
            index + rng.normal(0, 0.05, len(values)),
            values,
            s=13,
            alpha=0.28,
            color=colors[distance],
            edgecolors="none",
        )
        axis.errorbar(
            index,
            values.mean(),
            yerr=values.std(ddof=1) / np.sqrt(len(values)),
            fmt="o",
            color=colors[distance],
            markersize=8,
            capsize=4,
        )
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(
        range(3), ["Simple–\nComplex", "Simple–\nControl", "Complex–\nControl"]
    )
    axis.set_ylabel("Diagonal crossnobis distance")
    axis.set_title("Visual network subject distances", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Cross-session representational geometry", fontsize=17, fontweight="bold"
    )
    figure.tight_layout(rect=[0, 0, 1, 0.965])
    figure.savefig(output / "crossnobis_rsa.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "crossnobis_rsa.pdf", bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = build_four_session_manifest(Path(args.first_level_dir)).copy()
    table = table.sort_values(["subject_id", "session", "condition"]).reset_index(
        drop=True
    )
    X_all, union_img, flat_indices, columns = load_task_network_features(table)
    first_level_dir = Path(args.first_level_dir)
    resms_by_subject = {}
    resms_qc_rows = []
    for subject in sorted(table.subject_id.unique()):
        path, volume = load_resms_volume(first_level_dir, subject, union_img)
        variance = volume.ravel()[flat_indices]
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            raise RuntimeError(
                f"ResMS must be finite and positive in every retained voxel: {path}"
            )
        resms_by_subject[int(subject)] = variance
        resms_qc_rows.append(
            {
                "subject_id": int(subject),
                "resms_path": str(path.resolve()),
                "n_voxels": int(variance.size),
                "minimum": float(variance.min()),
                "median": float(np.median(variance)),
                "maximum": float(variance.max()),
            }
        )
    pd.DataFrame(resms_qc_rows).to_csv(output / "resms_qc.csv", index=False)
    atomic_write_json(
        output / "crossnobis_run_metadata.json",
        {
            "first_level_dir": str(first_level_dir.resolve()),
            "noise_normalization": "subject_first_level_ResMS_diagonal",
            "noise_estimate_is_external_to_condition_beta_array": True,
            "diagonal_shrinkage": args.diagonal_shrinkage,
            "spatial_mean_center": args.spatial_mean_center,
            "bootstrap": args.bootstrap,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "runtime": runtime_metadata(),
        },
    )
    subject_rows = []
    for network in ALL_NETWORKS:
        X = X_all[:, columns[network]]
        for subject, part in table.groupby("subject_id", sort=True):
            patterns = np.empty((4, 3, X.shape[1]), dtype=np.float32)
            for session in (1, 2, 3, 4):
                for condition_index, condition in enumerate(CONDITIONS):
                    match = part[
                        (part.session == session) & (part.condition == condition)
                    ]
                    if len(match) != 1:
                        raise RuntimeError(
                            f"Missing cell: {subject}, {session}, {condition}"
                        )
                    patterns[session - 1, condition_index] = X[match.index[0]]
            variance = resms_by_subject[int(subject)][columns[network]]
            distances = subject_crossnobis(
                patterns,
                variance,
                args.diagonal_shrinkage,
                spatial_mean_center=args.spatial_mean_center,
            )
            for distance, value in distances.items():
                subject_rows.append(
                    {
                        "network": network,
                        "subject_id": subject,
                        "distance": distance,
                        "crossnobis": value,
                    }
                )
        print(f"Completed RSA: {network}", flush=True)
    subject_table = pd.DataFrame(subject_rows)
    subject_table.to_csv(output / "crossnobis_subject_distances.csv", index=False)

    summary_rows = []
    for (network, distance), part in subject_table.groupby(["network", "distance"]):
        values = part.crossnobis.to_numpy()
        rng = np.random.default_rng(args.seed)
        boot = rng.choice(values, (args.bootstrap, len(values)), replace=True).mean(
            axis=1
        )
        summary_rows.append(
            {
                "network": network,
                "distance": distance,
                "n_subjects": len(values),
                "mean_crossnobis": values.mean(),
                "sd": values.std(ddof=1),
                "cohens_dz_vs_zero": cohens_dz(values),
                "ci_low": np.quantile(boot, 0.025),
                "ci_high": np.quantile(boot, 0.975),
                "p_signflip_vs_zero": paired_sign_flip_pvalue(
                    values, args.n_permutations, args.seed
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    primary = summary.network.isin(PRIMARY_NETWORKS)
    summary.loc[primary, "q_fdr_primary"] = bh_fdr(
        summary.loc[primary, "p_signflip_vs_zero"]
    )
    summary.to_csv(output / "crossnobis_summary.csv", index=False)

    contrast_rows = []
    for network in ALL_NETWORKS:
        wide = subject_table[subject_table.network == network].pivot(
            index="subject_id", columns="distance", values="crossnobis"
        )
        contrasts = {
            "task_control_mean_minus_simple_complex": (
                (wide.simple_control + wide.complex_control) / 2 - wide.simple_complex
            ),
            "complex_control_minus_simple_control": (
                wide.complex_control - wide.simple_control
            ),
        }
        friedman = friedmanchisquare(
            wide.simple_complex, wide.simple_control, wide.complex_control
        )
        for name, values in contrasts.items():
            contrast_rows.append(
                {
                    "network": network,
                    "contrast": name,
                    "mean_difference": values.mean(),
                    "cohens_dz": cohens_dz(values.to_numpy()),
                    "p_signflip": paired_sign_flip_pvalue(
                        values.to_numpy(), args.n_permutations, args.seed
                    ),
                    "friedman_chi2_three_distances": friedman.statistic,
                    "friedman_p_three_distances": friedman.pvalue,
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    primary_contrast = contrasts.network.isin(PRIMARY_NETWORKS)
    contrasts.loc[primary_contrast, "q_fdr_primary"] = bh_fdr(
        contrasts.loc[primary_contrast, "p_signflip"]
    )
    contrasts.to_csv(output / "crossnobis_distance_contrasts.csv", index=False)
    plot_rsa(subject_table, summary, output)
    print(summary.to_string(index=False))
    print(contrasts.to_string(index=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument("--output-dir", default=str(RESULTS_DIR / "rsa/crossnobis"))
    p.add_argument("--diagonal-shrinkage", type=float, default=0.2)
    p.add_argument("--spatial-mean-center", action="store_true")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260731)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
