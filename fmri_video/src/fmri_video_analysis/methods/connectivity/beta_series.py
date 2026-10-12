#!/usr/bin/env python3
"""Exploratory condition-specific beta-series covariance using four session betas."""

from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fmri_video_analysis.core.config import ORIGINAL_MVPA_RESULTS_DIR, RESULTS_DIR
from fmri_video_analysis.core.statistics import (
    bh_fdr,
    cohens_dz,
    paired_sign_flip_pvalue,
)


CONDITIONS = ("Simple", "Complex", "Control")
NETWORKS = ("visual", "dorsal_attention", "salience_attention", "control")


def run(args: argparse.Namespace):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.parcel_beta_values)
    values = pd.read_csv(source)
    parcel_info = values[
        ["parcel_id", "parcel_name", "network_family"]
    ].drop_duplicates()
    parcel_info = parcel_info.sort_values("parcel_id").reset_index(drop=True)
    parcel_ids = parcel_info.parcel_id.to_numpy()
    network = parcel_info.network_family.to_numpy()
    subject_rows, matrices = [], {condition: [] for condition in CONDITIONS}
    edge_names = list(combinations_with_replacement(NETWORKS, 2))
    for subject, subject_part in values.groupby("subject_id", sort=True):
        for condition in CONDITIONS:
            part = subject_part[subject_part.condition == condition]
            wide = part.pivot(index="session", columns="parcel_id", values="mean_beta")
            wide = wide.reindex(index=[1, 2, 3, 4], columns=parcel_ids)
            if wide.isna().any().any() or wide.shape[0] != 4:
                raise RuntimeError(
                    f"Incomplete beta series: subject {subject}, {condition}"
                )
            correlation = np.corrcoef(wide.to_numpy(), rowvar=False)
            z = np.arctanh(np.clip(correlation, -0.999999, 0.999999))
            np.fill_diagonal(z, np.nan)
            matrices[condition].append(z)
            for first, second in edge_names:
                left = np.flatnonzero(network == first)
                right = np.flatnonzero(network == second)
                block = z[np.ix_(left, right)]
                if first == second:
                    block = block[np.triu_indices_from(block, k=1)]
                subject_rows.append(
                    {
                        "subject_id": subject,
                        "condition": condition,
                        "network_edge": f"{first}--{second}",
                        "mean_fisher_z": float(np.nanmean(block)),
                        "beta_observations_per_condition": 4,
                    }
                )
    subject_table = pd.DataFrame(subject_rows)
    subject_table.to_csv(output / "beta_series_network_subject_values.csv", index=False)

    summary = (
        subject_table.groupby(["condition", "network_edge"])
        .mean_fisher_z.agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(output / "beta_series_network_summary.csv", index=False)
    contrasts = [
        ("Complex", "Simple", "complex_minus_simple"),
        ("Simple", "Control", "simple_minus_control"),
        ("Complex", "Control", "complex_minus_control"),
    ]
    rows = []
    for edge in subject_table.network_edge.unique():
        wide = subject_table[subject_table.network_edge == edge].pivot(
            index="subject_id", columns="condition", values="mean_fisher_z"
        )
        for first, second, name in contrasts:
            difference = wide[first] - wide[second]
            rows.append(
                {
                    "network_edge": edge,
                    "contrast": name,
                    "mean_difference": difference.mean(),
                    "cohens_dz": cohens_dz(difference.to_numpy()),
                    "p_signflip": paired_sign_flip_pvalue(
                        difference.to_numpy(), args.n_permutations, args.seed
                    ),
                }
            )
    tests = pd.DataFrame(rows)
    tests["q_fdr_all_30_tests"] = bh_fdr(tests.p_signflip)
    tests.to_csv(output / "beta_series_network_condition_tests.csv", index=False)

    mean_matrices = {}
    for condition in CONDITIONS:
        mean = np.nanmean(np.stack(matrices[condition]), axis=0)
        mean_matrices[condition] = mean
        pd.DataFrame(mean, index=parcel_ids, columns=parcel_ids).to_csv(
            output / f"{condition.lower()}_parcel_connectivity_mean.csv"
        )
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5.2),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.35]},
        constrained_layout=True,
    )
    limits = np.nanpercentile(np.abs(np.stack(list(mean_matrices.values()))), 98)
    for axis, condition in zip(axes[:3], CONDITIONS):
        image = axis.imshow(
            mean_matrices[condition], cmap="RdBu_r", vmin=-limits, vmax=limits
        )
        axis.set_title(condition, fontweight="bold")
        axis.set_xlabel("Schaefer task parcels")
        axis.set_ylabel("Schaefer task parcels")
    significant = tests.sort_values("q_fdr_all_30_tests").head(8).iloc[::-1].copy()
    abbreviations = {
        "visual": "Vis",
        "dorsal_attention": "DAN",
        "salience_attention": "Sal",
        "control": "Con",
    }
    contrast_abbreviations = {
        "complex_minus_simple": "C−S",
        "simple_minus_control": "S−Ctrl",
        "complex_minus_control": "C−Ctrl",
    }
    significant["plot_label"] = [
        "–".join(abbreviations[item] for item in edge.split("--"))
        + " | "
        + contrast_abbreviations[contrast]
        for edge, contrast in zip(significant.network_edge, significant.contrast)
    ]
    axes[3].barh(
        significant.plot_label,
        significant.mean_difference,
        color=np.where(significant.mean_difference >= 0, "#E67E22", "#2E86DE"),
    )
    axes[3].axvline(0, color="black", linewidth=1)
    axes[3].set_xlabel("Paired Fisher-z difference")
    axes[3].set_title("Smallest FDR q (exploratory)", fontweight="bold")
    axes[3].tick_params(axis="y", labelsize=8)
    figure.colorbar(image, ax=axes[:3], fraction=0.018, pad=0.02, label="Mean Fisher z")
    figure.suptitle(
        "Condition-specific beta-series covariance (4 beta observations/condition)",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(
        output / "beta_series_connectivity.png", dpi=300, bbox_inches="tight"
    )
    figure.savefig(output / "beta_series_connectivity.pdf", bbox_inches="tight")
    plt.close(figure)
    (output / "METHOD_LIMITATION.txt").write_text(
        "Each condition-specific correlation uses only four session-level beta estimates. "
        "This is beta-series covariance, not conventional BOLD time-series functional "
        "connectivity; estimates are low-degree-of-freedom and exploratory. Condition "
        "differences may reflect video/session variation and must not support claims of "
        "dynamic network reconfiguration.\n",
        encoding="utf-8",
    )
    print(tests.sort_values("q_fdr_all_30_tests").head(15).to_string(index=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parcel-beta-values",
        default=str(
            ORIGINAL_MVPA_RESULTS_DIR / "descriptive_responses/parcel_beta_values.csv"
        ),
    )
    p.add_argument(
        "--output-dir", default=str(RESULTS_DIR / "connectivity/beta_series")
    )
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260801)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
