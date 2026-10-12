#!/usr/bin/env python3
"""Large exploratory post-hoc screening of Schaefer task parcels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fmri_video_analysis.core.config import ORIGINAL_MVPA_RESULTS_DIR, RESULTS_DIR


def run(args: argparse.Namespace):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(args.descriptive_root)
    omnibus = pd.read_csv(root / "parcel_repeated_measures_friedman.csv")
    pairwise = pd.read_csv(root / "parcel_paired_permutation.csv")
    beta = pd.read_csv(root / "parcel_beta_values.csv")
    condition_means = (
        beta.groupby(["parcel_id", "parcel_name", "network_family", "condition"])
        .mean_beta.mean()
        .unstack("condition")
        .reset_index()
    )
    minimum = (
        pairwise.groupby("parcel_id")
        .agg(
            min_pairwise_q=("q_fdr_all_pairwise", "min"),
            max_abs_pairwise_dz=("cohens_dz", lambda x: float(np.max(np.abs(x)))),
            n_fdr_pairwise=("significant_fdr_all_05", "sum"),
        )
        .reset_index()
    )
    screened = omnibus.merge(minimum, on="parcel_id").merge(
        condition_means, on=["parcel_id", "parcel_name", "network_family"]
    )
    screened["selected_any_fdr"] = screened.q_fdr.lt(0.05) | screened.min_pairwise_q.lt(
        0.05
    )
    screened = screened.sort_values(
        ["selected_any_fdr", "min_pairwise_q", "q_fdr", "max_abs_pairwise_dz"],
        ascending=[False, True, True, False],
    )
    screened["posthoc_rank"] = np.arange(1, len(screened) + 1)
    screened.to_csv(output / "posthoc_roi_screening_all_parcels.csv", index=False)
    selected = screened[screened.selected_any_fdr]
    selected.to_csv(output / "posthoc_roi_selected_fdr05.csv", index=False)

    top_ids = screened.head(args.top_n).parcel_id
    heat = condition_means[condition_means.parcel_id.isin(top_ids)].copy()
    heat = heat.set_index("parcel_id").loc[top_ids]
    contrast = (
        pairwise[pairwise.parcel_id.isin(top_ids)]
        .pivot(index="parcel_id", columns="contrast", values="cohens_dz")
        .loc[top_ids]
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, max(7, args.top_n * 0.32)))
    image = axes[0].imshow(
        heat[["Simple", "Complex", "Control"]], aspect="auto", cmap="viridis"
    )
    axes[0].set_xticks(range(3), ["Simple", "Complex", "Control"])
    axes[0].set_yticks(
        range(len(heat)),
        heat.parcel_name.str.replace("17Networks_", "", regex=False),
        fontsize=7,
    )
    axes[0].set_title("A. Condition mean beta", fontweight="bold")
    figure.colorbar(image, ax=axes[0], fraction=0.035, pad=0.02)
    wanted = ["complex_minus_simple", "simple_minus_control", "complex_minus_control"]
    image2 = axes[1].imshow(
        contrast[wanted], aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5
    )
    axes[1].set_xticks(range(3), ["C−S", "S−Ctrl", "C−Ctrl"])
    axes[1].set_yticks(range(len(contrast)), [])
    axes[1].set_title("B. Paired effect size dz", fontweight="bold")
    figure.colorbar(image2, ax=axes[1], fraction=0.035, pad=0.02)
    figure.suptitle(
        f"Post-hoc screening of {len(screened)} task-network parcels (top {args.top_n})",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.975])
    figure.savefig(output / "posthoc_roi_screening.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "posthoc_roi_screening.pdf", bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(
        [
            {
                "parcels_screened": len(screened),
                "selected_any_fdr05": len(selected),
                "selection_data_independent": False,
                "confirmatory_roi_inference_valid": False,
            }
        ]
    ).to_csv(output / "posthoc_roi_screening_summary.csv", index=False)
    (output / "METHOD_LIMITATION.txt").write_text(
        "ROIs were ranked and interpreted using the same subjects and condition effects. "
        "FDR controls the tested parcel family but does not make these ROIs independent. "
        "They are hypothesis-generating and require preregistered or independent replication.\n",
        encoding="utf-8",
    )
    print(
        pd.read_csv(output / "posthoc_roi_screening_summary.csv").to_string(index=False)
    )
    print(screened.head(args.top_n).to_string(index=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--descriptive-root",
        default=str(ORIGINAL_MVPA_RESULTS_DIR / "descriptive_responses"),
    )
    p.add_argument(
        "--output-dir", default=str(RESULTS_DIR / "activation/roi_screening")
    )
    p.add_argument("--top-n", type=int, default=30)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
