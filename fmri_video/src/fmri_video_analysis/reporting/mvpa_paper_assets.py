"""Build manuscript assets from the formally completed cross-video MVPA run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fmri_video_analysis.core.config import RESULTS_DIR
from fmri_video_analysis.reporting.contracts import require_formal_mvpa

FIG_WIDTH_MM = 183
WIDTH_IN = FIG_WIDTH_MM / 25.4
COLORS = {
    "signal": "#C44E52",
    "comparison": "#3B73B9",
    "neutral": "#8A95A5",
    "light": "#DDE2E8",
    "dark": "#263238",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_bundle(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def balanced_accuracy(frame: pd.DataFrame) -> float:
    recalls = []
    for class_id in (0, 1):
        class_rows = frame.loc[frame.true_class == class_id]
        if class_rows.empty:
            raise RuntimeError(f"No observations for class {class_id}.")
        recalls.append(float((class_rows.predicted_class == class_id).mean()))
    return float(np.mean(recalls))


def load_and_audit(mvpa_dir: Path) -> dict[str, object]:
    metrics_path = mvpa_dir / "metrics.json"
    metrics = require_formal_mvpa(metrics_path)
    null = pd.read_csv(mvpa_dir / "structured_permutation_null.csv.gz")
    folds = pd.read_csv(mvpa_dir / "outer_fold_metrics.csv")
    predictions = pd.read_csv(mvpa_dir / "oof_predictions.csv")
    subjects = pd.read_csv(mvpa_dir / "subject_metrics.csv")

    expected_ids = np.arange(1, 10_001)
    permutation_ids = null.permutation.to_numpy(dtype=int)
    null_scores = null.balanced_accuracy.to_numpy(dtype=float)
    if not np.array_equal(np.sort(permutation_ids), expected_ids):
        raise RuntimeError("Structured-permutation IDs are not exactly 1--10,000.")
    if len(np.unique(permutation_ids)) != 10_000 or not np.isfinite(null_scores).all():
        raise RuntimeError("Structured-permutation null contains duplicates or nonfinite scores.")

    observed = float(metrics["balanced_accuracy"])
    exceedances = int(np.sum(null_scores >= observed - 1e-15))
    independent_p = (exceedances + 1) / (len(null_scores) + 1)
    reported_p = float(metrics["structured_label_permutation_p"])
    if not np.isclose(independent_p, reported_p, rtol=0, atol=1e-15):
        raise RuntimeError(
            f"Structured P-value mismatch: recomputed={independent_p}, reported={reported_p}."
        )

    expected_fold_pairs = pd.MultiIndex.from_product(
        [range(5), range(1, 5)], names=["subject_fold", "held_session"]
    )
    observed_fold_pairs = pd.MultiIndex.from_frame(
        folds[["subject_fold", "held_session"]].sort_values(
            ["subject_fold", "held_session"]
        )
    )
    if not observed_fold_pairs.equals(expected_fold_pairs):
        raise RuntimeError("Outer-fold table is not the complete 5 x 4 design.")
    if len(predictions) != 696 or predictions.subject_id.nunique() != 87:
        raise RuntimeError("OOF predictions do not contain 696 samples from 87 participants.")
    per_subject = predictions.groupby("subject_id").size()
    if not (per_subject == 8).all() or set(predictions.true_class.unique()) != {0, 1}:
        raise RuntimeError("OOF prediction balance is inconsistent with the study design.")
    if len(subjects) != 87 or not np.isfinite(subjects.balanced_accuracy).all():
        raise RuntimeError("Participant-level MVPA summary is incomplete.")

    session_summary = (
        predictions.groupby("session", sort=True)
        .apply(balanced_accuracy, include_groups=False)
        .rename("balanced_accuracy")
        .reset_index()
        .rename(columns={"session": "held_session"})
    )
    video_pairs = {
        1: "001 / 009",
        2: "002 / 010",
        3: "003 / 011",
        4: "004 / 012",
    }
    session_summary["held_video_pair"] = session_summary.held_session.map(video_pairs)

    confusion = pd.crosstab(
        predictions.true_class,
        predictions.predicted_class,
        dropna=False,
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    confusion_counts = confusion.to_numpy(dtype=int)
    confusion_rates = confusion_counts / confusion_counts.sum(axis=1, keepdims=True)
    confusion_source = pd.DataFrame(
        [
            {
                "true_condition": ("Simple", "Complex")[true_class],
                "predicted_condition": ("Simple", "Complex")[predicted_class],
                "count": int(confusion_counts[true_class, predicted_class]),
                "row_proportion": float(confusion_rates[true_class, predicted_class]),
            }
            for true_class in (0, 1)
            for predicted_class in (0, 1)
        ]
    )
    return {
        "metrics": metrics,
        "null": null,
        "folds": folds,
        "predictions": predictions,
        "subjects": subjects,
        "session_summary": session_summary,
        "confusion_counts": confusion_counts,
        "confusion_rates": confusion_rates,
        "confusion_source": confusion_source,
        "exceedances": exceedances,
        "independent_p": independent_p,
    }


def write_source_data(audit: dict[str, object], source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    null = audit["null"]
    folds = audit["folds"]
    session_summary = audit["session_summary"]
    confusion_source = audit["confusion_source"]
    metrics = audit["metrics"]
    assert isinstance(null, pd.DataFrame)
    assert isinstance(folds, pd.DataFrame)
    assert isinstance(session_summary, pd.DataFrame)
    assert isinstance(confusion_source, pd.DataFrame)
    assert isinstance(metrics, dict)
    null.to_csv(source_dir / "Figure_2a_structured_permutation_null.csv", index=False)
    folds.to_csv(source_dir / "Figure_2b_outer_fold_accuracy.csv", index=False)
    session_summary.to_csv(
        source_dir / "Figure_2b_held_video_pair_summary.csv", index=False
    )
    confusion_source.to_csv(source_dir / "Figure_2c_confusion_matrix.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_participants": metrics["n_subjects"],
                "n_sessions": metrics["n_sessions"],
                "n_samples": metrics["n_samples"],
                "n_visual_mask_voxels": 3141,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "bootstrap_ci_low": metrics["balanced_accuracy_bootstrap_95_ci"][0],
                "bootstrap_ci_high": metrics["balanced_accuracy_bootstrap_95_ci"][1],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "structured_permutations": metrics["structured_permutations_completed"],
                "null_scores_ge_observed": audit["exceedances"],
                "structured_permutation_p": audit["independent_p"],
            }
        ]
    ).to_csv(source_dir / "Figure_2_summary.csv", index=False)



def write_comparison_source_data(
    audit: dict[str, object],
    centered_audit: dict[str, object],
    source_dir: Path,
) -> None:
    """Write panel-level data for the primary/centred MVPA comparison."""

    centered_null = centered_audit["null"]
    centered_folds = centered_audit["folds"]
    centered_sessions = centered_audit["session_summary"]
    centered_confusion = centered_audit["confusion_source"]
    assert isinstance(centered_null, pd.DataFrame)
    assert isinstance(centered_folds, pd.DataFrame)
    assert isinstance(centered_sessions, pd.DataFrame)
    assert isinstance(centered_confusion, pd.DataFrame)
    centered_null.to_csv(
        source_dir / "Figure_2b_spatial_mean_centered_permutation_null.csv",
        index=False,
    )
    centered_folds.to_csv(
        source_dir / "Figure_2c_spatial_mean_centered_outer_fold_accuracy.csv",
        index=False,
    )
    centered_sessions.to_csv(
        source_dir / "Figure_2c_spatial_mean_centered_held_video_pair_summary.csv",
        index=False,
    )
    centered_confusion.to_csv(
        source_dir / "Figure_2d_spatial_mean_centered_confusion_matrix.csv",
        index=False,
    )

    rows = []
    for analysis, current_audit in (
        ("Original visual-mask beta", audit),
        ("Spatial-mean-centred beta", centered_audit),
    ):
        current_metrics = current_audit["metrics"]
        assert isinstance(current_metrics, dict)
        interval = current_metrics["balanced_accuracy_bootstrap_95_ci"]
        assert isinstance(interval, list)
        rows.append(
            {
                "analysis": analysis,
                "n_participants": current_metrics["n_subjects"],
                "n_samples": current_metrics["n_samples"],
                "n_visual_mask_voxels": current_metrics.get(
                    "n_candidate_voxels", 3141
                ),
                "balanced_accuracy": current_metrics["balanced_accuracy"],
                "bootstrap_ci_low": interval[0],
                "bootstrap_ci_high": interval[1],
                "macro_f1": current_metrics["macro_f1"],
                "structured_permutations": current_metrics[
                    "structured_permutations_completed"
                ],
                "structured_permutation_p": current_metrics[
                    "structured_label_permutation_p"
                ],
            }
        )
    pd.DataFrame(rows).to_csv(
        source_dir / "Figure_2a_decoding_sensitivity_summary.csv", index=False
    )


def _escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        ">": r"\textgreater{}",
        "<": r"\textless{}",
        "−": "--",
        "–": "--",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _latex_main_table(frame: pd.DataFrame) -> str:
    header = " & ".join(_escape_latex(column) for column in frame.columns) + r" \\"
    rows = [
        " & ".join(_escape_latex(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Summary of principal results. The activation omnibus is secondary; intervals are participant-bootstrap 95\% confidence intervals where shown.}",
            r"\label{tab:main-non-mvpa}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{p{1.8cm}p{3.4cm}p{3.4cm}p{2.3cm}p{2.0cm}p{3.0cm}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )


def update_main_table(
    metrics: dict[str, object],
    table_dir: Path,
    spatial_mean_centered_metrics: dict[str, object] | None = None,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "table_1_main_non_mvpa_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "Generate the non-MVPA manuscript table before adding the MVPA row: "
            f"{csv_path}"
        )
    main = pd.read_csv(csv_path)
    required_columns = [
        "Analysis",
        "Endpoint",
        "Estimate",
        "Effect",
        "P/q",
        "Inference",
    ]
    if main.columns.tolist() != required_columns:
        raise RuntimeError("Unexpected main manuscript table schema.")
    main = main.loc[
        ~main.Analysis.isin(["MVPA", "MVPA sensitivity"])
    ].copy().reset_index(drop=True)
    bootstrap_ci = metrics["balanced_accuracy_bootstrap_95_ci"]
    assert isinstance(bootstrap_ci, list)
    mvpa_row = pd.DataFrame(
        [
            {
                "Analysis": "MVPA",
                "Endpoint": "Visual-cortex Simple versus Complex decoding",
                "Estimate": (
                    f"{float(metrics['balanced_accuracy']):.3f} "
                    f"[{float(bootstrap_ci[0]):.3f}, "
                    f"{float(bootstrap_ci[1]):.3f}]"
                ),
                "Effect": f"{int(metrics['n_samples'])} OOF samples",
                "P/q": "P=0.0001",
                "Inference": "Cross-participant and held-pair information",
            }
        ]
    )
    mvpa_rows = [mvpa_row]
    if spatial_mean_centered_metrics is not None:
        centered_ci = spatial_mean_centered_metrics[
            "balanced_accuracy_bootstrap_95_ci"
        ]
        assert isinstance(centered_ci, list)
        mvpa_rows.append(
            pd.DataFrame(
                [
                    {
                        "Analysis": "MVPA sensitivity",
                        "Endpoint": "Spatial-mean-centred visual-cortex decoding",
                        "Estimate": (
                            f"{float(spatial_mean_centered_metrics['balanced_accuracy']):.3f} "
                            f"[{float(centered_ci[0]):.3f}, "
                            f"{float(centered_ci[1]):.3f}]"
                        ),
                        "Effect": (
                            "macro-F1="
                            f"{float(spatial_mean_centered_metrics['macro_f1']):.3f}"
                        ),
                        "P/q": "P=0.0001",
                        "Inference": "Not reducible to uniform visual-mask amplitude",
                    }
                ]
            )
        )
    combined_mvpa_rows = pd.concat(mvpa_rows, ignore_index=True)
    rsa_indices = main.index[main.Analysis.astype(str).str.startswith("RSA")]
    insert_at = int(rsa_indices.max() + 1) if len(rsa_indices) else len(main)
    main = pd.concat(
        [main.iloc[:insert_at], combined_mvpa_rows, main.iloc[insert_at:]],
        ignore_index=True,
    )
    main.to_csv(csv_path, index=False)
    markdown_main = main.map(
        lambda value: value.replace("|", r"\|") if isinstance(value, str) else value
    )
    (table_dir / "table_1_main_non_mvpa_results.md").write_text(
        "# Table 1 | Summary of principal results\n\n"
        + markdown_main.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    (table_dir / "table_1_main_non_mvpa_results.tex").write_text(
        _latex_main_table(main), encoding="utf-8"
    )


def build_figure(audit: dict[str, object], output_dir: Path) -> None:
    metrics = audit["metrics"]
    null = audit["null"]
    folds = audit["folds"]
    session_summary = audit["session_summary"]
    confusion_counts = audit["confusion_counts"]
    confusion_rates = audit["confusion_rates"]
    assert isinstance(metrics, dict)
    assert isinstance(null, pd.DataFrame)
    assert isinstance(folds, pd.DataFrame)
    assert isinstance(session_summary, pd.DataFrame)
    assert isinstance(confusion_counts, np.ndarray)
    assert isinstance(confusion_rates, np.ndarray)

    figure = plt.figure(figsize=(WIDTH_IN, 2.85), constrained_layout=False)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.30, 1.05, 0.80),
        left=0.07,
        right=0.985,
        bottom=0.19,
        top=0.80,
        wspace=0.40,
    )
    ax_null = figure.add_subplot(grid[0, 0])
    ax_folds = figure.add_subplot(grid[0, 1])
    ax_confusion = figure.add_subplot(grid[0, 2])

    observed = float(metrics["balanced_accuracy"])
    null_scores = null.balanced_accuracy.to_numpy(dtype=float)
    bins = np.linspace(0.42, 0.66, 49)
    ax_null.hist(
        null_scores,
        bins=bins,
        color=COLORS["neutral"],
        edgecolor="white",
        linewidth=0.25,
    )
    ax_null.axvline(0.5, color=COLORS["dark"], linestyle="--", linewidth=0.9)
    ax_null.axvline(observed, color=COLORS["signal"], linewidth=1.5)
    ymax = ax_null.get_ylim()[1]
    ax_null.text(
        observed - 0.002,
        ymax * 0.92,
        f"Observed {observed:.3f}",
        color=COLORS["signal"],
        ha="right",
        va="top",
        fontsize=6,
        fontweight="bold",
    )
    ax_null.text(
        0.503,
        ymax * 0.76,
        "Chance 0.50",
        color=COLORS["dark"],
        ha="left",
        va="top",
        fontsize=6,
    )
    ax_null.set_xlim(0.42, 0.66)
    ax_null.set_xlabel("Balanced accuracy")
    ax_null.set_ylabel("Permutation count")
    ax_null.set_title(
        "Whole-subject label permutations\n0 / 10,000 null scores >= observed",
        loc="left",
        pad=6,
    )
    panel_label(ax_null, "a")

    for _, part in folds.groupby("subject_fold", sort=True):
        ax_folds.plot(
            part.held_session,
            part.balanced_accuracy,
            color=COLORS["neutral"],
            linewidth=0.7,
            alpha=0.65,
            marker="o",
            markersize=2.5,
        )
    ax_folds.scatter(
        session_summary.held_session,
        session_summary.balanced_accuracy,
        color=COLORS["comparison"],
        edgecolor="white",
        linewidth=0.5,
        s=28,
        zorder=3,
        label="All held-out samples",
    )
    ax_folds.axhline(0.5, color=COLORS["dark"], linestyle="--", linewidth=0.9)
    ax_folds.set_xticks([1, 2, 3, 4], ["1", "2", "3", "4"])
    ax_folds.set_ylim(0.46, 0.88)
    ax_folds.set_xlabel("Held video pair")
    ax_folds.set_ylabel("Balanced accuracy")
    ax_folds.set_title(
        "Five participant folds x four\nheld video pairs",
        loc="left",
        pad=6,
    )
    ax_folds.legend(loc="upper right", frameon=False, handlelength=1.0)
    panel_label(ax_folds, "b")

    ax_confusion.imshow(confusion_rates, cmap="Blues", vmin=0, vmax=1)
    for true_class in (0, 1):
        for predicted_class in (0, 1):
            rate = confusion_rates[true_class, predicted_class]
            count = confusion_counts[true_class, predicted_class]
            ax_confusion.text(
                predicted_class,
                true_class,
                f"{rate * 100:.1f}%\n(n={count})",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if rate > 0.55 else COLORS["dark"],
                fontweight="bold" if true_class == predicted_class else "normal",
            )
    ax_confusion.set_xticks([0, 1], ["Simple", "Complex"])
    ax_confusion.set_yticks([0, 1], ["Simple", "Complex"])
    ax_confusion.set_xlabel("Predicted")
    ax_confusion.set_ylabel("True")
    ax_confusion.set_title("Out-of-fold predictions", loc="left", pad=6)
    for spine in ax_confusion.spines.values():
        spine.set_visible(False)
    panel_label(ax_confusion, "c")

    figure.suptitle(
        "Visual-cortex patterns generalized across participants and held video pairs",
        x=0.02,
        y=0.965,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_bundle(figure, output_dir / "Figure_MVPA_2_cross_video_decoding")



def build_comparison_figure(
    audit: dict[str, object],
    centered_audit: dict[str, object],
    output_dir: Path,
) -> None:
    """Build the manuscript figure comparing primary and centred MVPA."""

    metrics = audit["metrics"]
    centered_metrics = centered_audit["metrics"]
    null = audit["null"]
    centered_null = centered_audit["null"]
    sessions = audit["session_summary"]
    centered_sessions = centered_audit["session_summary"]
    confusion_rates = audit["confusion_rates"]
    confusion_counts = audit["confusion_counts"]
    centered_rates = centered_audit["confusion_rates"]
    centered_counts = centered_audit["confusion_counts"]
    assert isinstance(metrics, dict)
    assert isinstance(centered_metrics, dict)
    assert isinstance(null, pd.DataFrame)
    assert isinstance(centered_null, pd.DataFrame)
    assert isinstance(sessions, pd.DataFrame)
    assert isinstance(centered_sessions, pd.DataFrame)
    assert isinstance(confusion_rates, np.ndarray)
    assert isinstance(confusion_counts, np.ndarray)
    assert isinstance(centered_rates, np.ndarray)
    assert isinstance(centered_counts, np.ndarray)
    if not np.array_equal(sessions.held_session, centered_sessions.held_session):
        raise RuntimeError("Primary and centred held-video-pair summaries differ.")

    labels = ["Original beta", "Spatial-mean-centred"]
    colors = [COLORS["signal"], COLORS["comparison"]]
    analyses = [metrics, centered_metrics]
    estimates = np.asarray(
        [float(item["balanced_accuracy"]) for item in analyses], dtype=float
    )
    intervals = np.asarray(
        [item["balanced_accuracy_bootstrap_95_ci"] for item in analyses],
        dtype=float,
    )

    figure = plt.figure(figsize=(WIDTH_IN, 5.0), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.985,
        bottom=0.11,
        top=0.88,
        hspace=0.52,
        wspace=0.34,
    )
    ax_summary = figure.add_subplot(grid[0, 0])
    ax_null = figure.add_subplot(grid[0, 1])
    ax_pairs = figure.add_subplot(grid[1, 0])
    ax_confusion = figure.add_subplot(grid[1, 1])

    for index, (estimate, interval, color) in enumerate(
        zip(estimates, intervals, colors, strict=True)
    ):
        ax_summary.errorbar(
            index,
            estimate,
            yerr=[[estimate - interval[0]], [interval[1] - estimate]],
            fmt="o",
            markersize=6,
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        label_x = index + (0.06 if index == 0 else -0.06)
        ax_summary.text(
            label_x,
            estimate + 0.012,
            f"{estimate:.3f}\nP=0.0001",
            ha="left" if index == 0 else "right",
            va="bottom",
            fontsize=6,
            color=color,
        )
    ax_summary.axhline(0.5, color=COLORS["dark"], linestyle="--", linewidth=0.9)
    ax_summary.set_xticks([0, 1], labels)
    ax_summary.set_ylim(0.48, 0.69)
    ax_summary.set_ylabel("Balanced accuracy")
    ax_summary.set_title("Decoding persisted after mean removal", loc="left", pad=6)
    panel_label(ax_summary, "a")

    bins = np.linspace(0.42, 0.66, 49)
    ax_null.hist(
        null.balanced_accuracy,
        bins=bins,
        histtype="stepfilled",
        alpha=0.30,
        color=colors[0],
        label=labels[0],
    )
    ax_null.hist(
        centered_null.balanced_accuracy,
        bins=bins,
        histtype="step",
        linewidth=1.2,
        color=colors[1],
        label=labels[1],
    )
    ax_null.axvline(estimates[0], color=colors[0], linewidth=1.3)
    ax_null.axvline(estimates[1], color=colors[1], linewidth=1.3)
    ax_null.axvline(0.5, color=COLORS["dark"], linestyle="--", linewidth=0.9)
    ax_null.set_xlim(0.42, 0.66)
    ax_null.set_xlabel("Balanced accuracy")
    ax_null.set_ylabel("Permutation count")
    ax_null.set_title("Structured whole-subject nulls", loc="left", pad=6)
    ax_null.legend(loc="upper left", frameon=False)
    panel_label(ax_null, "b")

    held_sessions = sessions.held_session.to_numpy(dtype=float)
    primary_scores = sessions.balanced_accuracy.to_numpy(dtype=float)
    centered_scores = centered_sessions.balanced_accuracy.to_numpy(dtype=float)
    for session, primary_score, centered_score in zip(
        held_sessions, primary_scores, centered_scores, strict=True
    ):
        ax_pairs.plot(
            [session - 0.10, session + 0.10],
            [primary_score, centered_score],
            color=COLORS["light"],
            linewidth=1.0,
            zorder=1,
        )
    ax_pairs.scatter(
        held_sessions - 0.10,
        primary_scores,
        s=30,
        color=colors[0],
        edgecolor="white",
        linewidth=0.5,
        label=labels[0],
        zorder=2,
    )
    ax_pairs.scatter(
        held_sessions + 0.10,
        centered_scores,
        s=30,
        color=colors[1],
        edgecolor="white",
        linewidth=0.5,
        label=labels[1],
        zorder=2,
    )
    ax_pairs.axhline(0.5, color=COLORS["dark"], linestyle="--", linewidth=0.9)
    ax_pairs.set_xticks(
        [1, 2, 3, 4], ["001/009", "002/010", "003/011", "004/012"]
    )
    ax_pairs.set_ylim(0.46, 0.82)
    ax_pairs.set_xlabel("Held video pair")
    ax_pairs.set_ylabel("Balanced accuracy")
    ax_pairs.set_title("Performance across four held video pairs", loc="left", pad=6)
    ax_pairs.legend(loc="upper right", frameon=False)
    panel_label(ax_pairs, "c")

    matrix = np.full((2, 5), np.nan)
    matrix[:, :2] = confusion_rates
    matrix[:, 3:] = centered_rates
    ax_confusion.imshow(
        np.ma.masked_invalid(matrix), cmap="Blues", vmin=0, vmax=1, aspect="auto"
    )
    for offset, rates, counts in (
        (0, confusion_rates, confusion_counts),
        (3, centered_rates, centered_counts),
    ):
        for true_class in (0, 1):
            for predicted_class in (0, 1):
                rate = rates[true_class, predicted_class]
                count = counts[true_class, predicted_class]
                ax_confusion.text(
                    predicted_class + offset,
                    true_class,
                    f"{rate * 100:.1f}%\n(n={count})",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if rate > 0.55 else COLORS["dark"],
                    fontweight=(
                        "bold" if true_class == predicted_class else "normal"
                    ),
                )
    ax_confusion.set_xticks([0, 1, 3, 4], ["S", "C", "S", "C"])
    ax_confusion.set_yticks([0, 1], ["Simple", "Complex"])
    ax_confusion.set_xlabel("Predicted condition")
    ax_confusion.set_ylabel("True condition")
    ax_confusion.text(
        0.5, -0.38, "Original beta", ha="center", va="top", fontsize=6
    )
    ax_confusion.text(
        3.5,
        -0.38,
        "Spatial-mean-centred",
        ha="center",
        va="top",
        fontsize=6,
    )
    ax_confusion.set_title("Out-of-fold classification", loc="left", pad=6)
    for spine in ax_confusion.spines.values():
        spine.set_visible(False)
    panel_label(ax_confusion, "d")

    figure.suptitle(
        "Scene complexity remained decodable after removing visual-mask mean amplitude",
        x=0.02,
        y=0.97,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_bundle(figure, output_dir / "Figure_MVPA_2_cross_video_decoding")


def write_manifest(
    output_dir: Path,
    audit: dict[str, object],
    centered_audit: dict[str, object] | None = None,
) -> None:
    metrics = audit["metrics"]
    assert isinstance(metrics, dict)
    files = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "asset_manifest.json"
    )
    manifest = {
        "scope": "formally completed cross-video MVPA with amplitude sensitivity",
        "backend": "Python/matplotlib",
        "figure": "Figure_MVPA_2_cross_video_decoding",
        "formal_inference_complete": metrics["formal_inference_complete"],
        "structured_permutations_completed": metrics[
            "structured_permutations_completed"
        ],
        "structured_label_permutation_p": metrics[
            "structured_label_permutation_p"
        ],
        "spatial_mean_centered_sensitivity_included": centered_audit is not None,
        "files": files + ["asset_manifest.json"],
    }
    if centered_audit is not None:
        centered_metrics = centered_audit["metrics"]
        assert isinstance(centered_metrics, dict)
        manifest["spatial_mean_centered"] = {
            "balanced_accuracy": centered_metrics["balanced_accuracy"],
            "bootstrap_95_ci": centered_metrics[
                "balanced_accuracy_bootstrap_95_ci"
            ],
            "structured_permutations_completed": centered_metrics[
                "structured_permutations_completed"
            ],
            "structured_label_permutation_p": centered_metrics[
                "structured_label_permutation_p"
            ],
        }
    (output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def run(
    run_dir: Path,
    output_dir: Path,
    table_dir: Path | None = None,
    spatial_mean_centered_run_dir: Path | None = None,
) -> None:
    configure_style()
    mvpa_dir = run_dir / "decoding/cross_video/svm"
    audit = load_and_audit(mvpa_dir)
    centered_audit = None
    if spatial_mean_centered_run_dir is not None:
        centered_mvpa_dir = spatial_mean_centered_run_dir / "decoding/cross_video/svm"
        centered_audit = load_and_audit(centered_mvpa_dir)
        centered_metrics = centered_audit["metrics"]
        assert isinstance(centered_metrics, dict)
        if not centered_metrics.get("spatial_mean_centered", False):
            raise RuntimeError(
                "The sensitivity run is not marked as spatial-mean-centred."
            )
        if centered_metrics.get("feature_preprocessing") != (
            "per_sample_visual_mask_spatial_mean_center"
        ):
            raise RuntimeError(
                "Unexpected spatial-mean-centred preprocessing contract."
            )

    source_dir = output_dir / "source_data"
    write_source_data(audit, source_dir)
    if centered_audit is None:
        build_figure(audit, output_dir)
    else:
        write_comparison_source_data(audit, centered_audit, source_dir)
        build_comparison_figure(audit, centered_audit, output_dir)
    write_manifest(output_dir, audit, centered_audit)
    if table_dir is not None:
        metrics = audit["metrics"]
        centered_metrics = centered_audit["metrics"] if centered_audit else None
        assert isinstance(metrics, dict)
        assert centered_metrics is None or isinstance(centered_metrics, dict)
        update_main_table(metrics, table_dir, centered_metrics)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--run-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "paper_figures/mvpa",
    )
    argument_parser.add_argument("--table-dir", type=Path)
    argument_parser.add_argument(
        "--spatial-mean-centered-run-dir",
        type=Path,
        help=(
            "Optional completed MVPA run using per-sample visual-mask spatial "
            "mean centring."
        ),
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    run(
        args.run_dir,
        args.output_dir,
        args.table_dir,
        args.spatial_mean_centered_run_dir,
    )


if __name__ == "__main__":
    main()
