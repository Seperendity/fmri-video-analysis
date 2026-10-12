#!/usr/bin/env python3
"""Three-class network MVPA generalising jointly across subjects and videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from fmri_video_analysis.core.config import (
    FIRST_LEVEL_DIR,
    RESULTS_DIR,
    build_four_session_manifest,
)
from fmri_video_analysis.methods.decoding.cross_video import subject_folds
from fmri_video_analysis.core.data_loading import load_task_network_features
from fmri_video_analysis.core.statistics import (
    bh_fdr,
    cohens_dz,
    paired_sign_flip_pvalue,
)


CLASS_MAP = {"Simple": 0, "Complex": 1, "Control": 2}
PRIMARY_NETWORKS = ("visual", "dorsal_attention", "salience_attention", "control")
ALL_NETWORKS = PRIMARY_NETWORKS + ("task_combined",)


def model_grid(n_features: int):
    k_values = sorted(set(min(n_features, k) for k in (100, 300, 500)))
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif)),
            (
                "model",
                LinearSVC(
                    class_weight="balanced", dual="auto", max_iter=50000, tol=1e-3
                ),
            ),
        ]
    )
    # C=0.1 is prespecified from the completed binary nested-CV analysis;
    # only the number of selected voxels is tuned within each training fold.
    return pipeline, {"select__k": k_values, "model__C": [0.1]}


def run_network(
    network: str,
    X: np.ndarray,
    table: pd.DataFrame,
    fold_ids: np.ndarray,
    args: argparse.Namespace,
    output: Path,
):
    y = table.class_id.to_numpy()
    subjects = table.subject_id.to_numpy()
    sessions = table.session.to_numpy()
    predictions = np.full(len(table), -1, dtype=int)
    selected_counts = []
    fold_rows = []
    for subject_fold in range(args.subject_folds):
        for held_session in (1, 2, 3, 4):
            train = (fold_ids != subject_fold) & (sessions != held_session)
            test = (fold_ids == subject_fold) & (sessions == held_session)
            pipeline, grid = model_grid(X.shape[1])
            inner = GroupKFold(n_splits=3)
            search = GridSearchCV(
                pipeline,
                grid,
                scoring="balanced_accuracy",
                cv=inner,
                n_jobs=args.n_jobs,
                refit=True,
                error_score="raise",
            )
            search.fit(X[train], y[train], groups=subjects[train])
            prediction = search.predict(X[test])
            predictions[test] = prediction
            selected_counts.append(int(search.best_params_["select__k"]))
            fold_rows.append(
                {
                    "network": network,
                    "subject_fold": subject_fold,
                    "held_session": held_session,
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "balanced_accuracy": balanced_accuracy_score(y[test], prediction),
                    "macro_f1": f1_score(y[test], prediction, average="macro"),
                    "best_k": int(search.best_params_["select__k"]),
                    "best_C": float(search.best_params_["model__C"]),
                }
            )
            print(
                f"{network}: subject fold {subject_fold + 1}/{args.subject_folds}, "
                f"session {held_session}, BA={fold_rows[-1]['balanced_accuracy']:.3f}",
                flush=True,
            )
    if np.any(predictions < 0):
        raise RuntimeError(f"Incomplete out-of-fold predictions for {network}")

    prediction_table = table[
        ["subject_id", "session", "condition", "video_id", "beta_path"]
    ].copy()
    prediction_table["true_class"] = y
    prediction_table["predicted_class"] = predictions
    prediction_table["predicted_condition"] = [
        ("Simple", "Complex", "Control")[value] for value in predictions
    ]
    prediction_table.to_csv(output / f"{network}_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / f"{network}_outer_folds.csv", index=False)

    subject_rows = []
    for subject, part in prediction_table.groupby("subject_id"):
        subject_rows.append(
            {
                "network": network,
                "subject_id": subject,
                "balanced_accuracy": balanced_accuracy_score(
                    part.true_class, part.predicted_class
                ),
            }
        )
    subject_table = pd.DataFrame(subject_rows)
    subject_table.to_csv(output / f"{network}_subject_metrics.csv", index=False)
    subject_values = subject_table.balanced_accuracy.to_numpy()
    rng = np.random.default_rng(args.seed)
    boot = rng.choice(
        subject_values, size=(args.bootstrap, len(subject_values)), replace=True
    ).mean(axis=1)
    cm = confusion_matrix(y, predictions, labels=[0, 1, 2], normalize="true")
    np.savetxt(output / f"{network}_confusion_matrix.csv", cm, delimiter=",")
    return (
        {
            "network": network,
            "n_voxels": X.shape[1],
            "n_samples": len(y),
            "balanced_accuracy": balanced_accuracy_score(y, predictions),
            "macro_f1": f1_score(y, predictions, average="macro"),
            "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
            "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
            "p_signflip_vs_chance": paired_sign_flip_pvalue(
                subject_values - 1 / 3, args.n_permutations, args.seed
            ),
            "cohens_dz_vs_chance": cohens_dz(subject_values - 1 / 3),
            "mean_selected_k": float(np.mean(selected_counts)),
        },
        subject_table,
        cm,
    )


def plot_results(
    summary: pd.DataFrame, subjects: pd.DataFrame, cms: dict, output: Path
):
    order = list(ALL_NETWORKS)
    colors = ["#2E86DE", "#8E44AD", "#E67E22", "#27AE60", "#34495E"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axis = axes[0]
    rng = np.random.default_rng(17)
    for index, (network, color) in enumerate(zip(order, colors)):
        values = subjects.loc[
            subjects.network == network, "balanced_accuracy"
        ].to_numpy()
        axis.scatter(
            index + rng.normal(0, 0.055, len(values)),
            values,
            s=12,
            alpha=0.25,
            color=color,
            edgecolors="none",
        )
        row = summary[summary.network == network].iloc[0]
        axis.errorbar(
            index,
            row.balanced_accuracy,
            yerr=[
                [row.balanced_accuracy - row.bootstrap_ci_low],
                [row.bootstrap_ci_high - row.balanced_accuracy],
            ],
            fmt="o",
            markersize=9,
            color=color,
            capsize=4,
            linewidth=2.2,
        )
    axis.axhline(1 / 3, color="black", linestyle="--", linewidth=1.4, label="Chance")
    axis.set_xticks(range(len(order)), [x.replace("_", "\n") for x in order])
    axis.set_ylabel("Three-class balanced accuracy")
    axis.set_title("A. Cross-subject + cross-video network decoding", fontweight="bold")
    axis.set_ylim(0.15, 1.0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)

    visual_cm = cms["visual"]
    image = axes[1].imshow(visual_cm, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[1].text(
                j,
                i,
                f"{visual_cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if visual_cm[i, j] > 0.55 else "black",
            )
    axes[1].set_xticks(range(3), ["Simple", "Complex", "Control"])
    axes[1].set_yticks(range(3), ["Simple", "Complex", "Control"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("B. Visual-network confusion matrix", fontweight="bold")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output / "threeclass_network_mvpa.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "threeclass_network_mvpa.pdf", bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = build_four_session_manifest(Path(args.first_level_dir)).copy()
    table = table.sort_values(["subject_id", "session", "condition"]).reset_index(
        drop=True
    )
    table["class_id"] = table.condition.map(CLASS_MAP).astype(int)
    table.to_csv(output / "threeclass_labels.csv", index=False)
    X_all, _, _, columns = load_task_network_features(table)
    fold_ids, mapping = subject_folds(
        table.subject_id.to_numpy(), args.subject_folds, args.seed
    )
    (output / "subject_fold_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    summaries, subject_tables, cms = [], [], {}
    for network in ALL_NETWORKS:
        X = X_all[:, columns[network]]
        summary, subject_table, cm = run_network(
            network, X, table, fold_ids, args, output
        )
        summaries.append(summary)
        subject_tables.append(subject_table)
        cms[network] = cm
    summary = pd.DataFrame(summaries)
    primary = summary.network.isin(PRIMARY_NETWORKS)
    summary.loc[primary, "q_fdr_primary_networks"] = bh_fdr(
        summary.loc[primary, "p_signflip_vs_chance"]
    )
    subject_metrics = pd.concat(subject_tables, ignore_index=True)
    visual = subject_metrics[subject_metrics.network == "visual"].set_index(
        "subject_id"
    )
    comparison_rows = []
    for network in PRIMARY_NETWORKS[1:]:
        other = subject_metrics[subject_metrics.network == network].set_index(
            "subject_id"
        )
        diff = visual.balanced_accuracy - other.balanced_accuracy
        comparison_rows.append(
            {
                "contrast": f"visual_minus_{network}",
                "mean_difference": diff.mean(),
                "cohens_dz": cohens_dz(diff.to_numpy()),
                "p_signflip": paired_sign_flip_pvalue(
                    diff.to_numpy(), args.n_permutations, args.seed
                ),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["q_fdr"] = bh_fdr(comparisons.p_signflip)
    summary.to_csv(output / "threeclass_network_summary.csv", index=False)
    subject_metrics.to_csv(
        output / "threeclass_network_subject_metrics.csv", index=False
    )
    comparisons.to_csv(output / "visual_network_comparisons.csv", index=False)
    plot_results(summary, subject_metrics, cms, output)
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument(
        "--output-dir", default=str(RESULTS_DIR / "decoding/three_class_networks")
    )
    p.add_argument("--subject-folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260731)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
