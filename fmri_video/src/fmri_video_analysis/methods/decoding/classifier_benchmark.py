#!/usr/bin/env python3
"""Leakage-controlled three-class benchmark across subjects and held videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
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


def estimators(seed: int, n_jobs: int, k: int):
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Classifier benchmarking requires xgboost; install the project dependencies."
        ) from exc

    scaled = lambda model: Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("model", model),
        ]
    )
    selected = lambda model: Pipeline(
        [("select", SelectKBest(f_classif, k=k)), ("model", model)]
    )
    return {
        "linear_svm": scaled(
            LinearSVC(
                C=0.1, class_weight="balanced", dual="auto", max_iter=50000, tol=1e-3
            )
        ),
        "mlp_64": scaled(
            MLPClassifier(
                hidden_layer_sizes=(64,),
                alpha=1e-3,
                early_stopping=True,
                max_iter=500,
                random_state=seed,
            )
        ),
        "mlp_128_64": scaled(
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                alpha=1e-3,
                early_stopping=True,
                max_iter=500,
                random_state=seed,
            )
        ),
        "mlp_256_128_32": scaled(
            MLPClassifier(
                hidden_layer_sizes=(256, 128, 32),
                alpha=1e-3,
                early_stopping=True,
                max_iter=500,
                random_state=seed,
            )
        ),
        "random_forest": selected(
            RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
                random_state=seed,
            )
        ),
        "xgboost": selected(
            XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_jobs=n_jobs,
                random_state=seed,
            )
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
    X = X_all[:, columns["visual"]]
    k = min(args.k_features, X.shape[1])
    y = table.class_id.to_numpy()
    subjects = table.subject_id.to_numpy()
    sessions = table.session.to_numpy()
    fold_ids, mapping = subject_folds(subjects, args.subject_folds, args.seed)
    model_set = estimators(args.seed, args.n_jobs, k)
    predictions = {name: np.full(len(table), -1, dtype=int) for name in model_set}
    fold_rows = []
    for subject_fold in range(args.subject_folds):
        for held_session in (1, 2, 3, 4):
            train = (fold_ids != subject_fold) & (sessions != held_session)
            test = (fold_ids == subject_fold) & (sessions == held_session)
            for name, model in model_set.items():
                model.fit(X[train], y[train])
                pred = model.predict(X[test])
                predictions[name][test] = pred
                fold_rows.append(
                    {
                        "classifier": name,
                        "subject_fold": subject_fold,
                        "held_session": held_session,
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "balanced_accuracy": balanced_accuracy_score(y[test], pred),
                        "macro_f1": f1_score(y[test], pred, average="macro"),
                    }
                )
                print(
                    f"fold {subject_fold + 1}/5 session {held_session}: {name} "
                    f"BA={fold_rows[-1]['balanced_accuracy']:.3f}",
                    flush=True,
                )

    pd.DataFrame(fold_rows).to_csv(
        output / "classifier_outer_fold_metrics.csv", index=False
    )
    subject_rows, summary_rows = [], []
    for name, pred in predictions.items():
        if np.any(pred < 0):
            raise RuntimeError(f"Incomplete predictions: {name}")
        part = table[
            ["subject_id", "session", "condition", "video_id", "beta_path"]
        ].copy()
        part["true_class"] = y
        part["predicted_class"] = pred
        part.to_csv(output / f"{name}_oof_predictions.csv", index=False)
        values = []
        for subject, indices in table.groupby("subject_id").groups.items():
            value = balanced_accuracy_score(y[indices], pred[indices])
            values.append(value)
            subject_rows.append(
                {"classifier": name, "subject_id": subject, "balanced_accuracy": value}
            )
        values = np.asarray(values)
        rng = np.random.default_rng(args.seed)
        boot = rng.choice(values, (args.bootstrap, len(values)), replace=True).mean(
            axis=1
        )
        summary_rows.append(
            {
                "classifier": name,
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro"),
                "bootstrap_ci_low": np.quantile(boot, 0.025),
                "bootstrap_ci_high": np.quantile(boot, 0.975),
                "cohens_dz_vs_chance": cohens_dz(values - 1 / 3),
                "p_signflip_vs_chance": paired_sign_flip_pvalue(
                    values - 1 / 3, args.n_permutations, args.seed
                ),
            }
        )
    subject_table = pd.DataFrame(subject_rows)
    summary = pd.DataFrame(summary_rows)
    summary["q_fdr_vs_chance"] = bh_fdr(summary.p_signflip_vs_chance)
    subject_table.to_csv(output / "classifier_subject_metrics.csv", index=False)
    summary.to_csv(output / "classifier_summary.csv", index=False)

    wide = subject_table.pivot(
        index="subject_id", columns="classifier", values="balanced_accuracy"
    )
    comparison_rows = []
    for name in model_set:
        if name == "linear_svm":
            continue
        difference = wide[name] - wide["linear_svm"]
        comparison_rows.append(
            {
                "contrast": f"{name}_minus_linear_svm",
                "mean_difference": difference.mean(),
                "cohens_dz": cohens_dz(difference.to_numpy()),
                "p_signflip": paired_sign_flip_pvalue(
                    difference.to_numpy(), args.n_permutations, args.seed
                ),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["q_fdr"] = bh_fdr(comparisons.p_signflip)
    comparisons.to_csv(output / "classifier_vs_svm_comparisons.csv", index=False)

    order = summary.sort_values(
        "balanced_accuracy", ascending=False
    ).classifier.tolist()
    figure, axis = plt.subplots(figsize=(10, 5.8))
    rng = np.random.default_rng(19)
    for index, name in enumerate(order):
        values = wide[name].to_numpy()
        row = summary.set_index("classifier").loc[name]
        axis.scatter(
            index + rng.normal(0, 0.055, len(values)),
            values,
            s=12,
            alpha=0.25,
            color="#2E86DE",
            edgecolors="none",
        )
        axis.errorbar(
            index,
            row.balanced_accuracy,
            yerr=[
                [row.balanced_accuracy - row.bootstrap_ci_low],
                [row.bootstrap_ci_high - row.balanced_accuracy],
            ],
            fmt="o",
            color="#C0392B",
            capsize=4,
            markersize=8,
        )
    axis.axhline(1 / 3, color="black", linestyle="--", label="Chance")
    axis.set_xticks(range(len(order)), [x.replace("_", "\n") for x in order])
    axis.set_ylabel("Three-class balanced accuracy")
    axis.set_title(
        "Classifier sensitivity: cross-subject + cross-video visual ROI",
        fontweight="bold",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "classifier_benchmark.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "classifier_benchmark.pdf", bbox_inches="tight")
    plt.close(figure)
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument(
        "--output-dir", default=str(RESULTS_DIR / "decoding/classifier_benchmark")
    )
    p.add_argument("--k-features", type=int, default=300)
    p.add_argument("--subject-folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260801)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
