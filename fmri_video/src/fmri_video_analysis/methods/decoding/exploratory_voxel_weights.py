#!/usr/bin/env python3
"""Exploratory non-independent voxel search and descriptive SVM weight anatomy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from fmri_video_analysis.core.config import (
    ATLAS_DIR,
    FIRST_LEVEL_DIR,
    RESULTS_DIR,
    ROI_REFERENCE_DIR,
    build_four_session_manifest,
)
from fmri_video_analysis.methods.decoding.cross_video import load_visual_features


def classifier():
    return make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=0.1, class_weight="balanced", dual="auto", max_iter=50000, tol=1e-3
        ),
    )


def parcel_labels() -> dict[int, str]:
    path = ATLAS_DIR / "schaefer_2018/Schaefer2018_200Parcels_17Networks_order.txt"
    labels = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 1], names=["id", "name"]
    )
    return dict(zip(labels.id.astype(int), labels.name))


def save_map(values: np.ndarray, flat_indices: np.ndarray, reference, path: Path):
    volume = np.zeros(np.prod(reference.shape), dtype=np.float32)
    volume[flat_indices] = values.astype(np.float32)
    nib.save(
        nib.Nifti1Image(
            volume.reshape(reference.shape), reference.affine, reference.header
        ),
        path,
    )


def run(args: argparse.Namespace):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = build_four_session_manifest(Path(args.first_level_dir))
    table = table[table.condition.isin(["Simple", "Complex"])].copy()
    table = table.sort_values(["subject_id", "session", "class_id"]).reset_index(
        drop=True
    )
    mask_path = (
        ROI_REFERENCE_DIR / "roi_masks/schaefer200_17net_visual_commonmask.nii.gz"
    )
    X, reference, flat_indices = load_visual_features(table, mask_path)
    y = table.class_id.to_numpy()
    groups = table.subject_id.to_numpy()

    # Deliberately non-independent: labels from the full dataset determine both
    # candidate order and the accept/reject trajectory, and the same CV is reported.
    f_values, _ = f_classif(X, y)
    ranking = np.argsort(np.nan_to_num(f_values, nan=-np.inf))[::-1]
    candidates = ranking[: min(args.candidate_voxels, X.shape[1])]
    cv = GroupKFold(n_splits=5)
    selected, best = [], 0.5
    rows = []
    for step, candidate in enumerate(candidates, start=1):
        proposed = selected + [int(candidate)]
        score = float(
            cross_val_score(
                classifier(),
                X[:, proposed],
                y,
                groups=groups,
                cv=cv,
                scoring="balanced_accuracy",
                n_jobs=args.n_jobs,
            ).mean()
        )
        accepted = score > best + args.minimum_improvement
        if accepted:
            selected, best = proposed, score
        rows.append(
            {
                "step": step,
                "candidate_column": int(candidate),
                "candidate_flat_index": int(flat_indices[candidate]),
                "candidate_full_data_F": float(f_values[candidate]),
                "proposed_cv_balanced_accuracy": score,
                "accepted": accepted,
                "retained_voxels": len(selected),
                "best_reused_cv_balanced_accuracy": best,
            }
        )
    trajectory = pd.DataFrame(rows)
    trajectory.to_csv(
        output / "nonindependent_forward_search_trajectory.csv", index=False
    )
    final = classifier().fit(X[:, selected], y)
    apparent = balanced_accuracy_score(y, final.predict(X[:, selected]))
    pd.DataFrame(
        [
            {
                "candidate_voxels": len(candidates),
                "selected_voxels": len(selected),
                "reused_selection_cv_balanced_accuracy": best,
                "resubstitution_balanced_accuracy": apparent,
                "independent_test_available": False,
                "valid_for_confirmatory_inference": False,
            }
        ]
    ).to_csv(output / "nonindependent_forward_search_summary.csv", index=False)

    # Fit a full-data SVM for descriptive raw weights and a Haufe-style pattern.
    k = min(args.weight_voxels, X.shape[1])
    selected_weight = ranking[:k]
    scaler = StandardScaler().fit(X[:, selected_weight])
    Z = scaler.transform(X[:, selected_weight])
    svm = LinearSVC(
        C=0.1, class_weight="balanced", dual="auto", max_iter=50000, tol=1e-3
    ).fit(Z, y)
    raw_standardized = svm.coef_.ravel()
    raw_beta_units = raw_standardized / scaler.scale_
    decision = svm.decision_function(Z)
    haufe = (Z.T @ (decision - decision.mean())) / (len(decision) - 1)
    raw_all = np.zeros(X.shape[1])
    raw_all[selected_weight] = raw_beta_units
    raw_std_all = np.zeros(X.shape[1])
    raw_std_all[selected_weight] = raw_standardized
    haufe_all = np.zeros(X.shape[1])
    haufe_all[selected_weight] = haufe
    save_map(
        raw_all, flat_indices, reference, output / "svm_raw_weight_beta_units.nii.gz"
    )
    save_map(
        raw_std_all,
        flat_indices,
        reference,
        output / "svm_raw_weight_standardized.nii.gz",
    )
    save_map(haufe_all, flat_indices, reference, output / "svm_haufe_pattern.nii.gz")

    atlas_img = nib.load(
        str(
            ROI_REFERENCE_DIR
            / "roi_masks/schaefer200_17net_2mm_resampled_to_beta.nii.gz"
        )
    )
    atlas = np.asarray(atlas_img.dataobj).ravel()[flat_indices].astype(int)
    names = parcel_labels()
    parcel_rows = []
    for parcel in sorted(set(atlas) - {0}):
        keep = atlas == parcel
        parcel_rows.append(
            {
                "parcel_id": parcel,
                "parcel_name": names.get(parcel, f"parcel_{parcel}"),
                "n_visual_voxels": int(keep.sum()),
                "mean_raw_weight": raw_all[keep].mean(),
                "mean_abs_raw_weight": np.abs(raw_all[keep]).mean(),
                "mean_haufe_pattern": haufe_all[keep].mean(),
                "mean_abs_haufe_pattern": np.abs(haufe_all[keep]).mean(),
            }
        )
    parcels = pd.DataFrame(parcel_rows).sort_values(
        "mean_abs_raw_weight", ascending=False
    )
    parcels.to_csv(output / "svm_weight_parcel_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(
        trajectory.step,
        trajectory.best_reused_cv_balanced_accuracy,
        color="#8E44AD",
        linewidth=2,
    )
    accepted_rows = trajectory[trajectory.accepted]
    axes[0].scatter(
        accepted_rows.step,
        accepted_rows.best_reused_cv_balanced_accuracy,
        color="#E67E22",
        s=26,
    )
    axes[0].axhline(0.5, color="black", linestyle="--")
    axes[0].set_xlabel("Full-data-ranked candidate step")
    axes[0].set_ylabel("Reused 5-fold subject-CV BA")
    axes[0].set_title("A. Non-independent forward search", fontweight="bold")
    top = parcels.head(12).iloc[::-1]
    axes[1].barh(
        top.parcel_name.str.replace("17Networks_", "", regex=False),
        top.mean_abs_raw_weight,
        color="#2E86DE",
    )
    axes[1].set_xlabel("Mean absolute raw SVM weight (beta units)")
    axes[1].set_title("B. Descriptive parcel ranking", fontweight="bold")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Exploratory voxel selection and SVM weight anatomy",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(
        output / "exploratory_voxel_weights.png", dpi=300, bbox_inches="tight"
    )
    figure.savefig(output / "exploratory_voxel_weights.pdf", bbox_inches="tight")
    plt.close(figure)
    print(
        pd.read_csv(output / "nonindependent_forward_search_summary.csv").to_string(
            index=False
        )
    )
    print(parcels.head(12).to_string(index=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument(
        "--output-dir", default=str(RESULTS_DIR / "decoding/exploratory_voxel_weights")
    )
    p.add_argument("--candidate-voxels", type=int, default=100)
    p.add_argument("--weight-voxels", type=int, default=300)
    p.add_argument("--minimum-improvement", type=float, default=1e-4)
    p.add_argument("--n-jobs", type=int, default=8)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
