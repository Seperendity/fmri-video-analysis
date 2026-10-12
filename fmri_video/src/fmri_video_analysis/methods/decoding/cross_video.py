"""MVPA validation that simultaneously generalises across subjects and videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from fmri_video_analysis.core.config import (
    FIRST_LEVEL_DIR,
    RESULTS_DIR,
    ROI_REFERENCE_DIR,
    build_four_session_manifest,
)
from fmri_video_analysis.core.provenance import atomic_write_json, manifest_differences
from fmri_video_analysis.core.statistics import paired_sign_flip_pvalue
from fmri_video_analysis.core.validation import validate_nifti_geometry


def load_visual_features(
    manifest: pd.DataFrame, mask_path: Path
) -> tuple[np.ndarray, nib.Nifti1Image, np.ndarray]:
    mask_img = nib.load(str(mask_path))
    mask = np.asarray(mask_img.dataobj) > 0
    indices = np.flatnonzero(mask.ravel())
    features = np.empty((len(manifest), len(indices)), dtype=np.float32)
    for row_index, path in enumerate(manifest.beta_path):
        img = nib.load(path)
        validate_nifti_geometry(
            mask_img,
            img,
            context=f"MVPA mask and beta {path}",
        )
        features[row_index] = np.asarray(img.dataobj, dtype=np.float32).ravel()[indices]
        if (row_index + 1) % 100 == 0:
            print(f"Loaded {row_index + 1}/{len(manifest)} beta maps", flush=True)
    finite = np.all(np.isfinite(features), axis=0)
    variable = features.std(axis=0) > 1e-8
    keep = finite & variable
    return features[:, keep], mask_img, indices[keep]


def spatial_mean_center_features(features: np.ndarray) -> np.ndarray:
    """Remove each sample's mean across the candidate visual-mask voxels."""

    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("features must be a non-empty samples-by-voxels matrix.")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must be finite before spatial mean centering.")
    centered = features - features.mean(axis=1, keepdims=True)
    return centered.astype(features.dtype, copy=False)


def subject_folds(subject_ids: np.ndarray, n_folds: int, seed: int):
    unique = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    mapping = {}
    for fold, subjects in enumerate(np.array_split(shuffled, n_folds)):
        mapping.update({int(s): fold for s in subjects})
    return np.array([mapping[int(s)] for s in subject_ids]), mapping


def svm_search(n_features: int, seed: int):
    k_values = sorted({min(n_features, k) for k in (100, 300, 500)})
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif)),
            (
                "model",
                LinearSVC(
                    class_weight="balanced",
                    dual="auto",
                    max_iter=200000,
                    random_state=seed,
                    tol=1e-3,
                ),
            ),
        ]
    )
    grid = {"select__k": k_values, "model__C": [0.1, 1.0]}
    return pipeline, grid


def mlp_search(n_features: int, seed: int):
    k_values = sorted({min(n_features, k) for k in (100, 300, 500)})
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif)),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    alpha=1e-3,
                    early_stopping=True,
                    max_iter=400,
                    random_state=seed,
                ),
            ),
        ]
    )
    return pipeline, {"select__k": k_values}


def bootstrap_subject_metric(
    table: pd.DataFrame, metric, n_bootstrap: int, seed: int
) -> tuple[float, float]:
    # Resample the independent unit (subject), not individual beta maps.
    # With the balanced four-simple/four-complex design, the mean of the
    # subject-level balanced accuracies is the corresponding group estimate.
    subject_values = np.asarray(
        [
            metric(part.true_class, part.predicted_class)
            for _, part in table.groupby("subject_id", sort=True)
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        subject_values,
        size=(n_bootstrap, len(subject_values)),
        replace=True,
    )
    values = sampled.mean(axis=1)
    return tuple(np.quantile(values, [0.025, 0.975]))


def subjectwise_label_permutations(
    labels: np.ndarray,
    subject_ids: np.ndarray,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Swap both condition labels as a unit within each subject."""

    labels = np.asarray(labels, dtype=np.int8)
    unique_subjects, subject_index = np.unique(subject_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    swaps = rng.integers(
        0,
        2,
        size=(n_permutations, len(unique_subjects)),
        dtype=np.int8,
    )
    return np.where(swaps[:, subject_index] == 1, 1 - labels, labels)


def _permuted_outer_cv_score(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    sessions: np.ndarray,
    sample_subject_fold: np.ndarray,
    n_subject_folds: int,
    seed: int,
) -> float:
    predictions = np.full(len(labels), -1, dtype=np.int8)
    for subject_fold in range(n_subject_folds):
        for held_session in sorted(np.unique(sessions)):
            test = (sample_subject_fold == subject_fold) & (sessions == held_session)
            train = (sample_subject_fold != subject_fold) & (sessions != held_session)
            estimator, grid = svm_search(X.shape[1], seed)
            inner = GroupKFold(n_splits=3)
            search = GridSearchCV(
                estimator,
                grid,
                scoring="balanced_accuracy",
                cv=inner.split(X[train], labels[train], groups=subject_ids[train]),
                n_jobs=1,
                refit=True,
            )
            search.fit(X[train], labels[train])
            predictions[test] = search.predict(X[test])
    if np.any(predictions < 0):
        raise RuntimeError("Permutation left samples without predictions.")
    return float(balanced_accuracy_score(labels, predictions))


def structured_label_permutation_test(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    sessions: np.ndarray,
    sample_subject_fold: np.ndarray,
    n_subject_folds: int,
    observed_score: float,
    n_permutations: int,
    n_jobs: int,
    seed: int,
    checkpoint_path: Path,
    batch_size: int,
    force: bool = False,
    feature_preprocessing: str = "raw_visual_mask_beta",
) -> tuple[np.ndarray, float]:
    """Refit the complete nested CV under subject-wise label swaps.

    Results are committed after each deterministic batch so a long formal run
    can resume without changing the generated permutations or repeating work.
    """

    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero.")
    permuted_labels = subjectwise_label_permutations(
        labels, subject_ids, n_permutations, seed
    )

    def digest(values: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(values)
        return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()

    config = {
        "schema_version": 2,
        "seed": seed,
        "n_subject_folds": n_subject_folds,
        "x_shape": list(X.shape),
        "x_sha256": digest(X),
        "labels_sha256": digest(labels),
        "subject_ids_sha256": digest(subject_ids),
        "sessions_sha256": digest(sessions),
        "sample_subject_fold_sha256": digest(sample_subject_fold),
        "feature_preprocessing": feature_preprocessing,
        "estimator": (
            "StandardScaler + SelectKBest(f_classif; k=100,300,500) + "
            "LinearSVC(C=0.1,1; tol=1e-3; max_iter=200000)"
        ),
        "outer_cv": "subject folds x held session",
        "inner_cv": "3-fold GroupKFold by subject",
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_path.with_name("structured_permutation_config.json")

    def archive_incompatible_cache(reason: str, observed_config: dict | None) -> None:
        archive_root = checkpoint_path.parent / "superseded_structured_cache"
        archive_key = hashlib.sha256(
            json.dumps(
                {
                    "reason": reason,
                    "observed_config": observed_config,
                    "expected_config": config,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        archive_dir = archive_root / archive_key
        suffix = 1
        while archive_dir.exists():
            suffix += 1
            archive_dir = archive_root / f"{archive_key}-{suffix}"
        archive_dir.mkdir(parents=True)
        if config_path.exists():
            config_path.replace(archive_dir / config_path.name)
        if checkpoint_path.exists():
            checkpoint_path.replace(archive_dir / checkpoint_path.name)
        atomic_write_json(
            archive_dir / "replacement_record.json",
            {
                "reason": reason,
                "observed_config": observed_config,
                "expected_config": config,
            },
        )
        print(f"Archived incompatible checkpoint at {archive_dir}.", flush=True)

    if config_path.exists():
        observed_config = json.loads(config_path.read_text(encoding="utf-8"))
        differences = manifest_differences(observed_config, config)
        if differences:
            preview = "\n".join(f"- {item}" for item in differences[:30])
            if not force:
                raise RuntimeError(
                    "Structured-permutation checkpoint configuration does not "
                    f"match the current data or estimator:\n{preview}\n"
                    "Pass --force to archive an incompatible checkpoint or "
                    "upgrade a schema-only mismatch."
                )
            observed_scientific = {
                key: value
                for key, value in observed_config.items()
                if key != "schema_version"
            }
            expected_scientific = {
                key: value for key, value in config.items() if key != "schema_version"
            }
            if observed_scientific == expected_scientific:
                atomic_write_json(config_path, config)
                atomic_write_json(
                    checkpoint_path.with_name(
                        "structured_permutation_config_migration.json"
                    ),
                    {
                        "from_schema_version": observed_config.get("schema_version"),
                        "to_schema_version": config["schema_version"],
                        "scientific_configuration_unchanged": True,
                    },
                )
                print(
                    "Upgraded compatible checkpoint configuration schema; "
                    "scientific fields are unchanged.",
                    flush=True,
                )
            else:
                archive_incompatible_cache(
                    "scientific configuration mismatch", observed_config
                )
    elif checkpoint_path.exists():
        if not force:
            raise RuntimeError(
                "Structured-permutation checkpoint has no configuration manifest. "
                "Pass --force to archive it and start a validated checkpoint."
            )
        archive_incompatible_cache("missing configuration manifest", None)
    if not config_path.exists():
        atomic_write_json(config_path, config)

    null_scores = np.full(n_permutations, np.nan, dtype=float)
    if checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path)
        required = {"permutation", "balanced_accuracy"}
        if not required.issubset(checkpoint.columns):
            raise RuntimeError("Invalid structured-permutation checkpoint columns.")
        permutation_ids = checkpoint.permutation.to_numpy(dtype=int)
        scores = checkpoint.balanced_accuracy.to_numpy(dtype=float)
        if (
            len(np.unique(permutation_ids)) != len(permutation_ids)
            or np.any(permutation_ids < 1)
            or np.any(permutation_ids > n_permutations)
            or not np.all(np.isfinite(scores))
        ):
            raise RuntimeError("Invalid structured-permutation checkpoint rows.")
        null_scores[permutation_ids - 1] = scores
        print(
            f"Resuming structured permutations: {len(permutation_ids)}/"
            f"{n_permutations} already complete.",
            flush=True,
        )

    missing = np.flatnonzero(~np.isfinite(null_scores))
    if batch_size <= 0:
        batch_size = max(abs(n_jobs), 1) * 4
    for batch_start in range(0, len(missing), batch_size):
        batch_indices = missing[batch_start : batch_start + batch_size]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            batch_scores = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_permuted_outer_cv_score)(
                    X,
                    permuted_labels[permutation_index],
                    subject_ids,
                    sessions,
                    sample_subject_fold,
                    n_subject_folds,
                    seed,
                )
                for permutation_index in batch_indices
            )
        null_scores[batch_indices] = batch_scores
        completed = np.flatnonzero(np.isfinite(null_scores))
        checkpoint = pd.DataFrame(
            {
                "permutation": completed + 1,
                "balanced_accuracy": null_scores[completed],
            }
        )
        temporary = checkpoint_path.with_name(
            f".{checkpoint_path.stem}.checkpoint.csv.gz"
        )
        checkpoint.to_csv(temporary, index=False)
        temporary.replace(checkpoint_path)
        print(
            f"Structured permutations checkpointed: {len(completed)}/{n_permutations}.",
            flush=True,
        )
    p_value = (int(np.sum(null_scores >= observed_score - 1e-15)) + 1) / (
        n_permutations + 1
    )
    return null_scores, float(p_value)


def run_classifier(
    X: np.ndarray,
    table: pd.DataFrame,
    classifier: str,
    n_subject_folds: int,
    n_bootstrap: int,
    n_permutations: int,
    structured_permutations: int,
    permutation_jobs: int,
    permutation_batch_size: int,
    seed: int,
    output_dir: Path,
    force: bool = False,
    feature_preprocessing: str = "raw_visual_mask_beta",
):
    y = table.class_id.to_numpy()
    subject_ids = table.subject_id.to_numpy()
    sessions = table.session.to_numpy()
    sample_subject_fold, fold_mapping = subject_folds(
        subject_ids, n_subject_folds, seed
    )
    predictions = np.full(len(table), -1, dtype=int)
    selected_count = np.zeros(X.shape[1], dtype=int)
    fold_rows = []
    for subject_fold in range(n_subject_folds):
        for held_session in sorted(np.unique(sessions)):
            test = (sample_subject_fold == subject_fold) & (sessions == held_session)
            train = (sample_subject_fold != subject_fold) & (sessions != held_session)
            if classifier == "svm":
                estimator, grid = svm_search(X.shape[1], seed)
            else:
                estimator, grid = mlp_search(X.shape[1], seed)
            inner = GroupKFold(n_splits=3)
            search = GridSearchCV(
                estimator,
                grid,
                scoring="balanced_accuracy",
                cv=inner.split(X[train], y[train], groups=subject_ids[train]),
                n_jobs=1,
                refit=True,
            )
            search.fit(X[train], y[train])
            predictions[test] = search.predict(X[test])
            selected = search.best_estimator_.named_steps["select"].get_support()
            selected_count += selected.astype(int)
            fold_rows.append(
                {
                    "subject_fold": subject_fold,
                    "held_session": int(held_session),
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "balanced_accuracy": balanced_accuracy_score(
                        y[test], predictions[test]
                    ),
                    "best_params": json.dumps(search.best_params_),
                }
            )
            print(
                f"{classifier}: subject fold {subject_fold + 1}/{n_subject_folds}, "
                f"held session {held_session}, BA="
                f"{fold_rows[-1]['balanced_accuracy']:.3f}",
                flush=True,
            )
    if np.any(predictions < 0):
        raise RuntimeError("Some samples did not receive an outer-fold prediction.")
    prediction_table = table.copy()
    prediction_table["true_class"] = y
    prediction_table["predicted_class"] = predictions
    prediction_table["predicted_condition"] = np.where(
        predictions == 0, "Simple", "Complex"
    )
    prediction_table.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "outer_fold_metrics.csv", index=False)

    subject_rows = []
    for subject, part in prediction_table.groupby("subject_id"):
        subject_rows.append(
            {
                "subject_id": subject,
                "balanced_accuracy": balanced_accuracy_score(
                    part.true_class, part.predicted_class
                ),
                "accuracy": accuracy_score(part.true_class, part.predicted_class),
            }
        )
    subject_metrics = pd.DataFrame(subject_rows)
    subject_metrics.to_csv(output_dir / "subject_metrics.csv", index=False)
    ba = balanced_accuracy_score(y, predictions)
    ci = bootstrap_subject_metric(
        prediction_table,
        balanced_accuracy_score,
        n_bootstrap,
        seed,
    )
    p = paired_sign_flip_pvalue(
        subject_metrics.balanced_accuracy.to_numpy() - 0.5,
        n_permutations,
        seed,
    )
    if classifier == "svm" and structured_permutations > 0:
        _null_scores, structured_p = structured_label_permutation_test(
            X,
            y,
            subject_ids,
            sessions,
            sample_subject_fold,
            n_subject_folds,
            ba,
            structured_permutations,
            permutation_jobs,
            seed,
            output_dir / "structured_permutation_null.csv.gz",
            permutation_batch_size,
            force=force,
            feature_preprocessing=feature_preprocessing,
        )
    else:
        structured_p = np.nan
    metrics = {
        "classifier": classifier,
        "n_subjects": int(table.subject_id.nunique()),
        "n_sessions": int(table.session.nunique()),
        "n_samples": len(table),
        "n_candidate_voxels": int(X.shape[1]),
        "feature_preprocessing": feature_preprocessing,
        "spatial_mean_centered": (
            feature_preprocessing == "per_sample_visual_mask_spatial_mean_center"
        ),
        "balanced_accuracy": ba,
        "balanced_accuracy_bootstrap_95_ci": list(ci),
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "subject_level_sign_flip_p_vs_chance": p,
        "descriptive_subject_sign_flip_p_vs_chance": p,
        "n_permutations": n_permutations,
        "structured_label_permutation_p": (
            float(structured_p) if np.isfinite(structured_p) else None
        ),
        "structured_permutation_p": (
            float(structured_p) if np.isfinite(structured_p) else None
        ),
        "n_structured_label_permutations": structured_permutations,
        "structured_permutations_requested": structured_permutations,
        "structured_permutations_completed": structured_permutations,
        "formal_inference_complete": bool(
            classifier == "svm" and structured_permutations >= 10000
        ),
        "inference_method": (
            "structured_whole_subject_label_permutation_nested_cv"
            if structured_permutations > 0
            else "descriptive_subject_accuracy_sign_flip_only"
        ),
        "structured_label_permutation_scheme": (
            "whole-subject Simple/Complex label swap; complete nested "
            "cross-validation refitted per permutation"
            if structured_permutations > 0
            else "not_run"
        ),
        "cross_validation": (
            f"{n_subject_folds} subject folds x 4 held-video-pair folds; "
            "training excludes both held subjects and the held video pair"
        ),
    }
    atomic_write_json(output_dir / "metrics.json", metrics)
    cm = confusion_matrix(y, predictions, labels=[0, 1], normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Simple", "Complex"]).plot(
        ax=ax, cmap="Blues", values_format=".2f"
    )
    ax.set_title(f"Cross-subject + cross-video {classifier.upper()}")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=300)
    fig.savefig(output_dir / "confusion_matrix.pdf")
    plt.close(fig)
    return metrics, selected_count, len(fold_rows), fold_mapping


def feature_stability_table(
    selected_count: np.ndarray,
    n_folds: int,
    flat_indices: np.ndarray,
    mask_img: nib.Nifti1Image,
    output_path: Path,
):
    atlas_path = (
        ROI_REFERENCE_DIR / "roi_masks/schaefer200_17net_2mm_resampled_to_beta.nii.gz"
    )
    atlas_img = nib.load(str(atlas_path))
    validate_nifti_geometry(
        mask_img, atlas_img, context=f"feature-stability atlas {atlas_path}"
    )
    atlas_data = np.rint(np.asarray(atlas_img.dataobj)).astype(int)
    label_table = pd.read_csv(ROI_REFERENCE_DIR / "schaefer_task_parcels.tsv", sep="\t")
    name_map = dict(zip(label_table.parcel_id.astype(int), label_table.parcel_name))
    ijk = np.column_stack(np.unravel_index(flat_indices, mask_img.shape))
    xyz = nib.affines.apply_affine(mask_img.affine, ijk)
    parcel_ids = atlas_data.ravel()[flat_indices]
    table = pd.DataFrame(
        {
            "flat_index": flat_indices,
            "voxel_i": ijk[:, 0],
            "voxel_j": ijk[:, 1],
            "voxel_k": ijk[:, 2],
            "mni_x": xyz[:, 0],
            "mni_y": xyz[:, 1],
            "mni_z": xyz[:, 2],
            "parcel_id": parcel_ids,
            "parcel_name": [name_map.get(int(i), "") for i in parcel_ids],
            "selection_count": selected_count,
            "selection_frequency": selected_count / n_folds,
        }
    )
    table.sort_values("selection_frequency", ascending=False).to_csv(
        output_path, index=False
    )
    frequency_map = np.zeros(mask_img.shape, dtype=np.float32)
    frequency_map.ravel()[flat_indices] = selected_count / n_folds
    nib.save(
        nib.Nifti1Image(frequency_map, mask_img.affine, mask_img.header),
        output_path.parent / "selected_voxel_frequency_cross_generalized.nii.gz",
    )


def run(args: argparse.Namespace):
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_root / "run_configuration.json",
        {
            "entrypoint": "fmri_video_analysis.methods.decoding.cross_video",
            "arguments": vars(args),
        },
    )

    manifest = build_four_session_manifest(Path(args.first_level_dir))
    table = manifest.loc[manifest.condition.isin(["Simple", "Complex"])].copy()
    table["class_id"] = table.condition.map({"Simple": 0, "Complex": 1})
    table.to_csv(output_root / "labels_with_video_id.csv", index=False)
    mask_path = (
        ROI_REFERENCE_DIR / "roi_masks/schaefer200_17net_visual_commonmask.nii.gz"
    )
    X, mask_img, flat_indices = load_visual_features(table, mask_path)
    spatial_means_before = X.mean(axis=1, dtype=np.float64)
    spatial_mean_center = bool(getattr(args, "spatial_mean_center", False))
    if spatial_mean_center:
        X = spatial_mean_center_features(X)
        feature_preprocessing = "per_sample_visual_mask_spatial_mean_center"
    else:
        feature_preprocessing = "raw_visual_mask_beta"
    spatial_means_after = X.mean(axis=1, dtype=np.float64)
    centering_audit = table[["subject_id", "session", "condition", "video_id"]].copy()
    centering_audit["spatial_mean_before"] = spatial_means_before
    centering_audit["spatial_mean_after"] = spatial_means_after
    centering_audit.to_csv(
        output_root / "spatial_mean_centering_audit.csv", index=False
    )
    atomic_write_json(
        output_root / "feature_preprocessing.json",
        {
            "feature_preprocessing": feature_preprocessing,
            "spatial_mean_centered": spatial_mean_center,
            "centering_axis": "voxels_within_each_beta_sample",
            "n_samples": int(X.shape[0]),
            "n_candidate_voxels": int(X.shape[1]),
            "mask_path": str(mask_path.resolve()),
            "max_abs_spatial_mean_after": float(np.max(np.abs(spatial_means_after))),
        },
    )
    classifiers = ["svm"] + (["mlp"] if args.include_mlp else [])
    summaries = []
    for classifier in classifiers:
        classifier_dir = output_root / classifier
        classifier_dir.mkdir(parents=True, exist_ok=True)
        metrics, selected, n_folds, mapping = run_classifier(
            X,
            table,
            classifier,
            args.subject_folds,
            args.bootstrap,
            args.n_permutations,
            args.structured_permutations,
            args.permutation_jobs,
            args.structured_permutation_batch_size,
            args.seed,
            classifier_dir,
            getattr(args, "force", False),
            feature_preprocessing,
        )
        feature_stability_table(
            selected,
            n_folds,
            flat_indices,
            mask_img,
            classifier_dir / "selected_voxels_cross_generalized.csv",
        )
        (classifier_dir / "subject_fold_mapping.json").write_text(
            json.dumps(mapping, indent=2), encoding="utf-8"
        )
        summaries.append(metrics)
    pd.DataFrame(summaries).to_csv(output_root / "cross_video_summary.csv", index=False)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument("--output-dir", default=str(RESULTS_DIR / "decoding/cross_video"))
    p.add_argument("--subject-folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--structured-permutations", type=int, default=0)
    p.add_argument("--permutation-jobs", type=int, default=1)
    p.add_argument("--structured-permutation-batch-size", type=int, default=0)
    p.add_argument(
        "--spatial-mean-center",
        action="store_true",
        help=(
            "Subtract each beta sample's mean across all retained visual-mask "
            "voxels before nested cross-validation."
        ),
    )
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--force", action="store_true")
    p.add_argument("--include-mlp", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
