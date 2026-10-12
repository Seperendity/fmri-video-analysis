#!/usr/bin/env python3
"""MVPA for the Simple/Complex/Control natural-video fMRI experiment.

Each first-level beta NIfTI image is one sample.  Voxel-wise beta values from
predefined task-related Schaefer-200 parcels are used as features.  All feature
selection and scaling are fitted inside subject-grouped cross-validation.

Default paths are resolved from this file, so the fmri_video directory can move with the
repository.

Examples
--------
# 完整运行全部分析（MVPA、重复测量统计和结果整理）：
python \
  scripts/run_original_mvpa.py \
  --analysis all \
  --n-permutations 10000

# 如果 MVPA 已经跑完，只补充被试层面统计：
python \
  scripts/run_original_mvpa.py \
  --analysis statistics \
  --n-permutations 10000

# 只生成 pycortex 全皮层展开图：
ARIAL_FONT_PATH=/usr/share/fonts/truetype/msttcorefonts/Arial.ttf \
python \
  scripts/run_original_mvpa.py \
  --analysis flatmaps \
  --pycortex-model-set best \
  --pycortex-formats png pdf \
  --pycortex-dpi 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine
from nibabel.processing import resample_from_to
from nilearn import datasets, surface
from nilearn.datasets import fetch_atlas_schaefer_2018
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from fmri_video_analysis.core.config import ATLAS_DIR, FIRST_LEVEL_DIR, RESULTS_DIR
from fmri_video_analysis.core.study_design import INCLUDED_SUBJECT_IDS
from fmri_video_analysis.core.validation import validate_subject_ids

DEFAULT_WORK_DIR = RESULTS_DIR / "original_mvpa"
DEFAULT_DATA_DIR = FIRST_LEVEL_DIR
DEFAULT_ATLAS_DIR = ATLAS_DIR

# The mapping is verified against each NIfTI header before analysis.
CONDITION_BETAS = {
    "Simple": {1: 1, 2: 10, 3: 19, 4: 28},
    "Complex": {1: 2, 2: 11, 3: 20, 4: 29},
    "Control": {1: 3, 2: 12, 3: 21, 4: 30},
}
CLASS_IDS = {"Simple": 0, "Complex": 1, "Control": 2}
ID_TO_CLASS = {value: key for key, value in CLASS_IDS.items()}

# Schaefer 17-network labels are assigned a priori.  This avoids circular
# selection of networks based on the present classification results.
NETWORK_TOKENS = {
    "visual": ("VisCent", "VisPeri"),
    "dorsal_attention": ("DorsAttnA", "DorsAttnB"),
    "salience_attention": ("SalVentAttnA", "SalVentAttnB"),
    "control": ("ContA", "ContB", "ContC"),
}

TASKS = {
    "three_way": ("Simple", "Complex", "Control"),
    "simple_vs_complex": ("Simple", "Complex"),
    "simple_vs_control": ("Simple", "Control"),
    "complex_vs_control": ("Complex", "Control"),
}

# Paper-facing models selected from the prespecified comparisons, not by
# searching for the single highest observed score after the analysis.
PYCORTEX_BEST_MODELS = (
    ("three_way", "visual", "svm"),
    ("simple_vs_complex", "visual", "svm"),
    ("simple_vs_control", "combined", "svm"),
    ("complex_vs_control", "dorsal_attention", "svm"),
)


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """ANOVA ranking that safely caps k and ignores non-finite F statistics."""

    def __init__(self, k: int = 100):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        scores, pvalues = f_classif(X, y)
        scores = np.asarray(scores, dtype=np.float64)
        self.scores_ = np.nan_to_num(
            scores,
            nan=-np.inf,
            posinf=np.finfo(np.float64).max,
            neginf=-np.inf,
        )
        self.pvalues_ = pvalues
        finite_count = int(np.isfinite(self.scores_).sum())
        if finite_count == 0:
            raise ValueError("No voxel has a finite ANOVA F statistic.")
        self.k_ = min(int(self.k), finite_count, X.shape[1])
        order = np.argsort(self.scores_, kind="stable")
        self.selected_indices_ = np.sort(order[-self.k_ :])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_support(self, indices: bool = False):
        if indices:
            return self.selected_indices_
        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.selected_indices_] = True
        return support

    @property
    def n_features_in_(self) -> int:
        if not hasattr(self, "scores_"):
            raise AttributeError("The selector has not been fitted.")
        return int(self.scores_.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Schaefer-200 ROI MVPA for Simple, Complex and Control beta maps.")
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--work-dir", "--mvpa-dir", dest="work_dir", type=Path, default=DEFAULT_WORK_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--atlas-dir", type=Path, default=None)
    parser.add_argument(
        "--analysis",
        choices=("prepare", "cross", "within", "all", "statistics", "flatmaps"),
        default="all",
        help=(
            "prepare writes labels/audit only; all runs cross- and within-subject "
            "MVPA; statistics analyzes existing subject-level condition means; "
            "flatmaps renders existing results through pycortex without retraining."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        type=int,
        help="Optional subject IDs used for analysis; labels still describe all subjects.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASKS),
        default=list(TASKS),
    )
    parser.add_argument(
        "--roi-groups",
        nargs="+",
        choices=tuple(NETWORK_TOKENS) + ("combined",),
        default=list(NETWORK_TOKENS) + ["combined"],
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=("mlp", "svm"),
        default=["mlp", "svm"],
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument(
        "--k-grid", nargs="+", type=int, default=[25, 50, 100, 200, 500]
    )
    parser.add_argument(
        "--mlp-alpha-grid", nargs="+", type=float, default=[0.001, 0.01]
    )
    parser.add_argument("--svm-c-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help=(
            "Monte Carlo sign-flip permutations for paired condition tests "
            "(default: 10000)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 12 subjects and small grids for an end-to-end smoke test.",
    )
    parser.add_argument(
        "--pycortex-model-set",
        choices=("best", "all"),
        default="best",
        help=(
            "For --analysis flatmaps, render four prespecified paper models or "
            "all existing cross-subject models (default: best)."
        ),
    )
    parser.add_argument(
        "--pycortex-selection-threshold",
        type=float,
        default=0.6,
        help=(
            "Hide projected voxel-selection frequencies below this value "
            "(default: 0.6, selected in at least 3/5 outer folds)."
        ),
    )
    parser.add_argument(
        "--pycortex-parcel-threshold",
        type=float,
        default=0.05,
        help="Hide parcels with mean selection frequency below this value.",
    )
    parser.add_argument("--pycortex-subject", default="fsaverage")
    parser.add_argument("--freesurfer-subject", default="fsaverage5")
    parser.add_argument(
        "--subjects-dir",
        type=Path,
        default=Path("/usr/local/freesurfer/subjects"),
    )
    parser.add_argument(
        "--pycortex-with-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pycortex-recache", action="store_true")
    parser.add_argument(
        "--pycortex-formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png", "pdf"],
    )
    parser.add_argument("--pycortex-dpi", type=int, default=300)
    return parser.parse_args()


def setup_logging(output_dir: Path, filename: str = "mvpa.log") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(output_dir / filename, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    # Matplotlib's PDF backend can otherwise emit hundreds of font-subsetting
    # messages for each pycortex figure.
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def nifti_description(img: nib.spatialimages.SpatialImage) -> str:
    raw = img.header["descrip"].tobytes().split(b"\x00", 1)[0]
    return raw.decode("utf-8", errors="replace").strip()


def numeric_subject_dirs(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(
        (p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )


def build_manifest(
    data_dir: Path, label_file: Path, audit_file: Path
) -> tuple[pd.DataFrame, list[Path], nib.Nifti1Image]:
    """Build the label CSV and strictly audit all 12 beta maps per subject."""

    subject_dirs = numeric_subject_dirs(data_dir)
    if not subject_dirs:
        raise RuntimeError(f"No numeric subject folders found in {data_dir}")
    validate_subject_ids(
        (int(path.name) for path in subject_dirs),
        expected_ids=INCLUDED_SUBJECT_IDS,
        context=f"original MVPA cohort under {data_dir}",
    )

    rows: list[dict] = []
    audit_rows: list[dict] = []
    errors: list[str] = []
    reference_shape = None
    reference_affine = None
    reference_img = None

    for subject_dir in subject_dirs:
        subject = int(subject_dir.name)
        for condition, session_map in CONDITION_BETAS.items():
            for session, beta_index in session_map.items():
                beta_path = subject_dir / f"beta_{beta_index:04d}.nii"
                audit = {
                    "subject_id": subject,
                    "session": session,
                    "expected_condition": condition,
                    "beta_index": beta_index,
                    "beta_path": str(beta_path),
                    "exists": beta_path.is_file(),
                    "header_description": "",
                    "header_match": False,
                    "geometry_match": False,
                    "error": "",
                }
                if not beta_path.is_file():
                    message = f"Missing file: {beta_path}"
                    audit["error"] = message
                    errors.append(message)
                    audit_rows.append(audit)
                    continue

                try:
                    img = nib.load(str(beta_path))
                    description = nifti_description(img)
                    expected_pattern = re.compile(
                        rf"Sn\({session}\)\s+{re.escape(condition)}(?:\*bf\(1\))?",
                        flags=re.IGNORECASE,
                    )
                    header_match = bool(expected_pattern.search(description))

                    if reference_shape is None:
                        reference_shape = img.shape
                        reference_affine = img.affine.copy()
                        reference_img = img
                    geometry_match = img.shape == reference_shape and np.allclose(
                        img.affine, reference_affine, atol=1e-5
                    )

                    audit["header_description"] = description
                    audit["header_match"] = header_match
                    audit["geometry_match"] = geometry_match

                    if not header_match:
                        message = (
                            f"Header mismatch for {beta_path}: expected "
                            f"Sn({session}) {condition}, found {description!r}"
                        )
                        audit["error"] = message
                        errors.append(message)
                    elif not geometry_match:
                        message = f"Geometry mismatch: {beta_path}"
                        audit["error"] = message
                        errors.append(message)
                    else:
                        rows.append(
                            {
                                "sample_id": (
                                    f"sub-{subject:03d}_ses-{session}_"
                                    f"{condition.lower()}"
                                ),
                                "subject_id": subject,
                                "session": session,
                                "condition": condition,
                                "class_id": CLASS_IDS[condition],
                                "beta_index": beta_index,
                                "beta_path": str(beta_path),
                                "header_description": description,
                            }
                        )
                except Exception as exc:  # audit should survive one corrupt image
                    message = f"Cannot read {beta_path}: {exc}"
                    audit["error"] = message
                    errors.append(message)
                audit_rows.append(audit)

    manifest = pd.DataFrame(rows).sort_values(["subject_id", "session", "class_id"])
    audit_df = pd.DataFrame(audit_rows).sort_values(
        ["subject_id", "session", "expected_condition"]
    )
    label_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(label_file, index=False)
    audit_df.to_csv(audit_file, index=False)

    expected_rows = len(subject_dirs) * 12
    if len(manifest) != expected_rows:
        errors.append(
            f"Manifest contains {len(manifest)} valid rows; expected {expected_rows}."
        )
    class_counts = manifest["condition"].value_counts().to_dict()
    expected_per_class = len(subject_dirs) * 4
    for condition in CLASS_IDS:
        if class_counts.get(condition, 0) != expected_per_class:
            errors.append(
                f"{condition} has {class_counts.get(condition, 0)} rows; "
                f"expected {expected_per_class}."
            )

    if errors:
        preview = "\n".join(f"- {item}" for item in errors[:20])
        raise RuntimeError(
            f"Data audit failed with {len(errors)} issue(s). "
            f"See {audit_file}.\n{preview}"
        )

    logging.info(
        "Label manifest: %d subjects, %d samples (%s)",
        len(subject_dirs),
        len(manifest),
        class_counts,
    )
    assert reference_img is not None
    return manifest.reset_index(drop=True), subject_dirs, reference_img


def _decode_label(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def prepare_schaefer_masks(
    atlas_dir: Path,
    output_dir: Path,
    reference_img: nib.Nifti1Image,
    subject_dirs: Sequence[Path],
) -> tuple[
    nib.Nifti1Image,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    """Fetch Schaefer-200/17-network/2-mm and create task-network masks."""

    atlas_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Loading Schaefer-200, 17-network, 2-mm atlas")
    atlas = fetch_atlas_schaefer_2018(
        n_rois=200,
        yeo_networks=17,
        resolution_mm=2,
        data_dir=str(atlas_dir),
        verbose=1,
    )
    atlas_img = nib.load(str(atlas.maps))
    atlas_resampled_img = resample_from_to(
        atlas_img, (reference_img.shape, reference_img.affine), order=0
    )
    atlas_data = np.rint(np.asarray(atlas_resampled_img.dataobj)).astype(np.int16)

    if hasattr(atlas, "lut") and atlas.lut is not None:
        lut = atlas.lut.copy()
        lut.columns = [str(c).lower() for c in lut.columns]
        index_col = "index"
        name_col = "name"
        parcels = pd.DataFrame(
            {
                "parcel_id": lut[index_col].astype(int),
                "parcel_name": lut[name_col].map(_decode_label),
            }
        )
    else:
        labels = [_decode_label(v) for v in atlas.labels]
        parcels = pd.DataFrame(
            {"parcel_id": np.arange(len(labels), dtype=int), "parcel_name": labels}
        )

    parcels = parcels[parcels["parcel_id"] > 0].copy()
    if parcels["parcel_id"].nunique() != 200:
        raise RuntimeError(
            "Schaefer atlas did not expose exactly 200 non-background parcels."
        )

    def family_for_name(name: str) -> str | None:
        for family, tokens in NETWORK_TOKENS.items():
            if any(
                re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", name)
                for token in tokens
            ):
                return family
        return None

    parcels["network_family"] = parcels["parcel_name"].map(family_for_name)
    task_parcels = parcels.dropna(subset=["network_family"]).copy()
    if task_parcels.empty:
        raise RuntimeError("No task-network parcels were found in Schaefer labels.")

    # A strict common mask provides aligned, observed voxel features for all
    # participants and avoids zero-filled, subject-specific coverage artifacts.
    common_mask = np.ones(reference_img.shape, dtype=bool)
    for subject_dir in subject_dirs:
        mask_path = subject_dir / "mask.nii"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing subject brain mask: {mask_path}")
        mask_img = nib.load(str(mask_path))
        if mask_img.shape != reference_img.shape or not np.allclose(
            mask_img.affine, reference_img.affine, atol=1e-5
        ):
            raise RuntimeError(f"Subject mask geometry mismatch: {mask_path}")
        common_mask &= np.asarray(mask_img.dataobj) > 0

    family_masks: dict[str, np.ndarray] = {}
    mask_dir = output_dir / "roi_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for family in NETWORK_TOKENS:
        ids = task_parcels.loc[
            task_parcels["network_family"] == family, "parcel_id"
        ].to_numpy(dtype=int)
        mask = np.isin(atlas_data, ids) & common_mask
        if not mask.any():
            raise RuntimeError(f"ROI mask is empty after intersection: {family}")
        family_masks[family] = mask
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), reference_img.affine),
            mask_dir / f"schaefer200_17net_{family}_commonmask.nii.gz",
        )

    combined_mask = np.logical_or.reduce(list(family_masks.values()))
    family_masks["combined"] = combined_mask
    nib.save(
        nib.Nifti1Image(combined_mask.astype(np.uint8), reference_img.affine),
        mask_dir / "schaefer200_17net_task_combined_commonmask.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(atlas_data, reference_img.affine),
        mask_dir / "schaefer200_17net_2mm_resampled_to_beta.nii.gz",
    )

    task_parcels.to_csv(output_dir / "schaefer_task_parcels.tsv", sep="\t", index=False)
    counts = {name: int(mask.sum()) for name, mask in family_masks.items()}
    with (output_dir / "roi_voxel_counts.json").open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    logging.info("ROI voxel counts after common-mask intersection: %s", counts)
    return atlas_resampled_img, atlas_data, common_mask, family_masks, task_parcels


def extract_combined_features(
    manifest: pd.DataFrame, combined_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    flat_indices = np.flatnonzero(combined_mask.ravel())
    X = np.empty((len(manifest), len(flat_indices)), dtype=np.float32)
    for row_index, row in enumerate(manifest.itertuples(index=False)):
        img = nib.load(row.beta_path)
        values = np.asarray(img.dataobj, dtype=np.float32).ravel()[flat_indices]
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite beta value detected: {row.beta_path}")
        X[row_index] = values
        if (row_index + 1) % 100 == 0 or row_index + 1 == len(manifest):
            logging.info("Extracted beta features: %d/%d", row_index + 1, len(manifest))
    return X, flat_indices


def roi_column_indices(
    family_masks: dict[str, np.ndarray], combined_flat_indices: np.ndarray
) -> dict[str, np.ndarray]:
    result = {}
    for family, mask in family_masks.items():
        inside = mask.ravel()[combined_flat_indices]
        result[family] = np.flatnonzero(inside)
    return result


def parcel_column_indices(
    atlas_data: np.ndarray,
    combined_flat_indices: np.ndarray,
    task_parcels: pd.DataFrame,
) -> dict[int, np.ndarray]:
    parcel_values = atlas_data.ravel()[combined_flat_indices]
    result = {}
    for parcel_id in task_parcels["parcel_id"].astype(int):
        columns = np.flatnonzero(parcel_values == parcel_id)
        if len(columns):
            result[int(parcel_id)] = columns
    return result


def make_pipeline(
    classifier_name: str, max_iter: int, seed: int
) -> tuple[Pipeline, dict[str, list]]:
    if classifier_name == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            early_stopping=False,
            random_state=seed,
        )
        params = {"classifier__alpha": []}
    elif classifier_name == "svm":
        classifier = LinearSVC(
            class_weight="balanced",
            dual="auto",
            max_iter=max(5000, max_iter),
            random_state=seed,
        )
        params = {"classifier__C": []}
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    pipeline = Pipeline(
        [
            ("selector", SafeSelectKBest()),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    return pipeline, params


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if n_bootstrap <= 0:
        return {}
    unique_groups = np.unique(groups)
    group_indices = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[g] for g in sampled])
        metrics = metric_dict(y_true[indices], y_pred[indices])
        for name, value in metrics.items():
            values[name].append(value)
    return {
        name: {
            "lower_95": float(np.percentile(metric_values, 2.5)),
            "upper_95": float(np.percentile(metric_values, 97.5)),
        }
        for name, metric_values in values.items()
    }


def json_ready(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_confusion_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: Sequence[int],
    output_path: Path,
    title: str,
) -> None:
    names = [ID_TO_CLASS[int(value)] for value in class_ids]
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(names)),
        yticks=np.arange(len(names)),
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_one_grouped_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_table: pd.DataFrame,
    reference_img: nib.Nifti1Image,
    global_flat_indices: np.ndarray,
    atlas_data: np.ndarray,
    task_parcels: pd.DataFrame,
    classifier_name: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_groups = np.unique(groups)
    outer_folds = min(args.outer_folds, len(unique_groups))
    if outer_folds < 2:
        raise ValueError("At least two subjects are required for grouped CV.")

    outer_cv = StratifiedGroupKFold(
        n_splits=outer_folds, shuffle=True, random_state=args.seed
    )
    oof = np.full(len(y), -1, dtype=int)
    selection_counts = np.zeros(X.shape[1], dtype=np.int16)
    fold_rows = []
    curve_rows = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        train_groups = groups[train_idx]
        test_groups = groups[test_idx]
        overlap = np.intersect1d(np.unique(train_groups), np.unique(test_groups))
        if len(overlap):
            raise AssertionError(f"Subject leakage in fold {fold}: {overlap}")

        inner_folds = min(args.inner_folds, len(np.unique(train_groups)))
        if inner_folds < 2:
            raise ValueError(
                "At least two training subjects are required for inner CV."
            )
        inner_cv = StratifiedGroupKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=args.seed + fold,
        )

        pipeline, params = make_pipeline(
            classifier_name, args.max_iter, args.seed + fold
        )
        params["selector__k"] = sorted(
            set(min(int(k), X.shape[1]) for k in args.k_grid if int(k) > 0)
        )
        if classifier_name == "mlp":
            params["classifier__alpha"] = args.mlp_alpha_grid
        else:
            params["classifier__C"] = args.svm_c_grid

        search = GridSearchCV(
            pipeline,
            param_grid=params,
            scoring="balanced_accuracy",
            cv=inner_cv,
            n_jobs=args.n_jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            search.fit(
                X[train_idx],
                y[train_idx],
                groups=train_groups,
            )
        predictions = search.predict(X[test_idx])
        oof[test_idx] = predictions

        selected = search.best_estimator_.named_steps["selector"].get_support(
            indices=True
        )
        selection_counts[selected] += 1
        fold_metrics = metric_dict(y[test_idx], predictions)
        fold_rows.append(
            {
                "fold": fold,
                "n_train_samples": len(train_idx),
                "n_test_samples": len(test_idx),
                "n_train_subjects": len(np.unique(train_groups)),
                "n_test_subjects": len(np.unique(test_groups)),
                "train_subjects": ",".join(map(str, sorted(np.unique(train_groups)))),
                "test_subjects": ",".join(map(str, sorted(np.unique(test_groups)))),
                "best_inner_balanced_accuracy": float(search.best_score_),
                "best_params": json.dumps(json_ready(search.best_params_)),
                **fold_metrics,
            }
        )

        results = pd.DataFrame(search.cv_results_)
        results["k"] = results["param_selector__k"].astype(int)
        curve = results.groupby("k", as_index=False)["mean_test_score"].max()
        for row in curve.itertuples(index=False):
            curve_rows.append(
                {
                    "outer_fold": fold,
                    "k": int(row.k),
                    "best_inner_balanced_accuracy": float(row.mean_test_score),
                }
            )
        logging.info(
            "%s fold %d/%d: test balanced accuracy %.3f, params=%s",
            classifier_name,
            fold,
            outer_folds,
            fold_metrics["balanced_accuracy"],
            search.best_params_,
        )

    if np.any(oof < 0):
        raise AssertionError("Some samples did not receive an out-of-fold prediction.")

    overall = metric_dict(y, oof)
    overall["per_class_recall"] = {
        ID_TO_CLASS[int(class_id)]: float(value)
        for class_id, value in zip(
            sorted(np.unique(y)),
            recall_score(y, oof, labels=sorted(np.unique(y)), average=None),
        )
    }
    overall["bootstrap_95"] = bootstrap_metrics(
        y, oof, groups, args.bootstrap, args.seed
    )
    overall["n_samples"] = int(len(y))
    overall["n_subjects"] = int(len(unique_groups))
    overall["chance_accuracy"] = 1.0 / len(np.unique(y))

    prediction_table = sample_table.copy()
    prediction_table["true_class_id"] = y
    prediction_table["predicted_class_id"] = oof
    prediction_table["predicted_condition"] = [ID_TO_CLASS[int(value)] for value in oof]
    prediction_table.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(
        output_dir / "voxel_count_validation_curve.csv", index=False
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(overall), f, indent=2)

    save_confusion_plot(
        y,
        oof,
        sorted(np.unique(y)),
        output_dir / "confusion_matrix.png",
        f"{classifier_name.upper()} out-of-fold confusion matrix",
    )

    frequency = selection_counts.astype(np.float32) / outer_folds
    frequency_volume = np.zeros(reference_img.shape, dtype=np.float32)
    frequency_volume.ravel()[global_flat_indices] = frequency
    nib.save(
        nib.Nifti1Image(frequency_volume, reference_img.affine),
        output_dir / "selected_voxel_frequency.nii.gz",
    )

    selected_local = np.flatnonzero(selection_counts > 0)
    selected_global_flat = global_flat_indices[selected_local]
    ijk = np.column_stack(np.unravel_index(selected_global_flat, reference_img.shape))
    xyz = apply_affine(reference_img.affine, ijk)
    parcel_ids = atlas_data.ravel()[selected_global_flat].astype(int)
    parcel_name_map = dict(zip(task_parcels["parcel_id"], task_parcels["parcel_name"]))
    network_map = dict(zip(task_parcels["parcel_id"], task_parcels["network_family"]))
    selected_table = pd.DataFrame(
        {
            "i": ijk[:, 0],
            "j": ijk[:, 1],
            "k": ijk[:, 2],
            "mni_x": xyz[:, 0],
            "mni_y": xyz[:, 1],
            "mni_z": xyz[:, 2],
            "parcel_id": parcel_ids,
            "parcel_name": [parcel_name_map.get(v, "") for v in parcel_ids],
            "network_family": [network_map.get(v, "") for v in parcel_ids],
            "selection_count": selection_counts[selected_local],
            "selection_frequency": frequency[selected_local],
        }
    ).sort_values(["selection_frequency", "parcel_id"], ascending=[False, True])
    selected_table.to_csv(output_dir / "selected_voxels.csv", index=False)
    return overall


def run_cross_subject(
    X: np.ndarray,
    manifest: pd.DataFrame,
    roi_columns: dict[str, np.ndarray],
    reference_img: nib.Nifti1Image,
    combined_flat_indices: np.ndarray,
    atlas_data: np.ndarray,
    task_parcels: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    summary_rows = []
    for task_name in args.tasks:
        conditions = TASKS[task_name]
        sample_mask = manifest["condition"].isin(conditions).to_numpy()
        task_table = manifest.loc[sample_mask].reset_index(drop=True)
        y = task_table["class_id"].to_numpy(dtype=int)
        groups = task_table["subject_id"].to_numpy(dtype=int)

        for roi_name in args.roi_groups:
            columns = roi_columns[roi_name]
            roi_X = X[sample_mask][:, columns]
            roi_global_flat = combined_flat_indices[columns]
            for classifier_name in args.classifiers:
                # MLP is focused on the main three-way problem and the central
                # Simple-vs-Complex hypothesis. Linear SVM covers all contrasts.
                if classifier_name == "mlp" and task_name not in (
                    "three_way",
                    "simple_vs_complex",
                ):
                    continue
                model_dir = (
                    output_dir
                    / "cross_subject"
                    / task_name
                    / roi_name
                    / classifier_name
                )
                logging.info(
                    "Cross-subject model: task=%s ROI=%s classifier=%s "
                    "(%d samples, %d voxels)",
                    task_name,
                    roi_name,
                    classifier_name,
                    len(y),
                    roi_X.shape[1],
                )
                metrics = run_one_grouped_model(
                    roi_X,
                    y,
                    groups,
                    task_table,
                    reference_img,
                    roi_global_flat,
                    atlas_data,
                    task_parcels,
                    classifier_name,
                    model_dir,
                    args,
                )
                summary_rows.append(
                    {
                        "task": task_name,
                        "roi_group": roi_name,
                        "classifier": classifier_name,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if isinstance(value, (int, float))
                        },
                    }
                )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "cross_subject_summary.csv", index=False
    )


def network_mean_features(
    X: np.ndarray, roi_columns: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    names = list(NETWORK_TOKENS)
    means = np.column_stack([X[:, roi_columns[name]].mean(axis=1) for name in names])
    return means.astype(np.float32), names


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values, preserving NaNs."""

    pvalues = np.asarray(pvalues, dtype=float)
    qvalues = np.full(pvalues.shape, np.nan, dtype=float)
    valid = np.isfinite(pvalues)
    p = pvalues[valid]
    if not len(p):
        return qvalues
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    qvalues[valid] = restored
    return qvalues


def paired_sign_flip_pvalues(
    differences: np.ndarray,
    n_permutations: int,
    seed: int,
    chunk_size: int = 500,
) -> np.ndarray:
    """Two-sided paired permutation p-values for one or many features.

    Parameters
    ----------
    differences
        Subject × feature matrix for condition A minus condition B.
    """

    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim == 1:
        differences = differences[:, None]
    if not np.isfinite(differences).all():
        raise ValueError("Paired permutation input contains non-finite values.")
    n_subjects, n_features = differences.shape
    if n_subjects < 2:
        raise ValueError("Paired permutation testing requires at least two subjects.")
    if n_permutations < 1:
        raise ValueError("--n-permutations must be at least 1.")

    observed = np.abs(differences.mean(axis=0))
    exceedances = np.zeros(n_features, dtype=np.int64)
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < n_permutations:
        current = min(chunk_size, n_permutations - completed)
        signs = rng.choice((-1.0, 1.0), size=(current, n_subjects))
        permuted = np.abs(signs @ differences / n_subjects)
        exceedances += np.sum(permuted >= observed[None, :], axis=0)
        completed += current
    return (exceedances + 1.0) / (n_permutations + 1.0)


def repeated_measures_statistics(
    subject_condition_table: pd.DataFrame,
    region_columns: Sequence[str],
    value_column: str,
    region_kind: str,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Friedman omnibus and paired sign-flip tests for repeated conditions."""

    condition_order = ("Simple", "Complex", "Control")
    region_columns = list(region_columns)
    required = {"subject_id", "condition", value_column, *region_columns}
    missing = required.difference(subject_condition_table.columns)
    if missing:
        raise ValueError(
            f"Repeated-measures table is missing columns: {sorted(missing)}"
        )

    omnibus_rows = []
    region_keys = (
        subject_condition_table[region_columns]
        .drop_duplicates()
        .sort_values(region_columns)
    )
    for key_values in region_keys.itertuples(index=False, name=None):
        key_filter = np.ones(len(subject_condition_table), dtype=bool)
        key_dict = dict(zip(region_columns, key_values))
        for column, value in key_dict.items():
            key_filter &= subject_condition_table[column].to_numpy() == value
        subset = subject_condition_table.loc[key_filter]
        pivot = subset.pivot(
            index="subject_id", columns="condition", values=value_column
        ).dropna(subset=list(condition_order))
        statistic, pvalue = stats.friedmanchisquare(
            *[pivot[condition].to_numpy() for condition in condition_order]
        )
        omnibus_rows.append(
            {
                **key_dict,
                "n_subjects": len(pivot),
                "friedman_chi2": float(statistic),
                "df": 2,
                "p_uncorrected": float(pvalue),
            }
        )
    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["q_fdr"] = benjamini_hochberg(omnibus["p_uncorrected"])
    omnibus["significant_fdr_05"] = omnibus["q_fdr"] < 0.05

    contrasts = (
        ("Complex", "Simple", "complex_minus_simple"),
        ("Simple", "Control", "simple_minus_control"),
        ("Complex", "Control", "complex_minus_control"),
    )
    pairwise_rows = []
    for contrast_index, (condition_a, condition_b, contrast_name) in enumerate(
        contrasts
    ):
        differences = []
        metadata = []
        subject_counts = []
        for key_values in region_keys.itertuples(index=False, name=None):
            key_filter = np.ones(len(subject_condition_table), dtype=bool)
            key_dict = dict(zip(region_columns, key_values))
            for column, value in key_dict.items():
                key_filter &= subject_condition_table[column].to_numpy() == value
            subset = subject_condition_table.loc[key_filter]
            pivot = subset.pivot(
                index="subject_id", columns="condition", values=value_column
            ).dropna(subset=[condition_a, condition_b])
            differences.append(
                pivot[condition_a].to_numpy() - pivot[condition_b].to_numpy()
            )
            metadata.append(key_dict)
            subject_counts.append(len(pivot))

        if len(set(subject_counts)) != 1:
            raise RuntimeError(
                f"{region_kind} regions have unequal paired subject counts."
            )
        difference_matrix = np.column_stack(differences)
        pvalues = paired_sign_flip_pvalues(
            difference_matrix,
            n_permutations=n_permutations,
            seed=seed + 1000 * contrast_index,
        )
        means = difference_matrix.mean(axis=0)
        stds = difference_matrix.std(axis=0, ddof=1)
        effect_sizes = np.divide(
            means,
            stds,
            out=np.full_like(means, np.nan),
            where=stds > 0,
        )
        for feature_index, key_dict in enumerate(metadata):
            pairwise_rows.append(
                {
                    **key_dict,
                    "contrast": contrast_name,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "n_subjects": subject_counts[feature_index],
                    "mean_difference_a_minus_b": float(means[feature_index]),
                    "cohens_dz": float(effect_sizes[feature_index]),
                    "p_permutation": float(pvalues[feature_index]),
                    "n_permutations": int(n_permutations),
                }
            )

    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["q_fdr_all_pairwise"] = benjamini_hochberg(pairwise["p_permutation"])
    pairwise["q_fdr_within_contrast"] = np.nan
    for contrast_name, indices in pairwise.groupby("contrast").groups.items():
        pairwise.loc[indices, "q_fdr_within_contrast"] = benjamini_hochberg(
            pairwise.loc[indices, "p_permutation"]
        )
    pairwise["significant_fdr_all_05"] = pairwise["q_fdr_all_pairwise"] < 0.05
    pairwise["significant_fdr_contrast_05"] = pairwise["q_fdr_within_contrast"] < 0.05
    return omnibus, pairwise


def save_parcel_statistical_maps(
    parcel_pairwise: pd.DataFrame,
    atlas_path: Path,
    output_dir: Path,
) -> None:
    """Write unthresholded and FDR-thresholded parcel beta-contrast maps."""

    if not atlas_path.is_file():
        raise FileNotFoundError(
            f"Resampled Schaefer atlas needed for statistical maps: {atlas_path}"
        )
    atlas_img = nib.load(str(atlas_path))
    atlas_data = np.rint(np.asarray(atlas_img.dataobj)).astype(np.int16)
    map_dir = output_dir / "statistical_maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    for contrast, table in parcel_pairwise.groupby("contrast"):
        unthresholded = np.zeros(atlas_data.shape, dtype=np.float32)
        thresholded = np.zeros(atlas_data.shape, dtype=np.float32)
        qmap = np.ones(atlas_data.shape, dtype=np.float32)
        for row in table.itertuples(index=False):
            mask = atlas_data == int(row.parcel_id)
            difference = float(row.mean_difference_a_minus_b)
            qvalue = float(row.q_fdr_all_pairwise)
            unthresholded[mask] = difference
            qmap[mask] = qvalue
            if qvalue < 0.05:
                thresholded[mask] = difference
        nib.save(
            nib.Nifti1Image(unthresholded, atlas_img.affine),
            map_dir / f"{contrast}_parcel_beta_difference.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(thresholded, atlas_img.affine),
            map_dir / f"{contrast}_parcel_beta_difference_fdr05.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(qmap, atlas_img.affine),
            map_dir / f"{contrast}_parcel_q_fdr_all.nii.gz",
        )


def run_existing_repeated_measures_statistics(
    output_dir: Path, n_permutations: int, seed: int
) -> None:
    """Analyze existing subject-level CSVs without repeating MVPA training."""

    descriptive_dir = output_dir / "descriptive_responses"
    network_path = descriptive_dir / "network_subject_condition_means.csv"
    parcel_path = descriptive_dir / "parcel_subject_condition_means.csv"
    if not network_path.is_file() or not parcel_path.is_file():
        raise FileNotFoundError(
            "Subject-level descriptive CSVs are missing. Run --analysis all first."
        )

    network_subject = pd.read_csv(network_path)
    parcel_subject = pd.read_csv(parcel_path)
    network_omnibus, network_pairwise = repeated_measures_statistics(
        network_subject,
        region_columns=["network_family"],
        value_column="session_mean_beta",
        region_kind="network",
        n_permutations=n_permutations,
        seed=seed,
    )
    parcel_omnibus, parcel_pairwise = repeated_measures_statistics(
        parcel_subject,
        region_columns=["parcel_id", "parcel_name", "network_family"],
        value_column="session_mean_beta",
        region_kind="parcel",
        n_permutations=n_permutations,
        seed=seed + 10000,
    )
    network_omnibus.to_csv(
        descriptive_dir / "network_repeated_measures_friedman.csv", index=False
    )
    network_pairwise.to_csv(
        descriptive_dir / "network_paired_permutation.csv", index=False
    )
    parcel_omnibus.to_csv(
        descriptive_dir / "parcel_repeated_measures_friedman.csv", index=False
    )
    parcel_pairwise.to_csv(
        descriptive_dir / "parcel_paired_permutation.csv", index=False
    )

    atlas_path = (
        output_dir / "roi_masks" / "schaefer200_17net_2mm_resampled_to_beta.nii.gz"
    )
    save_parcel_statistical_maps(parcel_pairwise, atlas_path, descriptive_dir)
    logging.info(
        "Repeated-measures statistics complete: %d network and %d parcel "
        "pairwise tests; %d permutations.",
        len(network_pairwise),
        len(parcel_pairwise),
        n_permutations,
    )


def run_within_subject(
    network_X: np.ndarray,
    network_names: Sequence[str],
    manifest: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    root = output_dir / "within_subject"
    root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for task_name in args.tasks:
        conditions = TASKS[task_name]
        mask = manifest["condition"].isin(conditions).to_numpy()
        table = manifest.loc[mask].reset_index(drop=True)
        task_X = network_X[mask]
        task_y = table["class_id"].to_numpy(dtype=int)
        all_predictions = np.full(len(table), -1, dtype=int)
        subject_rows = []

        for subject in sorted(table["subject_id"].unique()):
            subject_mask = table["subject_id"].to_numpy() == subject
            indices = np.flatnonzero(subject_mask)
            Xs = task_X[indices]
            ys = task_y[indices]
            sessions = table.loc[subject_mask, "session"].to_numpy(dtype=int)
            logo = LeaveOneGroupOut()
            subject_predictions = np.full(len(indices), -1, dtype=int)

            for train, test in logo.split(Xs, ys, sessions):
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            LinearSVC(
                                C=1.0,
                                class_weight="balanced",
                                dual="auto",
                                max_iter=5000,
                                random_state=args.seed,
                            ),
                        ),
                    ]
                )
                model.fit(Xs[train], ys[train])
                subject_predictions[test] = model.predict(Xs[test])
            all_predictions[indices] = subject_predictions
            metrics = metric_dict(ys, subject_predictions)
            subject_rows.append({"subject_id": subject, **metrics})

        overall = metric_dict(task_y, all_predictions)
        overall["bootstrap_95"] = bootstrap_metrics(
            task_y,
            all_predictions,
            table["subject_id"].to_numpy(dtype=int),
            args.bootstrap,
            args.seed,
        )
        task_dir = root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        prediction_table = table.copy()
        prediction_table["predicted_class_id"] = all_predictions
        prediction_table["predicted_condition"] = [
            ID_TO_CLASS[int(v)] for v in all_predictions
        ]
        prediction_table.to_csv(task_dir / "oof_predictions.csv", index=False)
        pd.DataFrame(subject_rows).to_csv(task_dir / "subject_metrics.csv", index=False)
        with (task_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(json_ready(overall), f, indent=2)
        save_confusion_plot(
            task_y,
            all_predictions,
            sorted(np.unique(task_y)),
            task_dir / "confusion_matrix.png",
            f"Within-subject {task_name}",
        )
        summary_rows.append({"task": task_name, **metric_dict(task_y, all_predictions)})

    pd.DataFrame(summary_rows).to_csv(root / "within_subject_summary.csv", index=False)
    pd.DataFrame({"feature": list(network_names)}).to_csv(
        root / "network_features.csv", index=False
    )


def save_descriptive_responses(
    X: np.ndarray,
    manifest: pd.DataFrame,
    roi_columns: dict[str, np.ndarray],
    parcel_columns: dict[int, np.ndarray],
    task_parcels: pd.DataFrame,
    output_dir: Path,
) -> None:
    descriptive_dir = output_dir / "descriptive_responses"
    descriptive_dir.mkdir(parents=True, exist_ok=True)

    network_X, network_names = network_mean_features(X, roi_columns)
    network_wide = manifest[
        ["sample_id", "subject_id", "session", "condition", "class_id"]
    ].copy()
    for column_index, name in enumerate(network_names):
        network_wide[name] = network_X[:, column_index]
    network_long = network_wide.melt(
        id_vars=["sample_id", "subject_id", "session", "condition", "class_id"],
        value_vars=network_names,
        var_name="network_family",
        value_name="mean_beta",
    )
    network_long.to_csv(descriptive_dir / "network_beta_values.csv", index=False)

    subject_condition = (
        network_long.groupby(
            ["subject_id", "condition", "network_family"], as_index=False
        )["mean_beta"]
        .mean()
        .rename(columns={"mean_beta": "session_mean_beta"})
    )
    subject_condition.to_csv(
        descriptive_dir / "network_subject_condition_means.csv", index=False
    )
    network_summary = (
        subject_condition.groupby(["condition", "network_family"])["session_mean_beta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    network_summary["sem"] = network_summary["std"] / np.sqrt(network_summary["count"])
    network_summary.to_csv(
        descriptive_dir / "network_condition_summary.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    condition_order = ["Simple", "Complex", "Control"]
    colors = {"Simple": "#4C78A8", "Complex": "#E45756", "Control": "#72B7B2"}
    for ax, network in zip(axes.ravel(), network_names):
        subset = network_summary.set_index(["network_family", "condition"])
        means = [subset.loc[(network, c), "mean"] for c in condition_order]
        sems = [subset.loc[(network, c), "sem"] for c in condition_order]
        ax.bar(
            condition_order,
            means,
            yerr=sems,
            color=[colors[c] for c in condition_order],
            capsize=4,
        )
        ax.set_title(network.replace("_", " ").title())
        ax.set_ylabel("Mean beta")
    fig.suptitle("Schaefer task-network beta responses (mean ± SEM)")
    fig.tight_layout()
    fig.savefig(descriptive_dir / "network_condition_responses.png", dpi=180)
    plt.close(fig)

    parcel_name_map = dict(zip(task_parcels["parcel_id"], task_parcels["parcel_name"]))
    network_map = dict(zip(task_parcels["parcel_id"], task_parcels["network_family"]))
    parcel_frames = []
    base_columns = manifest[
        ["sample_id", "subject_id", "session", "condition", "class_id"]
    ]
    for parcel_id, columns in parcel_columns.items():
        frame = base_columns.copy()
        frame["parcel_id"] = parcel_id
        frame["parcel_name"] = parcel_name_map.get(parcel_id, "")
        frame["network_family"] = network_map.get(parcel_id, "")
        frame["mean_beta"] = X[:, columns].mean(axis=1)
        parcel_frames.append(frame)
    parcel_long = pd.concat(parcel_frames, ignore_index=True)
    parcel_long.to_csv(descriptive_dir / "parcel_beta_values.csv", index=False)
    parcel_subject = (
        parcel_long.groupby(
            [
                "subject_id",
                "condition",
                "parcel_id",
                "parcel_name",
                "network_family",
            ],
            as_index=False,
        )["mean_beta"]
        .mean()
        .rename(columns={"mean_beta": "session_mean_beta"})
    )
    parcel_subject.to_csv(
        descriptive_dir / "parcel_subject_condition_means.csv", index=False
    )
    parcel_summary = (
        parcel_subject.groupby(
            ["condition", "parcel_id", "parcel_name", "network_family"]
        )["session_mean_beta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    parcel_summary["sem"] = parcel_summary["std"] / np.sqrt(parcel_summary["count"])
    parcel_summary.to_csv(descriptive_dir / "parcel_condition_summary.csv", index=False)


def configure_paper_font() -> None:
    """Use an optional supplied Arial font and embed editable vector text."""

    font_path = os.environ.get("ARIAL_FONT_PATH")
    if font_path:
        if not Path(font_path).is_file():
            raise FileNotFoundError(f"ARIAL_FONT_PATH does not exist: {font_path}")
        import matplotlib.font_manager as font_manager

        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        family = "Arial"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_pycortex(subject: str):
    """Import pycortex lazily so regular MVPA runs do not require it."""

    try:
        import cortex
    except ImportError as exc:
        raise RuntimeError(
            "Pycortex flatmaps require a Python environment with pycortex installed."
        ) from exc
    try:
        cortex.db.get_surf(subject, "flat", merge=True)
    except Exception as exc:
        raise RuntimeError(
            f"Pycortex subject {subject!r} has no usable flat surface."
        ) from exc
    return cortex


def prepare_fsaverage5_vector(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    expected = 10242 * 2
    if data.size != expected:
        raise ValueError(f"Expected 20,484 fsaverage5 vertices, received {data.size}.")
    return data


def project_volume_to_fsaverage5(volume_path: Path) -> np.ndarray:
    """Project an MNI volume to fsaverage5, left hemisphere followed by right."""

    if not volume_path.is_file():
        raise FileNotFoundError(volume_path)
    fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage5")
    image = nib.load(str(volume_path))
    hemispheres = []
    for hemi in ("left", "right"):
        projected = surface.vol_to_surf(
            image,
            fsaverage5[f"pial_{hemi}"],
            inner_mesh=fsaverage5[f"white_{hemi}"],
            kind="depth",
            n_samples=5,
            interpolation="linear",
        )
        hemispheres.append(np.asarray(projected, dtype=np.float32))
    result = np.concatenate(hemispheres)
    if result.size != 20484:
        raise RuntimeError(f"Unexpected fsaverage5 projection length: {result.size}")
    return result


def render_pycortex_flatmap(
    fsaverage5_data: np.ndarray,
    title: str,
    output_stem: Path,
    vmin: float,
    vmax: float,
    cmap: str,
    colorbar_label: str,
    args: argparse.Namespace,
) -> None:
    """Render one bilateral paper flatmap using the established project style."""

    configure_paper_font()
    cortex = load_pycortex(args.pycortex_subject)
    subjects_dir = args.subjects_dir.resolve()
    source_subject = subjects_dir / args.freesurfer_subject
    if not source_subject.is_dir():
        raise FileNotFoundError(f"FreeSurfer subject not found: {source_subject}")

    data_fsaverage = cortex.freesurfer.upsample_to_fsaverage(
        prepare_fsaverage5_vector(fsaverage5_data),
        args.freesurfer_subject,
        freesurfer_subjects_dir=str(subjects_dir),
    )
    vertex = cortex.Vertex(
        data_fsaverage,
        args.pycortex_subject,
        vmin=float(vmin),
        vmax=float(vmax),
        cmap=cmap,
    )
    inkscape_available = shutil.which("inkscape") is not None
    fig = cortex.quickflat.make_figure(
        vertex,
        with_curvature=True,
        with_colorbar=False,
        with_rois=bool(args.pycortex_with_rois and inkscape_available),
        recache=bool(args.pycortex_recache),
        with_sulci=inkscape_available,
        with_labels=False,
        with_borders=inkscape_available,
        with_dropout=False,
        extra_disp=None,
        with_connected_vertices=False,
        overlay_file=None,
    )
    fig.set_size_inches(14, 5.8)
    main_ax = fig.axes[0]
    main_ax.set_position([0.015, 0.19, 0.97, 0.66])
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.965)

    colorbar_ax = fig.add_axes([0.34, 0.075, 0.32, 0.032])
    scalar_map = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=float(vmin), vmax=float(vmax)),
        cmap=plt.get_cmap(cmap),
    )
    scalar_map.set_array([])
    colorbar = fig.colorbar(
        scalar_map,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=np.linspace(float(vmin), float(vmax), 6),
    )
    colorbar.set_label(colorbar_label, fontsize=15, labelpad=3)
    colorbar.ax.tick_params(labelsize=12, length=3, pad=2)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in args.pycortex_formats:
        output_path = output_stem.with_suffix(f".{suffix}")
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            save_kwargs["dpi"] = args.pycortex_dpi
        fig.savefig(output_path, **save_kwargs)
        logging.info("Saved pycortex flatmap: %s", output_path)
    plt.close(fig)


def roi_mask_for_model(output_dir: Path, roi_name: str) -> Path:
    mask_dir = output_dir / "roi_masks"
    if roi_name == "combined":
        return mask_dir / "schaefer200_17net_task_combined_commonmask.nii.gz"
    return mask_dir / f"schaefer200_17net_{roi_name}_commonmask.nii.gz"


def parcel_stability_from_voxel_map(
    frequency_path: Path,
    atlas_path: Path,
    roi_mask_path: Path,
    task_parcels: pd.DataFrame,
    stable_threshold: float,
    output_dir: Path,
) -> Path:
    """Aggregate voxel selection frequency within each Schaefer parcel."""

    frequency_img = nib.load(str(frequency_path))
    frequency = np.nan_to_num(
        np.asarray(frequency_img.dataobj, dtype=np.float32), nan=0.0
    )
    atlas = np.rint(np.asarray(nib.load(str(atlas_path)).dataobj)).astype(int)
    roi_mask = np.asarray(nib.load(str(roi_mask_path)).dataobj) > 0
    parcel_volume = np.zeros(frequency.shape, dtype=np.float32)
    rows = []
    name_map = dict(zip(task_parcels["parcel_id"], task_parcels["parcel_name"]))
    family_map = dict(zip(task_parcels["parcel_id"], task_parcels["network_family"]))
    for parcel_id in sorted(np.unique(atlas[roi_mask])):
        if parcel_id <= 0:
            continue
        mask = (atlas == parcel_id) & roi_mask
        values = frequency[mask]
        if not len(values):
            continue
        mean_frequency = float(values.mean())
        stable_fraction = float(np.mean(values >= stable_threshold))
        parcel_volume[mask] = mean_frequency
        rows.append(
            {
                "parcel_id": int(parcel_id),
                "parcel_name": name_map.get(parcel_id, ""),
                "network_family": family_map.get(parcel_id, ""),
                "n_roi_voxels": int(len(values)),
                "mean_selection_frequency": mean_frequency,
                "max_selection_frequency": float(values.max()),
                "stable_voxel_count": int(np.sum(values >= stable_threshold)),
                "stable_voxel_fraction": stable_fraction,
                "stable_threshold": float(stable_threshold),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        ["stable_voxel_fraction", "mean_selection_frequency"],
        ascending=False,
    ).to_csv(output_dir / "parcel_selection_stability.csv", index=False)
    output_path = output_dir / "parcel_mean_selection_frequency.nii.gz"
    nib.save(
        nib.Nifti1Image(parcel_volume, frequency_img.affine),
        output_path,
    )
    return output_path


def discover_pycortex_models(
    output_dir: Path, model_set: str
) -> list[tuple[str, str, str]]:
    if model_set == "best":
        models = list(PYCORTEX_BEST_MODELS)
    else:
        models = []
        root = output_dir / "cross_subject"
        for metrics_path in sorted(root.glob("*/*/*/metrics.json")):
            models.append(tuple(metrics_path.parts[-4:-1]))
    missing = [
        model
        for model in models
        if not (
            output_dir
            / "cross_subject"
            / model[0]
            / model[1]
            / model[2]
            / "selected_voxel_frequency.nii.gz"
        ).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"MVPA result maps are missing for: {missing}")
    return models


def run_pycortex_flatmaps(output_dir: Path, args: argparse.Namespace) -> None:
    """Create paper-ready cortical flatmaps from existing MVPA results."""

    load_pycortex(args.pycortex_subject)
    atlas_path = (
        output_dir / "roi_masks" / "schaefer200_17net_2mm_resampled_to_beta.nii.gz"
    )
    task_parcels = pd.read_csv(output_dir / "schaefer_task_parcels.tsv", sep="\t")
    flatmap_root = output_dir / "pycortex_flatmaps"

    for task, roi_name, classifier in discover_pycortex_models(
        output_dir, args.pycortex_model_set
    ):
        model_dir = output_dir / "cross_subject" / task / roi_name / classifier
        figure_dir = flatmap_root / f"{task}__{roi_name}__{classifier}"
        frequency_path = model_dir / "selected_voxel_frequency.nii.gz"
        fsaverage5_frequency = project_volume_to_fsaverage5(frequency_path)
        stable_surface = fsaverage5_frequency.copy()
        stable_surface[stable_surface < args.pycortex_selection_threshold] = np.nan
        render_pycortex_flatmap(
            stable_surface,
            (
                f"{task.replace('_', ' ').title()} | "
                f"{roi_name.replace('_', ' ').title()} | "
                f"{classifier.upper()}\nStable voxel selection"
            ),
            figure_dir / "stable_voxel_selection_frequency_flatmap",
            args.pycortex_selection_threshold,
            1.0,
            "inferno",
            "Outer-fold selection frequency",
            args,
        )

        parcel_map_path = parcel_stability_from_voxel_map(
            frequency_path,
            atlas_path,
            roi_mask_for_model(output_dir, roi_name),
            task_parcels,
            args.pycortex_selection_threshold,
            figure_dir,
        )
        parcel_surface = project_volume_to_fsaverage5(parcel_map_path)
        parcel_surface[parcel_surface < args.pycortex_parcel_threshold] = np.nan
        render_pycortex_flatmap(
            parcel_surface,
            (
                f"{task.replace('_', ' ').title()} | "
                f"{roi_name.replace('_', ' ').title()} | "
                f"{classifier.upper()}\nSchaefer parcel mean stability"
            ),
            figure_dir / "parcel_mean_selection_frequency_flatmap",
            args.pycortex_parcel_threshold,
            1.0,
            "viridis",
            "Mean voxel selection frequency",
            args,
        )

    statistical_dir = output_dir / "descriptive_responses" / "statistical_maps"
    contrast_dir = flatmap_root / "paired_beta_contrasts_fdr05"
    for map_path in sorted(
        statistical_dir.glob("*_parcel_beta_difference_fdr05.nii.gz")
    ):
        contrast = map_path.name.replace("_parcel_beta_difference_fdr05.nii.gz", "")
        contrast_surface = project_volume_to_fsaverage5(map_path)
        contrast_surface[np.isclose(contrast_surface, 0.0, atol=1e-8)] = np.nan
        finite = np.isfinite(contrast_surface)
        if not finite.any():
            logging.warning("No FDR-significant parcel for %s", contrast)
            continue
        limit = float(np.nanpercentile(np.abs(contrast_surface[finite]), 98))
        if limit <= 0:
            continue
        render_pycortex_flatmap(
            contrast_surface,
            f"{contrast.replace('_', ' ').title()} | paired permutation FDR < .05",
            contrast_dir / f"{contrast}_fdr05_flatmap",
            -limit,
            limit,
            "RdBu_r",
            "Mean beta difference",
            args,
        )
    logging.info("Pycortex paper flatmaps complete: %s", flatmap_root)


def apply_quick_mode(
    args: argparse.Namespace, all_subject_ids: Sequence[int]
) -> argparse.Namespace:
    if not args.quick:
        return args
    if not args.subjects:
        args.subjects = list(all_subject_ids[:12])
    args.outer_folds = min(args.outer_folds, 3)
    args.inner_folds = min(args.inner_folds, 2)
    args.k_grid = [10, 25]
    args.mlp_alpha_grid = [0.01]
    args.svm_c_grid = [0.1]
    args.max_iter = min(args.max_iter, 80)
    args.bootstrap = min(args.bootstrap, 100)
    args.n_permutations = min(args.n_permutations, 1000)
    args.tasks = ["three_way"]
    args.roi_groups = ["combined"]
    logging.info("Quick mode enabled for subjects: %s", args.subjects)
    return args


def main() -> None:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else args.work_dir
    )
    args.atlas_dir = (
        args.atlas_dir.resolve() if args.atlas_dir is not None else DEFAULT_ATLAS_DIR
    )
    log_filename = (
        "mvpa_flatmaps.log"
        if args.analysis == "flatmaps"
        else "mvpa_statistics.log"
        if args.analysis == "statistics"
        else "mvpa.log"
    )
    setup_logging(args.output_dir, filename=log_filename)

    label_file = args.output_dir / "labels.csv"
    audit_file = args.output_dir / "data_audit.csv"
    manifest, subject_dirs, reference_img = build_manifest(
        args.data_dir.resolve(), label_file, audit_file
    )
    logging.info("Wrote %s and %s", label_file, audit_file)
    if args.analysis == "prepare":
        return
    if args.analysis == "statistics":
        run_existing_repeated_measures_statistics(
            args.output_dir, args.n_permutations, args.seed
        )
        return
    if args.analysis == "flatmaps":
        # Generate inferential tables/maps if the user has older MVPA results
        # created before this post-processing step was added.
        stats_sentinel = (
            args.output_dir / "descriptive_responses" / "parcel_paired_permutation.csv"
        )
        if not stats_sentinel.is_file():
            run_existing_repeated_measures_statistics(
                args.output_dir, args.n_permutations, args.seed
            )
        run_pycortex_flatmaps(args.output_dir, args)
        return

    all_subject_ids = sorted(manifest["subject_id"].unique())
    args = apply_quick_mode(args, all_subject_ids)
    if args.subjects:
        requested = set(args.subjects)
        missing = sorted(requested.difference(all_subject_ids))
        if missing:
            raise ValueError(f"Requested subject IDs not found: {missing}")
        analysis_manifest = manifest[
            manifest["subject_id"].isin(requested)
        ].reset_index(drop=True)
    else:
        analysis_manifest = manifest.copy()

    # The atlas/common mask is defined from the full audited cohort, even for
    # quick tests, so feature definitions remain stable across runs.
    _, atlas_data, _, family_masks, task_parcels = prepare_schaefer_masks(
        args.atlas_dir,
        args.output_dir,
        reference_img,
        subject_dirs,
    )
    X, combined_flat_indices = extract_combined_features(
        analysis_manifest, family_masks["combined"]
    )
    roi_columns = roi_column_indices(family_masks, combined_flat_indices)
    parcel_columns = parcel_column_indices(
        atlas_data, combined_flat_indices, task_parcels
    )

    save_descriptive_responses(
        X,
        analysis_manifest,
        roi_columns,
        parcel_columns,
        task_parcels,
        args.output_dir,
    )
    run_existing_repeated_measures_statistics(
        args.output_dir, args.n_permutations, args.seed
    )

    if args.analysis in ("cross", "all"):
        run_cross_subject(
            X,
            analysis_manifest,
            roi_columns,
            reference_img,
            combined_flat_indices,
            atlas_data,
            task_parcels,
            args.output_dir,
            args,
        )

    if args.analysis in ("within", "all"):
        network_X, network_names = network_mean_features(X, roi_columns)
        run_within_subject(
            network_X,
            network_names,
            analysis_manifest,
            args.output_dir,
            args,
        )

    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(run_config), f, indent=2)
    logging.info("Analysis completed. Results: %s", args.output_dir)


if __name__ == "__main__":
    main()
