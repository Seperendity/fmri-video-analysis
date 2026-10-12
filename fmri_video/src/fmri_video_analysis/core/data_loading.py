"""Shared loading utilities for task-network beta-pattern analyses."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fmri_video_analysis.core.config import ROI_REFERENCE_DIR
from fmri_video_analysis.core.validation import validate_nifti_geometry

NETWORK_MASKS = {
    "visual": "schaefer200_17net_visual_commonmask.nii.gz",
    "dorsal_attention": "schaefer200_17net_dorsal_attention_commonmask.nii.gz",
    "salience_attention": "schaefer200_17net_salience_attention_commonmask.nii.gz",
    "control": "schaefer200_17net_control_commonmask.nii.gz",
    "task_combined": "schaefer200_17net_task_combined_commonmask.nii.gz",
}


def load_resms_volume(
    first_level_dir: Path,
    subject_id: int,
    reference_img,
) -> tuple[Path, np.ndarray]:
    """Load and validate a subject's first-level GLM residual mean-square map."""

    path = Path(first_level_dir) / str(int(subject_id)) / "ResMS.nii"
    if not path.is_file():
        raise FileNotFoundError(f"Missing first-level residual variance map: {path}")
    image = nib.load(str(path))
    validate_nifti_geometry(
        reference_img,
        image,
        context=f"analysis reference and ResMS {path}",
    )
    variance = np.asarray(image.dataobj, dtype=np.float64)
    return path, variance


def load_task_network_features(table: pd.DataFrame):
    mask_dir = ROI_REFERENCE_DIR / "roi_masks"
    union_img = nib.load(str(mask_dir / NETWORK_MASKS["task_combined"]))
    union = np.asarray(union_img.dataobj) > 0
    flat_indices = np.flatnonzero(union.ravel())
    X = np.empty((len(table), len(flat_indices)), dtype=np.float32)
    for row, path in enumerate(table.beta_path):
        image = nib.load(str(path))
        validate_nifti_geometry(
            union_img, image, context=f"task-network union mask and beta {path}"
        )
        X[row] = np.asarray(image.dataobj, dtype=np.float32).ravel()[flat_indices]
        if (row + 1) % 100 == 0:
            print(f"Loaded {row + 1}/{len(table)} beta maps", flush=True)
    keep = np.all(np.isfinite(X), axis=0) & (X.std(axis=0) > 1e-8)
    X = X[:, keep]
    flat_indices = flat_indices[keep]
    network_columns = {}
    for network, filename in NETWORK_MASKS.items():
        mask_img = nib.load(str(mask_dir / filename))
        validate_nifti_geometry(
            union_img, mask_img, context=f"network mask {network}"
        )
        mask = np.asarray(mask_img.dataobj) > 0
        network_columns[network] = mask.ravel()[flat_indices]
        if not np.any(network_columns[network]):
            raise RuntimeError(f"Empty network mask after filtering: {network}")
    return X, union_img, flat_indices, network_columns
