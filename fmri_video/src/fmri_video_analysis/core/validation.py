"""Shared data-contract validation for maintained analyses."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np

from fmri_video_analysis.core.study_design import INCLUDED_SUBJECT_IDS


def validate_subject_ids(
    actual_ids: Iterable[int],
    *,
    expected_ids: Iterable[int] = INCLUDED_SUBJECT_IDS,
    context: str = "subject cohort",
    allow_subset: bool = False,
) -> tuple[int, ...]:
    """Validate a cohort against the frozen sample and return sorted IDs."""

    actual_list = [int(value) for value in actual_ids]
    duplicates = sorted(
        {value for value in actual_list if actual_list.count(value) > 1}
    )
    actual = set(actual_list)
    expected = {int(value) for value in expected_ids}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if duplicates or extra or (missing and not allow_subset):
        details = []
        if missing and not allow_subset:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        if duplicates:
            details.append(f"duplicates={duplicates}")
        raise RuntimeError(f"Invalid {context}: " + "; ".join(details))
    return tuple(sorted(actual))


def _as_image(image_or_path):
    if isinstance(image_or_path, (str, Path)):
        return nib.load(str(image_or_path))
    return image_or_path


def validate_nifti_geometry(
    reference,
    candidate,
    *,
    context: str = "NIfTI geometry",
    affine_atol: float = 1e-5,
) -> None:
    """Require matching spatial shape, affine, orientation and voxel sizes."""

    reference_image = _as_image(reference)
    candidate_image = _as_image(candidate)
    reference_shape = tuple(reference_image.shape[:3])
    candidate_shape = tuple(candidate_image.shape[:3])
    reference_zooms = tuple(float(v) for v in reference_image.header.get_zooms()[:3])
    candidate_zooms = tuple(float(v) for v in candidate_image.header.get_zooms()[:3])
    reference_orientation = nib.aff2axcodes(reference_image.affine)
    candidate_orientation = nib.aff2axcodes(candidate_image.affine)

    differences = []
    if candidate_shape != reference_shape:
        differences.append(f"shape {candidate_shape} != {reference_shape}")
    if not np.allclose(
        candidate_image.affine,
        reference_image.affine,
        atol=affine_atol,
        rtol=0,
    ):
        differences.append("affine differs")
    if candidate_orientation != reference_orientation:
        differences.append(
            f"orientation {candidate_orientation} != {reference_orientation}"
        )
    if not np.allclose(candidate_zooms, reference_zooms, atol=1e-6, rtol=0):
        differences.append(f"voxel sizes {candidate_zooms} != {reference_zooms}")
    if differences:
        raise RuntimeError(f"Geometry mismatch for {context}: " + "; ".join(differences))
