#!/usr/bin/env python3
"""Shared constants and audits for the four-session fMRI revision analyses."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fmri_video_analysis.core.environment import runtime_metadata
from fmri_video_analysis.core.provenance import atomic_write_json
from fmri_video_analysis.core.study_design import INCLUDED_SUBJECT_IDS
from fmri_video_analysis.core.validation import validate_subject_ids

# Resolve all maintained paths from the reorganized project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT  # Compatibility name used by reporting modules.
DATA_DIR = PROJECT_ROOT / "data"
FIRST_LEVEL_DIR = DATA_DIR / "derivatives/spm_first_level_4sessions"
STIMULUS_FRAMES_DIR = DATA_DIR / "stimuli/video_frames"
RESOURCES_DIR = PROJECT_ROOT / "resources"
ATLAS_DIR = RESOURCES_DIR / "atlases"
ROI_REFERENCE_DIR = DATA_DIR / "derivatives/mvpa_reference"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
HANDOFF_RESULTS_DIR = OUTPUTS_DIR / "handoff_baseline"
ORIGINAL_MVPA_RESULTS_DIR = HANDOFF_RESULTS_DIR / "original_mvpa"
CODE_DIR = PROJECT_ROOT / "src/fmri_video_analysis"
RUN_ID = os.environ.get("FMRI_VIDEO_RUN_ID") or datetime.now(timezone.utc).strftime(
    "run_%Y%m%d_%H%M%S"
)
RESULTS_DIR = OUTPUTS_DIR / "runs" / RUN_ID

CONDITION_BETAS = {
    "Simple": {1: 1, 2: 10, 3: 19, 4: 28},
    "Complex": {1: 2, 2: 11, 3: 20, 4: 29},
    "Control": {1: 3, 2: 12, 3: 21, 4: 30},
}

SESSION_VIDEOS = {
    1: {"Simple": "001", "Complex": "009", "Control": "017"},
    2: {"Simple": "002", "Complex": "010", "Control": "018"},
    3: {"Simple": "003", "Complex": "011", "Control": "019"},
    4: {"Simple": "004", "Complex": "012", "Control": "020"},
}

CONTRAST_FILES = {
    "simple": ("con_0002.nii", 1.0),
    "complex": ("con_0003.nii", 1.0),
    "control": ("con_0004.nii", 1.0),
    "complex_minus_simple": ("con_0005.nii", -1.0),
    "simple_minus_control": ("con_0006.nii", 1.0),
    "complex_minus_control": ("con_0007.nii", 1.0),
}

EXPECTED_SUBJECTS = 87
EXPECTED_FRAMES_PER_VIDEO = 1499


def numeric_subject_dirs(root: Path = FIRST_LEVEL_DIR) -> list[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def nifti_description(path: Path) -> str:
    import nibabel as nib

    img = nib.load(str(path))
    raw = img.header["descrip"].tobytes().split(b"\x00", 1)[0]
    return raw.decode("utf-8", errors="replace").strip()


def build_four_session_manifest(
    first_level_dir: Path = FIRST_LEVEL_DIR,
) -> pd.DataFrame:
    """Audit and return one row per subject/session/condition beta image."""

    rows: list[dict] = []
    errors: list[str] = []
    subjects = numeric_subject_dirs(first_level_dir)
    try:
        validate_subject_ids(
            (int(path.name) for path in subjects),
            context=f"four-session beta cohort under {first_level_dir}",
        )
    except RuntimeError as exc:
        errors.append(str(exc))
    for subject_dir in subjects:
        for condition, session_map in CONDITION_BETAS.items():
            for session, beta_index in session_map.items():
                path = subject_dir / f"beta_{beta_index:04d}.nii"
                if not path.is_file():
                    errors.append(f"Missing {path}")
                    continue
                description = nifti_description(path)
                expected = re.compile(
                    rf"Sn\({session}\)\s+{condition}(?:\*bf\(1\))?",
                    re.IGNORECASE,
                )
                if not expected.search(description):
                    errors.append(
                        f"Header mismatch {path}: expected session {session} "
                        f"{condition}, got {description!r}"
                    )
                rows.append(
                    {
                        "subject_id": int(subject_dir.name),
                        "session": session,
                        "video_id": SESSION_VIDEOS[session][condition],
                        "condition": condition,
                        "class_id": {"Simple": 0, "Complex": 1, "Control": 2}[
                            condition
                        ],
                        "beta_index": beta_index,
                        "beta_path": str(path),
                        "header_description": description,
                    }
                )
    manifest = pd.DataFrame(rows).sort_values(["subject_id", "session", "class_id"])
    expected_rows = len(subjects) * 12
    if len(manifest) != expected_rows:
        errors.append(f"Expected {expected_rows} beta rows, found {len(manifest)}.")
    if errors:
        raise RuntimeError("\n".join(errors[:50]))
    return manifest


def audit_contrasts(first_level_dir: Path = FIRST_LEVEL_DIR) -> pd.DataFrame:
    rows = []
    errors = []
    subjects = numeric_subject_dirs(first_level_dir)
    try:
        validate_subject_ids(
            (int(path.name) for path in subjects),
            context=f"contrast cohort under {first_level_dir}",
        )
    except RuntimeError as exc:
        errors.append(str(exc))
    for subject_dir in subjects:
        for contrast, (filename, sign) in CONTRAST_FILES.items():
            path = subject_dir / filename
            exists = path.is_file()
            rows.append(
                {
                    "subject_id": int(subject_dir.name),
                    "contrast": contrast,
                    "contrast_file": str(path),
                    "sign": sign,
                    "exists": exists,
                }
            )
            if not exists:
                errors.append(f"Missing {path}")
    table = pd.DataFrame(rows)
    if errors:
        raise RuntimeError("\n".join(errors[:50]))
    return table


def write_analysis_config(
    output_dir: Path = RESULTS_DIR,
    *,
    first_level_dir: Path = FIRST_LEVEL_DIR,
    stimulus_frames_dir: Path = STIMULUS_FRAMES_DIR,
    timeseries_root: Path | None = None,
    subject_ids: tuple[int, ...] = INCLUDED_SUBJECT_IDS,
) -> Path:
    """Write the resolved configuration actually used by the revision audit."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "analysis_config.json"
    run_dir = output_dir.parent if output_dir.name == "metadata" else output_dir
    payload = {
        "schema_version": 2,
        "run_id": run_dir.resolve().name,
        "project_root": str(PROJECT_ROOT),
        "first_level_dir": str(Path(first_level_dir).resolve()),
        "stimulus_frame_dir": str(Path(stimulus_frames_dir).resolve()),
        "timeseries_root": (
            str(Path(timeseries_root).resolve()) if timeseries_root else None
        ),
        "atlas_dir": str(ATLAS_DIR.resolve()),
        "roi_reference_dir": str(ROI_REFERENCE_DIR.resolve()),
        "output_dir": str(output_dir.resolve()),
        "analysis_scope": f"{len(subject_ids)} subjects; sessions 1-4 only",
        "subject_ids": [int(subject_id) for subject_id in subject_ids],
        "condition_betas": CONDITION_BETAS,
        "session_videos": SESSION_VIDEOS,
        "contrast_files": CONTRAST_FILES,
        "frame_rate_hz": 25,
        "frame_duration_ms": 40,
        "expected_frames_per_video": EXPECTED_FRAMES_PER_VIDEO,
        "runtime": runtime_metadata(),
    }
    return atomic_write_json(path, payload)
