"""Background functional connectivity for the frozen four-run design."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.linalg import qr

from fmri_video_analysis.core.provenance import (
    atomic_write_json,
    cache_manifest,
    file_fingerprint,
    require_matching_cache,
)
from fmri_video_analysis.core.provenance import (
    sha256_file as provenance_sha256,
)
from fmri_video_analysis.core.study_design import (
    BLOCK_DURATION_TR,
    CONDITIONS,
    N_VOLUMES_PER_RUN,
    RUN_IDS,
    TR_SECONDS,
    blocks_for_run,
)
from fmri_video_analysis.methods.connectivity.composite_atlas import sha256_file

MOTION_COLUMN_NAMES = ("x", "y", "z", "pitch", "roll", "yaw")


def input_paths(timeseries_root: Path, subject_id: int, run_id: str) -> tuple[Path, Path]:
    subject_dir = timeseries_root / str(subject_id)
    image = subject_dir / f"wraSMS_BOLD_2MM_REST_{run_id}.nii"
    motion = subject_dir / f"rp_aSMS_BOLD_2MM_REST_{run_id}.txt"
    return image, motion


def _header_sha256(image: nib.spatialimages.SpatialImage) -> str:
    digest = hashlib.sha256()
    digest.update(image.header.binaryblock)
    digest.update(np.asarray(image.affine, dtype=np.float64).tobytes())
    return digest.hexdigest()


def audit_timeseries_inputs(
    timeseries_root: Path,
    subject_ids: tuple[int, ...] | list[int],
    hash_inputs: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Audit one wra image and one realignment file per subject and run."""

    rows: list[dict] = []
    errors: list[str] = []
    reference_shape: tuple[int, int, int] | None = None
    reference_affine: np.ndarray | None = None
    for subject_id in subject_ids:
        for run_id in RUN_IDS:
            image_path, motion_path = input_paths(
                timeseries_root, subject_id, run_id
            )
            row = {
                "subject_id": subject_id,
                "run_id": run_id,
                "image_path": str(image_path),
                "motion_path": str(motion_path),
                "image_exists": image_path.is_file(),
                "motion_exists": motion_path.is_file(),
            }
            if not image_path.is_file() or not motion_path.is_file():
                if not image_path.is_file():
                    errors.append(f"Missing image: {image_path}")
                if not motion_path.is_file():
                    errors.append(f"Missing motion: {motion_path}")
                rows.append(row)
                continue
            try:
                image = nib.load(str(image_path))
                row.update(
                    {
                        "image_shape": "x".join(str(value) for value in image.shape),
                        "n_volumes": image.shape[3] if len(image.shape) == 4 else None,
                        "voxel_x_mm": float(image.header.get_zooms()[0]),
                        "voxel_y_mm": float(image.header.get_zooms()[1]),
                        "voxel_z_mm": float(image.header.get_zooms()[2]),
                        "tr_seconds": (
                            float(image.header.get_zooms()[3])
                            if len(image.shape) == 4
                            else None
                        ),
                        "image_bytes": image_path.stat().st_size,
                        "header_sha256": _header_sha256(image),
                        "image_sha256": (
                            sha256_file(image_path) if hash_inputs else None
                        ),
                    }
                )
                if len(image.shape) != 4 or image.shape[3] != N_VOLUMES_PER_RUN:
                    errors.append(
                        f"{image_path}: expected 4D/130 TR, got {image.shape}."
                    )
                if len(image.shape) == 4 and not np.isclose(
                    image.header.get_zooms()[3], TR_SECONDS
                ):
                    errors.append(
                        f"{image_path}: expected TR={TR_SECONDS}, got "
                        f"{image.header.get_zooms()[3]}."
                    )
                if reference_shape is None:
                    reference_shape = image.shape[:3]
                    reference_affine = image.affine
                elif image.shape[:3] != reference_shape or not np.allclose(
                    image.affine, reference_affine, atol=1e-5
                ):
                    errors.append(
                        f"{image_path}: spatial grid differs from the first input."
                    )
            except Exception as exc:  # noqa: BLE001 - audit every unreadable input
                errors.append(f"Cannot read image {image_path}: {exc}")
            try:
                motion = np.loadtxt(motion_path, dtype=float)
                row.update(
                    {
                        "motion_rows": motion.shape[0] if motion.ndim == 2 else None,
                        "motion_columns": (
                            motion.shape[1] if motion.ndim == 2 else None
                        ),
                        "motion_sha256": sha256_file(motion_path),
                    }
                )
                if motion.shape != (N_VOLUMES_PER_RUN, 6):
                    errors.append(
                        f"{motion_path}: expected (130, 6), got {motion.shape}."
                    )
                elif not np.all(np.isfinite(motion)):
                    errors.append(f"{motion_path}: non-finite values.")
            except Exception as exc:  # noqa: BLE001 - audit every unreadable input
                errors.append(f"Cannot read motion {motion_path}: {exc}")
            rows.append(row)
    return pd.DataFrame(rows), errors


def motion_24_parameters(motion: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return Friston-style 24P regressors with derivatives reset per run."""

    motion = np.asarray(motion, dtype=float)
    if motion.ndim != 2 or motion.shape[1] != 6:
        raise ValueError(f"Motion must have shape (n, 6), got {motion.shape}.")
    derivative = np.vstack([np.zeros((1, 6)), np.diff(motion, axis=0)])
    values = np.column_stack(
        (motion, derivative, np.square(motion), np.square(derivative))
    )
    names = tuple(
        [f"motion_{name}" for name in MOTION_COLUMN_NAMES]
        + [f"motion_derivative_{name}" for name in MOTION_COLUMN_NAMES]
        + [f"motion_squared_{name}" for name in MOTION_COLUMN_NAMES]
        + [
            f"motion_derivative_squared_{name}"
            for name in MOTION_COLUMN_NAMES
        ]
    )
    return values, names


def power_fd(motion: np.ndarray, head_radius_mm: float = 50.0) -> np.ndarray:
    """Calculate Power FD from SPM translations (mm) and rotations (radians)."""

    motion = np.asarray(motion, dtype=float)
    if motion.ndim != 2 or motion.shape[1] != 6:
        raise ValueError(f"Motion must have shape (n, 6), got {motion.shape}.")
    derivative = np.vstack([np.zeros((1, 6)), np.diff(motion, axis=0)])
    fd = np.abs(derivative[:, :3]).sum(axis=1)
    fd += head_radius_mm * np.abs(derivative[:, 3:]).sum(axis=1)
    fd[0] = 0.0
    return fd


def dct_high_pass_basis(
    n_volumes: int,
    tr_seconds: float = TR_SECONDS,
    cutoff_seconds: float = 128.0,
) -> np.ndarray:
    """SPM-compatible DCT drift terms, excluding the constant column."""

    if n_volumes < 2 or tr_seconds <= 0 or cutoff_seconds <= 0:
        raise ValueError("Invalid DCT high-pass parameters.")
    n_terms = int(np.floor(2 * n_volumes * tr_seconds / cutoff_seconds))
    sample = np.arange(n_volumes, dtype=float) + 0.5
    frequencies = np.arange(1, n_terms + 1, dtype=float)
    if n_terms == 0:
        return np.empty((n_volumes, 0), dtype=float)
    return np.sqrt(2.0 / n_volumes) * np.cos(
        np.pi * sample[:, None] * frequencies[None, :] / n_volumes
    )


def fir_task_design(
    run_ids: tuple[str, ...] = RUN_IDS,
    n_volumes_per_run: int = N_VOLUMES_PER_RUN,
    fir_lags: int = 35,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct condition-specific FIR columns across all four runs."""

    if fir_lags < BLOCK_DURATION_TR:
        raise ValueError("FIR must cover at least the 30-TR task block.")
    total_volumes = len(run_ids) * n_volumes_per_run
    matrix = np.zeros((total_volumes, len(CONDITIONS) * fir_lags), dtype=float)
    names: list[str] = []
    for condition_index, condition in enumerate(CONDITIONS):
        for lag in range(fir_lags):
            column = condition_index * fir_lags + lag
            names.append(f"fir_{condition}_lag_{lag:02d}")
            for run_index, run_id in enumerate(run_ids):
                onset = next(
                    onset
                    for block_condition, onset, _ in blocks_for_run(run_id)
                    if block_condition == condition
                )
                row = run_index * n_volumes_per_run + onset + lag
                if row < (run_index + 1) * n_volumes_per_run:
                    matrix[row, column] = 1.0
    return matrix, tuple(names)


def build_joint_design(
    motions: dict[str, np.ndarray],
    fd_threshold: float = 0.5,
    head_radius_mm: float = 50.0,
    high_pass_seconds: float = 128.0,
    fir_lags: int = 35,
    motion_model: str = "24P",
    include_fir: bool = True,
    add_fd_spikes: bool = True,
    global_signals: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, np.ndarray], dict]:
    """Build the joint 520-TR nuisance and FIR design matrix."""

    if tuple(motions) != RUN_IDS:
        raise ValueError(f"Motion runs must be ordered exactly as {RUN_IDS}.")
    total_volumes = len(RUN_IDS) * N_VOLUMES_PER_RUN
    columns: list[np.ndarray] = []
    names: list[str] = []
    fd_by_run: dict[str, np.ndarray] = {}

    for run_index, run_id in enumerate(RUN_IDS):
        row_slice = slice(
            run_index * N_VOLUMES_PER_RUN,
            (run_index + 1) * N_VOLUMES_PER_RUN,
        )
        intercept = np.zeros(total_volumes)
        intercept[row_slice] = 1.0
        columns.append(intercept[:, None])
        names.append(f"run_{run_id}_intercept")

        if motion_model == "24P":
            motion_regressors, motion_names = motion_24_parameters(
                motions[run_id]
            )
        elif motion_model == "6P":
            motion_regressors = np.asarray(motions[run_id], dtype=float)
            motion_names = tuple(
                f"motion_{name}" for name in MOTION_COLUMN_NAMES
            )
        else:
            raise ValueError(
                f"motion_model must be '24P' or '6P', got {motion_model!r}."
            )
        if motion_regressors.shape[0] != N_VOLUMES_PER_RUN:
            raise ValueError(
                f"Run {run_id}: expected 130 motion rows, got "
                f"{motion_regressors.shape[0]}."
            )
        run_motion = np.zeros(
            (total_volumes, motion_regressors.shape[1])
        )
        run_motion[row_slice] = motion_regressors
        columns.append(run_motion)
        names.extend(f"run_{run_id}_{name}" for name in motion_names)

        dct = dct_high_pass_basis(
            N_VOLUMES_PER_RUN, TR_SECONDS, high_pass_seconds
        )
        run_dct = np.zeros((total_volumes, dct.shape[1]))
        run_dct[row_slice] = dct
        columns.append(run_dct)
        names.extend(
            f"run_{run_id}_dct_{index:02d}"
            for index in range(1, dct.shape[1] + 1)
        )
        if global_signals is not None:
            global_signal = np.asarray(
                global_signals[run_id], dtype=float
            ).reshape(-1)
            if len(global_signal) != N_VOLUMES_PER_RUN:
                raise ValueError(
                    f"Run {run_id}: invalid global signal length "
                    f"{len(global_signal)}."
                )
            run_global_signal = np.zeros((total_volumes, 1))
            run_global_signal[row_slice, 0] = global_signal
            columns.append(run_global_signal)
            names.append(f"run_{run_id}_global_signal")
        fd_by_run[run_id] = power_fd(motions[run_id], head_radius_mm)

    spike_locations: list[tuple[str, int, int]] = []
    for run_index, run_id in enumerate(RUN_IDS):
        for volume in np.flatnonzero(fd_by_run[run_id] > fd_threshold):
            global_row = run_index * N_VOLUMES_PER_RUN + int(volume)
            spike_locations.append((run_id, int(volume), global_row))
    if add_fd_spikes and spike_locations:
        spikes = np.zeros((total_volumes, len(spike_locations)))
        for column, (run_id, volume, global_row) in enumerate(spike_locations):
            spikes[global_row, column] = 1.0
            names.append(f"fd_spike_run_{run_id}_volume_{volume:03d}")
        columns.append(spikes)

    if include_fir:
        fir, fir_names = fir_task_design(fir_lags=fir_lags)
        columns.append(fir)
        names.extend(fir_names)
    design = np.column_stack(columns)
    raw_n_columns = design.shape[1]
    raw_rank = int(np.linalg.matrix_rank(design))
    dropped_dependent_columns: list[str] = []
    if raw_rank < raw_n_columns:
        _, _, pivot = qr(design, mode="economic", pivoting=True)
        keep = sorted(int(index) for index in pivot[:raw_rank])
        drop = sorted(set(range(raw_n_columns)) - set(keep))
        dropped_dependent_columns = [names[index] for index in drop]
        design = design[:, keep]
        names = [names[index] for index in keep]
    rank = int(np.linalg.matrix_rank(design))
    diagnostics = {
        "n_rows": design.shape[0],
        "n_columns": design.shape[1],
        "n_columns_before_rank_reduction": raw_n_columns,
        "rank_before_rank_reduction": raw_rank,
        "dropped_dependent_columns": dropped_dependent_columns,
        "rank": rank,
        "full_rank": rank == design.shape[1],
        "residual_dof": design.shape[0] - rank,
        "n_fd_spikes": len(spike_locations),
        "n_fd_spike_regressors": (
            len(spike_locations) if add_fd_spikes else 0
        ),
        "fd_threshold_mm": fd_threshold,
        "head_radius_mm": head_radius_mm,
        "high_pass_seconds": high_pass_seconds,
        "fir_lags": fir_lags,
        "motion_model": motion_model,
        "include_fir": include_fir,
        "add_fd_spikes": add_fd_spikes,
        "global_signal_regression": global_signals is not None,
        "spikes": [
            {"run_id": run_id, "volume": volume, "global_row": global_row}
            for run_id, volume, global_row in spike_locations
        ],
    }
    return design, tuple(names), fd_by_run, diagnostics


def residualize_roi_timeseries(
    roi_by_run: dict[str, np.ndarray], design: np.ndarray
) -> tuple[np.ndarray, dict]:
    """OLS residualize concatenated ROI time series using the frozen design."""

    if tuple(roi_by_run) != RUN_IDS:
        raise ValueError(f"ROI runs must be ordered exactly as {RUN_IDS}.")
    response = np.vstack([roi_by_run[run_id] for run_id in RUN_IDS])
    if response.shape[0] != design.shape[0]:
        raise ValueError(
            f"Design/response row mismatch: {design.shape[0]} vs {response.shape[0]}."
        )
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, response, rcond=None
    )
    residuals = response - design @ coefficients
    diagnostics = {
        "ols_rank": int(rank),
        "residual_dof": int(design.shape[0] - rank),
        "condition_number": (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 0
            else float("inf")
        ),
        "finite_residuals": bool(np.all(np.isfinite(residuals))),
    }
    return residuals, diagnostics


def _fisher_z_correlation(timeseries: np.ndarray) -> np.ndarray:
    if timeseries.ndim != 2 or timeseries.shape[0] < 3:
        raise ValueError("At least three time points are required for FC.")
    variances = timeseries.var(axis=0)
    if np.any(~np.isfinite(variances)) or np.any(variances <= 1e-12):
        bad = np.flatnonzero(
            (~np.isfinite(variances)) | (variances <= 1e-12)
        )
        raise RuntimeError(f"Zero/non-finite variance ROI columns: {bad.tolist()}")
    correlation = np.corrcoef(timeseries, rowvar=False)
    if not np.all(np.isfinite(correlation)):
        raise RuntimeError("Non-finite correlation matrix.")
    fisher_z = np.arctanh(np.clip(correlation, -0.999999, 0.999999))
    np.fill_diagonal(fisher_z, 0.0)
    if not np.allclose(fisher_z, fisher_z.T, atol=1e-7):
        raise RuntimeError("FC matrix is not symmetric.")
    return fisher_z


def condition_connectivity(
    residuals: np.ndarray,
    fd_by_run: dict[str, np.ndarray],
    fd_threshold: float = 0.5,
    hrf_lag_trs: int = 2,
    min_condition_trs: int = 90,
    censor_fd: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Censor, block-demean, concatenate by condition, and calculate FC."""

    expected_rows = len(RUN_IDS) * N_VOLUMES_PER_RUN
    if residuals.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} residual rows, got {residuals.shape[0]}."
        )
    condition_blocks: dict[str, list[np.ndarray]] = {
        condition: [] for condition in CONDITIONS
    }
    block_matrices: list[np.ndarray] = []
    block_rows: list[dict] = []
    for run_index, run_id in enumerate(RUN_IDS):
        run_start = run_index * N_VOLUMES_PER_RUN
        run_residuals = residuals[
            run_start : run_start + N_VOLUMES_PER_RUN
        ]
        fd = fd_by_run[run_id]
        for condition, onset, duration in blocks_for_run(run_id):
            start = onset + hrf_lag_trs
            stop = start + duration
            if start < 0 or stop > N_VOLUMES_PER_RUN:
                raise ValueError(
                    f"Run {run_id} {condition}: shifted window [{start}, {stop}) "
                    "falls outside the run."
                )
            retained = (
                fd[start:stop] <= fd_threshold
                if censor_fd
                else np.ones(duration, dtype=bool)
            )
            block = run_residuals[start:stop][retained]
            if len(block) < 3:
                raise RuntimeError(
                    f"Run {run_id} {condition}: fewer than three retained TRs."
                )
            block = block - block.mean(axis=0, keepdims=True)
            condition_blocks[condition].append(block)
            block_matrices.append(_fisher_z_correlation(block))
            block_rows.append(
                {
                    "run_id": run_id,
                    "condition": condition,
                    "onset_tr": onset,
                    "duration_tr": duration,
                    "hrf_lag_trs": hrf_lag_trs,
                    "window_start_tr": start,
                    "window_stop_tr_exclusive": stop,
                    "retained_trs": int(retained.sum()),
                    "censored_trs": int((~retained).sum()),
                    "mean_fd_mm": float(fd[start:stop].mean()),
                    "max_fd_mm": float(fd[start:stop].max()),
                }
            )

    condition_matrices: list[np.ndarray] = []
    for condition in CONDITIONS:
        combined = np.vstack(condition_blocks[condition])
        if len(combined) < min_condition_trs:
            raise RuntimeError(
                f"{condition}: {len(combined)} retained TRs is below "
                f"the minimum {min_condition_trs}."
            )
        condition_matrices.append(_fisher_z_correlation(combined))
    return (
        np.stack(condition_matrices).astype(np.float32),
        np.stack(block_matrices).astype(np.float32),
        pd.DataFrame(block_rows),
    )


def load_atlas_labels(
    atlas_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(atlas_path))
    labels = np.rint(np.asarray(image.dataobj)).astype(np.int16)
    n_nodes = int(labels.max())
    counts = np.bincount(labels.ravel(), minlength=n_nodes + 1)
    if set(np.unique(labels).tolist()) - {0} != set(range(1, n_nodes + 1)):
        raise RuntimeError("Atlas labels must be contiguous and start at one.")
    if np.any(counts[1 : n_nodes + 1] == 0):
        raise RuntimeError("Atlas contains an empty node.")
    return labels, counts


def extract_roi_timeseries(
    image_path: Path, atlas_labels: np.ndarray, atlas_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean ROI signals and nonzero-temporal-variance coverage."""

    image = nib.load(str(image_path))
    if image.shape[:3] != atlas_labels.shape:
        raise RuntimeError(
            f"{image_path}: image/atlas shape mismatch "
            f"{image.shape[:3]} vs {atlas_labels.shape}."
        )
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.shape != (*atlas_labels.shape, N_VOLUMES_PER_RUN):
        raise RuntimeError(f"{image_path}: unexpected data shape {data.shape}.")
    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"{image_path}: non-finite voxel data.")
    flat_labels = atlas_labels.ravel()
    flat_data = data.reshape((-1, N_VOLUMES_PER_RUN))
    n_nodes = len(atlas_counts) - 1
    variable_voxels = np.std(flat_data, axis=1) > 1e-8
    variable_labels = flat_labels[variable_voxels]
    variable_data = flat_data[variable_voxels]
    variable_counts = np.bincount(
        variable_labels, minlength=n_nodes + 1
    )
    if np.any(variable_counts[1 : n_nodes + 1] == 0):
        empty = (
            np.flatnonzero(variable_counts[1 : n_nodes + 1] == 0) + 1
        )
        raise RuntimeError(
            f"{image_path}: no valid functional voxels for atlas nodes "
            f"{empty.tolist()}."
        )
    roi = np.empty((N_VOLUMES_PER_RUN, n_nodes), dtype=np.float64)
    for volume in range(N_VOLUMES_PER_RUN):
        sums = np.bincount(
            variable_labels,
            weights=variable_data[:, volume],
            minlength=n_nodes + 1,
        )
        roi[volume] = (
            sums[1 : n_nodes + 1] / variable_counts[1 : n_nodes + 1]
        )
    coverage = (
        variable_counts[1 : n_nodes + 1]
        / atlas_counts[1 : n_nodes + 1]
    )
    return roi, coverage


def subject_cache_manifest(
    subject_id: int,
    timeseries_root: Path,
    atlas_path: Path,
    parameters: dict,
    *,
    hash_inputs: bool,
) -> dict:
    """Build the complete cache identity for one subject extraction."""

    run_inputs = {}
    for run_id in RUN_IDS:
        image_path, motion_path = input_paths(timeseries_root, subject_id, run_id)
        run_inputs[run_id] = {
            "image": file_fingerprint(image_path, full_hash=hash_inputs),
            "motion": file_fingerprint(motion_path, full_hash=True),
        }
    code_files = {
        "background_fc.py": provenance_sha256(Path(__file__)),
        "study_design.py": provenance_sha256(
            Path(__file__).resolve().parents[2] / "core/study_design.py"
        ),
    }
    return cache_manifest(
        stage="background_fc_subject",
        parameters=parameters,
        inputs={
            "subject_id": subject_id,
            "timeseries_root": str(Path(timeseries_root).resolve()),
            "atlas": file_fingerprint(atlas_path, full_hash=True),
            "runs": run_inputs,
        },
        code=code_files,
    )


def process_subject(
    subject_id: int,
    timeseries_root: Path,
    atlas_path: Path,
    output_root: Path,
    fd_threshold: float = 0.5,
    head_radius_mm: float = 50.0,
    high_pass_seconds: float = 128.0,
    fir_lags: int = 35,
    hrf_lag_trs: int = 2,
    min_condition_trs: int = 90,
    min_residual_dof: int = 100,
    motion_model: str = "24P",
    include_fir: bool = True,
    censor_fd: bool = True,
    global_signal_regression: bool = False,
    hash_inputs: bool = False,
    force: bool = False,
) -> dict:
    """Run the complete background-FC extraction for one subject."""

    subject_output = output_root / "subjects" / f"sub-{subject_id:03d}"
    archive_path = subject_output / "background_fc.npz"
    qc_path = subject_output / "qc.json"
    cache_path = subject_output / "cache_manifest.json"
    parameters = {
        "fd_threshold_mm": fd_threshold,
        "head_radius_mm": head_radius_mm,
        "high_pass_seconds": high_pass_seconds,
        "fir_lags": fir_lags,
        "hrf_lag_trs": hrf_lag_trs,
        "min_condition_trs": min_condition_trs,
        "min_residual_dof": min_residual_dof,
        "motion_model": motion_model,
        "include_fir": include_fir,
        "censor_fd": censor_fd,
        "global_signal_regression": global_signal_regression,
        "hash_inputs": hash_inputs,
    }
    expected_cache = subject_cache_manifest(
        subject_id,
        timeseries_root,
        atlas_path,
        parameters,
        hash_inputs=hash_inputs,
    )
    if require_matching_cache(
        cache_path,
        expected_cache,
        artifacts_exist=archive_path.is_file() and qc_path.is_file(),
        force=force,
    ):
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("status") != "pass":
            raise RuntimeError(f"Cached subject {subject_id} did not pass QC.")
        return qc
    subject_output.mkdir(parents=True, exist_ok=True)
    atlas_labels, atlas_counts = load_atlas_labels(atlas_path)
    roi_by_run: dict[str, np.ndarray] = {}
    coverage_by_run: dict[str, np.ndarray] = {}
    motions: dict[str, np.ndarray] = {}
    for run_id in RUN_IDS:
        image_path, motion_path = input_paths(
            timeseries_root, subject_id, run_id
        )
        roi_by_run[run_id], coverage_by_run[run_id] = extract_roi_timeseries(
            image_path, atlas_labels, atlas_counts
        )
        motions[run_id] = np.loadtxt(motion_path, dtype=float)

    global_signals = None
    if global_signal_regression:
        global_signals = {}
        for run_id in RUN_IDS:
            valid_counts = np.rint(
                coverage_by_run[run_id] * atlas_counts[1:]
            ).astype(int)
            global_signals[run_id] = np.average(
                roi_by_run[run_id], axis=1, weights=valid_counts
            )

    design, design_names, fd_by_run, design_qc = build_joint_design(
        motions,
        fd_threshold=fd_threshold,
        head_radius_mm=head_radius_mm,
        high_pass_seconds=high_pass_seconds,
        fir_lags=fir_lags,
        motion_model=motion_model,
        include_fir=include_fir,
        add_fd_spikes=censor_fd,
        global_signals=global_signals,
    )
    if not design_qc["full_rank"]:
        raise RuntimeError(
            f"Subject {subject_id}: design is rank deficient "
            f"({design_qc['rank']}/{design_qc['n_columns']})."
        )
    residuals, ols_qc = residualize_roi_timeseries(roi_by_run, design)
    if ols_qc["residual_dof"] < min_residual_dof:
        raise RuntimeError(
            f"Subject {subject_id}: residual DoF {ols_qc['residual_dof']} "
            f"is below {min_residual_dof}."
        )
    matrices, block_matrices, block_table = condition_connectivity(
        residuals,
        fd_by_run,
        fd_threshold=fd_threshold,
        hrf_lag_trs=hrf_lag_trs,
        min_condition_trs=min_condition_trs,
        censor_fd=censor_fd,
    )
    condition_counts = (
        block_table.groupby("condition")["retained_trs"]
        .sum()
        .reindex(CONDITIONS)
        .astype(int)
    )
    min_coverage = float(
        min(np.min(values) for values in coverage_by_run.values())
    )
    minimum_valid_roi_voxels = int(
        min(
            np.min(np.rint(values * atlas_counts[1:]).astype(int))
            for values in coverage_by_run.values()
        )
    )
    qc = {
        "status": "pass",
        "subject_id": subject_id,
        "n_nodes": len(atlas_counts) - 1,
        "n_conditions": len(CONDITIONS),
        "condition_retained_trs": {
            condition: int(condition_counts[condition])
            for condition in CONDITIONS
        },
        "minimum_roi_functional_coverage": min_coverage,
        "minimum_valid_roi_voxels": minimum_valid_roi_voxels,
        "design": design_qc,
        "ols": ols_qc,
        "mean_fd_mm_by_run": {
            run_id: float(fd.mean()) for run_id, fd in fd_by_run.items()
        },
        "max_fd_mm_by_run": {
            run_id: float(fd.max()) for run_id, fd in fd_by_run.items()
        },
        "fd_spikes_by_run": {
            run_id: int(np.sum(fd > fd_threshold))
            for run_id, fd in fd_by_run.items()
        },
        "parameters": parameters,
    }
    temporary_archive = archive_path.with_suffix(".npz.tmp")
    with temporary_archive.open("wb") as stream:
        np.savez_compressed(
            stream,
            condition_matrices=matrices,
            block_matrices=block_matrices,
            conditions=np.asarray(CONDITIONS),
            block_run_ids=block_table["run_id"].to_numpy(),
            block_conditions=block_table["condition"].to_numpy(),
            residuals=residuals.astype(np.float32),
            roi_timeseries=np.stack(
                [roi_by_run[run_id] for run_id in RUN_IDS]
            ).astype(np.float32),
            fd=np.stack([fd_by_run[run_id] for run_id in RUN_IDS]).astype(
                np.float32
            ),
            roi_coverage=np.stack(
                [coverage_by_run[run_id] for run_id in RUN_IDS]
            ).astype(np.float32),
            design=design.astype(np.float32),
            design_names=np.asarray(design_names),
        )
    temporary_archive.replace(archive_path)
    block_table.to_csv(subject_output / "block_qc.csv", index=False)
    atomic_write_json(qc_path, qc)
    atomic_write_json(cache_path, expected_cache)
    return qc


def assemble_connectivity(
    subject_ids: tuple[int, ...] | list[int],
    output_root: Path,
) -> Path:
    """Assemble validated per-subject files into the group FC array."""

    condition_matrices = []
    block_matrices = []
    qc_rows = []
    for subject_id in subject_ids:
        subject_output = output_root / "subjects" / f"sub-{subject_id:03d}"
        qc = json.loads((subject_output / "qc.json").read_text(encoding="utf-8"))
        if qc.get("status") != "pass":
            raise RuntimeError(f"Subject {subject_id} did not pass QC.")
        archive = np.load(subject_output / "background_fc.npz", allow_pickle=False)
        matrices = archive["condition_matrices"]
        blocks = archive["block_matrices"]
        if matrices.shape != (3, 249, 249) or blocks.shape != (12, 249, 249):
            raise RuntimeError(
                f"Subject {subject_id}: invalid FC shapes "
                f"{matrices.shape}, {blocks.shape}."
            )
        condition_matrices.append(matrices)
        block_matrices.append(blocks)
        qc_rows.append(
            {
                "subject_id": subject_id,
                "minimum_roi_functional_coverage": qc[
                    "minimum_roi_functional_coverage"
                ],
                "minimum_valid_roi_voxels": qc["minimum_valid_roi_voxels"],
                "design_columns": qc["design"]["n_columns"],
                "design_rank": qc["design"]["rank"],
                "residual_dof": qc["ols"]["residual_dof"],
                **{
                    f"{condition.lower()}_retained_trs": qc[
                        "condition_retained_trs"
                    ][condition]
                    for condition in CONDITIONS
                },
                **{
                    f"run_{run_id}_mean_fd_mm": qc["mean_fd_mm_by_run"][
                        run_id
                    ]
                    for run_id in RUN_IDS
                },
                **{
                    f"run_{run_id}_fd_spikes": qc["fd_spikes_by_run"][run_id]
                    for run_id in RUN_IDS
                },
            }
        )
    group_path = output_root / "subject_condition_connectivity.npz"
    temporary_path = group_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            matrices=np.stack(condition_matrices).astype(np.float32),
            block_matrices=np.stack(block_matrices).astype(np.float32),
            subject_ids=np.asarray(subject_ids, dtype=int),
            conditions=np.asarray(CONDITIONS),
        )
    temporary_path.replace(group_path)
    pd.DataFrame(qc_rows).to_csv(output_root / "subject_qc.csv", index=False)
    return group_path
