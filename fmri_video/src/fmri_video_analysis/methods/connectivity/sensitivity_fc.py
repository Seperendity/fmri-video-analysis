"""Prespecified sensitivity variants using cached main-analysis ROI signals."""

from __future__ import annotations

import json
import traceback
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import stats

from fmri_video_analysis.core.provenance import (
    atomic_write_json,
    cache_manifest,
    file_fingerprint,
    require_matching_cache,
    sha256_file,
)
from fmri_video_analysis.core.statistics import (
    bh_fdr,
    cohens_dz,
    paired_sign_flip_pvalue,
)
from fmri_video_analysis.core.study_design import (
    CONDITIONS,
    RUN_IDS,
    blocks_for_run,
)
from fmri_video_analysis.methods.connectivity.background_fc import (
    build_joint_design,
    condition_connectivity,
    input_paths,
    load_atlas_labels,
    residualize_roi_timeseries,
)
from fmri_video_analysis.methods.connectivity.group_fc import (
    CONTRASTS,
    nbs_analysis,
    topology_analysis,
)

RECOMPUTED_VARIANTS = {
    "motion_6p": {
        "motion_model": "6P",
        "include_fir": True,
        "censor_fd": True,
        "fd_threshold": 0.5,
        "hrf_lag_trs": 2,
        "global_signal_regression": False,
    },
    "no_fir": {
        "motion_model": "24P",
        "include_fir": False,
        "censor_fd": True,
        "fd_threshold": 0.5,
        "hrf_lag_trs": 2,
        "global_signal_regression": False,
    },
    "no_fd_censor": {
        "motion_model": "24P",
        "include_fir": True,
        "censor_fd": False,
        "fd_threshold": 0.5,
        "hrf_lag_trs": 2,
        "global_signal_regression": False,
    },
    "fd_0p3": {
        "motion_model": "24P",
        "include_fir": True,
        "censor_fd": True,
        "fd_threshold": 0.3,
        "hrf_lag_trs": 2,
        "global_signal_regression": False,
    },
    "lag_0": {
        "motion_model": "24P",
        "include_fir": True,
        "censor_fd": True,
        "fd_threshold": 0.5,
        "hrf_lag_trs": 0,
        "global_signal_regression": False,
    },
    "gsr": {
        "motion_model": "24P",
        "include_fir": True,
        "censor_fd": True,
        "fd_threshold": 0.5,
        "hrf_lag_trs": 2,
        "global_signal_regression": True,
    },
}
SLICED_VARIANTS = ("schaefer200", "exclude_high_motion")
HIGH_MOTION_SUBJECTS = frozenset({70, 76, 81, 87, 96})


def _variant_subject_output(
    run_dir: Path, variant: str, subject_id: int
) -> Path:
    return (
        run_dir
        / "sensitivity"
        / variant
        / "subjects"
        / f"sub-{subject_id:03d}"
    )


def process_cached_subject(
    run_dir: Path,
    timeseries_root: Path,
    subject_id: int,
    variant: str,
    parameters: dict,
    min_condition_trs: int = 90,
    min_residual_dof: int = 100,
    force: bool = False,
) -> dict:
    output_dir = _variant_subject_output(run_dir, variant, subject_id)
    archive_path = output_dir / "background_fc.npz"
    qc_path = output_dir / "qc.json"
    cache_path = output_dir / "cache_manifest.json"
    main_path = run_dir / "subjects" / f"sub-{subject_id:03d}" / "background_fc.npz"
    main_cache_path = (
        run_dir / "subjects" / f"sub-{subject_id:03d}" / "cache_manifest.json"
    )
    atlas_path = run_dir / "atlas/composite_249_3mm.nii.gz"
    motion_paths = {
        run_id: input_paths(timeseries_root, subject_id, run_id)[1]
        for run_id in RUN_IDS
    }
    cache_parameters = {
        **parameters,
        "min_condition_trs": min_condition_trs,
        "min_residual_dof": min_residual_dof,
    }
    expected_cache = cache_manifest(
        stage="sensitivity_fc_subject",
        parameters=cache_parameters,
        inputs={
            "subject_id": subject_id,
            "variant": variant,
            "timeseries_root": str(Path(timeseries_root).resolve()),
            "main_roi_archive": file_fingerprint(main_path, full_hash=False),
            "main_cache_manifest": (
                file_fingerprint(main_cache_path, full_hash=True)
                if main_cache_path.is_file()
                else None
            ),
            "atlas": file_fingerprint(atlas_path, full_hash=True),
            "motions": {
                run_id: file_fingerprint(path, full_hash=True)
                for run_id, path in motion_paths.items()
            },
        },
        code={"sensitivity_fc.py": sha256_file(Path(__file__))},
    )
    if require_matching_cache(
        cache_path,
        expected_cache,
        artifacts_exist=archive_path.is_file() and qc_path.is_file(),
        force=force,
    ):
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("status") != "pass":
            raise RuntimeError(f"Cached {variant} subject {subject_id} failed QC.")
        return qc
    output_dir.mkdir(parents=True, exist_ok=True)
    main = np.load(main_path, allow_pickle=False)
    roi_array = main["roi_timeseries"].astype(np.float64)
    coverage_array = main["roi_coverage"].astype(np.float64)
    roi_by_run = {
        run_id: roi_array[index] for index, run_id in enumerate(RUN_IDS)
    }
    motions = {
        run_id: np.loadtxt(path, dtype=float) for run_id, path in motion_paths.items()
    }
    _, atlas_counts = load_atlas_labels(atlas_path)
    global_signals = None
    if parameters["global_signal_regression"]:
        global_signals = {}
        for run_index, run_id in enumerate(RUN_IDS):
            valid_counts = np.rint(
                coverage_array[run_index] * atlas_counts[1:]
            ).astype(int)
            global_signals[run_id] = np.average(
                roi_by_run[run_id], axis=1, weights=valid_counts
            )

    design, design_names, fd_by_run, design_qc = build_joint_design(
        motions,
        fd_threshold=parameters["fd_threshold"],
        head_radius_mm=50.0,
        high_pass_seconds=128.0,
        fir_lags=35,
        motion_model=parameters["motion_model"],
        include_fir=parameters["include_fir"],
        add_fd_spikes=parameters["censor_fd"],
        global_signals=global_signals,
    )
    if not design_qc["full_rank"]:
        raise RuntimeError(
            f"{variant} subject {subject_id}: rank-deficient design."
        )
    residuals, ols_qc = residualize_roi_timeseries(roi_by_run, design)
    if ols_qc["residual_dof"] < min_residual_dof:
        raise RuntimeError(
            f"{variant} subject {subject_id}: residual DoF "
            f"{ols_qc['residual_dof']} < {min_residual_dof}."
        )
    matrices, block_matrices, block_table = condition_connectivity(
        residuals,
        fd_by_run,
        fd_threshold=parameters["fd_threshold"],
        hrf_lag_trs=parameters["hrf_lag_trs"],
        min_condition_trs=min_condition_trs,
        censor_fd=parameters["censor_fd"],
    )
    counts = (
        block_table.groupby("condition")["retained_trs"]
        .sum()
        .reindex(CONDITIONS)
        .astype(int)
    )
    qc = {
        "status": "pass",
        "subject_id": subject_id,
        "variant": variant,
        "parameters": parameters,
        "condition_retained_trs": {
            condition: int(counts[condition]) for condition in CONDITIONS
        },
        "design": design_qc,
        "ols": ols_qc,
    }
    temporary = archive_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            condition_matrices=matrices,
            block_matrices=block_matrices,
            residuals=residuals.astype(np.float32),
            design_names=np.asarray(design_names),
            fd=np.stack([fd_by_run[run_id] for run_id in RUN_IDS]).astype(
                np.float32
            ),
        )
    temporary.replace(archive_path)
    block_table.to_csv(output_dir / "block_qc.csv", index=False)
    atomic_write_json(qc_path, qc)
    atomic_write_json(cache_path, expected_cache)
    return qc


def _assemble_recomputed_variant(
    run_dir: Path,
    subject_ids: np.ndarray,
    variant: str,
) -> Path:
    matrices = []
    blocks = []
    qc_rows = []
    for subject_id in subject_ids:
        output_dir = _variant_subject_output(
            run_dir, variant, int(subject_id)
        )
        qc = json.loads((output_dir / "qc.json").read_text(encoding="utf-8"))
        archive = np.load(output_dir / "background_fc.npz", allow_pickle=False)
        matrices.append(archive["condition_matrices"])
        blocks.append(archive["block_matrices"])
        qc_rows.append(
            {
                "subject_id": int(subject_id),
                "design_columns": qc["design"]["n_columns"],
                "design_rank": qc["design"]["rank"],
                "residual_dof": qc["ols"]["residual_dof"],
                **{
                    f"{condition.lower()}_retained_trs": qc[
                        "condition_retained_trs"
                    ][condition]
                    for condition in CONDITIONS
                },
            }
        )
    variant_dir = run_dir / "sensitivity" / variant
    group_path = variant_dir / "subject_condition_connectivity.npz"
    temporary = group_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            matrices=np.stack(matrices).astype(np.float32),
            block_matrices=np.stack(blocks).astype(np.float32),
            subject_ids=subject_ids,
            conditions=np.asarray(CONDITIONS),
        )
    temporary.replace(group_path)
    pd.DataFrame(qc_rows).to_csv(variant_dir / "subject_qc.csv", index=False)
    return group_path


def _process_cached_subject_safely(*args, **kwargs) -> dict:
    subject_id = int(args[2])
    try:
        return {
            "passed": True,
            "subject_id": subject_id,
            "qc": process_cached_subject(*args, **kwargs),
        }
    except Exception as exc:  # noqa: BLE001 - isolate sensitivity exclusions
        return {
            "passed": False,
            "subject_id": subject_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_recomputed_variant(
    run_dir: Path,
    timeseries_root: Path,
    subject_ids: np.ndarray,
    variant: str,
    n_jobs: int = 1,
    force: bool = False,
) -> Path:
    parameters = RECOMPUTED_VARIANTS[variant]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_process_cached_subject_safely)(
                run_dir,
                timeseries_root,
                int(subject_id),
                variant,
                parameters,
                force=force,
            )
            for subject_id in subject_ids
        )
    if len(results) != len(subject_ids):
        raise RuntimeError(f"{variant}: incomplete subject results.")
    failures = [result for result in results if not result["passed"]]
    variant_dir = run_dir / "sensitivity" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "exclusions.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    passing_subjects = np.asarray(
        [
            result["subject_id"]
            for result in results
            if result["passed"]
        ],
        dtype=int,
    )
    if len(passing_subjects) < 2:
        raise RuntimeError(f"{variant}: fewer than two subjects passed QC.")
    return _assemble_recomputed_variant(
        run_dir, passing_subjects, variant
    )


def create_sliced_variants(
    run_dir: Path, matrices: np.ndarray, block_matrices: np.ndarray, subject_ids: np.ndarray
) -> dict[str, Path]:
    outputs = {}
    specifications = {
        "schaefer200": (
            matrices[:, :, :200, :200],
            block_matrices[:, :, :200, :200],
            subject_ids,
        ),
        "exclude_high_motion": (
            matrices[
                ~np.isin(subject_ids, list(HIGH_MOTION_SUBJECTS))
            ],
            block_matrices[
                ~np.isin(subject_ids, list(HIGH_MOTION_SUBJECTS))
            ],
            subject_ids[
                ~np.isin(subject_ids, list(HIGH_MOTION_SUBJECTS))
            ],
        ),
    }
    for variant, (variant_matrices, variant_blocks, variant_subjects) in (
        specifications.items()
    ):
        variant_dir = run_dir / "sensitivity" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        path = variant_dir / "subject_condition_connectivity.npz"
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                matrices=variant_matrices.astype(np.float32),
                block_matrices=variant_blocks.astype(np.float32),
                subject_ids=variant_subjects,
                conditions=np.asarray(CONDITIONS),
            )
        outputs[variant] = path
    return outputs


def _edge_statistics(differences: np.ndarray) -> dict:
    mean = differences.mean(axis=0)
    sd = differences.std(axis=0, ddof=1)
    t_values = np.divide(
        mean,
        sd / np.sqrt(len(differences)),
        out=np.zeros_like(mean),
        where=sd > 0,
    )
    p_values = 2 * stats.t.sf(np.abs(t_values), len(differences) - 1)
    q_values = bh_fdr(p_values)
    return {
        "n_fdr_edges": int(np.sum(q_values < 0.05)),
        "minimum_q": float(np.min(q_values)),
        "edge_mean_difference": mean,
    }


def summarize_variants(
    run_dir: Path,
    variants: list[str],
    n_permutations: int = 10000,
    seed: int = 20260731,
) -> pd.DataFrame:
    main = np.load(
        run_dir / "subject_condition_connectivity.npz", allow_pickle=False
    )
    main_matrices = main["matrices"].astype(np.float64)
    main_subjects = main["subject_ids"].astype(int)
    rows = []
    for variant in variants:
        archive = np.load(
            run_dir
            / "sensitivity"
            / variant
            / "subject_condition_connectivity.npz",
            allow_pickle=False,
        )
        matrices = archive["matrices"].astype(np.float64)
        subject_ids = archive["subject_ids"].astype(int)
        n_nodes = matrices.shape[-1]
        upper = np.triu_indices(n_nodes, 1)
        edges = matrices[:, :, upper[0], upper[1]]
        if not set(subject_ids).issubset(set(main_subjects)):
            raise RuntimeError(f"{variant}: subject IDs do not match main run.")
        main_edges = main_matrices[:, :, :n_nodes, :n_nodes][
            :, :, upper[0], upper[1]
        ]
        for contrast_name in ("complex_minus_simple", "task_mean_minus_control"):
            coefficients = CONTRASTS[contrast_name]
            differences = np.einsum(
                "sce,c->se", edges, coefficients, optimize=True
            )
            main_differences = np.einsum(
                "sce,c->se", main_edges, coefficients, optimize=True
            )
            global_differences = differences.mean(axis=1)
            edge_summary = _edge_statistics(differences)
            main_edge_mean = main_differences.mean(axis=0)
            variant_edge_mean = edge_summary["edge_mean_difference"]
            rows.append(
                {
                    "variant": variant,
                    "contrast": contrast_name,
                    "n_subjects": len(subject_ids),
                    "n_nodes": n_nodes,
                    "global_mean_difference": float(
                        global_differences.mean()
                    ),
                    "global_cohens_dz": cohens_dz(global_differences),
                    "global_p_sign_flip": paired_sign_flip_pvalue(
                        global_differences,
                        n_permutations,
                        int(
                            (
                                seed
                                + zlib.crc32(
                                    f"{variant}:{contrast_name}".encode()
                                )
                            )
                            % (2**32 - 1)
                        ),
                    ),
                    "n_fdr_edges": edge_summary["n_fdr_edges"],
                    "minimum_edge_q": edge_summary["minimum_q"],
                    "edge_effect_correlation_with_main": float(
                        stats.pearsonr(
                            variant_edge_mean, main_edge_mean
                        ).statistic
                    ),
                    "edge_direction_concordance_with_main": float(
                        np.mean(
                            np.sign(variant_edge_mean)
                            == np.sign(main_edge_mean)
                        )
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    output_dir = run_dir / "sensitivity"
    summary.to_csv(output_dir / "sensitivity_summary.csv", index=False)
    return summary


def _columnwise_correlation(
    predictor: np.ndarray, outcomes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predictor = np.asarray(predictor, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    predictor_centered = predictor - predictor.mean()
    outcomes_centered = outcomes - outcomes.mean(axis=0)
    denominator = np.sqrt(
        np.sum(np.square(predictor_centered))
        * np.sum(np.square(outcomes_centered), axis=0)
    )
    correlation = np.divide(
        predictor_centered @ outcomes_centered,
        denominator,
        out=np.zeros(outcomes.shape[1]),
        where=denominator > 0,
    )
    correlation = np.clip(correlation, -0.999999, 0.999999)
    t_values = correlation * np.sqrt(
        (len(predictor) - 2) / (1 - np.square(correlation))
    )
    p_values = 2 * stats.t.sf(np.abs(t_values), len(predictor) - 2)
    return correlation, p_values


def qc_fc_analysis(run_dir: Path) -> dict:
    """Relate residual FC to mean FD and inter-node distance."""

    archive = np.load(
        run_dir / "subject_condition_connectivity.npz", allow_pickle=False
    )
    matrices = archive["matrices"].astype(np.float64)
    subject_ids = archive["subject_ids"].astype(int)
    subject_qc = pd.read_csv(run_dir / "subject_qc.csv").set_index("subject_id")
    fd_columns = [
        column for column in subject_qc if column.endswith("_mean_fd_mm")
    ]
    mean_fd = subject_qc.loc[subject_ids, fd_columns].mean(axis=1).to_numpy()
    nodes = pd.read_csv(
        run_dir / "atlas/composite_249_nodes.tsv", sep="\t"
    )
    upper = np.triu_indices(len(nodes), 1)
    edges = matrices[:, :, upper[0], upper[1]]
    mean_edges = edges.mean(axis=1)
    qc_fc, p_values = _columnwise_correlation(mean_fd, mean_edges)
    q_values = bh_fdr(p_values)

    coordinates = nodes[
        ["centroid_x_mni", "centroid_y_mni", "centroid_z_mni"]
    ].to_numpy()
    distances = np.linalg.norm(
        coordinates[upper[0]] - coordinates[upper[1]], axis=1
    )
    edge_table = pd.DataFrame(
        {
            "node1_id": nodes["node_id"].to_numpy()[upper[0]],
            "node2_id": nodes["node_id"].to_numpy()[upper[1]],
            "distance_mm": distances,
            "qc_fc_r": qc_fc,
            "p": p_values,
            "q_fdr": q_values,
        }
    )
    output_dir = run_dir / "sensitivity" / "qc_fc"
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_table.to_csv(output_dir / "qc_fc_edges.csv.gz", index=False)

    global_edges = edges.mean(axis=2)
    global_rows = []
    for condition_index, condition in enumerate(CONDITIONS):
        result = stats.pearsonr(mean_fd, global_edges[:, condition_index])
        global_rows.append(
            {
                "condition": condition,
                "pearson_r_mean_fd_global_fc": result.statistic,
                "p": result.pvalue,
            }
        )
    pd.DataFrame(global_rows).to_csv(
        output_dir / "mean_fd_global_fc.csv", index=False
    )
    signed_distance = stats.spearmanr(distances, qc_fc)
    absolute_distance = stats.spearmanr(distances, np.abs(qc_fc))
    summary = {
        "n_subjects": len(subject_ids),
        "n_edges": len(qc_fc),
        "n_qc_fc_edges_fdr_0_05": int(np.sum(q_values < 0.05)),
        "median_absolute_qc_fc_r": float(np.median(np.abs(qc_fc))),
        "distance_signed_qc_fc_spearman_r": float(
            signed_distance.statistic
        ),
        "distance_signed_qc_fc_p": float(signed_distance.pvalue),
        "distance_absolute_qc_fc_spearman_r": float(
            absolute_distance.statistic
        ),
        "distance_absolute_qc_fc_p": float(absolute_distance.pvalue),
    }
    (output_dir / "qc_fc_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _icc_3_1(values: np.ndarray) -> np.ndarray:
    """Vectorized two-way mixed, consistency, single-measure ICC."""

    values = np.asarray(values, dtype=float)
    n_subjects, n_blocks, _ = values.shape
    grand_mean = values.mean(axis=(0, 1))
    subject_mean = values.mean(axis=1)
    block_mean = values.mean(axis=0)
    ss_subject = n_blocks * np.sum(
        np.square(subject_mean - grand_mean), axis=0
    )
    ss_block = n_subjects * np.sum(
        np.square(block_mean - grand_mean), axis=0
    )
    ss_total = np.sum(np.square(values - grand_mean), axis=(0, 1))
    ss_error = ss_total - ss_subject - ss_block
    ms_subject = ss_subject / (n_subjects - 1)
    ms_error = ss_error / ((n_subjects - 1) * (n_blocks - 1))
    return np.divide(
        ms_subject - ms_error,
        ms_subject + (n_blocks - 1) * ms_error,
        out=np.full_like(ms_subject, np.nan),
        where=(ms_subject + (n_blocks - 1) * ms_error) != 0,
    )


def block_reliability_analysis(run_dir: Path) -> pd.DataFrame:
    archive = np.load(
        run_dir / "subject_condition_connectivity.npz", allow_pickle=False
    )
    block_matrices = archive["block_matrices"].astype(np.float64)
    block_conditions = np.asarray(
        [
            condition
            for run_id in RUN_IDS
            for condition, _, _ in blocks_for_run(run_id)
        ]
    )
    n_nodes = block_matrices.shape[-1]
    upper = np.triu_indices(n_nodes, 1)
    block_edges = block_matrices[:, :, upper[0], upper[1]]
    output_dir = run_dir / "sensitivity" / "block_reliability"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    edge_tables = []
    for condition in CONDITIONS:
        values = block_edges[:, block_conditions == condition]
        icc = _icc_3_1(values)
        edge_tables.append(
            pd.DataFrame(
                {
                    "condition": condition,
                    "node1_id": upper[0] + 1,
                    "node2_id": upper[1] + 1,
                    "icc_3_1": icc,
                }
            )
        )
        summaries.append(
            {
                "condition": condition,
                "n_subjects": values.shape[0],
                "n_blocks": values.shape[1],
                "median_icc_3_1": float(np.nanmedian(icc)),
                "q25_icc_3_1": float(np.nanquantile(icc, 0.25)),
                "q75_icc_3_1": float(np.nanquantile(icc, 0.75)),
                "proportion_icc_ge_0_40": float(np.nanmean(icc >= 0.4)),
                "proportion_icc_ge_0_60": float(np.nanmean(icc >= 0.6)),
            }
        )
    pd.concat(edge_tables, ignore_index=True).to_csv(
        output_dir / "edgewise_block_icc.csv.gz", index=False
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "block_reliability_summary.csv", index=False)
    return summary


def sensitivity_nbs_analysis(
    run_dir: Path,
    variants: list[str],
    n_permutations: int = 10000,
    seed: int = 20260731,
) -> dict:
    nodes = pd.read_csv(
        run_dir / "atlas/composite_249_nodes.tsv", sep="\t"
    )
    summaries = {}
    for variant in variants:
        archive = np.load(
            run_dir
            / "sensitivity"
            / variant
            / "subject_condition_connectivity.npz",
            allow_pickle=False,
        )
        matrices = archive["matrices"].astype(np.float64)
        variant_nodes = nodes.iloc[: matrices.shape[-1]].reset_index(drop=True)
        summaries[variant] = nbs_analysis(
            matrices,
            variant_nodes,
            run_dir / "sensitivity" / variant / "nbs",
            n_permutations=n_permutations,
            thresholds=(2.5, 3.1, 3.5),
            seed=seed,
            batch_size=128,
        )
    (run_dir / "sensitivity" / "sensitivity_nbs_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summaries


def sensitivity_topology_analysis(
    run_dir: Path,
    variants: list[str],
    n_permutations: int = 10000,
    equivalence_bound_dz: float = 0.2,
    n_jobs: int = 1,
    seed: int = 20260731,
) -> pd.DataFrame:
    nodes = pd.read_csv(
        run_dir / "atlas/composite_249_nodes.tsv", sep="\t"
    )
    tables = []
    for variant in variants:
        archive = np.load(
            run_dir
            / "sensitivity"
            / variant
            / "subject_condition_connectivity.npz",
            allow_pickle=False,
        )
        matrices = archive["matrices"].astype(np.float64)
        subject_ids = archive["subject_ids"].astype(int)
        variant_nodes = nodes.iloc[: matrices.shape[-1]].reset_index(drop=True)
        _, statistics = topology_analysis(
            matrices,
            subject_ids,
            variant_nodes,
            run_dir / "sensitivity" / variant / "topology",
            n_permutations=n_permutations,
            equivalence_bound_dz=equivalence_bound_dz,
            n_jobs=n_jobs,
            seed=seed,
        )
        statistics.insert(0, "n_nodes", matrices.shape[-1])
        statistics.insert(0, "n_subjects", len(subject_ids))
        statistics.insert(0, "variant", variant)
        tables.append(statistics)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(
        run_dir / "sensitivity" / "sensitivity_topology_summary.csv",
        index=False,
    )
    return combined
