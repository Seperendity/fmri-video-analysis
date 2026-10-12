from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_video_analysis.core.statistics import paired_tost_dz
from fmri_video_analysis.core.study_design import RUN_IDS
from fmri_video_analysis.methods.activation.repeated_measures import (
    build_randomise_command as build_repeated_measures_randomise_command,
)
from fmri_video_analysis.methods.connectivity.background_fc import (
    build_joint_design,
    condition_connectivity,
    dct_high_pass_basis,
    extract_roi_timeseries,
    fir_task_design,
    motion_24_parameters,
    power_fd,
)
from fmri_video_analysis.methods.connectivity.block_stimulus import (
    connection_masks,
    exact_paired_sign_flip,
)
from fmri_video_analysis.methods.connectivity.group_fc import (
    _components_for_edges,
)
from fmri_video_analysis.methods.connectivity.sensitivity_fc import _icc_3_1
from fmri_video_analysis.methods.connectivity.timeseries_topology import (
    component_masses,
    graph_metrics,
    shifted_block_bounds,
)
from fmri_video_analysis.methods.decoding import cross_video as cross_video_module
from fmri_video_analysis.methods.decoding.cross_video import (
    spatial_mean_center_features,
    structured_label_permutation_test,
    subjectwise_label_permutations,
    svm_search,
)
from fmri_video_analysis.methods.rsa.searchlight import (
    run_randomise as run_searchlight_randomise,
)
from fmri_video_analysis.methods.stimulus.characterization import motion_metrics
from fmri_video_analysis.reporting.non_mvpa_statistics_audit import (
    finite_or_blank,
    p_display,
    truthy,
)


def test_nbs_mass_excludes_subthreshold_edges():
    t_matrix = np.array(
        [
            [0.0, 4.0, 2.0],
            [4.0, 0.0, 4.0],
            [2.0, 4.0, 0.0],
        ]
    )

    components = component_masses(t_matrix, threshold=3.1)

    assert len(components) == 1
    assert components[0]["direction"] == "positive"
    assert set(components[0]["nodes"]) == {0, 1, 2}
    assert components[0]["mass"] == pytest.approx(8.0)


def test_global_efficiency_uses_weighted_shortest_paths():
    matrix = np.array(
        [
            [0.0, 1.0, 0.25],
            [1.0, 0.0, 1.0],
            [0.25, 1.0, 0.0],
        ]
    )

    metrics = graph_metrics(matrix)

    assert metrics["global_efficiency"] == pytest.approx(5.0 / 6.0)
    assert metrics["global_efficiency"] != pytest.approx(1.0)


def test_motion_metrics_are_zero_for_identical_frames():
    frame = np.linspace(0, 1, 128 * 96).reshape(96, 128).astype(np.float32)

    metrics = motion_metrics(frame, frame)

    assert metrics == pytest.approx(
        {
            "optical_flow_mean": 0.0,
            "global_motion_median": 0.0,
            "residual_flow_mean": 0.0,
        },
        abs=1e-6,
    )


def test_cross_video_svm_uses_frozen_random_seed():
    estimator, _ = svm_search(n_features=600, seed=20260731)

    assert estimator.named_steps["model"].random_state == 20260731


def test_mvpa_spatial_mean_centering_removes_only_samplewise_common_mode():
    features = np.array([[1.0, 2.0, 4.0], [11.0, 12.0, 14.0]], dtype=np.float32)
    original = features.copy()

    centered = spatial_mean_center_features(features)

    assert features == pytest.approx(original)
    assert centered.mean(axis=1) == pytest.approx([0.0, 0.0], abs=1e-6)
    assert centered[0] == pytest.approx(centered[1], abs=1e-6)
    assert np.diff(centered, axis=1) == pytest.approx(
        np.diff(original, axis=1), abs=1e-6
    )


def test_subjectwise_label_permutation_swaps_whole_subjects():
    labels = np.tile([0, 1, 0, 1], 3)
    subject_ids = np.repeat([1, 2, 3], 4)

    permutations = subjectwise_label_permutations(
        labels, subject_ids, n_permutations=8, seed=7
    )

    assert permutations.shape == (8, 12)
    for permutation in permutations:
        for subject_id in np.unique(subject_ids):
            subject = subject_ids == subject_id
            unchanged = np.array_equal(permutation[subject], labels[subject])
            swapped = np.array_equal(permutation[subject], 1 - labels[subject])
            assert unchanged or swapped


def test_structured_permutation_checkpoint_resumes(tmp_path, monkeypatch):
    calls = []

    def fake_score(
        X,
        labels,
        subject_ids,
        sessions,
        sample_subject_fold,
        n_subject_folds,
        seed,
    ):
        calls.append(labels.copy())
        return float(labels.mean())

    monkeypatch.setattr(cross_video_module, "_permuted_outer_cv_score", fake_score)
    X = np.arange(24, dtype=np.float32).reshape(8, 3)
    labels = np.tile([0, 1], 4)
    subject_ids = np.repeat(np.arange(4), 2)
    sessions = np.tile([1, 2], 4)
    folds = np.repeat([0, 1, 0, 1], 2)
    checkpoint = tmp_path / "structured_permutation_null.csv.gz"

    first, first_p = structured_label_permutation_test(
        X,
        labels,
        subject_ids,
        sessions,
        folds,
        2,
        0.75,
        4,
        1,
        11,
        checkpoint,
        2,
    )
    assert len(calls) == 4
    assert checkpoint.is_file()
    config_path = tmp_path / "structured_permutation_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    old_config["schema_version"] = 1
    config_path.write_text(json.dumps(old_config), encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Completed permutations should be resumed.")

    monkeypatch.setattr(cross_video_module, "_permuted_outer_cv_score", fail_if_called)
    with pytest.raises(RuntimeError, match="Pass --force"):
        structured_label_permutation_test(
            X,
            labels,
            subject_ids,
            sessions,
            folds,
            2,
            0.75,
            4,
            1,
            11,
            checkpoint,
            2,
        )

    resumed, resumed_p = structured_label_permutation_test(
        X,
        labels,
        subject_ids,
        sessions,
        folds,
        2,
        0.75,
        4,
        1,
        11,
        checkpoint,
        2,
        force=True,
    )

    assert resumed == pytest.approx(first)
    assert resumed_p == first_p
    migrated_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert migrated_config["schema_version"] == 2
    assert (tmp_path / "structured_permutation_config_migration.json").is_file()


def test_repeated_measures_randomise_is_f_only_and_seeded(tmp_path):
    command = build_repeated_measures_randomise_command(
        "randomise",
        merged=tmp_path / "merged.nii.gz",
        prefix=tmp_path / "omnibus",
        design=tmp_path / "design.mat",
        contrast=tmp_path / "design.con",
        ftest=tmp_path / "design.fts",
        groups=tmp_path / "design.grp",
        mask=tmp_path / "mask.nii.gz",
        n_permutations=10_000,
        seed=20260731,
    )

    assert "--fonly" in command
    assert "--seed=20260731" in command
    assert command[command.index("-n") + 1] == "10000"
    assert command[command.index("-f") + 1].endswith("design.fts")
    assert command[command.index("-e") + 1].endswith("design.grp")


def test_searchlight_randomise_receives_explicit_seed(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, check, stdout, stderr):
        commands.append(command)
        prefix = Path(command[command.index("-o") + 1])
        Path(f"{prefix}_tfce_corrp_tstat1.nii.gz").touch()

    monkeypatch.setattr("subprocess.run", fake_run)
    merged = tmp_path / "subjects.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    merged.touch()
    mask.touch()

    corrected = run_searchlight_randomise(
        merged,
        mask,
        tmp_path,
        "simple_complex",
        n_permutations=17,
        seed=12345,
        force=True,
    )

    assert corrected.is_file()
    assert "--seed=12345" in commands[0]


@pytest.mark.parametrize(
    "matrix",
    [
        np.array(
            [
                [0.0, 0.8, 0.2, 0.0],
                [0.8, 0.0, 0.6, 0.3],
                [0.2, 0.6, 0.0, 0.7],
                [0.0, 0.3, 0.7, 0.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.8, 0.0, 0.0],
                [0.8, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.7],
                [0.0, 0.0, 0.7, 0.0],
            ]
        ),
    ],
)
def test_scipy_path_metrics_match_networkx_reference(matrix):
    graph = nx.from_numpy_array(matrix)
    for _, _, edge in graph.edges(data=True):
        edge["distance"] = 1.0 / edge["weight"]
    reference_efficiency = sum(
        1.0 / distance
        for source, lengths in nx.all_pairs_dijkstra_path_length(
            graph, weight="distance"
        )
        for target, distance in lengths.items()
        if target != source and distance > 0
    ) / (len(graph) * (len(graph) - 1))
    largest = graph.subgraph(max(nx.connected_components(graph), key=len))
    reference_path_length = nx.average_shortest_path_length(largest, weight="distance")

    metrics = graph_metrics(matrix)

    assert metrics["global_efficiency"] == pytest.approx(reference_efficiency)
    assert metrics["path_length"] == pytest.approx(reference_path_length)


@pytest.mark.parametrize("constant", [0.0, 1.0, -1.0])
def test_tost_does_not_claim_equivalence_for_zero_variance(constant):
    result = paired_tost_dz(np.full(8, constant))

    assert np.isnan(result["p_lower"])
    assert np.isnan(result["p_upper"])
    assert result["equivalent"] is False


def test_hrf_shift_preserves_block_duration():
    assert shifted_block_bounds(10, 30, 0) == (10, 40)
    assert shifted_block_bounds(10, 30, 3) == (13, 43)


def test_motion_24_and_power_fd_reset_at_run_start():
    motion = np.zeros((3, 6))
    motion[1, 0] = 0.1
    motion[1, 3] = 0.001
    motion[2] = motion[1]

    regressors, names = motion_24_parameters(motion)
    fd = power_fd(motion, head_radius_mm=50.0)

    assert regressors.shape == (3, 24)
    assert len(names) == 24
    assert regressors[0, 6:12] == pytest.approx(np.zeros(6))
    assert fd == pytest.approx([0.0, 0.15, 0.0])


def test_dct_high_pass_has_expected_four_nonconstant_terms():
    basis = dct_high_pass_basis(130, tr_seconds=2.0, cutoff_seconds=128.0)

    assert basis.shape == (130, 4)
    assert basis.sum(axis=0) == pytest.approx(np.zeros(4), abs=1e-12)
    assert basis.T @ basis == pytest.approx(np.eye(4), abs=1e-12)


def test_fir_design_models_each_condition_lag_once_per_run():
    design, names = fir_task_design(fir_lags=35)

    assert design.shape == (520, 105)
    assert len(names) == 105
    assert design.sum(axis=0) == pytest.approx(np.full(105, 4.0))
    assert design[10, names.index("fir_Simple_lag_00")] == 1
    assert design[130 + 10, names.index("fir_Complex_lag_00")] == 1


def test_joint_design_is_full_rank_without_spikes():
    rng = np.random.default_rng(7)
    motions = {
        run_id: np.cumsum(rng.normal(scale=1e-5, size=(130, 6)), axis=0)
        for run_id in RUN_IDS
    }

    design, names, fd_by_run, diagnostics = build_joint_design(motions)

    assert design.shape == (520, 221)
    assert len(names) == 221
    assert diagnostics["n_fd_spikes"] == 0
    assert diagnostics["full_rank"] is True
    assert diagnostics["residual_dof"] == 299
    assert all(fd[0] == 0 for fd in fd_by_run.values())


def test_joint_design_drops_only_exactly_dependent_columns():
    motions = {run_id: np.zeros((130, 6)) for run_id in RUN_IDS}
    # Four matching FIR positions are marked as spikes. Any one of those
    # spike columns is redundant with the FIR column and the other three.
    for run_id in RUN_IDS:
        motions[run_id][10, 0] = 1.0

    design, names, _, diagnostics = build_joint_design(motions)

    assert (
        diagnostics["rank_before_rank_reduction"]
        < diagnostics["n_columns_before_rank_reduction"]
    )
    assert diagnostics["dropped_dependent_columns"]
    assert diagnostics["full_rank"] is True
    assert design.shape[1] == len(names) == diagnostics["rank"]


def test_joint_design_supports_6p_and_no_fir_variants():
    rng = np.random.default_rng(17)
    motions = {
        run_id: np.cumsum(rng.normal(scale=1e-5, size=(130, 6)), axis=0)
        for run_id in RUN_IDS
    }

    six_p, six_p_names, _, six_p_qc = build_joint_design(motions, motion_model="6P")
    no_fir, no_fir_names, _, no_fir_qc = build_joint_design(motions, include_fir=False)

    assert six_p.shape == (520, 149)
    assert len(six_p_names) == 149
    assert six_p_qc["motion_model"] == "6P"
    assert no_fir.shape == (520, 116)
    assert len(no_fir_names) == 116
    assert no_fir_qc["include_fir"] is False


def test_condition_fc_can_disable_fd_censoring():
    rng = np.random.default_rng(19)
    residuals = rng.normal(size=(520, 6))
    fd_by_run = {run_id: np.ones(130) for run_id in RUN_IDS}

    _, _, block_table = condition_connectivity(
        residuals,
        fd_by_run,
        censor_fd=False,
    )

    retained = block_table.groupby("condition").retained_trs.sum()
    assert retained.to_dict() == {
        "Complex": 120,
        "Control": 120,
        "Simple": 120,
    }


def test_condition_fc_censors_volumes_but_keeps_all_runs():
    rng = np.random.default_rng(11)
    residuals = rng.normal(size=(520, 6))
    fd_by_run = {run_id: np.zeros(130) for run_id in RUN_IDS}
    fd_by_run["0006"][12] = 0.6

    matrices, block_matrices, block_table = condition_connectivity(
        residuals,
        fd_by_run,
        fd_threshold=0.5,
        hrf_lag_trs=2,
        min_condition_trs=90,
    )

    assert matrices.shape == (3, 6, 6)
    assert block_matrices.shape == (12, 6, 6)
    assert np.all(np.isfinite(matrices))
    assert np.allclose(matrices, matrices.transpose(0, 2, 1))
    retained = block_table.groupby("condition").retained_trs.sum()
    assert retained["Simple"] == 119
    assert retained["Complex"] == 120
    assert retained["Control"] == 120


def test_non_mvpa_audit_formats_permutation_p_values_without_zero():
    assert p_display(1 / 10001) == "1.00e-04"
    assert p_display(0.0230976902309769) == "0.0231"
    assert p_display("") == ""


def test_non_mvpa_audit_keeps_missing_numeric_fields_blank():
    assert finite_or_blank(None) == ""
    assert finite_or_blank(np.nan) == ""
    assert finite_or_blank(-0.00939) == pytest.approx(-0.00939)


def test_non_mvpa_audit_reads_boolean_csv_fields_robustly():
    assert truthy(True)
    assert truthy(np.bool_(True))
    assert truthy("True")
    assert not truthy(False)


def test_block_stimulus_connection_masks_and_exact_sign_flip():
    nodes = pd.DataFrame(
        {
            "domain": [
                "Cortex",
                "Cortex",
                "Cortex",
                "Cortex",
                "Subcortex",
                "Subcortex",
                "Cerebellum",
                "Cerebellum",
            ],
            "network": [
                "VisCent",
                "VisPeri",
                "DorsAttnA",
                "DorsAttnB",
                "Subcortex",
                "Subcortex",
                "Cerebellum",
                "Cerebellum",
            ],
        }
    )

    masks = connection_masks(nodes)

    assert masks["global_all_edges"].sum() == 28
    assert masks["cortex_within"].sum() == 6
    assert masks["cortex_subcortex"].sum() == 8
    assert masks["cortex_cerebellum"].sum() == 8
    assert masks["subcortex_cerebellum"].sum() == 4
    assert masks["visual_dorsal_attention"].sum() == 4
    assert exact_paired_sign_flip(np.ones(4)) == 0.125


def test_roi_extraction_excludes_zero_filled_voxels_from_mean(tmp_path):
    labels = np.arange(1, 250, dtype=np.int16)
    labels = np.insert(labels, 1, 1).reshape((250, 1, 1))
    counts = np.bincount(labels.ravel(), minlength=250)
    signal = np.arange(130, dtype=np.float32)
    data = np.empty((250, 1, 1, 130), dtype=np.float32)
    data[:, 0, 0, :] = signal
    data[1, 0, 0, :] = 0.0
    image_path = tmp_path / "functional.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), image_path)

    roi, coverage = extract_roi_timeseries(image_path, labels, counts)

    assert roi[:, 0] == pytest.approx(signal)
    assert coverage[0] == pytest.approx(0.5)
    assert np.all(coverage[1:] == 1.0)


def test_nbs_keeps_positive_and_negative_components_separate():
    upper = np.triu_indices(4, 1)
    t_edges = np.zeros(len(upper[0]))
    edge_lookup = {
        (int(first), int(second)): index
        for index, (first, second) in enumerate(zip(*upper))
    }
    t_edges[edge_lookup[(0, 1)]] = 4.0
    t_edges[edge_lookup[(1, 2)]] = 5.0
    t_edges[edge_lookup[(2, 3)]] = -6.0

    components = _components_for_edges(t_edges, upper, 4, threshold=3.1)

    assert [component["direction"] for component in components] == [
        "positive",
        "negative",
    ]
    assert [component["mass"] for component in components] == pytest.approx([9.0, 6.0])


def test_icc_3_1_is_one_for_perfect_consistency():
    subject_effect = np.arange(8, dtype=float)[:, None, None]
    fixed_block_effect = np.arange(4, dtype=float)[None, :, None]
    values = subject_effect + fixed_block_effect

    icc = _icc_3_1(values)

    assert icc == pytest.approx([1.0])
