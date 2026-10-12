from __future__ import annotations

import json
import os
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from fmri_video_analysis.core import config as config_module
from fmri_video_analysis.core.config import write_analysis_config
from fmri_video_analysis.core.provenance import (
    CacheMismatchError,
    atomic_write_json,
    cache_manifest,
    file_fingerprint,
    git_metadata,
    require_matching_cache,
)
from fmri_video_analysis.core.validation import (
    validate_nifti_geometry,
    validate_subject_ids,
)
from fmri_video_analysis.methods.connectivity import block_stimulus
from fmri_video_analysis.methods.stimulus.characterization import (
    canonical_semantic_tables,
)
from fmri_video_analysis.reporting.contracts import (
    require_formal_mvpa,
    result_dir,
)


def test_subject_contract_detects_missing_extra_and_duplicate_ids():
    with pytest.raises(
        RuntimeError, match=r"missing=\[2\].*extra=\[4\].*duplicates=\[1\]"
    ):
        validate_subject_ids([1, 1, 3, 4], expected_ids=[1, 2, 3], context="test")


def test_analysis_config_normalizes_numpy_subject_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "runtime_metadata", dict)
    path = write_analysis_config(
        tmp_path / "metadata",
        first_level_dir=tmp_path / "first_level",
        stimulus_frames_dir=tmp_path / "frames",
        timeseries_root=tmp_path / "timeseries",
        subject_ids=(np.int64(1), np.int64(2)),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["subject_ids"] == [1, 2]
    assert all(type(subject_id) is int for subject_id in payload["subject_ids"])


def test_nifti_contract_rejects_shifted_affine_with_same_shape():
    reference = nib.Nifti1Image(np.zeros((3, 4, 5)), np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 3.0
    candidate = nib.Nifti1Image(np.zeros((3, 4, 5)), shifted)

    with pytest.raises(RuntimeError, match="affine differs"):
        validate_nifti_geometry(reference, candidate, context="shift test")


def test_cache_contract_reports_parameter_differences(tmp_path):
    artifact = tmp_path / "result.dat"
    artifact.write_text("complete", encoding="utf-8")
    manifest_path = tmp_path / "cache.json"
    observed = cache_manifest(stage="test", parameters={"seed": 1}, inputs={})
    requested = cache_manifest(stage="test", parameters={"seed": 2}, inputs={})
    atomic_write_json(manifest_path, observed)

    with pytest.raises(
        CacheMismatchError, match=r"parameters.seed: cached=1, requested=2"
    ):
        require_matching_cache(
            manifest_path,
            requested,
            artifacts_exist=artifact.exists(),
            force=False,
        )

    assert not require_matching_cache(
        manifest_path,
        requested,
        artifacts_exist=True,
        force=True,
    )


def test_full_fingerprint_is_stable_when_only_mtime_changes(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"same bytes")
    first = file_fingerprint(path, full_hash=True)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    second = file_fingerprint(path, full_hash=True)

    assert first == second


def test_semantic_fresh_and_rebuild_paths_share_one_schema():
    sampled = pd.DataFrame(
        [
            {"session": 1, "condition": "Simple", "video_id": "001", "frame_index": 0},
            {"session": 1, "condition": "Simple", "video_id": "001", "frame_index": 25},
        ]
    )
    detections = pd.DataFrame(
        [
            {
                "session": 1,
                "condition": "Simple",
                "video_id": "001",
                "frame_index": 0,
                "label": "car",
                "score": 0.9,
                "x1": 0,
                "y1": 0,
                "x2": 50,
                "y2": 50,
            },
            {
                "session": 1,
                "condition": "Simple",
                "video_id": "001",
                "frame_index": 25,
                "label": "dog",
                "score": 0.99,
                "x1": 0,
                "y1": 0,
                "x2": 20,
                "y2": 20,
            },
        ]
    )

    fresh_frames, fresh_video = canonical_semantic_tables(sampled, detections)
    rebuilt_frames, rebuilt_video = canonical_semantic_tables(fresh_frames, detections)

    pd.testing.assert_frame_equal(fresh_frames, rebuilt_frames)
    pd.testing.assert_frame_equal(fresh_video, rebuilt_video)
    assert fresh_frames.target_count.tolist() == [1, 0]


def test_mixed_model_uses_crossed_subject_and_video_components(monkeypatch):
    captured = {}

    class FakeFit:
        params = pd.Series({"condition_complex": 0.1})
        bse = pd.Series({"condition_complex": 0.02})
        pvalues = pd.Series({"condition_complex": 0.01})
        aic = 1.0
        bic = 2.0
        converged = True

    class FakeModel:
        def fit(self, **kwargs):
            captured["fit"] = kwargs
            return FakeFit()

    def fake_mixedlm(formula, data, **kwargs):
        captured.update({"formula": formula, "data": data, **kwargs})
        return FakeModel()

    monkeypatch.setattr(block_stimulus.smf, "mixedlm", fake_mixedlm)
    table = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "video_id": ["001", "009", "001", "009"],
            "mean_fisher_z": [0.1, 0.2, 0.0, 0.1],
            "condition_complex": [0, 1, 0, 1],
        }
    )

    row = block_stimulus._mixed_model_row(
        table,
        "global",
        "base",
        "mean_fisher_z ~ condition_complex",
    )

    assert np.unique(captured["groups"]).tolist() == [1]
    assert captured["re_formula"] == "0"
    assert captured["vc_formula"] == {
        "subject": "0 + C(subject_id)",
        "video": "0 + C(video_id)",
    }
    assert (
        row["random_effects_structure"]
        == "crossed random intercepts for subject and video"
    )


def test_formal_report_gate_requires_completed_structured_permutations():
    incomplete = {
        "formal_inference_complete": False,
        "structured_permutations_completed": 0,
        "structured_label_permutation_p": None,
    }
    with pytest.raises(RuntimeError, match="Formal MVPA inference is incomplete"):
        require_formal_mvpa(incomplete)

    complete = {
        "formal_inference_complete": True,
        "structured_permutations_completed": 10_000,
        "structured_label_permutation_p": 0.012,
    }
    assert require_formal_mvpa(complete) is complete


def test_report_layout_never_falls_back_per_module(tmp_path):
    normalized = tmp_path / "normalized/module"
    legacy = tmp_path / "legacy_module"
    normalized.mkdir(parents=True)
    legacy.mkdir()

    assert (
        result_dir(
            tmp_path,
            "normalized/module",
            "legacy_module",
            layout="legacy",
        )
        == legacy
    )
    with pytest.raises(FileNotFoundError, match="cannot mix"):
        result_dir(
            tmp_path,
            "missing_normalized",
            "legacy_module",
            layout="normalized",
        )



def test_git_metadata_hashes_untracked_file_contents(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "new_source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    before = git_metadata(tmp_path)

    source.write_text("VALUE = 2\n", encoding="utf-8")
    after = git_metadata(tmp_path)

    assert before["status_sha256"] == after["status_sha256"]
    assert before["untracked_count"] == after["untracked_count"] == 1
    assert (
        before["untracked_content_sha256"]
        != after["untracked_content_sha256"]
    )
