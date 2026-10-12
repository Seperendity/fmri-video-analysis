from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fmri_video_analysis.core.data_loading import load_resms_volume
from fmri_video_analysis.methods.rsa.crossnobis import subject_crossnobis
from fmri_video_analysis.methods.rsa.searchlight import subject_maps


def test_external_noise_crossnobis_is_centered_at_zero_under_null():
    rng = np.random.default_rng(20260814)
    distances = []
    variance = np.ones(2_000)
    for _ in range(400):
        patterns = rng.normal(size=(4, 3, variance.size))
        distances.append(
            subject_crossnobis(patterns, variance, 0.2)["simple_complex"]
        )

    assert np.mean(distances) == pytest.approx(0.0, abs=0.01)


def test_external_noise_crossnobis_recovers_injected_pattern_effect():
    rng = np.random.default_rng(7)
    patterns = rng.normal(scale=0.5, size=(4, 3, 1_000))
    signal = np.tile([-1.0, 1.0], 500)
    patterns[:, 1] += signal

    distances = subject_crossnobis(patterns, np.ones(1_000), 0.2)

    assert distances["simple_complex"] > 0.8


def test_spatial_mean_centering_removes_uniform_activation_shift():
    patterns = np.zeros((4, 3, 200))
    patterns[:, 1] = 2.0

    uncentered = subject_crossnobis(patterns, np.ones(200), 0.2)
    centered = subject_crossnobis(
        patterns,
        np.ones(200),
        0.2,
        spatial_mean_center=True,
    )

    assert uncentered["simple_complex"] == pytest.approx(4.0)
    assert centered["simple_complex"] == pytest.approx(0.0, abs=1e-12)


def test_searchlight_uses_same_external_noise_distance_core():
    rng = np.random.default_rng(17)
    shape = (3, 3, 3)
    patterns = rng.normal(size=(4, 3) + shape)
    variance = rng.uniform(0.5, 2.0, size=shape)
    mask = np.ones(shape, dtype=bool)
    kernel = np.ones(shape, dtype=np.float32)

    maps = subject_maps(patterns, variance, mask, kernel, shrinkage=0.2)
    expected = subject_crossnobis(
        patterns.reshape(4, 3, -1),
        variance.ravel(),
        0.2,
    )

    centre = (1, 1, 1)
    for name, distance in expected.items():
        assert maps[name][centre] == pytest.approx(distance)


def test_resms_loader_rejects_geometry_mismatch(tmp_path: Path):
    subject_dir = tmp_path / "1"
    subject_dir.mkdir()
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)),
        subject_dir / "ResMS.nii",
    )
    reference = nib.Nifti1Image(np.ones((3, 2, 2)), np.eye(4))

    with pytest.raises(RuntimeError, match="Geometry mismatch"):
        load_resms_volume(tmp_path, 1, reference)
