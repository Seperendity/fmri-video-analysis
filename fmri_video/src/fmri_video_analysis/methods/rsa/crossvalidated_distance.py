"""Shared unbiased cross-validated distance helpers for RSA analyses."""

from __future__ import annotations

import numpy as np

CONDITIONS = ("Simple", "Complex", "Control")
PAIRS = (
    ("Simple", "Complex", "simple_complex"),
    ("Simple", "Control", "simple_control"),
    ("Complex", "Control", "complex_control"),
)


def stabilized_diagonal_variance(
    noise_variance: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """Shrink an externally estimated positive diagonal variance to its median."""

    variance = np.asarray(noise_variance, dtype=np.float64)
    if variance.ndim != 1 or variance.size == 0:
        raise ValueError("noise_variance must be a non-empty one-dimensional array")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if not np.all(np.isfinite(variance)):
        raise ValueError("noise_variance contains non-finite values")
    if np.any(variance <= 0):
        raise ValueError("noise_variance must be strictly positive")

    target = float(np.median(variance))
    stabilized = (1.0 - shrinkage) * variance + shrinkage * target
    return np.maximum(stabilized, target * 1e-6)


def whiten_patterns(
    patterns: np.ndarray,
    noise_variance: np.ndarray,
    shrinkage: float,
    *,
    spatial_mean_center: bool = False,
) -> np.ndarray:
    """Apply diagonal whitening estimated outside the condition-beta array."""

    values = np.asarray(patterns, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("patterns must have shape session x condition x feature")
    if values.shape[0] < 2:
        raise ValueError("at least two independent sessions are required")
    if values.shape[1] != len(CONDITIONS):
        raise ValueError(f"expected {len(CONDITIONS)} conditions")
    if not np.all(np.isfinite(values)):
        raise ValueError("patterns contain non-finite values")

    variance = stabilized_diagonal_variance(noise_variance, shrinkage)
    if variance.shape != (values.shape[-1],):
        raise ValueError(
            "noise_variance must contain exactly one value per pattern feature"
        )
    whitened = values / np.sqrt(variance)[None, None, :]
    if spatial_mean_center:
        whitened = whitened - whitened.mean(axis=-1, keepdims=True)
    return whitened


def ordered_crossproducts(differences: np.ndarray) -> np.ndarray:
    """Return per-feature ordered cross-products across independent sessions."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("differences must have shape session x feature")
    n_sessions = values.shape[0]
    return (
        np.sum(values, axis=0) ** 2 - np.sum(values**2, axis=0)
    ) / (n_sessions * (n_sessions - 1))


def pairwise_crossproducts(whitened_patterns: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate per-feature cross-products for the three condition pairs."""

    outputs: dict[str, np.ndarray] = {}
    for first, second, name in PAIRS:
        first_index = CONDITIONS.index(first)
        second_index = CONDITIONS.index(second)
        differences = (
            whitened_patterns[:, first_index] - whitened_patterns[:, second_index]
        )
        outputs[name] = ordered_crossproducts(differences)
    return outputs
