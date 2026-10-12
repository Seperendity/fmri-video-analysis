#!/usr/bin/env python3
"""Small statistical helpers shared by the revision analyses."""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def bh_fdr(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    values = p[valid]
    if not len(values):
        return q
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    q[valid] = restored
    return q


def cohens_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else np.nan


def paired_sign_flip_pvalue(
    diff: np.ndarray, n_permutations: int = 10000, seed: int = 20260731
) -> float:
    diff = np.asarray(diff, dtype=float)
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, n_permutations, 1000):
        n = min(1000, n_permutations - start)
        signs = rng.choice((-1.0, 1.0), size=(n, len(diff)))
        means = (signs * diff[None, :]).mean(axis=1)
        exceed += int(np.sum(np.abs(means) >= observed - 1e-15))
    return (exceed + 1) / (n_permutations + 1)


def exact_label_permutation_anova(values: np.ndarray, labels: np.ndarray) -> float:
    """Exact three-group label permutation p-value for 12 video-level values."""

    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    groups = np.unique(labels)
    if len(groups) != 3 or any(np.sum(labels == g) != 4 for g in groups):
        raise ValueError("Expected exactly three groups with four videos each.")

    def f_stat(index_groups):
        overall = values.mean()
        between = sum(
            len(idx) * (values[list(idx)].mean() - overall) ** 2 for idx in index_groups
        )
        within = sum(
            np.sum((values[list(idx)] - values[list(idx)].mean()) ** 2)
            for idx in index_groups
        )
        return (between / 2) / (within / 9) if within > 0 else np.inf

    observed_groups = [np.flatnonzero(labels == g) for g in groups]
    observed = f_stat(observed_groups)
    all_indices = set(range(12))
    exceed = total = 0
    for first in itertools.combinations(range(12), 4):
        remaining = sorted(all_indices.difference(first))
        for second in itertools.combinations(remaining, 4):
            third = tuple(sorted(set(remaining).difference(second)))
            if min(second) > min(third):  # remove swapped group-2/group-3 duplicates
                continue
            total += 1
            if f_stat((first, second, third)) >= observed - 1e-15:
                exceed += 1
    return exceed / total


def exact_two_group_permutation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided exact independent-label test and Hedges g."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pooled = np.concatenate([x, y])
    observed = abs(x.mean() - y.mean())
    exceed = total = 0
    for indices in itertools.combinations(range(len(pooled)), len(x)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(indices)] = True
        delta = abs(pooled[mask].mean() - pooled[~mask].mean())
        total += 1
        exceed += int(delta >= observed - 1e-15)
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    )
    d = (x.mean() - y.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return exceed / total, float(correction * d)


def paired_tost_dz(
    diff: np.ndarray, equivalence_bound_dz: float = 0.2
) -> dict[str, float | bool]:
    """Paired TOST using a standardized dz equivalence bound."""

    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    n = len(diff)
    if sd == 0:
        # dz is undefined for a constant difference, so equivalence cannot be
        # established from this degenerate sample (including an all-zero one).
        return {"p_lower": np.nan, "p_upper": np.nan, "equivalent": False}
    lower, upper = -equivalence_bound_dz * sd, equivalence_bound_dz * sd
    se = sd / np.sqrt(n)
    t_lower = (diff.mean() - lower) / se
    t_upper = (diff.mean() - upper) / se
    p_lower = stats.t.sf(t_lower, n - 1)
    p_upper = stats.t.cdf(t_upper, n - 1)
    return {
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "equivalent": bool(max(p_lower, p_upper) < 0.05),
    }
