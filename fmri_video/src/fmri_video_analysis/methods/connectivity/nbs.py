"""Shared signed-component construction for two-sided NBS analyses."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def signed_components_for_edges(
    t_edges: np.ndarray,
    upper: tuple[np.ndarray, np.ndarray],
    n_nodes: int,
    threshold: float,
) -> list[dict]:
    """Return sign-homogeneous supra-threshold connected components."""

    t_edges = np.asarray(t_edges, dtype=float)
    components: list[dict] = []
    for direction, selected in (
        ("positive", t_edges >= threshold),
        ("negative", t_edges <= -threshold),
    ):
        selected_indices = np.flatnonzero(selected)
        if len(selected_indices) == 0:
            continue
        first = upper[0][selected_indices]
        second = upper[1][selected_indices]
        adjacency = coo_matrix(
            (
                np.ones(2 * len(first), dtype=np.uint8),
                (
                    np.concatenate([first, second]),
                    np.concatenate([second, first]),
                ),
            ),
            shape=(n_nodes, n_nodes),
        ).tocsr()
        _, labels = connected_components(adjacency, directed=False)
        participating = np.concatenate([first, second])
        for component_label in np.unique(labels[participating]):
            component_edges = selected_indices[labels[first] == component_label]
            if len(component_edges) == 0:
                continue
            nodes = np.unique(
                np.concatenate(
                    [upper[0][component_edges], upper[1][component_edges]]
                )
            )
            components.append(
                {
                    "direction": direction,
                    "nodes": nodes,
                    "edge_indices": component_edges,
                    "mass": float(np.sum(np.abs(t_edges[component_edges]))),
                }
            )
    return components


def signed_components_from_matrix(
    t_matrix: np.ndarray, threshold: float
) -> list[dict]:
    """Matrix convenience wrapper around :func:`signed_components_for_edges`."""

    t_matrix = np.asarray(t_matrix, dtype=float)
    if t_matrix.ndim != 2 or t_matrix.shape[0] != t_matrix.shape[1]:
        raise ValueError(f"Expected a square t matrix, got {t_matrix.shape}.")
    upper = np.triu_indices(len(t_matrix), 1)
    return signed_components_for_edges(
        t_matrix[upper], upper, len(t_matrix), threshold
    )
