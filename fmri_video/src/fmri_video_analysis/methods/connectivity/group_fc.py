"""Group statistics for the frozen 249-node background-FC matrices."""

from __future__ import annotations

import json
import zlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from fmri_video_analysis.core.statistics import (
    bh_fdr,
    cohens_dz,
    paired_sign_flip_pvalue,
    paired_tost_dz,
)
from fmri_video_analysis.core.study_design import CONDITIONS
from fmri_video_analysis.core.validation import validate_subject_ids
from fmri_video_analysis.methods.connectivity.nbs import (
    signed_components_for_edges as _components_for_edges,
)
from fmri_video_analysis.methods.connectivity.timeseries_topology import (
    graph_metrics,
    threshold_positive_matrix,
)

CONTRASTS = {
    "complex_minus_simple": np.array([-1.0, 1.0, 0.0]),
    "task_mean_minus_control": np.array([0.5, 0.5, -1.0]),
    "complex_minus_control": np.array([0.0, 1.0, -1.0]),
    "simple_minus_control": np.array([1.0, 0.0, -1.0]),
}
PRIMARY_CONTRASTS = ("complex_minus_simple", "task_mean_minus_control")
YEO17_NETWORKS = (
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicB",
    "LimbicA",
    "ContA",
    "ContB",
    "ContC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "TempPar",
)


def _stable_seed(name: str, base_seed: int) -> int:
    return int((base_seed + zlib.crc32(name.encode("utf-8"))) % (2**32 - 1))


def load_group_connectivity(
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    archive = np.load(
        run_dir / "subject_condition_connectivity.npz", allow_pickle=False
    )
    matrices = archive["matrices"].astype(np.float64)
    subject_ids = archive["subject_ids"].astype(int)
    validate_subject_ids(
        subject_ids,
        context=f"group FC archive {run_dir}",
    )
    conditions = tuple(archive["conditions"].tolist())
    nodes = pd.read_csv(run_dir / "atlas/composite_249_nodes.tsv", sep="\t")
    if matrices.shape != (87, 3, 249, 249):
        raise RuntimeError(f"Expected (87, 3, 249, 249), got {matrices.shape}.")
    if conditions != CONDITIONS:
        raise RuntimeError(f"Unexpected condition order: {conditions}.")
    if len(nodes) != 249 or not np.all(np.isfinite(matrices)):
        raise RuntimeError("Invalid nodes or non-finite FC values.")
    if not np.allclose(matrices, matrices.transpose(0, 1, 3, 2), atol=1e-7):
        raise RuntimeError("FC matrices are not symmetric.")
    return matrices, subject_ids, nodes


def _edge_mean(
    matrices: np.ndarray, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    if len(first) == 0:
        raise ValueError("Metric contains no edges.")
    return matrices[:, :, first, second].mean(axis=2)


def predefined_metric_values(
    matrices: np.ndarray,
    subject_ids: np.ndarray,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate global, cortical-network, hemisphere, and domain FC metrics."""

    n_nodes = matrices.shape[-1]
    values: list[tuple[str, str, str, np.ndarray]] = []
    upper = np.triu_indices(n_nodes, 1)
    values.append(
        ("global", "whole_brain_signed_mean", "Whole brain", _edge_mean(matrices, *upper))
    )

    domain = nodes["domain"].to_numpy()
    hemisphere = nodes["hemisphere"].to_numpy()
    network = nodes["network"].to_numpy()
    cortex = np.flatnonzero(domain == "Cortex")
    left = np.flatnonzero((domain == "Cortex") & (hemisphere == "Left"))
    right = np.flatnonzero((domain == "Cortex") & (hemisphere == "Right"))
    cross_left, cross_right = np.meshgrid(left, right, indexing="ij")
    values.append(
        (
            "hemisphere",
            "cortical_cross_hemisphere",
            "Cortical left-right",
            _edge_mean(matrices, cross_left.ravel(), cross_right.ravel()),
        )
    )

    for first_domain, second_domain, metric_name in (
        ("Cortex", "Subcortex", "cortex_subcortex"),
        ("Cortex", "Cerebellum", "cortex_cerebellum"),
        ("Subcortex", "Cerebellum", "subcortex_cerebellum"),
    ):
        first_nodes = np.flatnonzero(domain == first_domain)
        second_nodes = np.flatnonzero(domain == second_domain)
        first_grid, second_grid = np.meshgrid(
            first_nodes, second_nodes, indexing="ij"
        )
        values.append(
            (
                "domain",
                metric_name,
                f"{first_domain}-{second_domain}",
                _edge_mean(matrices, first_grid.ravel(), second_grid.ravel()),
            )
        )

    for network_name in YEO17_NETWORKS:
        network_nodes = np.flatnonzero(
            (domain == "Cortex") & (network == network_name)
        )
        within = np.triu_indices(len(network_nodes), 1)
        within_values = _edge_mean(
            matrices,
            network_nodes[within[0]],
            network_nodes[within[1]],
        )
        values.append(
            (
                "network_within",
                f"within_{network_name}",
                network_name,
                within_values,
            )
        )
        other_nodes = cortex[network[cortex] != network_name]
        first_grid, second_grid = np.meshgrid(
            network_nodes, other_nodes, indexing="ij"
        )
        between_values = _edge_mean(
            matrices, first_grid.ravel(), second_grid.ravel()
        )
        values.append(
            (
                "network_separation",
                f"separation_{network_name}",
                network_name,
                within_values - between_values,
            )
        )

    for first_network, second_network in combinations(YEO17_NETWORKS, 2):
        first_nodes = np.flatnonzero(
            (domain == "Cortex") & (network == first_network)
        )
        second_nodes = np.flatnonzero(
            (domain == "Cortex") & (network == second_network)
        )
        first_grid, second_grid = np.meshgrid(
            first_nodes, second_nodes, indexing="ij"
        )
        values.append(
            (
                "network_between",
                f"between_{first_network}_{second_network}",
                f"{first_network}-{second_network}",
                _edge_mean(matrices, first_grid.ravel(), second_grid.ravel()),
            )
        )

    rows = []
    for family, metric, label, metric_values in values:
        for subject_index, subject_id in enumerate(subject_ids):
            for condition_index, condition in enumerate(CONDITIONS):
                rows.append(
                    {
                        "subject_id": int(subject_id),
                        "condition": condition,
                        "family": family,
                        "metric": metric,
                        "label": label,
                        "value": float(
                            metric_values[subject_index, condition_index]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _bootstrap_paired_summary(
    differences: np.ndarray, n_bootstrap: int, seed: int
) -> dict:
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(n_bootstrap, len(differences))
    )
    samples = differences[indices]
    sample_means = samples.mean(axis=1)
    sample_sd = samples.std(axis=1, ddof=1)
    sample_dz = np.divide(
        sample_means,
        sample_sd,
        out=np.full(n_bootstrap, np.nan),
        where=sample_sd > 0,
    )
    return {
        "mean_ci95_low": float(np.quantile(sample_means, 0.025)),
        "mean_ci95_high": float(np.quantile(sample_means, 0.975)),
        "dz_ci95_low": float(np.nanquantile(sample_dz, 0.025)),
        "dz_ci95_high": float(np.nanquantile(sample_dz, 0.975)),
    }


def predefined_metric_statistics(
    values: pd.DataFrame,
    output_dir: Path,
    n_permutations: int = 10000,
    n_bootstrap: int = 5000,
    seed: int = 20260731,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    values.to_csv(output_dir / "predefined_metric_values.csv.gz", index=False)
    rows = []
    for (family, metric, label), part in values.groupby(
        ["family", "metric", "label"], sort=False
    ):
        pivot = part.pivot(index="subject_id", columns="condition", values="value")
        condition_values = pivot.loc[:, CONDITIONS].to_numpy()
        for contrast_name, coefficients in CONTRASTS.items():
            differences = condition_values @ coefficients
            bootstrap = _bootstrap_paired_summary(
                differences,
                n_bootstrap,
                _stable_seed(f"{family}:{metric}:{contrast_name}:bootstrap", seed),
            )
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "label": label,
                    "contrast": contrast_name,
                    "n_subjects": len(differences),
                    "mean_difference": float(differences.mean()),
                    "sd_difference": float(differences.std(ddof=1)),
                    "cohens_dz": cohens_dz(differences),
                    "p_sign_flip": paired_sign_flip_pvalue(
                        differences,
                        n_permutations,
                        _stable_seed(f"{family}:{metric}:{contrast_name}", seed),
                    ),
                    **bootstrap,
                }
            )
    table = pd.DataFrame(rows)
    table["q_fdr_within_family"] = np.nan
    for indices in table.groupby(
        ["family", "contrast"], sort=False
    ).groups.values():
        table.loc[indices, "q_fdr_within_family"] = bh_fdr(
            table.loc[indices, "p_sign_flip"].to_numpy()
        )
    table.to_csv(output_dir / "predefined_metric_statistics.csv", index=False)
    return table


def edgewise_statistics(
    matrices: np.ndarray,
    nodes: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict]:
    """Paired edge tests with BH-FDR separately for the two main contrasts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    upper = np.triu_indices(matrices.shape[-1], 1)
    edges = matrices[:, :, upper[0], upper[1]]
    summaries: dict[str, dict] = {}
    for contrast_name in PRIMARY_CONTRASTS:
        differences = np.einsum(
            "sce,c->se", edges, CONTRASTS[contrast_name], optimize=True
        )
        mean_difference = differences.mean(axis=0)
        sd_difference = differences.std(axis=0, ddof=1)
        t_values = np.divide(
            mean_difference,
            sd_difference / np.sqrt(len(differences)),
            out=np.zeros_like(mean_difference),
            where=sd_difference > 0,
        )
        p_values = 2 * stats.t.sf(np.abs(t_values), len(differences) - 1)
        q_values = bh_fdr(p_values)
        dz = np.divide(
            mean_difference,
            sd_difference,
            out=np.full_like(mean_difference, np.nan),
            where=sd_difference > 0,
        )
        table = pd.DataFrame(
            {
                "node1_id": nodes["node_id"].to_numpy()[upper[0]],
                "node1_name": nodes["node_name"].to_numpy()[upper[0]],
                "node1_network": nodes["network"].to_numpy()[upper[0]],
                "node1_domain": nodes["domain"].to_numpy()[upper[0]],
                "node2_id": nodes["node_id"].to_numpy()[upper[1]],
                "node2_name": nodes["node_name"].to_numpy()[upper[1]],
                "node2_network": nodes["network"].to_numpy()[upper[1]],
                "node2_domain": nodes["domain"].to_numpy()[upper[1]],
                "mean_difference": mean_difference,
                "cohens_dz": dz,
                "t": t_values,
                "p": p_values,
                "q_fdr": q_values,
                "direction": np.where(
                    mean_difference > 0,
                    "positive",
                    np.where(mean_difference < 0, "negative", "zero"),
                ),
            }
        )
        table.to_csv(
            output_dir / f"edgewise_{contrast_name}.csv.gz", index=False
        )
        summaries[contrast_name] = {
            "n_edges": len(table),
            "n_fdr_significant": int(np.sum(q_values < 0.05)),
            "minimum_p": float(np.min(p_values)),
            "minimum_q": float(np.min(q_values)),
            "max_abs_dz": float(np.nanmax(np.abs(dz))),
        }
    (output_dir / "edgewise_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summaries


def _permuted_t_batches(
    differences: np.ndarray,
    n_permutations: int,
    seed: int,
    batch_size: int,
):
    """Yield sign-flipped t arrays, using CUDA when available."""

    try:
        import torch

        use_cuda = torch.cuda.is_available()
    except ImportError:
        torch = None
        use_cuda = False
    if use_cuda:
        device = torch.device("cuda:0")
        edge_tensor = torch.as_tensor(
            differences, dtype=torch.float32, device=device
        )
        sum_squares = torch.sum(torch.square(edge_tensor), dim=0)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        for start in range(0, n_permutations, batch_size):
            current = min(batch_size, n_permutations - start)
            signs = torch.randint(
                0,
                2,
                (current, len(differences)),
                generator=generator,
                device=device,
                dtype=torch.int8,
            ).to(torch.float32)
            signs = signs.mul_(2).sub_(1)
            sums = signs @ edge_tensor
            means = sums / len(differences)
            variances = (
                sum_squares[None, :] - len(differences) * torch.square(means)
            ) / (len(differences) - 1)
            t_values = means / torch.sqrt(
                torch.clamp(variances, min=1e-20) / len(differences)
            )
            yield t_values.cpu().numpy(), "cuda:0"
    else:
        rng = np.random.default_rng(seed)
        sum_squares = np.sum(np.square(differences), axis=0)
        for start in range(0, n_permutations, batch_size):
            current = min(batch_size, n_permutations - start)
            signs = rng.choice(
                (-1.0, 1.0), size=(current, len(differences))
            )
            sums = signs @ differences
            means = sums / len(differences)
            variances = (
                sum_squares[None, :] - len(differences) * np.square(means)
            ) / (len(differences) - 1)
            t_values = means / np.sqrt(
                np.maximum(variances, 1e-20) / len(differences)
            )
            yield t_values, "cpu"


def nbs_analysis(
    matrices: np.ndarray,
    nodes: pd.DataFrame,
    output_dir: Path,
    n_permutations: int = 10000,
    thresholds: tuple[float, ...] = (2.5, 3.1, 3.5),
    seed: int = 20260731,
    batch_size: int = 64,
) -> dict:
    """Run two-sided component-mass NBS for both prespecified contrasts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    n_nodes = matrices.shape[-1]
    upper = np.triu_indices(n_nodes, 1)
    edges = matrices[:, :, upper[0], upper[1]]
    metadata: dict[str, dict] = {}
    for contrast_name in PRIMARY_CONTRASTS:
        differences = np.einsum(
            "sce,c->se", edges, CONTRASTS[contrast_name], optimize=True
        )
        means = differences.mean(axis=0)
        sd = differences.std(axis=0, ddof=1)
        observed_t = np.divide(
            means,
            sd / np.sqrt(len(differences)),
            out=np.zeros_like(means),
            where=sd > 0,
        )
        observed = {
            threshold: _components_for_edges(
                observed_t, upper, n_nodes, threshold
            )
            for threshold in thresholds
        }
        null_max = {
            threshold: np.zeros(n_permutations, dtype=np.float32)
            for threshold in thresholds
        }
        completed = 0
        compute_device = "cpu"
        for t_batch, compute_device in _permuted_t_batches(
            differences,
            n_permutations,
            _stable_seed(f"nbs:{contrast_name}", seed),
            batch_size,
        ):
            for batch_index, t_edges in enumerate(t_batch):
                permutation = completed + batch_index
                for threshold in thresholds:
                    components = _components_for_edges(
                        t_edges, upper, n_nodes, threshold
                    )
                    null_max[threshold][permutation] = max(
                        (component["mass"] for component in components),
                        default=0.0,
                    )
            completed += len(t_batch)
            print(
                f"NBS {contrast_name}: {completed}/{n_permutations}",
                flush=True,
            )

        component_rows = []
        edge_rows = []
        for threshold in thresholds:
            for component_index, component in enumerate(
                observed[threshold], 1
            ):
                p_fwer = float(
                    (
                        np.sum(null_max[threshold] >= component["mass"]) + 1
                    )
                    / (n_permutations + 1)
                )
                component_rows.append(
                    {
                        "contrast": contrast_name,
                        "threshold_abs_t": threshold,
                        "component_id": component_index,
                        "direction": component["direction"],
                        "n_nodes": len(component["nodes"]),
                        "n_edges": len(component["edge_indices"]),
                        "component_mass": component["mass"],
                        "p_fwer": p_fwer,
                        "significant_fwer_0_05": p_fwer < 0.05,
                    }
                )
                for edge_index in component["edge_indices"]:
                    first, second = upper[0][edge_index], upper[1][edge_index]
                    edge_rows.append(
                        {
                            "contrast": contrast_name,
                            "threshold_abs_t": threshold,
                            "component_id": component_index,
                            "component_p_fwer": p_fwer,
                            "direction": component["direction"],
                            "node1_id": int(nodes.iloc[first]["node_id"]),
                            "node1_name": nodes.iloc[first]["node_name"],
                            "node2_id": int(nodes.iloc[second]["node_id"]),
                            "node2_name": nodes.iloc[second]["node_name"],
                            "mean_difference": means[edge_index],
                            "t": observed_t[edge_index],
                        }
                    )
        component_table = pd.DataFrame(component_rows)
        edge_table = pd.DataFrame(edge_rows)
        component_table.to_csv(
            output_dir / f"nbs_{contrast_name}_components.csv", index=False
        )
        edge_table.to_csv(
            output_dir / f"nbs_{contrast_name}_edges.csv.gz", index=False
        )
        pd.DataFrame(
            {
                f"threshold_{threshold:g}": null_max[threshold]
                for threshold in thresholds
            }
        ).to_csv(
            output_dir / f"nbs_{contrast_name}_null_max_mass.csv.gz",
            index=False,
        )
        metadata[contrast_name] = {
            "n_permutations": n_permutations,
            "thresholds": list(thresholds),
            "compute_device": compute_device,
            "n_observed_components": {
                str(threshold): len(observed[threshold])
                for threshold in thresholds
            },
            "n_significant_components": (
                int(component_table["significant_fwer_0_05"].sum())
                if len(component_table)
                else 0
            ),
        }
    (output_dir / "nbs_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def cortical_network_separation(
    matrix: np.ndarray, nodes: pd.DataFrame
) -> float:
    domain = nodes["domain"].to_numpy()
    network = nodes["network"].to_numpy()
    cortex = np.flatnonzero(domain == "Cortex")
    upper = np.triu_indices(len(cortex), 1)
    first, second = cortex[upper[0]], cortex[upper[1]]
    same_network = network[first] == network[second]
    values = matrix[first, second]
    return float(values[same_network].mean() - values[~same_network].mean())


def _topology_one(
    matrix: np.ndarray,
    subject_id: int,
    condition: str,
    density: float,
    nodes: pd.DataFrame,
) -> dict:
    thresholded = threshold_positive_matrix(matrix, density)
    return {
        "subject_id": subject_id,
        "condition": condition,
        "density": density,
        **graph_metrics(thresholded),
        "network_separation": cortical_network_separation(
            thresholded, nodes
        ),
    }


def topology_analysis(
    matrices: np.ndarray,
    subject_ids: np.ndarray,
    nodes: pd.DataFrame,
    output_dir: Path,
    n_permutations: int = 10000,
    equivalence_bound_dz: float = 0.2,
    n_jobs: int = 1,
    seed: int = 20260731,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    densities = np.round(np.arange(0.10, 0.301, 0.02), 2)
    tasks = (
        (matrices[subject_index, condition_index], int(subject_id), condition, density)
        for subject_index, subject_id in enumerate(subject_ids)
        for condition_index, condition in enumerate(CONDITIONS)
        for density in densities
    )
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_topology_one)(matrix, subject_id, condition, density, nodes)
        for matrix, subject_id, condition, density in tasks
    )
    by_density = pd.DataFrame(rows)
    by_density.to_csv(output_dir / "topology_by_density.csv.gz", index=False)

    metric_names = (
        "strength",
        "global_efficiency",
        "modularity",
        "network_separation",
        "clustering",
        "path_length",
    )
    auc_rows = []
    for (subject_id, condition), part in by_density.groupby(
        ["subject_id", "condition"], sort=False
    ):
        part = part.sort_values("density")
        row = {"subject_id": subject_id, "condition": condition}
        for metric in metric_names:
            row[f"{metric}_auc"] = float(
                np.trapezoid(part[metric], part["density"])
            )
        auc_rows.append(row)
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(output_dir / "topology_auc.csv", index=False)

    primary_metrics = {
        "strength_auc",
        "global_efficiency_auc",
        "modularity_auc",
        "network_separation_auc",
    }
    statistic_rows = []
    for metric in [f"{name}_auc" for name in metric_names]:
        pivot = auc.pivot(index="subject_id", columns="condition", values=metric)
        condition_values = pivot.loc[:, CONDITIONS].to_numpy()
        for contrast_name, coefficients in CONTRASTS.items():
            differences = condition_values @ coefficients
            row = {
                "metric": metric,
                "metric_role": (
                    "primary" if metric in primary_metrics else "secondary"
                ),
                "contrast": contrast_name,
                "mean_difference": float(differences.mean()),
                "cohens_dz": cohens_dz(differences),
                "p_sign_flip": paired_sign_flip_pvalue(
                    differences,
                    n_permutations,
                    _stable_seed(f"topology:{metric}:{contrast_name}", seed),
                ),
                **_bootstrap_paired_summary(
                    differences,
                    5000,
                    _stable_seed(
                        f"topology:{metric}:{contrast_name}:bootstrap", seed
                    ),
                ),
            }
            if contrast_name == "complex_minus_simple":
                tost = paired_tost_dz(differences, equivalence_bound_dz)
                standard_error = differences.std(ddof=1) / np.sqrt(
                    len(differences)
                )
                critical = stats.t.ppf(0.95, len(differences) - 1)
                row.update(
                    {
                        "tost_p_lower": tost["p_lower"],
                        "tost_p_upper": tost["p_upper"],
                        "tost_equivalent_dz_0_20": tost["equivalent"],
                        "mean_ci90_low": float(
                            differences.mean() - critical * standard_error
                        ),
                        "mean_ci90_high": float(
                            differences.mean() + critical * standard_error
                        ),
                    }
                )
            statistic_rows.append(row)
    statistic_table = pd.DataFrame(statistic_rows)
    statistic_table["q_fdr_by_role_contrast"] = np.nan
    for indices in statistic_table.groupby(
        ["metric_role", "contrast"], sort=False
    ).groups.values():
        statistic_table.loc[indices, "q_fdr_by_role_contrast"] = bh_fdr(
            statistic_table.loc[indices, "p_sign_flip"].to_numpy()
        )
    statistic_table.to_csv(
        output_dir / "topology_statistics.csv", index=False
    )
    primary_tost = statistic_table[
        (statistic_table["metric_role"] == "primary")
        & (statistic_table["contrast"] == "complex_minus_simple")
    ]
    summary = {
        "densities": densities.tolist(),
        "equivalence_bound_dz": equivalence_bound_dz,
        "all_four_primary_metrics_equivalent": bool(
            len(primary_tost) == 4
            and primary_tost["tost_equivalent_dz_0_20"].fillna(False).all()
        ),
    }
    (output_dir / "topology_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return auc, statistic_table
