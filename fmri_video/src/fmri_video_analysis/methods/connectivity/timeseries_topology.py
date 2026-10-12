"""Parameterized whole-brain connectivity and topology analysis.

The script intentionally refuses to run without the preprocessed 4D time-series
root. Beta images cannot be used to estimate within-block functional
connectivity.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import signal
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiLabelsMasker
from scipy import stats
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from fmri_video_analysis.core.config import (
    ATLAS_DIR,
    FIRST_LEVEL_DIR,
    RESULTS_DIR,
    numeric_subject_dirs,
)
from fmri_video_analysis.core.statistics import (
    bh_fdr,
    cohens_dz,
    paired_sign_flip_pvalue,
    paired_tost_dz,
)
from fmri_video_analysis.methods.connectivity.nbs import (
    signed_components_from_matrix,
)

CONDITIONS = ("Simple", "Complex", "Control")


def schaefer_resource():
    root = ATLAS_DIR / "schaefer_2018"
    image = root / "Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
    order = root / "Schaefer2018_200Parcels_17Networks_order.txt"
    names = {}
    for line in order.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            names[int(fields[0])] = fields[1]
    return image, names


def fsl_xml_resource(xml_name: str):
    atlas_root = Path("/usr/local/fsl/data/atlases")
    xml_root = ET.parse(atlas_root / xml_name).getroot()
    names = {
        int(node.attrib["index"]) + 1: (node.text or "").strip()
        for node in xml_root.findall(".//label")
    }
    candidates = []
    image_nodes = xml_root.findall(".//summaryimagefile")
    image_nodes = sorted(image_nodes, key=lambda node: "2mm" not in (node.text or ""))
    for node in image_nodes:
        stem = (node.text or "").strip().lstrip("/")
        candidates.extend([atlas_root / stem, atlas_root / f"{stem}.nii.gz"])
    image = next((p for p in candidates if p.exists()), None)
    if image is None:
        raise FileNotFoundError(f"Cannot resolve atlas image from {xml_name}")
    return image, names


def atlas_network(name: str, source: str) -> str:
    if source == "Schaefer200":
        match = re.search(r"17Networks_[LR]H_([^_]+)", name)
        return match.group(1) if match else "SchaeferOther"
    if source == "HarvardOxford":
        return "Subcortical"
    return "Cerebellum"


def build_composite_atlas(
    reference_img: nib.Nifti1Image, output_dir: Path, min_voxels: int = 20
):
    output_dir.mkdir(parents=True, exist_ok=True)
    resources = [
        (*schaefer_resource(), "Schaefer200"),
        (*fsl_xml_resource("HarvardOxford-Subcortical.xml"), "HarvardOxford"),
        (*fsl_xml_resource("Cerebellum_MNIfnirt.xml"), "Cerebellum"),
    ]
    composite = np.zeros(reference_img.shape[:3], dtype=np.int16)
    rows = []
    new_id = 1
    for image_path, names, source in resources:
        resampled = resample_to_img(
            nib.load(str(image_path)), reference_img, interpolation="nearest"
        )
        labels = np.rint(np.asarray(resampled.dataobj)).astype(int)
        for original_id in sorted(i for i in np.unique(labels) if i > 0):
            mask = (labels == original_id) & (composite == 0)
            n_voxels = int(mask.sum())
            if n_voxels < min_voxels:
                continue
            composite[mask] = new_id
            name = names.get(int(original_id), f"{source}_{original_id}")
            rows.append(
                {
                    "node_id": new_id,
                    "source_atlas": source,
                    "source_label_id": int(original_id),
                    "node_name": name,
                    "network": atlas_network(name, source),
                    "n_voxels": n_voxels,
                }
            )
            new_id += 1
    atlas_path = output_dir / "composite_schaefer200_subcortical_cerebellum.nii.gz"
    nib.save(
        nib.Nifti1Image(composite, reference_img.affine, reference_img.header),
        atlas_path,
    )
    nodes = pd.DataFrame(rows)
    nodes.to_csv(output_dir / "composite_atlas_nodes.tsv", sep="\t", index=False)
    return atlas_path, nodes


def spm_for_subject(subject_dir: Path):
    return loadmat(subject_dir / "SPM.mat", squeeze_me=True, struct_as_record=False)[
        "SPM"
    ]


def basename_from_spm_path(value) -> str:
    text = str(value).strip()
    text = text.split(",", 1)[0].strip()
    return text.replace("\\", "/").rsplit("/", 1)[-1]


def find_timeseries(timeseries_root: Path, subject_id: int, basename: str) -> Path:
    direct = [
        timeseries_root / str(subject_id) / basename,
        timeseries_root / f"sub-{subject_id:03d}" / basename,
        timeseries_root / basename,
    ]
    found = [p for p in direct if p.is_file()]
    if not found:
        found = list(timeseries_root.glob(f"**/{subject_id}/{basename}"))
    if not found:
        found = list(timeseries_root.glob(f"**/sub-{subject_id:03d}/**/{basename}"))
    unique = sorted(set(found))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Subject {subject_id}, {basename}: expected one 4D file under "
            f"{timeseries_root}, found {len(unique)}."
        )
    return unique[0]


def spm_session_details(spm, session_index: int):
    session = np.atleast_1d(spm.Sess)[session_index]
    row_indices = np.asarray(session.row, dtype=int) - 1
    first_spm_row = row_indices[0]
    basename = basename_from_spm_path(np.atleast_1d(spm.xY.P)[first_spm_row])
    motion = np.asarray(session.C.C, dtype=float)
    blocks = {}
    for condition in np.atleast_1d(session.U):
        name = str(np.atleast_1d(condition.name)[0])
        onset = round(float(np.atleast_1d(condition.ons)[0]))
        duration = round(float(np.atleast_1d(condition.dur)[0]))
        blocks[name] = (onset, duration)
    return basename, motion, blocks


def shifted_block_bounds(onset: int, duration: int, hrf_lag_trs: int):
    """Shift a block while preserving its modeled duration."""

    start = onset + hrf_lag_trs
    return start, start + duration


def extract_subject_connectivity(
    subject_dir: Path,
    timeseries_root: Path,
    atlas_path: Path,
    n_nodes: int,
    hrf_lag_trs: int,
):
    spm = spm_for_subject(subject_dir)
    per_condition = {condition: [] for condition in CONDITIONS}
    audit = []
    for session_index in range(4):
        basename, motion, blocks = spm_session_details(spm, session_index)
        image_path = find_timeseries(timeseries_root, int(subject_dir.name), basename)
        image = nib.load(str(image_path))
        if len(image.shape) != 4 or image.shape[3] != len(motion):
            raise RuntimeError(
                f"{image_path}: expected 4D with {len(motion)} volumes, got {image.shape}."
            )
        masker = NiftiLabelsMasker(
            labels_img=str(atlas_path),
            standardize=False,
            strategy="mean",
            keep_masked_labels=True,
        )
        roi = masker.fit_transform(image)
        if roi.shape[1] != n_nodes:
            raise RuntimeError(
                f"{image_path}: extracted {roi.shape[1]} nodes, expected {n_nodes}."
            )
        cleaned = signal.clean(
            roi,
            confounds=motion,
            detrend=True,
            standardize="zscore_sample",
            high_pass=1 / 128,
            t_r=2.0,
            ensure_finite=True,
        )
        for condition, (onset, duration) in blocks.items():
            if condition not in per_condition:
                continue
            start, stop = shifted_block_bounds(onset, duration, hrf_lag_trs)
            if start < 0 or stop > len(cleaned):
                raise RuntimeError(
                    f"Subject {subject_dir.name} session {session_index + 1} "
                    f"{condition}: shifted block [{start}:{stop}] falls outside "
                    f"the {len(cleaned)}-volume run."
                )
            block = cleaned[start:stop]
            if len(block) < 20:
                raise RuntimeError(
                    f"Subject {subject_dir.name} session {session_index + 1} "
                    f"{condition}: only {len(block)} retained volumes."
                )
            correlation = np.corrcoef(block, rowvar=False)
            correlation = np.nan_to_num(correlation)
            np.fill_diagonal(correlation, 0)
            z = np.arctanh(np.clip(correlation, -0.999999, 0.999999))
            per_condition[condition].append(z)
            audit.append(
                {
                    "subject_id": int(subject_dir.name),
                    "session": session_index + 1,
                    "condition": condition,
                    "timeseries_file": str(image_path),
                    "block_onset_tr": onset,
                    "block_duration_tr": duration,
                    "hrf_lag_trs": hrf_lag_trs,
                    "block_start_tr": start,
                    "block_stop_tr_exclusive": stop,
                    "retained_volumes": len(block),
                }
            )
    averaged = {
        condition: np.mean(per_condition[condition], axis=0) for condition in CONDITIONS
    }
    return averaged, audit


def extract_all(args, output_dir: Path):
    if not args.timeseries_root:
        raise RuntimeError(
            "--timeseries-root is required. Functional connectivity cannot be "
            "estimated from beta or contrast images."
        )
    timeseries_root = Path(args.timeseries_root)
    subjects = numeric_subject_dirs(Path(args.first_level_dir))
    first_spm = spm_for_subject(subjects[0])
    basename, _, _ = spm_session_details(first_spm, 0)
    first_4d = nib.load(
        str(find_timeseries(timeseries_root, int(subjects[0].name), basename))
    )
    atlas_path, nodes = build_composite_atlas(
        first_4d.slicer[..., 0], output_dir, args.min_roi_voxels
    )
    matrices = np.empty(
        (len(subjects), len(CONDITIONS), len(nodes), len(nodes)), dtype=np.float32
    )
    audits = []
    for subject_index, subject_dir in enumerate(subjects):
        averaged, audit = extract_subject_connectivity(
            subject_dir,
            timeseries_root,
            atlas_path,
            len(nodes),
            args.hrf_lag_trs,
        )
        for condition_index, condition in enumerate(CONDITIONS):
            matrices[subject_index, condition_index] = averaged[condition]
        audits.extend(audit)
        print(
            f"Connectivity extracted: {subject_index + 1}/{len(subjects)} "
            f"(subject {subject_dir.name})",
            flush=True,
        )
    np.savez_compressed(
        output_dir / "subject_condition_connectivity.npz",
        matrices=matrices,
        subject_ids=np.array([int(s.name) for s in subjects]),
        conditions=np.array(CONDITIONS),
        node_ids=nodes.node_id.to_numpy(),
        node_names=nodes.node_name.to_numpy(),
        networks=nodes.network.to_numpy(),
    )
    pd.DataFrame(audits).to_csv(output_dir / "timeseries_block_audit.csv", index=False)
    return matrices, nodes, np.array([int(s.name) for s in subjects])


def load_connectivity(output_dir: Path):
    path = output_dir / "subject_condition_connectivity.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run --analysis connectivity first."
        )
    archive = np.load(path, allow_pickle=True)
    nodes = pd.read_csv(output_dir / "composite_atlas_nodes.tsv", sep="\t")
    return archive["matrices"], nodes, archive["subject_ids"]


def repeated_measure_f(values: np.ndarray):
    """Vectorised one-way repeated-measures F for subjects x 3 x features."""
    n, k, _ = values.shape
    grand = values.mean(axis=(0, 1))
    condition_mean = values.mean(axis=0)
    subject_mean = values.mean(axis=1)
    ss_condition = n * np.sum((condition_mean - grand) ** 2, axis=0)
    ss_subject = k * np.sum((subject_mean - grand) ** 2, axis=0)
    ss_total = np.sum((values - grand) ** 2, axis=(0, 1))
    ss_error = ss_total - ss_condition - ss_subject
    f = (ss_condition / (k - 1)) / (ss_error / ((n - 1) * (k - 1)))
    p = stats.f.sf(f, k - 1, (n - 1) * (k - 1))
    eta = ss_condition / (ss_condition + ss_error)
    return f, p, eta


def connectivity_statistics(
    matrices: np.ndarray,
    nodes: pd.DataFrame,
    subject_ids: np.ndarray,
    output_dir: Path,
    n_permutations: int,
):
    n_nodes = matrices.shape[-1]
    upper = np.triu_indices(n_nodes, 1)
    edges = matrices[:, :, upper[0], upper[1]]
    f, p, eta = repeated_measure_f(edges)
    q = bh_fdr(p)
    edge_rows = pd.DataFrame(
        {
            "node1_id": nodes.node_id.to_numpy()[upper[0]],
            "node1_name": nodes.node_name.to_numpy()[upper[0]],
            "node2_id": nodes.node_id.to_numpy()[upper[1]],
            "node2_name": nodes.node_name.to_numpy()[upper[1]],
            "rm_f": f,
            "p_omnibus": p,
            "q_omnibus": q,
            "partial_eta_squared": eta,
        }
    )
    contrasts = [
        (1, 0, "complex_minus_simple"),
        (0, 2, "simple_minus_control"),
        (1, 2, "complex_minus_control"),
    ]
    for a, b, name in contrasts:
        diff = edges[:, a] - edges[:, b]
        t, pair_p = stats.ttest_1samp(diff, 0, axis=0)
        edge_rows[f"{name}_mean"] = diff.mean(axis=0)
        edge_rows[f"{name}_t"] = t
        edge_rows[f"{name}_p"] = pair_p
        edge_rows[f"{name}_q"] = bh_fdr(pair_p)
    edge_rows.to_csv(output_dir / "edgewise_statistics.csv.gz", index=False)

    global_values = edges.mean(axis=2)
    global_rows = []
    for condition_index, condition in enumerate(CONDITIONS):
        for subject_index, subject in enumerate(subject_ids):
            global_rows.append(
                {
                    "subject_id": subject,
                    "condition": condition,
                    "mean_fisher_z": global_values[subject_index, condition_index],
                }
            )
    global_table = pd.DataFrame(global_rows)
    global_table.to_csv(output_dir / "global_connectivity_values.csv", index=False)
    global_stats = []
    friedman = stats.friedmanchisquare(
        global_values[:, 0], global_values[:, 1], global_values[:, 2]
    )
    global_stats.append(
        {
            "test": "friedman_omnibus",
            "contrast": "three_conditions",
            "statistic": friedman.statistic,
            "p": friedman.pvalue,
        }
    )
    for contrast_index, (a, b, name) in enumerate(contrasts):
        diff = global_values[:, a] - global_values[:, b]
        global_stats.append(
            {
                "test": "paired_sign_flip",
                "contrast": name,
                "mean_difference": diff.mean(),
                "cohens_dz": cohens_dz(diff),
                "p": paired_sign_flip_pvalue(
                    diff, n_permutations, 20260731 + contrast_index
                ),
            }
        )
    stats_table = pd.DataFrame(global_stats)
    mask = stats_table.test == "paired_sign_flip"
    stats_table.loc[mask, "q_fdr"] = bh_fdr(stats_table.loc[mask, "p"])
    stats_table.to_csv(output_dir / "global_connectivity_statistics.csv", index=False)
    save_connectivity_figures(matrices, global_table, output_dir)
    nbs(matrices[:, 1] - matrices[:, 0], nodes, output_dir, min(n_permutations, 5000))


def save_connectivity_figures(matrices, global_table, output_dir):
    means = matrices.mean(axis=0)
    vmax = np.quantile(np.abs(means), 0.99)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, matrix, condition in zip(axes, means, CONDITIONS):
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(condition)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Mean Fisher z")
    fig.savefig(
        output_dir / "group_connectivity_matrices.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig(output_dir / "group_connectivity_matrices.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, condition in enumerate(CONDITIONS):
        values = global_table.loc[global_table.condition == condition, "mean_fisher_z"]
        ax.scatter(np.full(len(values), i), values, alpha=0.45, s=15)
        ax.hlines(values.mean(), i - 0.25, i + 0.25, color="black", linewidth=2)
    ax.set_xticks(range(3), CONDITIONS)
    ax.set_ylabel("Whole-network mean Fisher z")
    fig.tight_layout()
    fig.savefig(output_dir / "global_connectivity.png", dpi=300)
    fig.savefig(output_dir / "global_connectivity.pdf")
    plt.close(fig)


def component_masses(t_matrix: np.ndarray, threshold: float):
    return signed_components_from_matrix(t_matrix, threshold)


def nbs(diff_matrices, nodes, output_dir, n_permutations, threshold=3.1):
    mean = diff_matrices.mean(axis=0)
    sd = diff_matrices.std(axis=0, ddof=1)
    t = np.divide(
        mean, sd / np.sqrt(len(diff_matrices)), out=np.zeros_like(mean), where=sd > 0
    )
    components = component_masses(t, threshold)
    upper = np.triu_indices(len(t), 1)
    rng = np.random.default_rng(20260731)
    null = np.zeros(n_permutations)
    for permutation in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=(len(diff_matrices), 1, 1))
        perm = diff_matrices * signs
        pmean, psd = perm.mean(axis=0), perm.std(axis=0, ddof=1)
        pt = np.divide(
            pmean,
            psd / np.sqrt(len(perm)),
            out=np.zeros_like(pmean),
            where=psd > 0,
        )
        permuted_components = component_masses(pt, threshold)
        null[permutation] = max(
            (component["mass"] for component in permuted_components), default=0
        )
    rows = []
    for component_id, component in enumerate(components, 1):
        mass = component["mass"]
        p_fwer = (np.sum(null >= mass) + 1) / (n_permutations + 1)
        for edge_index in component["edge_indices"]:
            i = int(upper[0][edge_index])
            j = int(upper[1][edge_index])
            rows.append(
                {
                    "component_id": component_id,
                    "direction": component["direction"],
                    "component_mass": mass,
                    "component_p_fwer": p_fwer,
                    "node1": nodes.iloc[i].node_name,
                    "node2": nodes.iloc[j].node_name,
                    "mean_difference": mean[i, j],
                    "t": t[i, j],
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "nbs_complex_minus_simple.csv", index=False)


def threshold_positive_matrix(matrix: np.ndarray, density: float):
    matrix = np.maximum(matrix, 0)
    n = len(matrix)
    upper = np.triu_indices(n, 1)
    values = matrix[upper]
    n_keep = max(1, round(density * len(values)))
    order = np.argsort(values)[::-1][:n_keep]
    thresholded = np.zeros_like(matrix)
    thresholded[upper[0][order], upper[1][order]] = values[order]
    return thresholded + thresholded.T


def graph_metrics(matrix: np.ndarray):
    graph = nx.from_numpy_array(matrix)
    graph.remove_edges_from(
        [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] <= 0]
    )
    for u, v, d in graph.edges(data=True):
        d["distance"] = 1.0 / max(d["weight"], 1e-8)
    strength = np.mean([value for _, value in graph.degree(weight="weight")])
    clustering = nx.average_clustering(graph, weight="weight")
    n_nodes = len(graph)
    distance_matrix = np.zeros_like(matrix, dtype=np.float64)
    positive = matrix > 0
    distance_matrix[positive] = 1.0 / np.maximum(matrix[positive], 1e-8)
    sparse_distances = csr_matrix(distance_matrix)
    paths = shortest_path(
        sparse_distances, directed=False, unweighted=False, method="D"
    )
    valid_paths = np.isfinite(paths) & (paths > 0)
    inverse_paths = np.zeros_like(paths)
    np.reciprocal(paths, where=valid_paths, out=inverse_paths)
    efficiency = (
        float(inverse_paths.sum() / (n_nodes * (n_nodes - 1)))
        if n_nodes > 1
        else 0.0
    )
    _, component_labels = connected_components(
        sparse_distances, directed=False, return_labels=True
    )
    largest_label = int(np.argmax(np.bincount(component_labels)))
    largest_nodes = np.flatnonzero(component_labels == largest_label)
    if len(largest_nodes) < 2:
        path_length = 0.0
    else:
        largest_paths = paths[np.ix_(largest_nodes, largest_nodes)]
        path_length = float(
            largest_paths[np.triu_indices(len(largest_nodes), 1)].mean()
        )
    communities = nx.community.louvain_communities(
        graph, weight="weight", seed=20260731
    )
    modularity = nx.community.modularity(graph, communities, weight="weight")
    return {
        "strength": strength,
        "clustering": clustering,
        "global_efficiency": efficiency,
        "path_length": path_length,
        "modularity": modularity,
    }


def topology_analysis(
    matrices, subject_ids, output_dir, n_permutations, equivalence_bound=0.2
):
    densities = np.arange(0.10, 0.301, 0.02)
    rows = []
    for si, subject in enumerate(subject_ids):
        for ci, condition in enumerate(CONDITIONS):
            for density in densities:
                metrics = graph_metrics(
                    threshold_positive_matrix(matrices[si, ci], density)
                )
                rows.append(
                    {
                        "subject_id": subject,
                        "condition": condition,
                        "density": density,
                        **metrics,
                    }
                )
        print(f"Topology complete: {si + 1}/{len(subject_ids)}", flush=True)
    values = pd.DataFrame(rows)
    values.to_csv(output_dir / "topology_by_density.csv.gz", index=False)
    metric_names = (
        "strength",
        "clustering",
        "global_efficiency",
        "path_length",
        "modularity",
    )
    auc_rows = []
    for (subject, condition), part in values.groupby(["subject_id", "condition"]):
        row = {"subject_id": subject, "condition": condition}
        for metric in metric_names:
            row[metric + "_auc"] = np.trapz(part[metric], part.density)
        auc_rows.append(row)
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(output_dir / "topology_auc.csv", index=False)
    stats_rows = []
    for metric in [m + "_auc" for m in metric_names]:
        pivot = auc.pivot(index="subject_id", columns="condition", values=metric)
        friedman = stats.friedmanchisquare(pivot.Simple, pivot.Complex, pivot.Control)
        stats_rows.append(
            {
                "metric": metric,
                "test": "friedman",
                "contrast": "three_conditions",
                "statistic": friedman.statistic,
                "p": friedman.pvalue,
            }
        )
        for index, (a, b, name) in enumerate(
            [
                ("Complex", "Simple", "complex_minus_simple"),
                ("Simple", "Control", "simple_minus_control"),
                ("Complex", "Control", "complex_minus_control"),
            ]
        ):
            diff = pivot[a].to_numpy() - pivot[b].to_numpy()
            tost = paired_tost_dz(diff, equivalence_bound)
            stats_rows.append(
                {
                    "metric": metric,
                    "test": "paired_sign_flip",
                    "contrast": name,
                    "mean_difference": diff.mean(),
                    "cohens_dz": cohens_dz(diff),
                    "p": paired_sign_flip_pvalue(
                        diff, n_permutations, 20260731 + index
                    ),
                    "tost_p_lower": tost["p_lower"],
                    "tost_p_upper": tost["p_upper"],
                    "equivalent_dz_0_2": tost["equivalent"],
                }
            )
    stats_table = pd.DataFrame(stats_rows)
    pair = stats_table.test == "paired_sign_flip"
    stats_table.loc[pair, "q_fdr"] = bh_fdr(stats_table.loc[pair, "p"])
    omni = stats_table.test == "friedman"
    stats_table.loc[omni, "q_fdr"] = bh_fdr(stats_table.loc[omni, "p"])
    stats_table.to_csv(output_dir / "topology_statistics.csv", index=False)

    group = matrices.mean(axis=0)
    correlations = np.corrcoef(np.stack([g[np.triu_indices(len(g), 1)] for g in group]))
    pd.DataFrame(correlations, index=CONDITIONS, columns=CONDITIONS).to_csv(
        output_dir / "group_matrix_correlations.csv"
    )
    save_topology_figure(auc, metric_names, output_dir)


def save_topology_figure(auc, metric_names, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, metric in zip(axes.ravel(), metric_names):
        column = metric + "_auc"
        for i, condition in enumerate(CONDITIONS):
            values = auc.loc[auc.condition == condition, column]
            ax.scatter(np.full(len(values), i), values, alpha=0.35, s=12)
            ax.hlines(values.mean(), i - 0.25, i + 0.25, color="black", linewidth=2)
        ax.set_xticks(range(3), CONDITIONS)
        ax.set_title(metric.replace("_", " ").title())
    axes.ravel()[-1].axis("off")
    fig.suptitle("Weighted topology AUC across proportional densities 10–30%")
    fig.tight_layout()
    fig.savefig(output_dir / "topology_auc.png", dpi=300)
    fig.savefig(output_dir / "topology_auc.pdf")
    plt.close(fig)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.analysis in {"connectivity", "all"}:
        matrices, nodes, subjects = extract_all(args, output_dir)
        connectivity_statistics(
            matrices, nodes, subjects, output_dir, args.n_permutations
        )
    else:
        matrices, nodes, subjects = load_connectivity(output_dir)
    if args.analysis in {"topology", "all"}:
        topology_analysis(
            matrices,
            subjects,
            output_dir,
            args.n_permutations,
            args.equivalence_bound_dz,
        )


def parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--analysis", choices=["connectivity", "topology", "all"], default="all"
    )
    p.add_argument("--timeseries-root")
    p.add_argument("--first-level-dir", default=str(FIRST_LEVEL_DIR))
    p.add_argument("--output-dir", default=str(RESULTS_DIR / "connectivity/timeseries"))
    p.add_argument("--hrf-lag-trs", type=int, default=0)
    p.add_argument("--min-roi-voxels", type=int, default=20)
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--equivalence-bound-dz", type=float, default=0.2)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
