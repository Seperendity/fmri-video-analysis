"""Build frozen non-MVPA manuscript figures and result tables.

All numerical panels are generated from the formal run directory.  The module
does not read the MVPA output tree. Corrected RSA inputs are supplied separately
so that manuscript assets cannot silently fall back to the biased legacy
within-beta variance normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting

from fmri_video_analysis.core.config import PROJECT_ROOT, RESULTS_DIR

FIG_WIDTH_MM = 183
WIDTH_IN = FIG_WIDTH_MM / 25.4
P_VALUE_EPSILON = np.finfo(float).tiny
COLORS = {
    "positive": "#C44E52",
    "negative": "#3B73B9",
    "signal": "#7A5195",
    "visual": "#D95F59",
    "attention": "#E6A23C",
    "control": "#577590",
    "neutral": "#7D8597",
    "light": "#D9DEE7",
    "dark": "#263238",
    "cortex": "#4C78A8",
    "subcortex": "#F2A65A",
    "cerebellum": "#8F6F9F",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.08,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_bundle(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _load_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "activation": "activation/group_contrasts/complex_minus_simple_clusters_tfce_fwer05.csv",
        "activation_omnibus": "activation/repeated_measures/condition_omnibus_clusters_tfce_fwer05.csv",
        "edges": "group_statistics/edgewise/edgewise_complex_minus_simple.csv.gz",
        "nbs_components": "group_statistics/nbs/nbs_complex_minus_simple_components.csv",
        "nbs_edges": "group_statistics/nbs/nbs_complex_minus_simple_edges.csv.gz",
        "predefined": "group_statistics/predefined_metrics/predefined_metric_statistics.csv",
        "topology": "group_statistics/topology/topology_statistics.csv",
        "icc": "sensitivity/block_reliability/block_reliability_summary.csv",
        "stimulus": "stimulus/characterization/stimulus_pairwise_tests.csv",
        "stimulus_omnibus": "stimulus/characterization/stimulus_omnibus_tests.csv",
        "block_fixed": "sensitivity/block_stimulus/fixed_video_condition_tests.csv",
        "block_video": "sensitivity/block_stimulus/video_pair_exact_tests.csv",
        "block_mixed": "sensitivity/block_stimulus/mixed_effects_results.csv",
    }
    tables = {name: pd.read_csv(run_dir / path) for name, path in paths.items()}
    tables["nodes"] = pd.read_csv(run_dir / "atlas/composite_249_nodes.tsv", sep="\t")
    return tables


def _write_source(frame: pd.DataFrame, source_dir: Path, name: str) -> None:
    frame.to_csv(source_dir / f"{name}.csv", index=False)


def _load_corrected_rsa_tables(rsa_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the externally whitened RSA repair outputs or fail closed."""

    paths = {
        "rsa_primary": rsa_dir / "crossnobis_primary/crossnobis_summary.csv",
        "rsa_centered": (
            rsa_dir / "crossnobis_spatial_mean_centered/crossnobis_summary.csv"
        ),
        "rsa_searchlight": rsa_dir / "searchlight_primary/searchlight_summary.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Corrected RSA manuscript inputs are incomplete: " + ", ".join(missing)
        )
    return {name: pd.read_csv(path) for name, path in paths.items()}


def _thresholded_image(path: Path) -> nib.Nifti1Image:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.any(np.isfinite(data) & (data != 0)):
        raise RuntimeError(f"Thresholded statistical image is empty: {path}")
    return image


def figure_activation_rsa(
    run_dir: Path,
    rsa_dir: Path,
    output_dir: Path,
    source_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> None:
    """Figure 1: activation and corrected primary/sensitivity RSA results."""

    activation_path = (
        run_dir
        / "activation/group_contrasts/complex_minus_simple_tfce_fwer05_tstat.nii.gz"
    )
    activation_img = _thresholded_image(activation_path)

    figure = plt.figure(figsize=(WIDTH_IN, 5.35), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.15, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.30,
        hspace=0.38,
        left=0.05,
        right=0.98,
        bottom=0.10,
        top=0.89,
    )
    ax_activation = figure.add_subplot(grid[0, 0])
    ax_searchlight = figure.add_subplot(grid[1, 0])
    ax_primary = figure.add_subplot(grid[0, 1])
    ax_centered = figure.add_subplot(grid[1, 1])

    plotting.plot_glass_brain(
        activation_img,
        axes=ax_activation,
        display_mode="lyrz",
        plot_abs=False,
        cmap="YlOrRd",
        colorbar=True,
        annotate=False,
        threshold=1e-8,
    )
    ax_activation.set_title(
        "Mean response: Complex > Simple\nTFCE–FWER P < 0.05",
        loc="left",
        pad=7,
    )
    panel_label(ax_activation, "a")

    network_order = [
        "visual",
        "dorsal_attention",
        "salience_attention",
        "control",
    ]
    labels = ["Visual", "Dorsal attention", "Salience/attention", "Control"]
    primary = (
        tables["rsa_primary"]
        .query("distance == 'simple_complex' and network in @network_order")
        .set_index("network")
        .loc[network_order]
        .reset_index()
    )
    centered = (
        tables["rsa_centered"]
        .query("distance == 'simple_complex' and network in @network_order")
        .set_index("network")
        .loc[network_order]
        .reset_index()
    )

    def plot_network_distances(
        axis: plt.Axes,
        frame: pd.DataFrame,
        title: str,
        color: str,
        xlim: tuple[float, float],
    ) -> None:
        y = np.arange(len(frame))[::-1]
        axis.axvline(0, color=COLORS["dark"], linewidth=0.8, linestyle="--")
        axis.errorbar(
            frame.mean_crossnobis,
            y,
            xerr=np.vstack(
                [
                    frame.mean_crossnobis - frame.ci_low,
                    frame.ci_high - frame.mean_crossnobis,
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.3,
            markersize=4.5,
        )
        annotation_x = xlim[1] - 0.01 * (xlim[1] - xlim[0])
        for yi, row in zip(y, frame.itertuples()):
            q_text = (
                f"q={row.q_fdr_primary:.4f}"
                if row.q_fdr_primary >= 0.001
                else "q<0.001"
            )
            axis.text(annotation_x, yi, q_text, va="center", ha="right", fontsize=6)
        axis.set_yticks(y, labels)
        axis.set_xlim(*xlim)
        axis.set_xlabel("ResMS-normalized crossvalidated distance")
        axis.set_title(title, loc="left", pad=6)
        axis.grid(axis="x", color=COLORS["light"], linewidth=0.6)

    plot_network_distances(
        ax_primary,
        primary,
        "Primary Simple–Complex distance\nmean and participant-bootstrap 95% CI",
        COLORS["signal"],
        (-0.08, 0.31),
    )
    panel_label(ax_primary, "b")
    plot_network_distances(
        ax_centered,
        centered,
        "Spatial-mean-centred sensitivity\nmean and participant-bootstrap 95% CI",
        COLORS["attention"],
        (-0.025, 0.105),
    )
    panel_label(ax_centered, "c")

    contrast_order = ["simple_complex", "simple_control", "complex_control"]
    contrast_labels = ["Simple–\nComplex", "Simple–\nControl", "Complex–\nControl"]
    searchlight_rows = []
    for contrast in contrast_order:
        path = (
            rsa_dir
            / "searchlight_primary"
            / f"{contrast}_tfce_fwer_bonferroni3_tstat.nii.gz"
        )
        data = np.asarray(nib.load(str(path)).dataobj)
        summary_row = tables["rsa_searchlight"].query("distance == @contrast").iloc[0]
        searchlight_rows.append(
            {
                "distance": contrast,
                "n_significant_voxels": int(
                    np.count_nonzero(np.isfinite(data) & (data != 0))
                ),
                "mapwise_significant_clusters": int(
                    summary_row.mapwise_significant_clusters
                ),
                "threshold": "TFCE whole-brain FWER plus Bonferroni 0.05/3",
            }
        )
    searchlight = pd.DataFrame(searchlight_rows)
    bars = ax_searchlight.bar(
        np.arange(3),
        searchlight.n_significant_voxels,
        color=[COLORS["neutral"], COLORS["control"], COLORS["signal"]],
        width=0.68,
    )
    for bar, value in zip(bars, searchlight.n_significant_voxels):
        ax_searchlight.text(
            bar.get_x() + bar.get_width() / 2,
            max(value, 0) + 250,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax_searchlight.set_xticks(np.arange(3), contrast_labels)
    ax_searchlight.set_ylabel("Significant searchlight voxels")
    ax_searchlight.set_ylim(
        0, max(12500, int(searchlight.n_significant_voxels.max() * 1.18))
    )
    ax_searchlight.set_title(
        "Searchlight after TFCE–FWER + Bonferroni/3", loc="left", pad=6
    )
    ax_searchlight.grid(axis="y", color=COLORS["light"], linewidth=0.6)
    panel_label(ax_searchlight, "d")

    figure.suptitle(
        "Complexity-related mean responses were not accompanied by primary multivoxel-distance evidence",
        x=0.03,
        y=0.985,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_bundle(figure, output_dir / "Figure_nonMVPA_1_activation_rsa")
    _write_source(tables["activation"], source_dir, "Figure_1a_activation_clusters")
    _write_source(primary, source_dir, "Figure_1b_primary_crossnobis")
    _write_source(centered, source_dir, "Figure_1c_mean_centered_crossnobis")
    _write_source(
        searchlight,
        source_dir,
        "Figure_1d_searchlight_corrected_extent",
    )


def figure_activation_omnibus(
    run_dir: Path,
    output_dir: Path,
    source_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> None:
    """Extended Data Figure 3: three-condition repeated-measures omnibus F."""

    omnibus_path = (
        run_dir
        / "activation/repeated_measures/condition_omnibus_tfce_fwer05_fstat.nii.gz"
    )
    omnibus_img = _thresholded_image(omnibus_path)
    clusters = (
        tables["activation_omnibus"]
        .sort_values("n_voxels", ascending=False)
        .reset_index(drop=True)
    )

    figure = plt.figure(figsize=(WIDTH_IN, 3.55))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.75, 1.0),
        wspace=0.30,
        left=0.04,
        right=0.98,
        bottom=0.16,
        top=0.84,
    )
    ax_map = figure.add_subplot(grid[0, 0])
    plotting.plot_stat_map(
        omnibus_img,
        figure=figure,
        axes=ax_map,
        display_mode="z",
        cut_coords=[-12, 0, 12, 24, 36, 48, 60],
        cmap="inferno",
        threshold=1e-8,
        colorbar=True,
        annotate=False,
        draw_cross=False,
        black_bg=False,
    )
    ax_map.set_title(
        "Simple / Complex / Control omnibus F\nTFCE–FWER P<0.05",
        loc="left",
        pad=7,
    )
    panel_label(ax_map, "a")

    ax_clusters = figure.add_subplot(grid[0, 1])
    y = np.arange(len(clusters))[::-1]
    peak_values = clusters.peak_F.to_numpy(dtype=float)
    normalization = mpl.colors.Normalize(
        vmin=float(peak_values.min()), vmax=float(peak_values.max())
    )
    colors = plt.get_cmap("inferno")(normalization(peak_values))
    ax_clusters.barh(y, clusters.n_voxels, color=colors, height=0.68)
    ax_clusters.set_yticks(
        y, [f"Cluster {int(value)}" for value in clusters.cluster_id]
    )
    ax_clusters.set_xlabel("TFCE–FWER significant voxels")
    ax_clusters.set_title("Cluster extent and peak F", loc="left", pad=7)
    ax_clusters.grid(axis="x", color=COLORS["light"], linewidth=0.6)
    ax_clusters.set_xlim(0, float(clusters.n_voxels.max()) * 1.23)
    for yi, row in zip(y, clusters.itertuples()):
        ax_clusters.text(
            row.n_voxels + float(clusters.n_voxels.max()) * 0.025,
            yi,
            f"F={row.peak_F:.1f}",
            va="center",
            fontsize=6,
        )
    panel_label(ax_clusters, "b")

    figure.suptitle(
        "The three experimental conditions produced a distributed activation effect",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_bundle(figure, output_dir / "ExtendedData_nonMVPA_3_activation_omnibus")
    _write_source(
        clusters,
        source_dir,
        "ExtendedData_3_activation_omnibus_clusters",
    )


def _main_nbs_component(tables: dict[str, pd.DataFrame]) -> pd.Series:
    rows = tables["nbs_components"].query(
        "threshold_abs_t == 3.1 and direction == 'negative' and significant_fwer_0_05"
    )
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one main negative NBS component, found {len(rows)}"
        )
    return rows.iloc[0]


def _plot_significant_edge_effects(axis: plt.Axes, edges: pd.DataFrame) -> None:
    significant = edges.loc[edges.q_fdr < 0.05].copy()
    significant = significant.sort_values("mean_difference")
    y = np.arange(len(significant))
    colors = np.where(
        significant.mean_difference.gt(0), COLORS["positive"], COLORS["negative"]
    )
    axis.axvline(0, color=COLORS["dark"], linewidth=0.8, linestyle="--")
    axis.barh(y, significant.mean_difference, color=colors, height=0.68)
    axis.set_yticks(y, [f"E{index}" for index in range(1, len(significant) + 1)])
    axis.set_xlabel("Mean Fisher-z difference")
    axis.grid(axis="x", color=COLORS["light"], linewidth=0.6)


def figure_fc_reconfiguration(
    run_dir: Path, output_dir: Path, source_dir: Path, tables: dict[str, pd.DataFrame]
) -> None:
    """Figure 2: global, edgewise, and connected-component FC evidence."""

    predefined = tables["predefined"]
    selected_specs = [
        ("global", "whole_brain_signed_mean", "Whole brain"),
        ("network_within", "within_SalVentAttnB", "SalVentAttnB"),
        ("network_within", "within_DefaultA", "DefaultA"),
        ("network_separation", "separation_DefaultA", "DefaultA"),
    ]
    selected_rows = []
    for family, metric, label in selected_specs:
        row = predefined.query(
            "family == @family and metric == @metric and label == @label and contrast == 'complex_minus_simple'"
        )
        if len(row) != 1:
            raise RuntimeError(
                f"Cannot resolve predefined result: {family}/{metric}/{label}"
            )
        selected_rows.append(row.iloc[0])
    forest = pd.DataFrame(selected_rows)
    forest["display"] = [
        "Whole-brain mean",
        "Within SalVentAttnB",
        "Within DefaultA",
        "DefaultA separation",
    ]

    edges = tables["edges"].copy()
    significant = edges.loc[edges.q_fdr < 0.05].copy()
    nodes = tables["nodes"].copy()
    component = _main_nbs_component(tables)
    component_id = int(component.component_id)
    all_nbs_edges = tables["nbs_edges"]
    nbs_edges = all_nbs_edges.loc[
        all_nbs_edges.threshold_abs_t.eq(3.1)
        & all_nbs_edges.direction.eq("negative")
        & all_nbs_edges.component_id.eq(component_id)
    ].copy()

    figure = plt.figure(figsize=(WIDTH_IN, 6.4), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=(0.92, 1.35),
        width_ratios=(1.08, 1.02, 0.90),
        hspace=0.31,
        wspace=0.35,
    )
    ax_forest = figure.add_subplot(grid[0, 0])
    ax_volcano = figure.add_subplot(grid[0, 1])
    ax_connectome = figure.add_subplot(grid[0, 2])
    ax_matrix = figure.add_subplot(grid[1, :2])
    ax_domains = figure.add_subplot(grid[1, 2])

    y = np.arange(len(forest))[::-1]
    colors = [COLORS["neutral"]] + [COLORS["negative"]] * 3
    ax_forest.axvline(0, color=COLORS["dark"], linewidth=0.8, linestyle="--")
    for yi, (_, row), color in zip(y, forest.iterrows(), colors):
        ax_forest.errorbar(
            row.mean_difference,
            yi,
            xerr=[
                [row.mean_difference - row.mean_ci95_low],
                [row.mean_ci95_high - row.mean_difference],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2,
            markersize=4.5,
        )
    ax_forest.set_yticks(y, forest.display)
    ax_forest.set_xlabel("Mean Fisher-z difference")
    ax_forest.set_title("Prespecified summaries", loc="left")
    ax_forest.grid(axis="x", color=COLORS["light"], linewidth=0.6)
    panel_label(ax_forest, "a")

    y_volcano = -np.log10(np.clip(edges.p.to_numpy(), P_VALUE_EPSILON, 1.0))
    ax_volcano.scatter(
        edges.mean_difference,
        y_volcano,
        s=4,
        color="#B8BEC8",
        alpha=0.38,
        edgecolors="none",
        rasterized=True,
    )
    for direction, color in (
        ("positive", COLORS["positive"]),
        ("negative", COLORS["negative"]),
    ):
        part = significant.loc[significant.direction == direction]
        ax_volcano.scatter(
            part.mean_difference,
            -np.log10(np.clip(part.p.to_numpy(), P_VALUE_EPSILON, 1.0)),
            s=20,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            label=f"{direction.capitalize()} ({len(part)})",
            zorder=3,
        )
    ax_volcano.axvline(0, color=COLORS["dark"], linewidth=0.6)
    ax_volcano.set_xlabel("Mean Fisher-z difference")
    ax_volcano.set_ylabel("−log10(P)")
    ax_volcano.set_title("All 30,876 edges", loc="left")
    ax_volcano.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    panel_label(ax_volcano, "b")

    _plot_significant_edge_effects(ax_connectome, edges)
    ax_connectome.set_title("7 BH–FDR edges: 4 higher, 3 lower", loc="left")
    panel_label(ax_connectome, "c")

    order = nodes.sort_values(
        ["domain", "network", "hemisphere", "node_id"]
    ).node_id.to_numpy()
    position = {node_id: index for index, node_id in enumerate(order)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    for row in nbs_edges.itertuples():
        first, second = position[row.node1_id], position[row.node2_id]
        matrix[first, second] = matrix[second, first] = row.mean_difference
    limit = max(abs(float(matrix.min())), abs(float(matrix.max())), 0.01)
    image = ax_matrix.imshow(
        matrix,
        cmap="Blues_r",
        vmin=-limit,
        vmax=0,
        interpolation="none",
        rasterized=True,
        aspect="auto",
    )
    ordered_nodes = nodes.set_index("node_id").loc[order].reset_index()
    boundaries = []
    cursor = 0
    for _, part in ordered_nodes.groupby("domain", sort=False):
        cursor += len(part)
        boundaries.append(cursor)
    for boundary in boundaries[:-1]:
        ax_matrix.axhline(boundary - 0.5, color="white", linewidth=0.8)
        ax_matrix.axvline(boundary - 0.5, color="white", linewidth=0.8)
    centers = []
    domain_labels = []
    start = 0
    for domain, part in ordered_nodes.groupby("domain", sort=False):
        centers.append(start + (len(part) - 1) / 2)
        domain_labels.append(domain)
        start += len(part)
    ax_matrix.set_xticks(
        centers,
        domain_labels,
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )
    ax_matrix.set_yticks(centers, domain_labels)
    ax_matrix.set_title(
        f"Negative NBS component: {int(component.n_nodes)} nodes, {int(component.n_edges)} edges",
        loc="left",
    )
    cbar = figure.colorbar(image, ax=ax_matrix, fraction=0.025, pad=0.02)
    cbar.set_label("Mean Fisher-z difference")
    panel_label(ax_matrix, "d")

    involved = set(nbs_edges.node1_id) | set(nbs_edges.node2_id)
    node_counts = (
        nodes.assign(in_component=nodes.node_id.isin(involved))
        .groupby("domain", sort=False)
        .agg(component_nodes=("in_component", "sum"), atlas_nodes=("node_id", "size"))
        .reset_index()
    )
    domain_order = ["Cortex", "Subcortex", "Cerebellum"]
    node_counts = node_counts.set_index("domain").loc[domain_order].reset_index()
    ypos = np.arange(3)[::-1]
    domain_colors = [COLORS[domain.lower()] for domain in domain_order]
    ax_domains.barh(ypos, node_counts.atlas_nodes, color=COLORS["light"], height=0.62)
    ax_domains.barh(ypos, node_counts.component_nodes, color=domain_colors, height=0.62)
    for yi, row in zip(ypos, node_counts.itertuples()):
        ax_domains.text(
            row.component_nodes + 2,
            yi,
            f"{int(row.component_nodes)}/{int(row.atlas_nodes)}",
            va="center",
            fontsize=6,
        )
    ax_domains.set_yticks(ypos, domain_order)
    ax_domains.set_xlabel("Nodes")
    ax_domains.set_xlim(0, max(node_counts.atlas_nodes) * 1.18)
    ax_domains.set_title(
        f"Component coverage\nFWER P={component.p_fwer:.4f}", loc="left"
    )
    panel_label(ax_domains, "e")

    figure.suptitle(
        "Complexity is associated with distributed functional-connectivity reconfiguration",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_bundle(figure, output_dir / "Figure_nonMVPA_2_fc_reconfiguration")
    _write_source(
        forest[
            [
                "family",
                "metric",
                "label",
                "display",
                "n_subjects",
                "mean_difference",
                "mean_ci95_low",
                "mean_ci95_high",
                "cohens_dz",
                "p_sign_flip",
                "q_fdr_within_family",
            ]
        ],
        source_dir,
        "Figure_2a_predefined_fc",
    )
    _write_source(edges, source_dir, "Figure_2b_all_edges")
    edge_panel = significant.sort_values("mean_difference").copy()
    edge_panel.insert(
        0, "panel_edge_id", [f"E{i}" for i in range(1, len(edge_panel) + 1)]
    )
    _write_source(edge_panel, source_dir, "Figure_2c_fdr_edges")
    _write_source(nbs_edges, source_dir, "Figure_2d_nbs_edges")
    _write_source(node_counts, source_dir, "Figure_2e_nbs_domain_nodes")


def _sensitivity_nbs_table(run_dir: Path) -> pd.DataFrame:
    variants = [
        ("motion_6p", "6-parameter motion"),
        ("no_fir", "No FIR regression"),
        ("no_fd_censor", "No FD censoring"),
        ("fd_0p3", "FD threshold 0.3 mm"),
        ("lag_0", "No HRF shift"),
        ("gsr", "Global signal regression"),
        ("schaefer200", "Schaefer-200 only"),
        ("exclude_high_motion", "Motion-sensitivity subset"),
    ]
    rows = []
    for variant, label in variants:
        path = (
            run_dir
            / "sensitivity"
            / variant
            / "nbs/nbs_complex_minus_simple_components.csv"
        )
        table = pd.read_csv(path)
        part = table.query(
            "threshold_abs_t == 3.1 and direction == 'negative' and significant_fwer_0_05"
        )
        if len(part) != 1:
            raise RuntimeError(f"Expected one negative NBS component for {variant}")
        row = part.iloc[0].to_dict()
        archive = np.load(
            run_dir / "sensitivity" / variant / "subject_condition_connectivity.npz",
            allow_pickle=False,
        )
        row.update(
            {
                "variant": variant,
                "display": label,
                "n_subjects": len(archive["subject_ids"]),
                "n_atlas_nodes": archive["matrices"].shape[-1],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def figure_robustness(
    run_dir: Path, output_dir: Path, source_dir: Path, tables: dict[str, pd.DataFrame]
) -> None:
    """Extended Data Figure 1: NBS robustness, topology and block reliability."""

    sensitivity = _sensitivity_nbs_table(run_dir)
    topology = (
        tables["topology"]
        .query("metric_role == 'primary' and contrast == 'complex_minus_simple'")
        .copy()
    )
    topology_order = [
        "strength_auc",
        "global_efficiency_auc",
        "modularity_auc",
        "network_separation_auc",
    ]
    topology = topology.set_index("metric").loc[topology_order].reset_index()
    topology["display"] = [
        "Strength",
        "Global efficiency",
        "Modularity",
        "Network separation",
    ]
    icc = tables["icc"].copy()

    figure = plt.figure(figsize=(WIDTH_IN, 5.35))
    grid = figure.add_gridspec(2, 2, hspace=0.36, wspace=0.42)
    ax_sensitivity = figure.add_subplot(grid[:, 0])
    ax_topology = figure.add_subplot(grid[0, 1])
    ax_icc = figure.add_subplot(grid[1, 1])

    y = np.arange(len(sensitivity))[::-1]
    ax_sensitivity.axvline(0.05, color=COLORS["dark"], linestyle="--", linewidth=0.8)
    sizes = 18 + 65 * (sensitivity.n_edges / sensitivity.n_edges.max())
    ax_sensitivity.scatter(
        sensitivity.p_fwer,
        y,
        s=sizes,
        color=COLORS["negative"],
        alpha=0.88,
        edgecolors="white",
        linewidths=0.5,
    )
    for yi, row in zip(y, sensitivity.itertuples()):
        ax_sensitivity.text(
            row.p_fwer + 0.0015,
            yi,
            f"{int(row.n_nodes)} nodes / {int(row.n_edges)} edges",
            va="center",
            fontsize=5.5,
        )
    ax_sensitivity.set_yticks(y, sensitivity.display)
    ax_sensitivity.set_xlabel("NBS component FWER P")
    ax_sensitivity.set_xlim(0, 0.065)
    ax_sensitivity.set_title(
        "Negative NBS component across processing variants", loc="left"
    )
    panel_label(ax_sensitivity, "a")

    y_top = np.arange(len(topology))[::-1]
    ax_topology.axvspan(-0.2, 0.2, color="#EEF1F5", zorder=0)
    ax_topology.axvline(0, color=COLORS["dark"], linewidth=0.7)
    for yi, row in zip(y_top, topology.itertuples()):
        color = (
            COLORS["signal"] if bool(row.tost_equivalent_dz_0_20) else COLORS["neutral"]
        )
        ax_topology.errorbar(
            row.cohens_dz,
            yi,
            xerr=[
                [row.cohens_dz - row.dz_ci95_low],
                [row.dz_ci95_high - row.cohens_dz],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2,
            markersize=4.5,
        )
    ax_topology.set_yticks(y_top, topology.display)
    ax_topology.set_xlabel("Cohen dz (95% bootstrap CI)")
    ax_topology.set_xlim(-0.55, 0.40)
    ax_topology.set_title("Topology and equivalence bound (±0.20)", loc="left")
    panel_label(ax_topology, "b")

    condition_order = ["Simple", "Complex", "Control"]
    icc = icc.set_index("condition").loc[condition_order].reset_index()
    x = np.arange(3)
    ax_icc.errorbar(
        x,
        icc.median_icc_3_1,
        yerr=np.vstack(
            [icc.median_icc_3_1 - icc.q25_icc_3_1, icc.q75_icc_3_1 - icc.median_icc_3_1]
        ),
        fmt="o",
        color=COLORS["control"],
        ecolor=COLORS["control"],
        capsize=3,
        markersize=5,
    )
    ax_icc.set_xticks(x, condition_order)
    ax_icc.set_ylabel("Edgewise ICC(3,1)")
    ax_icc.set_ylim(0.20, 0.48)
    ax_icc.set_title("Four-block reliability: median and IQR", loc="left")
    ax_icc.grid(axis="y", color=COLORS["light"], linewidth=0.6)
    panel_label(ax_icc, "c")

    figure.suptitle(
        "Robustness analyses preserve the distributed FC pattern but not global topological equivalence",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_bundle(figure, output_dir / "ExtendedData_nonMVPA_1_robustness")
    _write_source(sensitivity, source_dir, "ExtendedData_1a_nbs_sensitivity")
    _write_source(topology, source_dir, "ExtendedData_1b_topology")
    _write_source(icc, source_dir, "ExtendedData_1c_icc")


def figure_stimulus_boundary(
    output_dir: Path, source_dir: Path, tables: dict[str, pd.DataFrame]
) -> None:
    """Extended Data Figure 2: stimulus checks and exploratory block models."""

    stimulus = tables["stimulus"].query("contrast == 'complex_minus_simple'").copy()
    stimulus = stimulus.sort_values("q_fdr_all", ascending=True).reset_index(drop=True)
    mixed = tables["block_mixed"].query("model == 'low_level_stimulus_adjusted'").copy()
    fixed = tables["block_fixed"].copy()
    video = tables["block_video"].copy()
    block_rows = pd.DataFrame(
        [
            {
                "display": "Global, 8 videos fixed",
                "estimate": fixed.query("fc_metric == 'global_all_edges'")
                .iloc[0]
                .mean_complex_minus_simple,
                "low": fixed.query("fc_metric == 'global_all_edges'").iloc[0].ci_low,
                "high": fixed.query("fc_metric == 'global_all_edges'").iloc[0].ci_high,
                "p": fixed.query("fc_metric == 'global_all_edges'")
                .iloc[0]
                .p_subject_sign_flip,
                "q": fixed.query("fc_metric == 'global_all_edges'")
                .iloc[0]
                .q_fdr_metric_family,
            },
            {
                "display": "Global, covariate adjusted",
                "estimate": mixed.query("fc_metric == 'global_all_edges'")
                .iloc[0]
                .condition_complex_beta,
                "low": mixed.query("fc_metric == 'global_all_edges'")
                .iloc[0]
                .condition_complex_beta
                - 1.96
                * mixed.query("fc_metric == 'global_all_edges'")
                .iloc[0]
                .condition_complex_se,
                "high": mixed.query("fc_metric == 'global_all_edges'")
                .iloc[0]
                .condition_complex_beta
                + 1.96
                * mixed.query("fc_metric == 'global_all_edges'")
                .iloc[0]
                .condition_complex_se,
                "p": mixed.query("fc_metric == 'global_all_edges'")
                .iloc[0]
                .condition_complex_p_asymptotic,
                "q": mixed.query("fc_metric == 'global_all_edges'")
                .iloc[0]
                .q_fdr_metric_family,
            },
            {
                "display": "Visual within, covariate adjusted",
                "estimate": mixed.query("fc_metric == 'visual_within'")
                .iloc[0]
                .condition_complex_beta,
                "low": mixed.query("fc_metric == 'visual_within'")
                .iloc[0]
                .condition_complex_beta
                - 1.96
                * mixed.query("fc_metric == 'visual_within'")
                .iloc[0]
                .condition_complex_se,
                "high": mixed.query("fc_metric == 'visual_within'")
                .iloc[0]
                .condition_complex_beta
                + 1.96
                * mixed.query("fc_metric == 'visual_within'")
                .iloc[0]
                .condition_complex_se,
                "p": mixed.query("fc_metric == 'visual_within'")
                .iloc[0]
                .condition_complex_p_asymptotic,
                "q": mixed.query("fc_metric == 'visual_within'")
                .iloc[0]
                .q_fdr_metric_family,
            },
        ]
    )

    figure = plt.figure(figsize=(WIDTH_IN, 4.35))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.2, 1.0), wspace=0.42)
    ax_stimulus = figure.add_subplot(grid[0, 0])
    ax_block = figure.add_subplot(grid[0, 1])

    y = np.arange(len(stimulus))[::-1]
    ax_stimulus.axvline(0.05, color=COLORS["dark"], linestyle="--", linewidth=0.8)
    ax_stimulus.scatter(
        stimulus.q_fdr_all,
        y,
        s=15,
        color=COLORS["neutral"],
        alpha=0.8,
        edgecolors="none",
    )
    ax_stimulus.set_yticks(y, stimulus.metric.str.replace("_", " "), fontsize=5)
    ax_stimulus.set_xlabel("BH–FDR q across 69 pairwise tests")
    ax_stimulus.set_xlim(0, 1.0)
    ax_stimulus.set_title(
        "Simple–Complex stimulus metrics (23/23 not significant)", loc="left"
    )
    panel_label(ax_stimulus, "a")

    y_block = np.arange(len(block_rows))[::-1]
    ax_block.axvline(0, color=COLORS["dark"], linestyle="--", linewidth=0.8)
    colors = [COLORS["neutral"], COLORS["negative"], COLORS["positive"]]
    for yi, row, color in zip(y_block, block_rows.itertuples(), colors):
        ax_block.errorbar(
            row.estimate,
            yi,
            xerr=[[row.estimate - row.low], [row.high - row.estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.5,
            markersize=5,
        )
        ax_block.text(row.high + 0.003, yi, f"q={row.q:.3f}", va="center", fontsize=6)
    ax_block.set_yticks(y_block, block_rows.display)
    ax_block.set_xlabel("Complex−Simple Fisher-z coefficient")
    ax_block.set_xlim(-0.035, 0.075)
    ax_block.set_title("Exploratory block-FC models", loc="left")
    panel_label(ax_block, "b")

    figure.suptitle(
        "Stimulus and video-level checks define the generalization boundary",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_bundle(figure, output_dir / "ExtendedData_nonMVPA_2_stimulus_boundary")
    _write_source(stimulus, source_dir, "ExtendedData_2a_stimulus_metrics")
    _write_source(block_rows, source_dir, "ExtendedData_2b_block_models")
    _write_source(video, source_dir, "ExtendedData_2_video_pair_exact_tests")


def _format_p(value: float | str | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        ">": r"\textgreater{}",
        "<": r"\textless{}",
        "−": "--",
        "–": "--",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _latex_table(
    frame: pd.DataFrame,
    caption: str,
    label: str,
    column_format: str | None = None,
) -> str:
    if column_format is None:
        column_format = "l" * len(frame.columns)
    header = " & ".join(_escape_latex(column) for column in frame.columns) + r" \\"
    rows = [
        " & ".join(_escape_latex(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{{column_format}}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )


def build_tables(
    run_dir: Path,
    manuscript_table_dir: Path,
    source_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> dict[str, int]:
    manuscript_table_dir.mkdir(parents=True, exist_ok=True)
    predefined = tables["predefined"]
    component = _main_nbs_component(tables)
    rsa = tables["rsa_primary"].query("distance == 'simple_complex'")
    rsa_centered = tables["rsa_centered"].query("distance == 'simple_complex'")
    top = tables["topology"].query(
        "metric_role == 'primary' and contrast == 'complex_minus_simple'"
    )
    activation = tables["activation"]
    omnibus = tables["activation_omnibus"]
    edges = tables["edges"].query("q_fdr < 0.05")
    n_positive_edges = int(edges.mean_difference.gt(0).sum())
    n_negative_edges = int(edges.mean_difference.lt(0).sum())

    main_rows = [
        {
            "Analysis": "Activation omnibus (secondary)",
            "Endpoint": "Simple / Complex / Control whole-brain map",
            "Estimate": f"{len(omnibus)} TFCE–FWER clusters",
            "Effect": f"peak F={omnibus.peak_F.max():.3f}",
            "P/q": f"peak P={_format_p(1 - omnibus.peak_corrp_1_minus_p.max())}",
            "Inference": "Any condition effect",
        },
        {
            "Analysis": "Activation",
            "Endpoint": "Complex > Simple whole-brain map",
            "Estimate": f"{len(activation)} TFCE–FWER clusters",
            "Effect": f"peak t={activation.peak_t.max():.3f}",
            "P/q": f"peak P={_format_p(1 - activation.peak_corrp_1_minus_p.max())}",
            "Inference": "Higher mean response",
        },
    ]
    rsa_labels = {
        "visual": "Visual Crossnobis",
        "dorsal_attention": "Dorsal-attention Crossnobis",
        "salience_attention": "Salience/attention Crossnobis",
        "control": "Control-network Crossnobis",
    }
    for network in ["visual", "dorsal_attention", "salience_attention", "control"]:
        row = rsa.query("network == @network").iloc[0]
        main_rows.append(
            {
                "Analysis": "RSA",
                "Endpoint": rsa_labels[network],
                "Estimate": f"{row.mean_crossnobis:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]",
                "Effect": f"dz={row.cohens_dz_vs_zero:.3f}",
                "P/q": f"q={_format_p(row.q_fdr_primary)}",
                "Inference": (
                    "FDR significant"
                    if row.q_fdr_primary < 0.05
                    else "Not FDR significant"
                ),
            }
        )
    centered_significant = rsa_centered.query(
        "network in ['visual', 'dorsal_attention', 'salience_attention', 'control'] and q_fdr_primary < 0.05"
    )
    main_rows.append(
        {
            "Analysis": "RSA sensitivity",
            "Endpoint": "Spatial-mean-centred Simple–Complex",
            "Estimate": f"{len(centered_significant)}/4 networks FDR significant",
            "Effect": "Visual and control",
            "P/q": "BH–FDR across 12 tests",
            "Inference": "Analysis-dependent sensitivity",
        }
    )
    global_row = predefined.query(
        "family == 'global' and metric == 'whole_brain_signed_mean' and contrast == 'complex_minus_simple'"
    ).iloc[0]
    main_rows.append(
        {
            "Analysis": "Background FC",
            "Endpoint": "Whole-brain mean",
            "Estimate": f"{global_row.mean_difference:.4f} [{global_row.mean_ci95_low:.4f}, {global_row.mean_ci95_high:.4f}]",
            "Effect": f"dz={global_row.cohens_dz:.3f}",
            "P/q": f"P={_format_p(global_row.p_sign_flip)}",
            "Inference": "Not significant",
        }
    )
    main_rows.append(
        {
            "Analysis": "Edgewise FC",
            "Endpoint": "All 30,876 edges",
            "Estimate": (
                f"{len(edges)} FDR edges "
                f"({n_positive_edges} positive, {n_negative_edges} negative)"
            ),
            "Effect": (
                f"absolute dz={edges.cohens_dz.abs().min():.3f}–"
                f"{edges.cohens_dz.abs().max():.3f}"
            ),
            "P/q": f"max q={_format_p(edges.q_fdr.max())}",
            "Inference": "Local increases and decreases",
        }
    )
    main_rows.append(
        {
            "Analysis": "NBS",
            "Endpoint": "Negative component, |t|≥3.1",
            "Estimate": f"{int(component.n_nodes)} nodes; {int(component.n_edges)} edges",
            "Effect": f"mass={component.component_mass:.2f}",
            "P/q": f"FWER P={_format_p(component.p_fwer)}",
            "Inference": "Distributed reduction",
        }
    )
    main_rows.append(
        {
            "Analysis": "Topology",
            "Endpoint": "Four primary AUC metrics",
            "Estimate": "1/4 equivalent at dz±0.20",
            "Effect": "Modularity only",
            "P/q": "TOST",
            "Inference": "No overall-stability claim",
        }
    )
    main = pd.DataFrame(main_rows)
    main.to_csv(manuscript_table_dir / "table_1_main_non_mvpa_results.csv", index=False)
    _write_source(main, source_dir, "Table_1_main_non_mvpa_results")
    markdown_main = main.map(
        lambda value: value.replace("|", r"\|") if isinstance(value, str) else value
    )
    (manuscript_table_dir / "table_1_main_non_mvpa_results.md").write_text(
        "# Table 1 | Summary of non-MVPA results\n\n"
        + markdown_main.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    (manuscript_table_dir / "table_1_main_non_mvpa_results.tex").write_text(
        _latex_table(
            main,
            r"Summary of non-MVPA results. The activation omnibus is secondary; intervals are participant-bootstrap 95\% confidence intervals where shown.",
            "tab:main-non-mvpa",
            "p{1.8cm}p{3.4cm}p{3.4cm}p{2.3cm}p{2.0cm}p{3.0cm}",
        ),
        encoding="utf-8",
    )

    activation.to_csv(
        manuscript_table_dir / "table_s1_activation_clusters.csv", index=False
    )
    omnibus.to_csv(
        manuscript_table_dir / "table_s1b_activation_omnibus_clusters.csv",
        index=False,
    )
    edges.to_csv(manuscript_table_dir / "table_s2_fdr_edges.csv", index=False)
    sensitivity = _sensitivity_nbs_table(run_dir)
    sensitivity.to_csv(
        manuscript_table_dir / "table_s3_nbs_sensitivity.csv", index=False
    )
    top.to_csv(manuscript_table_dir / "table_s4_topology.csv", index=False)
    boundary = pd.concat(
        [
            tables["block_fixed"].assign(analysis="fixed_video"),
            tables["block_video"].assign(analysis="video_pair_exact"),
        ],
        ignore_index=True,
        sort=False,
    )
    boundary.to_csv(
        manuscript_table_dir / "table_s5_stimulus_generalization.csv", index=False
    )

    supplement = "\n".join(
        [
            _latex_table(
                omnibus.assign(
                    volume_mm3=omnibus.volume_mm3.map(lambda value: f"{value:.0f}"),
                    peak_F=omnibus.peak_F.map(lambda value: f"{value:.3f}"),
                    peak_fwer_p=(1 - omnibus.peak_corrp_1_minus_p).map(_format_p),
                    mni_x=omnibus.mni_x.map(lambda value: f"{value:.0f}"),
                    mni_y=omnibus.mni_y.map(lambda value: f"{value:.0f}"),
                    mni_z=omnibus.mni_z.map(lambda value: f"{value:.0f}"),
                )[
                    [
                        "cluster_id",
                        "n_voxels",
                        "volume_mm3",
                        "peak_F",
                        "peak_fwer_p",
                        "mni_x",
                        "mni_y",
                        "mni_z",
                        "atlas_label",
                    ]
                ],
                r"Three-condition repeated-measures omnibus-F clusters after TFCE whole-brain FWER correction.",
                "tab:s1b-activation-omnibus",
                "rrrrrrrrp{5.2cm}",
            ),
            _latex_table(
                activation.assign(
                    volume_mm3=activation.volume_mm3.map(lambda value: f"{value:.0f}"),
                    peak_t=activation.peak_t.map(lambda value: f"{value:.3f}"),
                    peak_fwer_p=(1 - activation.peak_corrp_1_minus_p).map(_format_p),
                    mni_x=activation.mni_x.map(lambda value: f"{value:.0f}"),
                    mni_y=activation.mni_y.map(lambda value: f"{value:.0f}"),
                    mni_z=activation.mni_z.map(lambda value: f"{value:.0f}"),
                )[
                    [
                        "cluster_id",
                        "n_voxels",
                        "volume_mm3",
                        "peak_t",
                        "peak_fwer_p",
                        "mni_x",
                        "mni_y",
                        "mni_z",
                        "atlas_label",
                    ]
                ],
                r"Complex $>$ Simple activation clusters after TFCE whole-brain FWER correction.",
                "tab:s1-activation",
                "rrrrrrrrp{5.2cm}",
            ),
            _latex_table(
                edges.assign(
                    edge=edges.node1_name + " -- " + edges.node2_name,
                    mean_difference=edges.mean_difference.map(
                        lambda value: f"{value:.4f}"
                    ),
                    cohens_dz=edges.cohens_dz.map(lambda value: f"{value:.3f}"),
                    t=edges.t.map(lambda value: f"{value:.3f}"),
                    p_display=edges.p.map(_format_p),
                    q_display=edges.q_fdr.map(_format_p),
                )[
                    [
                        "edge",
                        "mean_difference",
                        "cohens_dz",
                        "t",
                        "p_display",
                        "q_display",
                    ]
                ],
                "Edges surviving BH--FDR across all 30,876 undirected edges.",
                "tab:s2-edges",
                "p{9cm}rrrrr",
            ),
            _latex_table(
                sensitivity.assign(
                    component_mass=sensitivity.component_mass.map(
                        lambda value: f"{value:.2f}"
                    ),
                    p_fwer=sensitivity.p_fwer.map(_format_p),
                )[
                    [
                        "display",
                        "n_subjects",
                        "n_atlas_nodes",
                        "n_nodes",
                        "n_edges",
                        "component_mass",
                        "p_fwer",
                    ]
                ],
                "Negative NBS components at |t| $\\geq$ 3.1 across processing variants.",
                "tab:s3-nbs-sensitivity",
                "p{5cm}rrrrrr",
            ),
            _latex_table(
                top.assign(
                    mean_difference=top.mean_difference.map(
                        lambda value: f"{value:.4f}"
                    ),
                    cohens_dz=top.cohens_dz.map(lambda value: f"{value:.3f}"),
                    p_sign_flip=top.p_sign_flip.map(_format_p),
                    q_fdr_by_role_contrast=top.q_fdr_by_role_contrast.map(_format_p),
                    tost_p_lower=top.tost_p_lower.map(_format_p),
                    tost_p_upper=top.tost_p_upper.map(_format_p),
                )[
                    [
                        "metric",
                        "mean_difference",
                        "cohens_dz",
                        "p_sign_flip",
                        "q_fdr_by_role_contrast",
                        "tost_p_lower",
                        "tost_p_upper",
                        "tost_equivalent_dz_0_20",
                    ]
                ],
                r"Primary topology AUC metrics and equivalence tests for Complex $-$ Simple.",
                "tab:s4-topology",
                "p{4cm}rrrrrrr",
            ),
        ]
    )
    (manuscript_table_dir / "supplementary_tables.tex").write_text(
        supplement, encoding="utf-8"
    )
    return {
        "main_rows": len(main),
        "activation_clusters": len(activation),
        "activation_omnibus_clusters": len(omnibus),
        "fdr_edges": len(edges),
        "sensitivity_variants": len(sensitivity),
        "topology_metrics": len(top),
    }


def write_manifest(output_dir: Path, counts: dict[str, int]) -> None:
    files = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    manifest = {
        "scope": "non-MVPA only",
        "backend": "Python/matplotlib+nilearn",
        "figures": [
            "Figure_nonMVPA_1_activation_rsa",
            "Figure_nonMVPA_2_fc_reconfiguration",
            "ExtendedData_nonMVPA_1_robustness",
            "ExtendedData_nonMVPA_2_stimulus_boundary",
            "ExtendedData_nonMVPA_3_activation_omnibus",
        ],
        "table_counts": counts,
        "files": files,
    }
    (output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    configure_style()
    run_dir = Path(args.run_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else run_dir / "paper_figures/non_mvpa"
    )
    source_dir = output_dir / "source_data"
    table_dir = Path(args.table_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    if not args.rsa_dir:
        raise ValueError(
            "--rsa-dir is required and must point to the corrected ResMS-whitened RSA output"
        )
    rsa_dir = Path(args.rsa_dir)
    tables = _load_tables(run_dir)
    tables.update(_load_corrected_rsa_tables(rsa_dir))
    figure_activation_rsa(run_dir, rsa_dir, output_dir, source_dir, tables)
    figure_activation_omnibus(run_dir, output_dir, source_dir, tables)
    figure_fc_reconfiguration(run_dir, output_dir, source_dir, tables)
    figure_robustness(run_dir, output_dir, source_dir, tables)
    figure_stimulus_boundary(output_dir, source_dir, tables)
    counts = build_tables(run_dir, table_dir, source_dir, tables)
    write_manifest(output_dir, counts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=str(RESULTS_DIR))
    p.add_argument("--output-dir")
    p.add_argument(
        "--rsa-dir",
        help="Corrected RSA directory containing crossnobis_primary and searchlight_primary",
    )
    p.add_argument(
        "--table-dir",
        default=str(PROJECT_ROOT.parent / "docs/manuscript/current/tables"),
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
