"""Generate the frozen non-MVPA statistics audit and canonical result table."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label

from fmri_video_analysis.core.config import PROJECT_ROOT, RESULTS_DIR

DEFAULT_RUN_DIR = RESULTS_DIR
DEFAULT_REPORT_DIR = PROJECT_ROOT.parent / "docs/reports"
TABLE_NAME = "非MVPA唯一结果表.csv"
DICTIONARY_NAME = "非MVPA唯一结果表_字段说明.md"
AUDIT_NAME = "非MVPA统计审计.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_source(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT.parent).as_posix()
    except ValueError:
        return str(path)


def finite_or_blank(value):
    if value is None:
        return ""
    try:
        return float(value) if np.isfinite(value) else ""
    except (TypeError, ValueError):
        return value


def p_display(value) -> str:
    if value in (None, "") or not np.isfinite(value):
        return ""
    value = float(value)
    if value == 0:
        return "0"
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def require_one(frame: pd.DataFrame, *, context: str, **criteria) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in criteria.items():
        mask &= frame[column].eq(value)
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one {context} row for {criteria}; found {len(rows)}."
        )
    return rows.iloc[0]



def truthy(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def add_record(
    records: list[dict],
    *,
    record_id: str,
    module: str,
    role: str,
    contrast: str,
    endpoint: str,
    independent_unit: str,
    n: int | str,
    estimate="",
    estimate_type: str = "",
    ci95_low="",
    ci95_high="",
    interval_type: str = "",
    effect_size="",
    effect_size_type: str = "",
    statistic="",
    statistic_type: str = "",
    df="",
    p_value="",
    multiplicity: str = "",
    q_value="",
    tost_p_lower="",
    tost_p_upper="",
    alpha="",
    threshold="",
    inference_status: str,
    direction: str = "",
    source_file: Path,
    source_filter: str,
    manuscript_location: str,
    conclusion_zh: str,
    limitation_zh: str,
) -> None:
    if not source_file.exists():
        raise FileNotFoundError(
            f"Canonical result source does not exist: {source_file}"
        )
    p_value = finite_or_blank(p_value)
    interval_low = finite_or_blank(ci95_low)
    interval_high = finite_or_blank(ci95_high)
    if not interval_type and interval_low != "" and interval_high != "":
        interval_type = "95% bootstrap CI"
    records.append(
        {
            "record_id": record_id,
            "module": module,
            "role": role,
            "contrast": contrast,
            "endpoint": endpoint,
            "independent_unit": independent_unit,
            "n": n,
            "estimate": finite_or_blank(estimate),
            "estimate_type": estimate_type,
            "interval_low": interval_low,
            "interval_high": interval_high,
            "interval_type": interval_type,
            "effect_size": finite_or_blank(effect_size),
            "effect_size_type": effect_size_type,
            "statistic": finite_or_blank(statistic),
            "statistic_type": statistic_type,
            "df": df,
            "p_value": p_value,
            "p_display": p_display(p_value),
            "multiplicity": multiplicity,
            "q_value": finite_or_blank(q_value),
            "tost_p_lower": finite_or_blank(tost_p_lower),
            "tost_p_upper": finite_or_blank(tost_p_upper),
            "alpha": alpha,
            "threshold": threshold,
            "inference_status": inference_status,
            "direction": direction,
            "source_file": relative_source(source_file),
            "source_filter": source_filter,
            "manuscript_location": manuscript_location,
            "conclusion_zh": conclusion_zh,
            "limitation_zh": limitation_zh,
        }
    )


def searchlight_bonferroni_summary(run_dir: Path, contrast: str) -> dict:
    path = (
        run_dir
        / "rsa/searchlight"
        / f"{contrast}_tfce_fwer_bonferroni3_tstat.nii.gz"
    )
    data = np.asarray(nib.load(str(path)).dataobj)
    components, n_components = label(
        data != 0,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(components.ravel())[1:]
    return {
        "path": path,
        "n_voxels": int(np.count_nonzero(data)),
        "n_components": int(n_components),
        "largest_component_voxels": int(sizes.max()) if len(sizes) else 0,
        "peak_t": float(np.max(data)) if np.any(data) else np.nan,
    }


def collect_records(run_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    stimulus_dir = run_dir / "stimulus/characterization"
    activation_dir = run_dir / "activation/group_contrasts"
    activation_rm_dir = run_dir / "activation/repeated_measures"
    rsa_dir = run_dir / "rsa"
    group_dir = run_dir / "group_statistics"
    sensitivity_dir = run_dir / "sensitivity"

    input_audit_path = run_dir / "input_audit.json"
    input_audit = load_json(input_audit_path)
    add_record(
        records,
        record_id="DESIGN-01",
        module="study_design",
        role="design",
        contrast="not_applicable",
        endpoint="included functional runs",
        independent_unit="participant",
        n=input_audit["n_subjects"],
        estimate=input_audit["n_runs"],
        estimate_type="total subject-run inputs",
        inference_status="frozen",
        source_file=input_audit_path,
        source_filter="complete input audit",
        manuscript_location="Methods—Participants and fMRI data",
        conclusion_zh=(
            f"当前分析纳入{input_audit['n_subjects']}名被试，共"
            f"{input_audit['n_runs']}个被试-run输入。"
        ),
        limitation_zh="该记录仅冻结当前分析所用的功能数据输入。",
    )

    atlas_path = run_dir / "atlas/composite_249_qc.json"
    atlas = load_json(atlas_path)
    add_record(
        records,
        record_id="ATLAS-01",
        module="atlas",
        role="design",
        contrast="not_applicable",
        endpoint="composite atlas",
        independent_unit="atlas node",
        n=atlas["node_count"],
        estimate=atlas["edge_count"],
        estimate_type="unique undirected edges",
        inference_status="frozen",
        source_file=atlas_path,
        source_filter="node_count and edge_count",
        manuscript_location="Methods—Parcellation",
        conclusion_zh=(
            "主分析使用Schaefer-200皮层、Tian Scale II 32个皮层下节点和"
            "Buckner-17小脑网络组成的249节点图谱。"
        ),
        limitation_zh=(
            "重采样后两个小脑节点各仅含2个3-mm体素；Schaefer-200纯皮层"
            "分析作为图谱敏感性检验。"
        ),
    )

    stimulus_path = stimulus_dir / "stimulus_pairwise_tests.csv"
    stimulus = pd.read_csv(stimulus_path)
    simple_complex_stimulus = stimulus[
        stimulus.contrast.eq("complex_minus_simple")
    ]
    minimum_stimulus = simple_complex_stimulus.loc[
        simple_complex_stimulus.q_fdr_all.idxmin()
    ]
    add_record(
        records,
        record_id="STIM-01",
        module="stimulus",
        role="primary",
        contrast="Complex−Simple",
        endpoint=f"{len(simple_complex_stimulus)} video-level visual and semantic metrics",
        independent_unit="video",
        n="4 Complex + 4 Simple videos",
        estimate=int((simple_complex_stimulus.q_fdr_all < 0.05).sum()),
        estimate_type=(
            "number significant after BH-FDR; reported p/q are the minimum "
            f"for metric={minimum_stimulus.metric}"
        ),
        p_value=minimum_stimulus.p_exact_permutation,
        multiplicity="BH-FDR across all 69 pairwise metric tests",
        q_value=minimum_stimulus.q_fdr_all,
        alpha=0.05,
        inference_status="not_significant",
        source_file=stimulus_path,
        source_filter="contrast == complex_minus_simple",
        manuscript_location="Results—Stimulus characterization",
        conclusion_zh="Simple与Complex之间23项视频级指标均未通过全配对检验族BH-FDR。",
        limitation_zh=(
            "每个条件仅4段视频，未显著不能证明刺激等效；自动语义检测未完成"
            "人工核验，只作描述。"
        ),
    )

    omnibus_path = stimulus_dir / "stimulus_omnibus_tests.csv"
    omnibus = pd.read_csv(omnibus_path)
    add_record(
        records,
        record_id="STIM-02",
        module="stimulus",
        role="secondary",
        contrast="Simple–Complex–Control omnibus",
        endpoint="23 video-level visual and semantic metrics",
        independent_unit="video",
        n="4 videos per condition",
        estimate=int((omnibus.q_fdr < 0.05).sum()),
        estimate_type="number significant after BH-FDR",
        p_value=omnibus.p_exact_permutation.min(),
        multiplicity="BH-FDR across 23 omnibus tests",
        q_value=omnibus.q_fdr.min(),
        alpha=0.05,
        inference_status="secondary_significant",
        source_file=omnibus_path,
        source_filter="all 23 metrics",
        manuscript_location="Supplementary Results—Control videos",
        conclusion_zh="三条件总体比较中15/23项指标通过FDR，主要反映Control视频的差异。",
        limitation_zh="该结果不能用于声称Simple和Complex之间存在相同的刺激差异。",
    )

    activation_omnibus_summary_path = activation_rm_dir / "omnibus_summary.csv"
    activation_omnibus_summary = pd.read_csv(
        activation_omnibus_summary_path
    ).iloc[0]
    activation_omnibus_clusters_path = (
        activation_rm_dir / "condition_omnibus_clusters_tfce_fwer05.csv"
    )
    activation_omnibus_clusters = pd.read_csv(
        activation_omnibus_clusters_path
    ).sort_values("peak_F", ascending=False)
    activation_omnibus_peak = activation_omnibus_clusters.iloc[0]
    add_record(
        records,
        record_id="ACT-OMNI-00",
        module="activation_omnibus",
        role="secondary",
        contrast="Simple–Complex–Control omnibus",
        endpoint="whole-brain mean BOLD condition effect",
        independent_unit="participant",
        n=int(activation_omnibus_summary.n_subjects),
        estimate=int(activation_omnibus_summary.n_significant_clusters),
        estimate_type="TFCE-FWER significant clusters",
        statistic=activation_omnibus_peak.peak_F,
        statistic_type="peak F",
        p_value=1.0 - activation_omnibus_peak.peak_corrp_1_minus_p,
        multiplicity=(
            "10,000 restricted permutations with participant exchangeability "
            "blocks; TFCE controls voxelwise whole-brain FWER"
        ),
        alpha=0.05,
        inference_status="secondary_significant",
        source_file=activation_omnibus_summary_path,
        source_filter="three_condition_repeated_measures_F",
        manuscript_location="Results—Univariate activation omnibus",
        conclusion_zh=(
            "三条件重复测量omnibus F检出"
            f"{int(activation_omnibus_summary.n_significant_clusters)}个"
            "全脑TFCE-FWER显著簇。"
        ),
        limitation_zh=(
            "总体F检验无方向，不能识别由哪一对条件驱动；预设状态待作者确认，"
            "当前按secondary分析报告。"
        ),
    )
    for index, row in enumerate(activation_omnibus_clusters.itertuples(), 1):
        add_record(
            records,
            record_id=f"ACT-OMNI-{index:02d}",
            module="activation_omnibus",
            role="secondary_detail",
            contrast="Simple–Complex–Control omnibus",
            endpoint=str(row.atlas_label),
            independent_unit="participant",
            n=int(activation_omnibus_summary.n_subjects),
            estimate=int(row.n_voxels),
            estimate_type="cluster voxels (3-mm isotropic)",
            statistic=row.peak_F,
            statistic_type="peak F",
            p_value=1.0 - row.peak_corrp_1_minus_p,
            multiplicity="TFCE whole-brain FWER",
            alpha=0.05,
            inference_status="secondary_significant",
            source_file=activation_omnibus_clusters_path,
            source_filter=f"cluster_id == {row.cluster_id}",
            manuscript_location=(
                "Extended Data Figure 3 / Supplementary Table—Activation omnibus"
            ),
            conclusion_zh=(
                f"总体条件效应显著簇峰值位于{row.atlas_label}，峰值MNI="
                f"({row.mni_x:g}, {row.mni_y:g}, {row.mni_z:g})。"
            ),
            limitation_zh=(
                "峰值标签仅用于空间描述；总体F检验不能提供特定条件对的方向。"
            ),
        )

    activation_summary_path = activation_dir / "activation_summary.csv"
    activation_summary = pd.read_csv(activation_summary_path).set_index("contrast")
    activation_row = activation_summary.loc["complex_minus_simple"]
    add_record(
        records,
        record_id="ACT-00",
        module="activation",
        role="primary",
        contrast="Complex>Simple",
        endpoint="whole-brain mean BOLD contrast",
        independent_unit="participant",
        n=int(activation_row.n_subjects),
        estimate=int(activation_row.n_significant_clusters),
        estimate_type="TFCE-FWER significant clusters",
        p_value="",
        multiplicity="10,000 sign flips; TFCE controls voxelwise whole-brain FWER",
        alpha=0.05,
        inference_status="significant",
        source_file=activation_summary_path,
        source_filter="contrast == complex_minus_simple",
        manuscript_location="Results—Univariate activation",
        conclusion_zh="Complex>Simple单变量激活检出6个全脑TFCE-FWER显著簇。",
        limitation_zh=(
            "这是Complex>Simple的单向空间检验；当前结果不能排除Simple>Complex区域，"
            "也不能解释为功能连接或因果机制。"
        ),
    )

    activation_clusters_path = (
        activation_dir / "complex_minus_simple_clusters_tfce_fwer05.csv"
    )
    activation_clusters = pd.read_csv(activation_clusters_path).sort_values(
        "peak_t", ascending=False
    )
    for index, row in enumerate(activation_clusters.itertuples(), 1):
        add_record(
            records,
            record_id=f"ACT-{index:02d}",
            module="activation",
            role="primary_detail",
            contrast="Complex>Simple",
            endpoint=str(row.atlas_label),
            independent_unit="participant",
            n=87,
            estimate=int(row.n_voxels),
            estimate_type="cluster voxels (3-mm isotropic)",
            statistic=row.peak_t,
            statistic_type="peak t",
            df=86,
            p_value=1.0 - row.peak_corrp_1_minus_p,
            multiplicity="TFCE whole-brain FWER",
            alpha=0.05,
            inference_status="significant",
            source_file=activation_clusters_path,
            source_filter=f"cluster_id == {row.cluster_id}",
            manuscript_location="Results table / Supplementary Table—Activation",
            conclusion_zh=(
                f"显著簇峰值位于{row.atlas_label}，峰值MNI=({row.mni_x:g}, "
                f"{row.mni_y:g}, {row.mni_z:g})。"
            ),
            limitation_zh="峰值标签仅用于空间描述，不能据此赋予整个簇单一功能。",
        )

    rsa_path = rsa_dir / "crossnobis/crossnobis_summary.csv"
    rsa = pd.read_csv(rsa_path)
    primary_networks = (
        "visual",
        "dorsal_attention",
        "salience_attention",
        "control",
    )
    rsa_sc = rsa[
        rsa.network.isin(primary_networks)
        & rsa.distance.eq("simple_complex")
    ].copy()
    for row in rsa_sc.itertuples():
        add_record(
            records,
            record_id=f"RSA-{row.network.upper()}",
            module="crossnobis_rsa",
            role="primary",
            contrast="Simple–Complex distance > 0",
            endpoint=f"{row.network} network Crossnobis distance",
            independent_unit="participant",
            n=int(row.n_subjects),
            estimate=row.mean_crossnobis,
            estimate_type="mean diagonal Crossnobis distance",
            ci95_low=row.ci_low,
            ci95_high=row.ci_high,
            effect_size=row.cohens_dz_vs_zero,
            effect_size_type="Cohen dz versus zero",
            p_value=row.p_signflip_vs_zero,
            multiplicity="BH-FDR across 4 primary networks × 3 distances",
            q_value=row.q_fdr_primary,
            alpha=0.05,
            inference_status="significant",
            direction="positive distance",
            source_file=rsa_path,
            source_filter=(
                f"network == {row.network}; distance == simple_complex"
            ),
            manuscript_location="Results—Network representational geometry",
            conclusion_zh=(
                f"{row.network}网络的Simple–Complex跨session表征距离大于零。"
            ),
            limitation_zh=(
                "正Crossnobis距离表示条件模式可区分，不等于分类泛化或机制证据。"
            ),
        )

    searchlight_summary_path = rsa_dir / "searchlight/searchlight_summary.csv"
    searchlight_summary = pd.read_csv(searchlight_summary_path)
    searchlight_row = searchlight_summary[
        searchlight_summary.distance.eq("simple_complex")
    ].iloc[0]
    searchlight = searchlight_bonferroni_summary(run_dir, "simple_complex")
    searchlight_cluster_path = (
        rsa_dir / "searchlight/simple_complex_tfce_fwer05_clusters.csv"
    )
    searchlight_clusters = pd.read_csv(searchlight_cluster_path)
    searchlight_peak = searchlight_clusters.iloc[0]
    add_record(
        records,
        record_id="SL-01",
        module="searchlight_rsa",
        role="primary_spatial_localization",
        contrast="Simple–Complex distance > 0",
        endpoint="9-mm whole-brain Crossnobis searchlight",
        independent_unit="participant",
        n=int(searchlight_row.n_subjects),
        estimate=searchlight["n_voxels"],
        estimate_type=(
            f"significant voxels in {searchlight['n_components']} connected components"
        ),
        statistic=searchlight["peak_t"],
        statistic_type="peak t",
        df=86,
        multiplicity=(
            "10,000 sign flips; TFCE whole-brain FWER plus Bonferroni across 3 maps"
        ),
        alpha=0.05,
        threshold="TFCE-FWER corrp > 0.98333 (Bonferroni 0.05/3)",
        inference_status="significant",
        direction="positive distance",
        source_file=searchlight["path"],
        source_filter="nonzero voxels in Bonferroni3 thresholded t map",
        manuscript_location="Results—Searchlight localization",
        conclusion_zh=(
            "Simple–Complex表征距离在全脑校正后显著，主峰位于"
            f"{searchlight_peak.atlas_label}。"
        ),
        limitation_zh=(
            "Searchlight定位局部信息分布；大而重叠的搜索球会扩大显著空间范围，"
            "不能把整个显著区视为一个功能单元。"
        ),
    )

    predefined_path = (
        group_dir / "predefined_metrics/predefined_metric_statistics.csv"
    )
    predefined = pd.read_csv(predefined_path)
    selected_metrics = (
        "whole_brain_signed_mean",
        "cortical_cross_hemisphere",
        "cortex_subcortex",
        "cortex_cerebellum",
        "subcortex_cerebellum",
    )
    selected = predefined[
        predefined.contrast.eq("complex_minus_simple")
        & predefined.metric.isin(selected_metrics)
    ]
    for row in selected.itertuples():
        add_record(
            records,
            record_id=f"FC-{row.metric.upper()}",
            module="background_fc_summary",
            role="primary" if row.family == "global" else "primary_domain",
            contrast="Complex−Simple",
            endpoint=row.label,
            independent_unit="participant",
            n=int(row.n_subjects),
            estimate=row.mean_difference,
            estimate_type="mean Fisher-z FC difference",
            ci95_low=row.mean_ci95_low,
            ci95_high=row.mean_ci95_high,
            effect_size=row.cohens_dz,
            effect_size_type="Cohen dz",
            p_value=row.p_sign_flip,
            multiplicity=(
                "10,000 paired sign flips; BH-FDR within metric family and contrast"
            ),
            q_value=row.q_fdr_within_family,
            alpha=0.05,
            inference_status="not_significant",
            direction="negative" if row.mean_difference < 0 else "positive",
            source_file=predefined_path,
            source_filter=(
                f"metric == {row.metric}; contrast == complex_minus_simple"
            ),
            manuscript_location="Results—Global and domain-level FC",
            conclusion_zh=f"{row.label}平均FC未显示可靠的Complex−Simple差异。",
            limitation_zh="未显著不等于等效；只有预设TOST可以支持等效性结论。",
        )

    significant_predefined = predefined[
        predefined.contrast.eq("complex_minus_simple")
        & predefined.q_fdr_within_family.lt(0.05)
    ]
    for row in significant_predefined.itertuples():
        add_record(
            records,
            record_id=f"FC-NET-{row.metric.upper()}",
            module="background_fc_network",
            role="primary_network",
            contrast="Complex−Simple",
            endpoint=row.label,
            independent_unit="participant",
            n=int(row.n_subjects),
            estimate=row.mean_difference,
            estimate_type="mean Fisher-z FC metric difference",
            ci95_low=row.mean_ci95_low,
            ci95_high=row.mean_ci95_high,
            effect_size=row.cohens_dz,
            effect_size_type="Cohen dz",
            p_value=row.p_sign_flip,
            multiplicity=(
                f"10,000 paired sign flips; BH-FDR within {row.family} family"
            ),
            q_value=row.q_fdr_within_family,
            alpha=0.05,
            inference_status="significant",
            direction="negative",
            source_file=predefined_path,
            source_filter=(
                f"metric == {row.metric}; contrast == complex_minus_simple"
            ),
            manuscript_location="Results—Prespecified network FC",
            conclusion_zh=(
                f"{row.label}的{row.metric}在Complex条件下降低。"
            ),
            limitation_zh=(
                "各网络指标在预定义但不同大小的FDR家族中校正，不能与30,876条"
                "边的全局FDR结果混为同一检验族。"
            ),
        )

    edge_path = group_dir / "edgewise/edgewise_complex_minus_simple.csv.gz"
    edges = pd.read_csv(edge_path)
    significant_edges = edges[edges.q_fdr.lt(0.05)].sort_values(
        "q_fdr"
    )
    for index, row in enumerate(significant_edges.itertuples(), 1):
        add_record(
            records,
            record_id=f"EDGE-{index:02d}",
            module="background_fc_edge",
            role="primary_edge",
            contrast="Complex−Simple",
            endpoint=f"{row.node1_name} — {row.node2_name}",
            independent_unit="participant",
            n=87,
            estimate=row.mean_difference,
            estimate_type="mean Fisher-z edge difference",
            effect_size=row.cohens_dz,
            effect_size_type="Cohen dz",
            statistic=row.t,
            statistic_type="paired t",
            df=86,
            p_value=row.p,
            multiplicity="BH-FDR across all 30,876 undirected edges",
            q_value=row.q_fdr,
            alpha=0.05,
            inference_status="significant",
            direction=row.direction,
            source_file=edge_path,
            source_filter=f"node1_id == {row.node1_id}; node2_id == {row.node2_id}",
            manuscript_location="Results—Edgewise FC / Supplementary Table",
            conclusion_zh=(
                f"该边在Complex条件下呈{('增强' if row.direction == 'positive' else '降低')}。"
            ),
            limitation_zh=(
                "边级显著性不等于该连接在所有被试中同向，也不能单独支持特定机制。"
            ),
        )

    nbs_path = group_dir / "nbs/nbs_complex_minus_simple_components.csv"
    nbs = pd.read_csv(nbs_path)
    nbs_main = nbs[
        nbs.threshold_abs_t.eq(3.1)
        & nbs.direction.eq("negative")
        & nbs.significant_fwer_0_05
    ].iloc[0]
    add_record(
        records,
        record_id="NBS-MAIN",
        module="nbs",
        role="primary",
        contrast="Complex−Simple",
        endpoint="negative connected component at |t|≥3.1",
        independent_unit="participant",
        n=87,
        estimate=int(nbs_main.n_edges),
        estimate_type=f"edges across {int(nbs_main.n_nodes)} nodes",
        statistic=nbs_main.component_mass,
        statistic_type="NBS component mass",
        p_value=nbs_main.p_fwer,
        multiplicity=(
            "10,000 sign flips; joint two-sided maximum-component-mass FWER"
        ),
        alpha=0.05,
        inference_status="significant",
        direction="negative",
        source_file=nbs_path,
        source_filter=(
            "threshold_abs_t == 3.1; direction == negative; FWER significant"
        ),
        manuscript_location="Results—Distributed FC reconfiguration",
        conclusion_zh=(
            f"Complex条件出现包含{int(nbs_main.n_nodes)}个节点、"
            f"{int(nbs_main.n_edges)}条边的分布式负向FC成分。"
        ),
        limitation_zh=(
            "NBS推断对象是整个连通成分，不保证成分内每一条边单独通过多重校正；"
            "成分范围依赖初始t阈值。"
        ),
    )

    for threshold in (2.5, 3.5):
        row = nbs[
            nbs.threshold_abs_t.eq(threshold)
            & nbs.direction.eq("negative")
            & nbs.significant_fwer_0_05
        ].iloc[0]
        add_record(
            records,
            record_id=f"NBS-THRESH-{str(threshold).replace('.', 'P')}",
            module="nbs",
            role="threshold_sensitivity",
            contrast="Complex−Simple",
            endpoint=f"negative component at |t|≥{threshold:g}",
            independent_unit="participant",
            n=87,
            estimate=int(row.n_edges),
            estimate_type=f"edges across {int(row.n_nodes)} nodes",
            statistic=row.component_mass,
            statistic_type="NBS component mass",
            p_value=row.p_fwer,
            multiplicity=(
                "10,000 sign flips; joint two-sided maximum-component-mass FWER"
            ),
            alpha=0.05,
            inference_status="sensitivity_significant",
            direction="negative",
            source_file=nbs_path,
            source_filter=(
                f"threshold_abs_t == {threshold:g}; negative FWER component"
            ),
            manuscript_location="Supplementary Results—NBS thresholds",
            conclusion_zh=f"负向NBS成分在|t|≥{threshold:g}阈值下仍显著。",
            limitation_zh="该行用于阈值稳健性，不作为新的独立发现计数。",
        )

    variants = (
        "motion_6p",
        "no_fir",
        "no_fd_censor",
        "fd_0p3",
        "lag_0",
        "gsr",
        "schaefer200",
        "exclude_high_motion",
    )
    variant_labels = {
        "motion_6p": "6-parameter motion regression",
        "no_fir": "without FIR task regression",
        "no_fd_censor": "without FD censoring",
        "fd_0p3": "FD threshold 0.3 mm",
        "lag_0": "HRF lag 0 TR",
        "gsr": "with global signal regression",
        "schaefer200": "Schaefer-200 cortex only",
        "exclude_high_motion": "high-motion participant exclusion",
    }
    sensitivity_path = sensitivity_dir / "sensitivity_summary.csv"
    sensitivity = pd.read_csv(sensitivity_path)
    for variant in variants:
        component_path = (
            sensitivity_dir
            / variant
            / "nbs/nbs_complex_minus_simple_components.csv"
        )
        components = pd.read_csv(component_path)
        component = components[
            components.threshold_abs_t.eq(3.1)
            & components.direction.eq("negative")
            & components.significant_fwer_0_05
        ].iloc[0]
        summary = sensitivity[
            sensitivity.variant.eq(variant)
            & sensitivity.contrast.eq("complex_minus_simple")
        ].iloc[0]
        add_record(
            records,
            record_id=f"SENS-{variant.upper()}",
            module="sensitivity",
            role="sensitivity",
            contrast="Complex−Simple",
            endpoint=variant_labels[variant],
            independent_unit="participant",
            n=int(summary.n_subjects),
            estimate=int(component.n_edges),
            estimate_type=f"negative NBS edges across {int(component.n_nodes)} nodes",
            effect_size=summary.global_cohens_dz,
            effect_size_type="global FC Cohen dz",
            p_value=component.p_fwer,
            multiplicity=(
                "10,000 sign flips; joint two-sided NBS "
                "maximum-component-mass FWER"
            ),
            alpha=0.05,
            inference_status="sensitivity_significant",
            direction="negative",
            source_file=component_path,
            source_filter="threshold_abs_t == 3.1; significant negative component",
            manuscript_location="Supplementary Results—FC sensitivity",
            conclusion_zh=(
                f"{variant_labels[variant]}方案保留显著负向NBS成分。"
            ),
            limitation_zh=(
                "敏感性方案共享同一批数据，说明处理稳健性，不构成独立重复验证。"
            ),
        )

    topology_path = group_dir / "topology/topology_statistics.csv"
    topology = pd.read_csv(topology_path)
    topology_primary = topology[
        topology.metric_role.eq("primary")
        & topology.contrast.eq("complex_minus_simple")
    ]
    for row in topology_primary.itertuples():
        equivalent = truthy(row.tost_equivalent_dz_0_20)
        add_record(
            records,
            record_id=f"TOPO-{row.metric.upper()}",
            module="topology",
            role="primary",
            contrast="Complex−Simple",
            endpoint=row.metric,
            independent_unit="participant",
            n=87,
            estimate=row.mean_difference,
            estimate_type="density 0.10–0.30 AUC difference; TOST 90% CI in source",
            ci95_low=row.mean_ci95_low,
            ci95_high=row.mean_ci95_high,
            effect_size=row.cohens_dz,
            effect_size_type="Cohen dz",
            p_value=row.p_sign_flip,
            multiplicity="BH-FDR across four primary topology metrics per contrast",
            q_value=row.q_fdr_by_role_contrast,
            tost_p_lower=row.tost_p_lower,
            tost_p_upper=row.tost_p_upper,
            alpha=0.05,
            inference_status=("equivalent" if equivalent else "not_equivalent"),
            direction="negative" if row.mean_difference < 0 else "positive",
            source_file=topology_path,
            source_filter=(
                f"metric == {row.metric}; contrast == complex_minus_simple"
            ),
            manuscript_location="Results—Network topology",
            conclusion_zh=(
                f"{row.metric}"
                + ("通过dz±0.20的TOST等效检验。" if equivalent else "未建立等效。")
            ),
            limitation_zh=(
                "预设判定要求四项主要指标全部通过TOST；实际仅模块度通过，"
                "不能表述为整体拓扑稳定。"
            ),
        )

    qc_fc_path = sensitivity_dir / "qc_fc/qc_fc_summary.json"
    qc_fc = load_json(qc_fc_path)
    add_record(
        records,
        record_id="QCFC-01",
        module="qc_fc",
        role="diagnostic",
        contrast="mean across all three conditions",
        endpoint="edgewise correlation with participant mean FD",
        independent_unit="participant",
        n=qc_fc["n_subjects"],
        estimate=qc_fc["n_qc_fc_edges_fdr_0_05"],
        estimate_type="edges significant after BH-FDR",
        effect_size=qc_fc["median_absolute_qc_fc_r"],
        effect_size_type="median absolute QC-FC Pearson r",
        multiplicity="BH-FDR across 30,876 edges",
        alpha=0.05,
        inference_status="residual_motion_detected",
        source_file=qc_fc_path,
        source_filter="complete QC-FC summary",
        manuscript_location="Results/Limitations—Motion diagnostics",
        conclusion_zh="残余QC-FC仍有2,759条边通过FDR，中位|r|=0.163。",
        limitation_zh=(
            "该诊断基于三条件平均FC而非Complex−Simple差值；它提示残余运动结构，"
            "但不能单独证明条件差异由运动造成，也不能声称头动影响已完全去除。"
        ),
    )

    icc_path = sensitivity_dir / "block_reliability/block_reliability_summary.csv"
    icc = pd.read_csv(icc_path)
    for row in icc.itertuples():
        add_record(
            records,
            record_id=f"ICC-{row.condition.upper()}",
            module="block_reliability",
            role="diagnostic",
            contrast="within-condition across 4 blocks",
            endpoint=f"{row.condition} edgewise ICC(3,1)",
            independent_unit="participant",
            n=int(row.n_subjects),
            estimate=row.median_icc_3_1,
            estimate_type="median edgewise ICC(3,1)",
            ci95_low=row.q25_icc_3_1,
            ci95_high=row.q75_icc_3_1,
            interval_type="IQR (25th–75th percentile)",
            inference_status="descriptive",
            source_file=icc_path,
            source_filter=f"condition == {row.condition}",
            manuscript_location="Supplementary Results—Block reliability",
            conclusion_zh=(
                f"{row.condition}条件边级ICC中位数为{row.median_icc_3_1:.3f}。"
            ),
            limitation_zh="该区间为四分位距而非95%置信区间；短区块FC可靠性有限。",
        )

    fixed_path = sensitivity_dir / "block_stimulus/fixed_video_condition_tests.csv"
    fixed = pd.read_csv(fixed_path).set_index("fc_metric").loc["global_all_edges"]
    add_record(
        records,
        record_id="BLOCK-GLOBAL-FIXED",
        module="block_stimulus",
        role="exploratory",
        contrast="Complex−Simple",
        endpoint="global block FC; eight observed videos fixed",
        independent_unit="participant",
        n=int(fixed.n_subjects),
        estimate=fixed.mean_complex_minus_simple,
        estimate_type="mean Fisher-z FC difference",
        ci95_low=fixed.ci_low,
        ci95_high=fixed.ci_high,
        effect_size=fixed.cohens_dz,
        effect_size_type="Cohen dz",
        p_value=fixed.p_subject_sign_flip,
        multiplicity="BH-FDR across 10 prespecified FC summaries",
        q_value=fixed.q_fdr_metric_family,
        alpha=0.05,
        inference_status="exploratory_not_significant",
        direction="negative",
        source_file=fixed_path,
        source_filter="fc_metric == global_all_edges",
        manuscript_location="Results—Stimulus-adjusted FC robustness",
        conclusion_zh="固定当前8段任务视频时，全局区块FC差异未达到显著。",
        limitation_zh="该检验推广到被试，但不推广到未见视频。",
    )

    pair_path = sensitivity_dir / "block_stimulus/video_pair_exact_tests.csv"
    pair = pd.read_csv(pair_path).set_index("fc_metric").loc["global_all_edges"]
    add_record(
        records,
        record_id="BLOCK-GLOBAL-VIDEO",
        module="block_stimulus",
        role="exploratory_generalization",
        contrast="Complex−Simple",
        endpoint="global block FC across session-matched video pairs",
        independent_unit="video pair",
        n=int(pair.n_video_pairs),
        estimate=pair.mean_complex_minus_simple,
        estimate_type="mean Fisher-z FC difference",
        effect_size=pair.cohens_dz_across_pairs,
        effect_size_type="Cohen dz across video pairs",
        p_value=pair.p_exact_video_pair_sign_flip,
        multiplicity="exact sign flips; BH-FDR across 10 FC summaries",
        q_value=pair.q_fdr_metric_family,
        alpha=0.05,
        inference_status="exploratory_not_significant",
        direction="negative",
        source_file=pair_path,
        source_filter="fc_metric == global_all_edges",
        manuscript_location="Results—Stimulus generalization boundary",
        conclusion_zh="4对视频的精确符号翻转检验未支持全局FC条件差异的视频层泛化。",
        limitation_zh="仅4对视频，两侧精确检验最小可达p=0.125，统计分辨率很低。",
    )

    mixed_path = sensitivity_dir / "block_stimulus/mixed_effects_results.csv"
    mixed = pd.read_csv(mixed_path)
    global_adjusted = mixed[
        mixed.fc_metric.eq("global_all_edges")
        & mixed.model.eq("low_level_stimulus_adjusted")
    ].iloc[0]
    add_record(
        records,
        record_id="BLOCK-GLOBAL-ADJUSTED",
        module="block_stimulus",
        role="exploratory_covariate_adjusted",
        contrast="Complex−Simple",
        endpoint="global block FC adjusted for stimulus PCs and motion",
        independent_unit="participant × video block",
        n=int(global_adjusted.n_observations),
        estimate=global_adjusted.condition_complex_beta,
        estimate_type="mixed-model condition coefficient",
        statistic=global_adjusted.condition_complex_se,
        statistic_type="standard error",
        p_value=global_adjusted.condition_complex_p_asymptotic,
        multiplicity="BH-FDR across 10 FC summaries within model",
        q_value=global_adjusted.q_fdr_metric_family,
        alpha=0.05,
        inference_status="exploratory_not_significant",
        direction="negative",
        source_file=mixed_path,
        source_filter=(
            "fc_metric == global_all_edges; model == low_level_stimulus_adjusted"
        ),
        manuscript_location="Results—Stimulus-adjusted FC robustness",
        conclusion_zh="控制低级视觉PC、平均FD和删失TR后，全局条件系数减小且不显著。",
        limitation_zh=(
            "只有8段视频且模型出现边界警告，渐近p值仅作探索；视频层精确检验"
            "才是主要泛化检查。"
        ),
    )

    visual_adjusted = mixed[
        mixed.fc_metric.eq("visual_within")
        & mixed.model.eq("low_level_stimulus_adjusted")
    ].iloc[0]
    visual_adjusted_passes_fdr = bool(
        float(visual_adjusted.q_fdr_metric_family) < 0.05
    )
    add_record(
        records,
        record_id="BLOCK-VISUAL-ADJUSTED",
        module="block_stimulus",
        role="exploratory_covariate_adjusted",
        contrast="Complex−Simple",
        endpoint="within-visual block FC adjusted for stimulus PCs and motion",
        independent_unit="participant × video block",
        n=int(visual_adjusted.n_observations),
        estimate=visual_adjusted.condition_complex_beta,
        estimate_type="mixed-model condition coefficient",
        statistic=visual_adjusted.condition_complex_se,
        statistic_type="standard error",
        p_value=visual_adjusted.condition_complex_p_asymptotic,
        multiplicity="BH-FDR across 10 FC summaries within model",
        q_value=visual_adjusted.q_fdr_metric_family,
        alpha=0.05,
        inference_status=(
            "exploratory_significant"
            if visual_adjusted_passes_fdr
            else "exploratory_not_significant"
        ),
        direction="positive",
        source_file=mixed_path,
        source_filter=(
            "fc_metric == visual_within; model == low_level_stimulus_adjusted"
        ),
        manuscript_location="Supplementary Results—Exploratory block FC",
        conclusion_zh=(
            "探索模型中，刺激调整后的视觉网络内FC条件系数为正，"
            + (
                "并通过模型内FDR。"
                if visual_adjusted_passes_fdr
                else "但未通过模型内FDR。"
            )
        ),
        limitation_zh=(
            "只有8段视频、4对配对视频且存在模型边界警告；该结果不能升级为"
            "视频总体泛化的主结论。"
        ),
    )

    task_control_predefined = predefined[
        predefined.metric.eq("whole_brain_signed_mean")
        & predefined.contrast.eq("task_mean_minus_control")
    ].iloc[0]
    add_record(
        records,
        record_id="CTRL-GLOBAL",
        module="control_contrast",
        role="secondary",
        contrast="mean(Simple, Complex)−Control",
        endpoint="whole-brain signed mean FC",
        independent_unit="participant",
        n=int(task_control_predefined.n_subjects),
        estimate=task_control_predefined.mean_difference,
        estimate_type="mean Fisher-z FC difference",
        ci95_low=task_control_predefined.mean_ci95_low,
        ci95_high=task_control_predefined.mean_ci95_high,
        effect_size=task_control_predefined.cohens_dz,
        effect_size_type="Cohen dz",
        p_value=task_control_predefined.p_sign_flip,
        multiplicity="10,000 paired sign flips",
        q_value=task_control_predefined.q_fdr_within_family,
        alpha=0.05,
        inference_status="secondary_not_significant",
        direction="positive",
        source_file=predefined_path,
        source_filter=(
            "metric == whole_brain_signed_mean; contrast == task_mean_minus_control"
        ),
        manuscript_location="Supplementary Results—Control comparison",
        conclusion_zh="两种任务条件平均与Control之间的全脑平均FC差异不显著。",
        limitation_zh="Control视频在多项刺激属性上不同，因此只作辅助参照。",
    )

    task_edge_path = group_dir / "edgewise/edgewise_task_mean_minus_control.csv.gz"
    task_edges = pd.read_csv(task_edge_path)
    task_significant = task_edges[task_edges.q_fdr.lt(0.05)]
    add_record(
        records,
        record_id="CTRL-EDGES",
        module="control_contrast",
        role="secondary",
        contrast="mean(Simple, Complex)−Control",
        endpoint="edgewise FC",
        independent_unit="participant",
        n=87,
        estimate=len(task_significant),
        estimate_type=(
            f"FDR-significant edges; {sum(task_significant.direction.eq('positive'))} positive, "
            f"{sum(task_significant.direction.eq('negative'))} negative"
        ),
        p_value=task_significant.p.min(),
        multiplicity="BH-FDR across all 30,876 undirected edges",
        q_value=task_significant.q_fdr.min(),
        alpha=0.05,
        inference_status="secondary_significant",
        source_file=task_edge_path,
        source_filter="q_fdr < 0.05",
        manuscript_location="Supplementary Results—Control comparison",
        conclusion_zh=f"任务均值与Control比较有{len(task_significant)}条FDR显著边。",
        limitation_zh="Control刺激与任务视频差异较大，不作为场景复杂度主结论。",
    )

    table = pd.DataFrame(records)
    if table.record_id.duplicated().any():
        duplicates = table.loc[table.record_id.duplicated(), "record_id"].tolist()
        raise RuntimeError(f"Duplicate canonical record IDs: {duplicates}")
    if table.module.str.contains("mvpa", case=False).any():
        raise RuntimeError("MVPA must remain outside the non-MVPA canonical table.")
    missing_sources = [
        source
        for source in table.source_file.unique()
        if not (PROJECT_ROOT.parent / source).exists()
    ]
    if missing_sources:
        raise FileNotFoundError(
            f"Canonical table contains missing source files: {missing_sources}"
        )
    expected_counts = {
        "background_fc_edge": 7,
        "crossnobis_rsa": 4,
        "sensitivity": 8,
        "topology": 4,
    }
    for module, expected in expected_counts.items():
        observed = int(table.module.eq(module).sum())
        if observed != expected:
            raise RuntimeError(
                f"Expected {expected} {module} rows, found {observed}."
            )

    return table

def markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    display = table.loc[:, columns].copy()
    display = display.fillna("")
    for column in display:
        display[column] = display[column].map(
            lambda value: (
                f"{value:.4g}"
                if isinstance(value, (float, np.floating))
                else str(value)
            )
        )
        display[column] = display[column].str.replace("|", "\\|", regex=False)
        display[column] = display[column].str.replace("\n", " ", regex=False)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in display.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_dictionary(table: pd.DataFrame, path: Path) -> None:
    fields = [
        ("record_id", "稳定且唯一的结果编号；正文、图和补充材料应引用此编号。"),
        ("module", "分析模块。"),
        ("role", "primary、secondary、diagnostic或sensitivity等证据角色。"),
        ("contrast", "条件对比及方向。"),
        ("endpoint", "被检验的结局、指标、边或空间结果。"),
        ("independent_unit", "推断所使用的独立统计单位。"),
        ("n", "独立单位数量；必要时写为分层样本结构。"),
        ("estimate / estimate_type", "估计值及其具体含义。"),
        ("interval_low / interval_high", "区间上下界。"),
        ("interval_type", "区间类型，例如95% bootstrap CI或四分位距。"),
        ("effect_size / effect_size_type", "效应量及定义。"),
        ("statistic / statistic_type / df", "检验统计量、类型和自由度。"),
        ("p_value / p_display", "机器可读精确p值及正文显示格式。"),
        ("multiplicity / q_value", "多重比较家族、校正方法和校正后q值。"),
        ("alpha / threshold", "显著性水平及采用的统计图阈值；阈值不是观测p值。"),
        ("inference_status", "冻结后的推断状态，不得按写作需要自行升级。"),
        ("tost_p_lower / tost_p_upper", "TOST两个单侧检验的p值。"),
        ("source_file / source_filter", "来源文件和准确筛选条件。"),
        ("manuscript_location", "推荐写入论文的位置。"),
        ("conclusion_zh", "当前证据允许的中文结论。"),
        ("limitation_zh", "与该结果绑定的解释边界。"),
    ]
    field_rows = pd.DataFrame(fields, columns=["字段", "含义"])
    status = (
        table.groupby(["module", "inference_status"], dropna=False)
        .size()
        .rename("记录数")
        .reset_index()
    )
    content = f"""# 非MVPA唯一结果表：字段说明

本文件说明 `{TABLE_NAME}` 的字段和使用规则。该CSV由正式结果目录直接生成，
是后续论文正文、图注、表格和补充材料中非MVPA统计数字的唯一抄录来源。

## 使用规则

1. 正文或图片中的数字必须能对应到一个 `record_id`。
2. 不从旧DOCX、旧AAL3结果或汇报PPT重新抄录统计数字。
3. `primary` 与 `secondary`、`diagnostic`、`sensitivity` 不得混为同等级证据。
4. p值必须同时保留检验家族和校正方式；不能只复制p值或星号。
5. 每个结果的结论必须与同一行的 `limitation_zh` 一起核对。
6. MVPA不在此表中；其10,000次结构化置换完成后另行审核，再并入最终总表。

## 字段

{markdown_table(field_rows, ["字段", "含义"])}

## 当前记录构成

{markdown_table(status, ["module", "inference_status", "记录数"])}

## 固定论文口径

- 当前分析纳入87名被试，每名被试4个功能run。
- FC中的FD删失仅指纳入run内部对高FD时间点的处理，不是删除整个run。
"""
    path.write_text(content, encoding="utf-8")


def write_audit(table: pd.DataFrame, run_dir: Path, path: Path) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    global_fc = require_one(
        table, context="canonical global FC", record_id="FC-WHOLE_BRAIN_SIGNED_MEAN"
    )
    nbs = require_one(table, context="canonical NBS", record_id="NBS-MAIN")
    edge_rows = table[table.module.eq("background_fc_edge")]
    rsa_rows = table[table.module.eq("crossnobis_rsa")]
    topology = table[table.module.eq("topology")]
    sensitivity_rows = table[table.module.eq("sensitivity")]
    activation = require_one(table, context="canonical activation", record_id="ACT-00")
    activation_omnibus = require_one(
        table,
        context="canonical activation omnibus",
        record_id="ACT-OMNI-00",
    )
    activation_omnibus_largest = (
        table[
            (table.module == "activation_omnibus")
            & (table.role == "secondary_detail")
        ]
        .sort_values("estimate", ascending=False)
        .iloc[0]
    )
    stimulus = require_one(table, context="canonical stimulus", record_id="STIM-01")
    qc_fc = require_one(table, context="canonical QC-FC", record_id="QCFC-01")
    block_global = require_one(table, context="adjusted global block FC", record_id="BLOCK-GLOBAL-ADJUSTED")
    block_visual = require_one(table, context="adjusted visual block FC", record_id="BLOCK-VISUAL-ADJUSTED")
    nbs_nodes = int(nbs.estimate_type.split()[2])
    edge_positive = int(edge_rows.direction.eq("positive").sum())
    edge_negative = int(edge_rows.direction.eq("negative").sum())
    edge_source = pd.read_csv(
        run_dir / "group_statistics/edgewise/edgewise_complex_minus_simple.csv.gz"
    ).query("q_fdr < 0.05")
    edge_cerebellar = int(
        (edge_source.node1_domain.eq("Cerebellum") | edge_source.node2_domain.eq("Cerebellum")).sum()
    )
    stimulus_pairwise = pd.read_csv(
        run_dir / "stimulus/characterization/stimulus_pairwise_tests.csv"
    )
    n_stimulus_metrics = int(stimulus_pairwise.metric.nunique())
    n_stimulus_tests = len(stimulus_pairwise)
    stimulus_omnibus = pd.read_csv(
        run_dir / "stimulus/characterization/stimulus_omnibus_tests.csv"
    )
    n_stimulus_omnibus_significant = int(stimulus_omnibus.q_fdr.lt(0.05).sum())
    primary_table = pd.concat(
        [
            table[table.record_id.isin(["STIM-01", "ACT-OMNI-00", "ACT-00", "SL-01"])],
            rsa_rows,
            table[table.record_id.eq("FC-WHOLE_BRAIN_SIGNED_MEAN")],
            table[table.module.eq("background_fc_network")],
            table[table.module.eq("background_fc_edge")],
            table[table.record_id.eq("NBS-MAIN")],
            topology,
        ],
        ignore_index=True,
    )

    def result_summary(row: pd.Series) -> str:
        parts = [f"estimate={row.estimate}"]
        if row.interval_type:
            parts.append(
                f"{row.interval_type}=[{row.interval_low}, {row.interval_high}]"
            )
        if row.effect_size != "":
            parts.append(f"{row.effect_size_type}={row.effect_size}")
        if row.statistic != "":
            parts.append(f"{row.statistic_type}={row.statistic}")
        if row.p_display:
            parts.append(f"p={row.p_display}")
        if row.q_value != "":
            parts.append(f"q={row.q_value}")
        if row.tost_p_lower != "" and row.tost_p_upper != "":
            parts.append(f"TOST p=({row.tost_p_lower}, {row.tost_p_upper})")
        return "; ".join(parts)

    primary_display = primary_table.assign(
        result=primary_table.apply(result_summary, axis=1)
    )
    risk_rows = pd.DataFrame(
        [
            {
                "级别": "P1",
                "问题": "空间检验方向与跨图多重性",
                "证据": (
                    "激活只检验Complex>Simple；Searchlight每张图先做TFCE-FWER，"
                    "本表另采用三张图Bonferroni阈值。"
                ),
                "处理": (
                    "正文明确单向假设；Searchlight主图使用Bonferroni3结果，"
                    "另外两项对比放补充。"
                ),
            },
            {
                "级别": "P1",
                "问题": "残余头动关联",
                "证据": (
                    f"三条件平均QC-FC有{int(qc_fc.estimate)}条边通过FDR，"
                    f"中位|r|={float(qc_fc.effect_size):.3f}。"
                ),
                "处理": (
                    "保留24P、FD删失和敏感性分析；论文不得写头动影响已完全去除。"
                ),
            },
            {
                "级别": "P1",
                "问题": "视频层泛化有限",
                "证据": (
                    "Simple和Complex各4段视频；4对视频精确检验p=0.25；"
                    f"刺激调整后全局beta={float(block_global.estimate):.5f}，"
                    f"p={block_global.p_display}。"
                ),
                "处理": "中心结论限定于当前自然视频集合，区块模型标为探索性。",
            },
            {
                "级别": "P1",
                "问题": "组合图谱的小脑节点体素过少",
                "证据": "Buckner节点233与246重采样后各含2个3-mm体素。",
                "处理": (
                    "保留纯Schaefer-200敏感性；小脑特异性结论不作为论文主张。"
                ),
            },
            {
                "级别": "P1",
                "问题": "拓扑稳定表述不成立",
                "证据": (
                    f"4项主要拓扑指标仅"
                    f"{int(topology.inference_status.eq('equivalent').sum())}项通过TOST。"
                ),
                "处理": "仅报告模块度等效，不写整体网络骨架稳定。",
            },
            {
                "级别": "P2",
                "问题": "三条件激活omnibus的预设状态待确认",
                "证据": (
                    "87名被试、每条件4个session平均、10,000次受限置换的正式"
                    "omnibus F已完成，但运行元数据不能证明该分析已预先设定。"
                ),
                "处理": (
                    "当前明确标为secondary且保持无方向解释；由作者确认预设状态"
                    "及最终放在Extended Data还是Supplementary Information。"
                ),
            },
            {
                "级别": "P2",
                "问题": "预定义FC存在多个不同大小的FDR家族",
                "证据": (
                    "global=1、domain=3、hemisphere=1、network-within=17、"
                    "network-separation=17、network-between=136。"
                ),
                "处理": (
                    "每个结果紧跟家族名称；网络级发现不得写成全175项统一FDR。"
                ),
            },
            {
                "级别": "P2",
                "问题": (
                    "探索模型出现FDR显著视觉网络内系数"
                    if float(block_visual.q_value) < 0.05
                    else "探索模型视觉网络内系数未通过FDR"
                ),
                "证据": (
                    f"视觉网络内FC调整后beta={float(block_visual.estimate):.5f}，"
                    f"p={block_visual.p_display}，q={float(block_visual.q_value):.4f}。"
                ),
                "处理": (
                    "完整报告未校正p值和FDR q值；由于只有8段视频、模型未完全收敛、"
                    "存在边界警告且视频层精确检验不显著，不作为确认性泛化证据。"
                ),
            },
        ]
    )
    content = f"""# 非MVPA统计审计

生成时间：{generated}
正式结果目录：`{relative_source(run_dir)}`
机器可读唯一结果表：[`{TABLE_NAME}`]({TABLE_NAME})

## 一、审计范围与固定边界

本次使用 `nature-statistics` 的统计报告框架，对刺激属性、三条件激活omnibus、单向激活、
Crossnobis RSA、Searchlight、背景功能连接、预定义网络指标、边级FDR、NBS、
拓扑与TOST、FC敏感性、QC-FC、区块ICC和刺激调整模型进行审计。MVPA分类及其
10,000次结构化置换明确排除在本次审计之外。

论文固定口径如下：

- 当前分析纳入87名被试，每名被试4个功能run。
- FC分析中的FD删失是纳入run内部的时间点处理，不是删除整个run。
- 独立推断单位以被试为主；刺激泛化分析以视频或配对视频为单位。
- 主对比统一写作 `Complex−Simple`；距离指标写作 `Simple–Complex distance`，
  因为距离没有正负方向。

## 二、总体结论

未发现会直接推翻非MVPA主结论的P0级错误。当前证据支持以下受限结论：

> 在当前自然动态视频集合中，Complex相较Simple伴随视觉—注意系统平均活动和
> 多体素表征的改变。全脑平均背景FC没有可靠的普遍增强，但边级结果表现为少量
> 局部增强与降低并存，并出现跨参数方案保留的分布式负向NBS成分。该模式应描述
> 为功能连接重配，而不是整体拓扑稳定、全脑通信增强或特定小脑通路机制。

核心数字为：三条件重复测量omnibus F检出
{int(activation_omnibus.estimate)}个TFCE-FWER显著簇；Complex>Simple激活
{int(activation.estimate)}个TFCE-FWER显著簇；
4个预定义功能网络的Simple–Complex Crossnobis距离均通过FDR；全脑平均FC差异
为{float(global_fc.estimate):.5f}（dz={float(global_fc.effect_size):.3f}，
p={global_fc.p_display}）；30,876条边中{len(edge_rows)}条通过FDR；主NBS负向成分
含{nbs_nodes}个节点、
{int(nbs.estimate)}条边（FWER p={nbs.p_display}）；{len(sensitivity_rows)}/{len(sensitivity_rows)}种处理变体保留显著负向
NBS成分。拓扑上仅模块度通过预设dz±0.20的等效检验。

## 三、主要统计结果清单

{markdown_table(primary_display, ["record_id", "module", "contrast", "endpoint", "n", "result", "inference_status"])}

## 四、统计设计审计

### 4.1 统计单位与重复测量

- 激活、RSA、Searchlight、FC、NBS和拓扑均以87名被试为独立单位，并使用被试内
  条件差值或符号翻转，符合重复测量结构。
- 30,876条边、体素、图谱节点和每名被试的多个区块均不是独立样本量。
- 区块混合模型含696个被试×视频观测，但独立信息仍受87名被试和8段视频限制；
  因此不能把 `n=696` 当作696个独立样本。
- 刺激属性以视频为单位，Simple和Complex各4段；语义检测的采样帧不是独立视频。

### 4.2 多重比较

- 三条件激活omnibus采用10,000次受限置换和被试exchangeability blocks；
  方向性激活图采用10,000次被试内符号翻转。两者均以TFCE控制图内全脑FWER。
- RSA：4个预设网络×3项距离构成12项主检验，采用BH-FDR。
- Searchlight：每张图采用TFCE-FWER；主表进一步使用 `0.05/3` 阈值控制三项距离图。
- 预定义FC：按事先定义的指标家族和对比分别BH-FDR；家族大小必须随结果报告。
- 边级FC：Complex−Simple在全部30,876条无向边上统一BH-FDR。
- NBS：正负成分分别形成；每次置换取两个方向全部成分中的最大质量，以单一零分布
  联合控制双侧FWER。主阈值|t|≥3.1，2.5和3.5只作阈值敏感性。
- 拓扑：四项主要指标在同一对比内BH-FDR；等效性用dz±0.20双单侧TOST。

### 4.3 效应量和区间

- 被试内条件差异使用Cohen dz；全局和预定义FC同时报告均值差及bootstrap 95% CI。
- 刺激属性使用Hedges g，但每组只有4段视频，其CI很宽，不作为刺激等效证据。
- NBS报告成分质量、节点数、边数和FWER p；不能为成分内每条边赋予单边显著性。
- ICC表的q25–q75是四分位距，不得误写成95%置信区间。

## 五、审计问题与处理

{markdown_table(risk_rows, ["级别", "问题", "证据", "处理"])}

## 六、分模块裁决

### 刺激属性

Simple与Complex的{n_stimulus_metrics}项视频级指标均未通过全{n_stimulus_tests}项配对检验族FDR；最小q值为
{float(stimulus.q_value):.4f}。这只能表述为“未检出经校正差异”，不能表述为
“刺激完全匹配”。三条件总体检验有{n_stimulus_omnibus_significant}/{len(stimulus_omnibus)}项通过FDR，主要由Control视频差异驱动，
因此Control只作辅助参照。

### 单变量激活、RSA与Searchlight

三条件重复测量omnibus F检出{int(activation_omnibus.estimate)}个
TFCE-FWER显著簇；其最大簇含{int(activation_omnibus_largest.estimate):,}体素，
峰值F={float(activation_omnibus_largest.statistic):.2f}。该分析当前按secondary
结果报告，且总体F检验不提供特定条件对方向。Complex>Simple方向性检验检出
{int(activation.estimate)}个TFCE-FWER显著簇，主峰涉及双侧视觉皮层、
右侧背侧注意和额叶控制区域。四个预定义网络的Simple–Complex Crossnobis距离均为正并通过FDR；
Searchlight在三张距离图Bonferroni后仍保留Simple–Complex显著体素。三类结果分别
回答平均活动、网络表征距离和局部空间定位，不能相互替代。

### 背景FC、边与NBS

全脑和跨域平均FC均未显示可靠Complex−Simple差异。预定义网络指标中，
SalVentAttnB网络内FC、DefaultA网络内FC和DefaultA网络分离度在各自家族内通过FDR，
方向均为降低。边级全家族FDR检出{len(edge_rows)}条边，其中{edge_positive}条增强、{edge_negative}条降低；
有{edge_cerebellar}条显著边涉及小脑节点。NBS主分析检出{nbs_nodes}节点、{int(nbs.estimate)}边的{nbs.direction}成分，{len(sensitivity_rows)}/{len(sensitivity_rows)}种敏感性处理均保留
负向成分。因此中心结论应是局部增减并存、以分布式降低为主的网络重配。

### 拓扑、运动与可靠性

四项主要拓扑指标只有模块度通过TOST，不能写“整体拓扑稳定”。残余QC-FC较明显，
但该诊断考察三条件平均FC与平均FD的关系，不直接检验Complex−Simple差值；因此应
同时报告它和8种稳健性结果，而不是据此完全否定或完全豁免条件效应。区块边级ICC
中位数为0.344–0.360，提示单个短区块的边估计可靠性有限。

### 刺激调整探索模型

固定8段视频的全局差异不显著；4对视频精确检验p=0.25；调整低级视觉PC、FD和删失
后全局beta={float(block_global.estimate):.5f}、p={block_global.p_display}。视觉网络内
FC调整系数方向为正，但q={float(block_visual.q_value):.4f}，未通过模型内FDR。该模型
未完全收敛并存在边界警告，且只有8段视频、视频层精确检验不显著，因此不作为确认性
泛化证据。

## 七、可直接用于论文的统计口径

### Methods简写

当前分析纳入87名被试，每名被试4个功能run。全脑背景功能连接使用249节点组合图谱
估计。ROI时间序列在四个run联合模型中回归run截距、24参数头动项、128秒高通DCT项、
FD>0.5 mm时间点对应的尖峰回归量以及条件特异FIR任务反应。随后按条件拼接保留的
区块时间点，计算Pearson相关并进行Fisher-z变换。被试内条件差异主要通过10,000次
符号翻转检验评估；边级比较使用BH-FDR，连通子网络使用最大成分质量NBS控制FWER。

### Results中心句

Complex相较Simple并未表现为全脑平均FC的普遍增强，而呈现少量局部连接增强与降低
并存，并伴随跨阈值和处理方案保留的分布式负向NBS成分。这一结果更符合网络重配，
而非整体网络骨架稳定或某条小脑通路的选择性增强。

### 必须避免

- “刺激完全匹配”；
- “头动影响已完全去除”；
- “整体拓扑稳定”；
- “NBS成分中的每条边均显著”；
- “已经证实楔前叶—小脑VI增强”；
- 由关联结果推出因果机制或所有自然场景的泛化。

## 八、尚待并入最终总表的唯一模块

MVPA分类及10,000次完整结构化置换未纳入本审计。其运行完成后，应以相同字段结构
新增MVPA记录，核对观察balanced accuracy、完整零分布、正式置换p值、外层折独立性、
置信区间和训练内预处理，再生成包含全部模块的最终唯一结果表。
"""
    path.write_text(content, encoding="utf-8")


def generate(run_dir: Path, report_dir: Path) -> tuple[Path, Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    table = collect_records(run_dir)
    table_path = report_dir / TABLE_NAME
    dictionary_path = report_dir / DICTIONARY_NAME
    audit_path = report_dir / AUDIT_NAME
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    write_dictionary(table, dictionary_path)
    write_audit(table, run_dir, audit_path)
    return table_path, dictionary_path, audit_path


def parser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser_.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser_


def main() -> None:
    args = parser().parse_args()
    paths = generate(args.run_dir, args.report_dir)
    print("Generated non-MVPA audit artifacts:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
