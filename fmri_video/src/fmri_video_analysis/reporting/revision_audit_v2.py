"""Generate the final Chinese experiment and teacher-comment implementation audit."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fmri_video_analysis.core.config import PROJECT_ROOT, RESULTS_DIR
from fmri_video_analysis.reporting.contracts import require_formal_mvpa

DEFAULT_RUN_DIR = RESULTS_DIR
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "docs/reports/完善实验方案与意见落实审计_v2.html"
REFERENCE_SEARCHLIGHT_DIR = (
    PROJECT_ROOT.parent
    / "collaborator_handoff/fmri_video_20260810/daima/results/"
    "searchlight_crossnobis"
)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_p(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    if value < 0.0001:
        return "&lt;0.0001"
    return f"{value:.4f}"


def badge(label: str, kind: str) -> str:
    return f'<span class="badge {kind}">{html.escape(label)}</span>'


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def relative_asset(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path, output.parent)).as_posix()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

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



def searchlight_replication_audit(current_dir: Path) -> dict:
    """Compare deterministic subject maps and inference-level conclusions."""

    distances = ("simple_complex", "simple_control", "complex_control")
    rows = []
    cluster_columns = (
        "cluster_id",
        "n_voxels",
        "volume_mm3",
        "peak_t",
        "mni_x",
        "mni_y",
        "mni_z",
        "atlas_label",
    )
    for distance in distances:
        current_map = current_dir / f"{distance}_subjects_4d.nii.gz"
        reference_map = REFERENCE_SEARCHLIGHT_DIR / current_map.name
        current_clusters = current_dir / f"{distance}_tfce_fwer05_clusters.csv"
        reference_clusters = REFERENCE_SEARCHLIGHT_DIR / current_clusters.name
        required = (
            current_map,
            reference_map,
            current_clusters,
            reference_clusters,
        )
        if not all(path.is_file() for path in required):
            return {"complete": False, "comparisons": rows}

        current_img = nib.load(str(current_map))
        reference_img = nib.load(str(reference_map))
        if current_img.shape != reference_img.shape:
            maximum_difference = np.inf
        else:
            current_data = np.asarray(current_img.dataobj, dtype=np.float32)
            reference_data = np.asarray(reference_img.dataobj, dtype=np.float32)
            maximum_difference = float(
                np.max(np.abs(current_data - reference_data))
            )

        current_table = pd.read_csv(current_clusters).sort_values("cluster_id")
        reference_table = pd.read_csv(reference_clusters).sort_values("cluster_id")
        cluster_match = len(current_table) == len(reference_table)
        if cluster_match:
            for column in cluster_columns:
                current_values = current_table[column].to_numpy()
                reference_values = reference_table[column].to_numpy()
                if np.issubdtype(current_values.dtype, np.number):
                    matches = np.allclose(
                        current_values,
                        reference_values,
                        rtol=0,
                        atol=1e-6,
                    )
                else:
                    matches = np.array_equal(current_values, reference_values)
                cluster_match = cluster_match and bool(matches)
        rows.append(
            {
                "distance": distance,
                "subject_map_max_abs_difference": maximum_difference,
                "fwer_cluster_conclusion_match": cluster_match,
                "n_current_fwer_clusters": len(current_table),
                "n_reference_fwer_clusters": len(reference_table),
            }
        )
    return {
        "complete": True,
        "subject_maps_exact": all(
            row["subject_map_max_abs_difference"] == 0 for row in rows
        ),
        "fwer_cluster_conclusions_match": all(
            row["fwer_cluster_conclusion_match"] for row in rows
        ),
        "comparisons": rows,
    }


def write_source_provenance(run_dir: Path) -> dict:
    """Freeze checksums for the source tree used by the final audit."""

    source_files = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    source_files.extend(
        path
        for path in (PROJECT_ROOT / "pyproject.toml",)
        if path.is_file()
    )
    entries = []
    aggregate = hashlib.sha256()
    for path in sorted(source_files):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(PROJECT_ROOT.parent).as_posix()
        entries.append({"path": relative, "sha256": digest})
        aggregate.update(f"{relative}\0{digest}\n".encode())

    def git_output(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = git_output(
        "status",
        "--short",
        "--",
        "fmri_video/src",
        "fmri_video/pyproject.toml",
    )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_output("rev-parse", "HEAD"),
        "scoped_worktree_dirty": bool(status),
        "scoped_git_status": status.splitlines(),
        "source_manifest_sha256": aggregate.hexdigest(),
        "files": entries,
    }
    provenance_path = run_dir / "metadata/source_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def generate(run_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stimulus_dir = run_dir / "stimulus/characterization"
    activation_dir = run_dir / "activation/group_contrasts"
    group_dir = run_dir / "group_statistics"
    sensitivity_dir = run_dir / "sensitivity"

    stimulus = pd.read_csv(stimulus_dir / "stimulus_pairwise_tests.csv")
    stimulus = stimulus[stimulus.contrast == "complex_minus_simple"].set_index("metric")
    activation = pd.read_csv(activation_dir / "activation_summary.csv").set_index(
        "contrast"
    )
    metrics = pd.read_csv(
        group_dir / "predefined_metrics/predefined_metric_statistics.csv"
    )
    global_row = require_one(
        metrics,
        context="whole-brain FC",
        metric="whole_brain_signed_mean",
        contrast="complex_minus_simple",
    )
    edge_summary = load_json(group_dir / "edgewise/edgewise_summary.json")
    significant_edges = pd.read_csv(
        group_dir / "edgewise/edgewise_complex_minus_simple.csv.gz"
    ).query("q_fdr < 0.05")
    n_positive_edges = int(significant_edges.direction.eq("positive").sum())
    n_negative_edges = int(significant_edges.direction.eq("negative").sum())
    n_cerebellar_edges = int(
        (
            significant_edges.node1_domain.eq("Cerebellum")
            | significant_edges.node2_domain.eq("Cerebellum")
        ).sum()
    )
    nbs = pd.read_csv(group_dir / "nbs/nbs_complex_minus_simple_components.csv")
    nbs_main = require_one(
        nbs[np.isclose(nbs.threshold_abs_t, 3.1)],
        context="primary NBS",
        direction="negative",
        significant_fwer_0_05=True,
    )
    topology = pd.read_csv(group_dir / "topology/topology_statistics.csv")
    topology_main = topology[
        (topology.contrast == "complex_minus_simple")
        & (topology.metric_role == "primary")
    ]
    qc_fc = load_json(sensitivity_dir / "qc_fc/qc_fc_summary.json")
    fd_global = pd.read_csv(sensitivity_dir / "qc_fc/mean_fd_global_fc.csv")
    reliability = pd.read_csv(
        sensitivity_dir / "block_reliability/block_reliability_summary.csv"
    )
    block_summary = load_json(
        sensitivity_dir / "block_stimulus/block_stimulus_summary.json"
    )
    run_metadata = load_json(group_dir / "metrics_run_metadata.json")
    runtime = run_metadata["runtime"]
    source_provenance = write_source_provenance(run_dir)
    rsa = pd.read_csv(run_dir / "rsa/crossnobis/crossnobis_summary.csv")
    rsa_visual = rsa[
        (rsa.network == "visual") & (rsa.distance == "simple_complex")
    ].iloc[0]

    mvpa_path = run_dir / "decoding/cross_video/svm/metrics.json"
    mvpa = require_formal_mvpa(mvpa_path)
    mvpa_status = badge("服务器正式复现完成", "ok")
    mvpa_structured_p = fmt_p(mvpa.get("structured_label_permutation_p", np.nan))

    searchlight_path = run_dir / "rsa/searchlight/searchlight_summary.csv"
    searchlight_complete = searchlight_path.exists()
    if searchlight_complete:
        searchlight = pd.read_csv(searchlight_path)
        searchlight_primary = searchlight[
            searchlight.distance == "simple_complex"
        ].iloc[0]
        searchlight_audit = searchlight_replication_audit(searchlight_path.parent)
        (searchlight_path.parent / "searchlight_replication_audit.json").write_text(
            json.dumps(searchlight_audit, indent=2),
            encoding="utf-8",
        )
        searchlight_reproduced = (
            searchlight_audit.get("complete", False)
            and searchlight_audit["subject_maps_exact"]
            and searchlight_audit["fwer_cluster_conclusions_match"]
        )
        audit_rows = searchlight_audit.get("comparisons", [])
        matched_contrasts = sum(
            row["fwer_cluster_conclusion_match"] for row in audit_rows
        )
        searchlight_text = (
            "三项被试级距离图与交接结果逐值一致；固定种子10,000次"
            f"TFCE-FWER后，{matched_contrasts}/3项显著簇结论一致。"
            "原交接运行未固定种子，因此校正p图不要求逐值相同；"
            f"Simple–Complex显著簇数="
            f"{int(searchlight_primary.mapwise_significant_clusters)}。"
        )
        searchlight_status = (
            badge("服务器结论复现通过", "ok")
            if searchlight_reproduced
            else badge("服务器复现存在差异", "warn")
        )
    else:
        searchlight_text = "87人距离图已生成，固定种子TFCE-FWER仍在执行。"
        searchlight_status = badge("正在执行", "warn")

    sensitivity_rows = []
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
    for variant in variants:
        component_path = (
            sensitivity_dir
            / variant
            / "nbs/nbs_complex_minus_simple_components.csv"
        )
        components = pd.read_csv(component_path)
        row = components[
            (np.isclose(components.threshold_abs_t, 3.1))
            & (components.direction == "negative")
            & components.significant_fwer_0_05
        ].iloc[0]
        sensitivity_rows.append(
            [variant, int(row.n_nodes), int(row.n_edges), fmt_p(row.p_fwer)]
        )

    topology_rows = []
    label_map = {
        "strength_auc": "平均强度AUC",
        "global_efficiency_auc": "全局效率AUC",
        "modularity_auc": "模块度AUC",
        "network_separation_auc": "网络分离度AUC",
    }
    for row in topology_main.itertuples():
        topology_rows.append(
            [
                label_map[row.metric],
                fmt(row.mean_difference, 5),
                fmt(row.cohens_dz),
                fmt_p(row.q_fdr_by_role_contrast),
                badge("等效", "ok")
                if row.tost_equivalent_dz_0_20
                else badge("未建立等效", "warn"),
            ]
        )

    teacher_rows = [
        [
            "1",
            "撤销驾驶叙事，聚焦自然动态场景目标跟踪与网络重组",
            "同学的新报告弱化驾驶应用；本轮冻结任务为“动态视觉目标跟踪”。",
            badge("已落实", "ok"),
            "论文引言和标题仍需按新核心问题统一改写。",
        ],
        [
            "2",
            "参考Graumann等（2022）",
            "同学借鉴了跨刺激MVPA/RSA/searchlight逻辑。",
            badge("意见合理但需限界", "info"),
            "该文是静态图像表征研究，不能直接支持动态FC、小脑或因果结论。",
        ],
        [
            "3",
            "统一全文样本、条件和结果数字",
            "87人、4 run、12视频、130 TR及对比方向已在代码和结果中冻结。",
            badge("分析层已完成", "ok"),
            "论文正文尚需从本次正式输出自动回填，不能继续沿用旧AAL3数值。",
        ],
        [
            "4",
            "自动流程；AAL3改为Schaefer-200",
            "同学在MVPA中使用Schaefer-200；本轮建立Schaefer-200+Tian-32+Buckner-17组合图谱和参数化流水线。",
            badge("已完成并修正", "ok"),
            "组合图谱是为补足皮层下和小脑覆盖；Schaefer-200单独作为图谱敏感性。",
        ],
        [
            "5",
            "行为数据大量缺失，删除行为结果",
            "同学未继续做RT条件推断；本轮所有主分析不依赖行为成绩。",
            badge("部分落实", "warn"),
            "仍需行为文件或缺失审计。方法必须说明按键任务，局限必须说明无法验证行为/眼动依从性。",
        ],
        [
            "6",
            "增加低级视觉和语义指标",
            "同学完成亮度、对比度、熵、边缘、帧差和目标检测；本轮新增光流、全局运动、残余光流及区块FC刺激调整。",
            badge("主体完成", "ok"),
            "语义检测仍需人工QC；遮挡指标需目标轨迹或逐帧标注。不能把非显著写成完全匹配。",
        ],
        [
            "7",
            "增加Complex/Simple MVPA",
            f"同学报告BA=0.622；服务器固定种子复现BA={fmt(mvpa['balanced_accuracy'])}，训练/测试同时隔离被试和视频。",
            mvpa_status,
            f"正式结构化置换p={mvpa_structured_p}；被试符号翻转只作补充。",
        ],
        [
            "8",
            "补全跨半球全脑FC",
            "同学未完成连续BOLD全脑FC；本轮已完成87×3×249×249背景FC，覆盖跨半球、皮层下和小脑。",
            badge("本轮已完成", "ok"),
            "旧半球内AAL3结果降为历史探索，不再承担正式结论。",
        ],
        [
            "9",
            "增加网络拓扑，检验骨架稳定",
            "本轮完成10%–30%密度AUC、置换、FDR和dz±0.20 TOST。",
            badge("分析完成，原预期不成立", "warn"),
            "只有模块度通过等效性，不能写“整体拓扑稳定”。",
        ],
        [
            "10",
            "按刺激→激活/MVPA→整体FC→边→拓扑组织",
            "五段证据链现均有正式结果。",
            badge("分析已具备", "ok"),
            "下一步按该顺序重写论文，结果方向必须服从数据。",
        ],
        [
            "11",
            "避免描述性结果汇编",
            "UMAP、beta-series和大规模ROI筛选被降为补充；正文保留与核心问题直接相关的检验。",
            badge("方案已落实", "ok"),
            "写作时继续限制网络/边表格数量，机器可读全集放补充。",
        ],
        [
            "12",
            "暂定稳定骨架+后部视觉空间—小脑增强",
            (
                "新结果显示全局均值无明显差异，但有稳健的分布式负向NBS成分"
                f"及少量双侧视觉正向边；{len(significant_edges)}条FDR边中有"
                f"{n_cerebellar_edges}条涉及小脑节点。"
            ),
            badge("需实质修正", "bad"),
            "改为“全局平均无明显改变、局部增减并存且以分布式连接降低为主”；不得保留楔前叶—小脑VI为已证实主结论。",
        ],
    ]

    colleague_rows = [
        [
            "刺激属性",
            "12视频、视频级统计、低级视觉及自动语义检测",
            badge("服务器复算通过并扩展", "ok"),
            "新增光流/相机运动；人工语义QC和遮挡仍缺。",
        ],
        [
            "单变量激活",
            "4 session一阶模型；Complex−Simple TFCE-FWER",
            badge("高度接近但非逐值相同", "warn"),
            (
                "固定种子服务器结果"
                f"{int(activation.loc['complex_minus_simple', 'n_significant_clusters'])}簇。"
            ),
        ],
        [
            "二分类MVPA",
            "视觉ROI线性SVM；跨被试且跨视频",
            mvpa_status,
            f"服务器BA={fmt(mvpa['balanced_accuracy'])}，95% CI [{fmt(mvpa['balanced_accuracy_bootstrap_95_ci'][0])}, {fmt(mvpa['balanced_accuracy_bootstrap_95_ci'][1])}]。",
        ],
        [
            "Crossnobis RSA",
            "5个网络、4 session独立partition",
            badge("服务器逐表复现通过", "ok"),
            f"视觉Simple–Complex距离={fmt(rsa_visual.mean_crossnobis)}，q={fmt_p(rsa_visual.q_fdr_primary)}。",
        ],
        [
            "全脑searchlight",
            "9 mm、三项距离、TFCE-FWER",
            searchlight_status,
            searchlight_text,
        ],
        [
            "UMAP/t-SNE、三分类、分类器比较",
            "同学已完成多项补充分析",
            badge("交接结果保留", "info"),
            "与网络重组核心问题关系较弱，不必优先服务器重跑，放补充材料。",
        ],
        [
            "每条件4 beta的网络协变",
            "同学称beta-series connectivity",
            badge("不作为FC证据", "bad"),
            "4个观测无法替代连续BOLD背景FC；本轮249节点时域FC已补齐。",
        ],
    ]

    stimulus_rows = []
    stimulus_labels = {
        "luminance_mean": "平均亮度",
        "rms_contrast_mean": "RMS对比度",
        "entropy_mean": "图像熵",
        "edge_density_mean": "边缘密度",
        "optical_flow_mean_mean": "平均光流",
        "global_motion_median_mean": "全局运动代理",
        "residual_flow_mean_mean": "残余光流",
        "mean_target_count": "自动检测目标均数",
    }
    for metric, label in stimulus_labels.items():
        row = stimulus.loc[metric]
        stimulus_rows.append(
            [
                label,
                fmt(row.mean_difference, 4),
                fmt(row.hedges_g),
                fmt_p(row.p_exact_permutation),
                fmt_p(row.q_fdr_all),
            ]
        )

    fc_rows = [
        [
            "全脑平均FC",
            fmt(global_row.mean_difference, 5),
            fmt(global_row.cohens_dz),
            fmt_p(global_row.p_sign_flip),
            "无普遍增强；均值方向为降低",
        ],
        [
            "边水平FDR",
            str(edge_summary["complex_minus_simple"]["n_fdr_significant"]),
            f"最大|dz|={fmt(edge_summary['complex_minus_simple']['max_abs_dz'])}",
            f"min q={fmt_p(edge_summary['complex_minus_simple']['minimum_q'])}",
            (
                f"{n_positive_edges}条正向、{n_negative_edges}条负向；"
                "正向集中在双侧周边视觉皮层"
            ),
        ],
        [
            "NBS |t|≥3.1",
            f"{int(nbs_main.n_nodes)}节点/{int(nbs_main.n_edges)}边",
            f"mass={fmt(nbs_main.component_mass, 1)}",
            f"FWER p={fmt_p(nbs_main.p_fwer)}",
            "显著负向分布式成分",
        ],
        [
            "区块FC+刺激调整",
            fmt(block_summary["global_stimulus_adjusted_beta"], 5),
            "8视频、4配对",
            f"探索性p={fmt_p(block_summary['global_stimulus_adjusted_asymptotic_p'])}",
            "调整后全局条件系数减半；视频泛化精确p=0.25",
        ],
    ]

    fd_rows = [
        [row.condition, fmt(row.pearson_r_mean_fd_global_fc), fmt_p(row.p)]
        for row in fd_global.itertuples()
    ]
    icc_rows = [
        [
            row.condition,
            fmt(row.median_icc_3_1),
            fmt(row.q25_icc_3_1),
            fmt(row.q75_icc_3_1),
        ]
        for row in reliability.itertuples()
    ]

    stimulus_figure = relative_asset(stimulus_dir / "stimulus_summary.png", output)
    activation_figure = relative_asset(
        activation_dir / "complex_minus_simple_glass.png", output
    )
    block_figure = relative_asset(
        sensitivity_dir
        / "block_stimulus/block_global_fc_by_stimulus_pc1.png",
        output,
    )
    searchlight_figure_path = (
        run_dir
        / "rsa/searchlight/simple_complex_tfce_fwer05_pycortex_flatmap.png"
    )
    searchlight_figure = (
        f'<figure><img src="{relative_asset(searchlight_figure_path, output)}" alt="Simple-Complex searchlight flatmap"><figcaption>Simple–Complex Crossnobis searchlight，TFCE-FWER校正。</figcaption></figure>'
        if searchlight_figure_path.exists()
        else ""
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>视觉场景复杂度fMRI：实验执行与意见落实审计 v2</title>
  <style>
    :root {{ --ink:#202426; --muted:#606a70; --line:#d9dee1; --paper:#ffffff; --wash:#f4f6f7; --green:#1f6b4f; --amber:#9a5b00; --red:#a23434; --blue:#285b8f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--wash); font:15px/1.68 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; letter-spacing:0; }}
    header {{ background:#202426; color:white; padding:42px max(24px,calc((100vw - 1120px)/2)); }}
    header h1 {{ margin:0 0 10px; max-width:900px; font-size:32px; line-height:1.3; letter-spacing:0; }}
    header p {{ margin:0; color:#d8dfe2; max-width:900px; }}
    nav {{ position:sticky; top:0; z-index:2; background:#fff; border-bottom:1px solid var(--line); overflow:auto; white-space:nowrap; padding:10px max(18px,calc((100vw - 1120px)/2)); }}
    nav a {{ color:var(--blue); text-decoration:none; margin-right:20px; font-size:14px; }}
    main {{ max-width:1120px; margin:0 auto; padding:28px 20px 72px; }}
    section {{ background:var(--paper); border:1px solid var(--line); border-radius:6px; padding:26px; margin:0 0 20px; }}
    h2 {{ font-size:22px; margin:0 0 14px; letter-spacing:0; }}
    h3 {{ font-size:17px; margin:24px 0 10px; letter-spacing:0; }}
    p {{ margin:8px 0 14px; }}
    .lead {{ font-size:17px; color:#354047; }}
    .callout {{ border-left:4px solid var(--blue); background:#f1f5f8; padding:14px 16px; margin:14px 0; }}
    .warning {{ border-left-color:var(--amber); background:#fff8ed; }}
    .danger {{ border-left-color:var(--red); background:#fff3f3; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:16px 0; }}
    .metric {{ border:1px solid var(--line); border-radius:5px; padding:14px; min-height:100px; }}
    .metric strong {{ display:block; font-size:22px; line-height:1.25; margin-bottom:6px; }}
    .metric span {{ color:var(--muted); font-size:13px; }}
    .badge {{ display:inline-block; border-radius:3px; padding:2px 7px; font-size:12px; font-weight:650; white-space:nowrap; }}
    .ok {{ color:var(--green); background:#eaf5ef; }} .warn {{ color:var(--amber); background:#fff1da; }} .bad {{ color:var(--red); background:#fdeaea; }} .info {{ color:var(--blue); background:#eaf1f8; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:5px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; min-width:760px; }}
    th {{ text-align:left; background:#edf1f2; padding:9px 10px; border-bottom:1px solid var(--line); }}
    td {{ vertical-align:top; padding:9px 10px; border-bottom:1px solid #e7eaec; }}
    tr:last-child td {{ border-bottom:0; }}
    code {{ background:#edf1f2; border-radius:3px; padding:1px 4px; overflow-wrap:anywhere; }}
    ul,ol {{ padding-left:22px; }} li {{ margin:5px 0; }}
    figure {{ margin:18px 0 4px; }} figure img {{ width:100%; max-height:560px; object-fit:contain; border:1px solid var(--line); background:white; }} figcaption {{ color:var(--muted); font-size:13px; margin-top:6px; }}
    a {{ color:var(--blue); }} .small {{ color:var(--muted); font-size:13px; }}
    @media (max-width:800px) {{ .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} section {{ padding:19px; }} header h1 {{ font-size:25px; }} }}
    @media (max-width:480px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>视觉场景复杂度对大脑功能网络重组的影响</h1>
  <p>完整实验执行结果、老师意见落实审计与论文下一步方案 · v2 · {generated}</p>
</header>
<nav><a href="#summary">结论</a><a href="#teacher">意见矩阵</a><a href="#peer">同学工作</a><a href="#methods">正式方案</a><a href="#results">结果</a><a href="#risks">风险</a><a href="#next">下一步</a><a href="#references">依据</a></nav>
<main>
<section id="summary">
  <h2>1. 当前结论与下一步</h2>
  <p class="lead">老师要求的“刺激控制 → 激活/MVPA → 全局FC → 边/NBS → 拓扑”五段分析链已经建立。数据不支持“复杂条件引起全脑连接普遍增强”，也不足以支持“整体拓扑稳定”；更准确的结论是：复杂与简单条件的全局平均FC差异较小，但存在少量局部增强边、更多局部降低边，以及跨多种处理方案稳定出现的分布式负向NBS成分。</p>
  <div class="grid">
    <div class="metric"><strong>{fmt(global_row.mean_difference, 4)}</strong><span>全脑平均FC Complex−Simple；p={fmt_p(global_row.p_sign_flip)}</span></div>
    <div class="metric"><strong>{edge_summary['complex_minus_simple']['n_fdr_significant']}条</strong><span>30,876条边中经BH-FDR显著</span></div>
    <div class="metric"><strong>{int(nbs_main.n_nodes)}节点</strong><span>负向NBS主成分；FWER p={fmt_p(nbs_main.p_fwer)}</span></div>
    <div class="metric"><strong>1 / 4</strong><span>主要拓扑指标通过TOST，仅模块度</span></div>
  </div>
  <div class="callout"><strong>论文中心命题建议：</strong>“复杂度增加没有伴随全脑平均通信的普遍增强，而表现为双侧视觉局部连接增强与更广泛连接降低并存的网络重配。”该表述是关联性结果，不能写成机制因果。</div>
  <div class="callout danger"><strong>必须删除或改写：</strong>“整体网络骨架稳定”“选择性增强后部视觉空间—小脑通路，尤其双侧楔前叶—小脑VI”均未被本次正式全脑结果证实。</div>
</section>

<section id="teacher">
  <h2>2. 老师意见逐条落实</h2>
  {table(["#", "老师意见", "另一位同学已做", "当前状态", "仍需处理"], teacher_rows)}
</section>

<section id="peer">
  <h2>3. 另一位同学的工作与流程审查</h2>
  <p>同学的主要贡献是建立刺激属性、激活、MVPA、RSA和searchlight证据链。流程整体合理，但原报告中的“beta-series connectivity”不能回答老师提出的连续BOLD全脑FC问题；本轮已用背景FC替代。</p>
  {table(["模块", "同学完成内容", "复现/裁决", "定位与剩余"], colleague_rows)}
</section>

<section id="methods">
  <h2>4. 已执行的正式实验方案</h2>
  <h3>设计与样本</h3>
  <ul>
    <li>100名被试中按既有名单排除13名，正式纳入87名；只使用前4个高质量run（0006–0009）。</li>
    <li>每个run交付前已删除最初13个volume，现有130 TR均为有效序列，不再次丢弃；TR=2 s。</li>
    <li>每个run含3个60 s区块；每条件4个区块、4段视频。任务称为“动态视觉目标跟踪”：目光跟随车辆，出现/消失时按键。</li>
    <li>主要对比Complex−Simple；次要对比(Complex+Simple)/2−Control。</li>
  </ul>
  <h3>图谱与FC</h3>
  <ul>
    <li>组合图谱：Schaefer-200/17网络皮层 + Tian S2 32个皮层下区 + Buckner 17个小脑网络，共249节点、30,876条边。三个空间域互斥，Schaefer-200单独作为敏感性。</li>
    <li>FC输入为未平滑<code>wra</code>；24参数运动模型、Power FD&gt;0.5 mm体积删失、run独立128 s高通、35-lag FIR去除任务诱发均值，不做主分析GSR。</li>
    <li>条件窗口延迟2 TR、每区块去均值、同条件区块拼接后计算Pearson相关和Fisher-z；同时保存区块级矩阵。</li>
  </ul>
  <h3>推断</h3>
  <ul>
    <li>全局/网络指标采用10,000次被试内符号翻转、dz与置信区间，并按预定义家族FDR。</li>
    <li>全部边分别BH-FDR；NBS采用|t|≥3.1、component mass、10,000次置换，2.5和3.5作阈值敏感性。</li>
    <li>正权重比例图密度10%–30%、步长2%，跨密度AUC；4个主要拓扑指标同时做差异检验和dz±0.20 TOST。</li>
    <li>固定完成8种FC处理变体、QC-FC、区块ICC及区块FC刺激属性模型。</li>
  </ul>
  <h3>运行环境与可追溯性</h3>
  <ul>
    <li>全部正式命令在Conda <code>{html.escape(str(runtime['conda_environment']))}</code>环境执行；Python {html.escape(str(runtime['python_version']))}，解释器<code>{html.escape(str(runtime['python_executable']))}</code>。</li>
    <li>关键版本：NumPy {runtime['packages']['numpy']}、SciPy {runtime['packages']['scipy']}、nilearn {runtime['packages']['nilearn']}、scikit-learn {runtime['packages']['scikit-learn']}；FSL {runtime['external_tools']['fsl_version']}。</li>
    <li>统一随机种子20260731；源码清单SHA-256为<code>{source_provenance['source_manifest_sha256']}</code>，逐文件校验值保存在<code>metadata/source_provenance.json</code>。</li>
    <li>当前分析源码工作树{'含未提交修改，校验清单已冻结本次实际字节' if source_provenance['scoped_worktree_dirty'] else '无未提交修改'}；正式投稿归档时仍应提交或打版本标签。</li>
  </ul>
</section>

<section id="results">
  <h2>5. 正式结果</h2>
  <h3>5.1 刺激属性</h3>
  <p>所有推断均以视频为独立单位，每条件只有4段视频。Simple–Complex指标没有一项通过全指标FDR，但多个效应量较大且置信区间宽，因此只能写“没有检出校正后差异”，不能写“刺激完全匹配”。</p>
  {table(["指标", "Complex−Simple", "Hedges g", "精确p", "全指标q"], stimulus_rows)}
  <figure><img src="{stimulus_figure}" alt="刺激属性汇总"><figcaption>12段正式视频的低级视觉、运动与自动语义指标。自动检测仍待人工QC。</figcaption></figure>

  <h3>5.2 激活、MVPA、RSA和searchlight</h3>
  <ul>
    <li>固定种子10,000次TFCE-FWER：{'; '.join(f"{contrast} {int(row.n_significant_clusters)}簇" for contrast, row in activation.iterrows())}。</li>
    <li>跨被试且跨视频二分类MVPA：BA={fmt(mvpa['balanced_accuracy'])}，95%被试bootstrap CI [{fmt(mvpa['balanced_accuracy_bootstrap_95_ci'][0])}, {fmt(mvpa['balanced_accuracy_bootstrap_95_ci'][1])}]；结构化置换p={mvpa_structured_p}。</li>
    <li>视觉网络Crossnobis Simple–Complex距离={fmt(rsa_visual.mean_crossnobis)}，q={fmt_p(rsa_visual.q_fdr_primary)}，服务器与交接表最大差不超过2.4×10<sup>−7</sup>。</li>
    <li>{searchlight_text}</li>
  </ul>
  <figure><img src="{activation_figure}" alt="Complex minus Simple activation"><figcaption>Complex−Simple单变量激活，TFCE-FWER p&lt;0.05。</figcaption></figure>
  {searchlight_figure}

  <h3>5.3 全脑FC、边与NBS</h3>
  {table(["分析", "估计/规模", "效应", "校正结果", "解释"], fc_rows)}
  <p>{len(significant_edges)}条FDR边由{n_positive_edges}条正向和{n_negative_edges}条负向组成。最强正向边连接双侧周边视觉皮层；正式显著边中有{n_cerebellar_edges}条涉及小脑节点。主NBS结果是{int(nbs_main.n_nodes)}节点、{int(nbs_main.n_edges)}边的{nbs_main.direction}分布式成分，说明“网络重组”主要不是单一后部视觉—小脑增强通路。</p>

  <h3>5.4 拓扑与等效性</h3>
  {table(["主要指标", "Complex−Simple", "dz", "FDR q", "TOST dz±0.20"], topology_rows)}
  <p>四项主要指标没有在FDR后显示明确Complex−Simple差异，但只有模块度通过等效检验。预设判定要求四项主要指标全部通过TOST，实际未达到；因此不能表述为“整体拓扑稳定”。</p>

  <h3>5.5 稳健性、头动与可靠性</h3>
  <p>8个预定义变体在主NBS阈值均保留显著负向Complex−Simple成分：</p>
  {table(["变体", "节点", "边", "FWER p"], sensitivity_rows)}
  <p>但残余运动关系仍明显：30,876条边中{qc_fc['n_qc_fc_edges_fdr_0_05']:,}条QC-FC关系经FDR显著，中位|r|={fmt(qc_fc['median_absolute_qc_fc_r'])}。平均FD与全局FC的关系如下：</p>
  {table(["条件", "Pearson r", "p"], fd_rows)}
  <p>区块FC可靠性偏低：</p>
  {table(["条件", "ICC中位数", "Q1", "Q3"], icc_rows)}
  <figure><img src="{block_figure}" alt="区块FC与刺激PC1"><figcaption>8个任务视频的全局区块FC与低级视觉PC1。误差线反映被试均值标准误，不代表刺激总体不确定性。</figcaption></figure>
</section>

<section id="risks">
  <h2>6. 解释边界与仍缺材料</h2>
  <ol>
    <li><strong>刺激数：</strong>每条件只有4段视频。87名被试提高被试层精度，不能补偿刺激总体只有4个水平。区块视频配对精确检验的最小双侧p为0.125。</li>
    <li><strong>头动：</strong>24P和FD删失后仍有QC-FC关联。处理变体中的NBS稳定降低了该风险，但不能宣称运动混杂已彻底消除。</li>
    <li><strong>可靠性：</strong>区块边ICC中位数约0.34–0.36，短区块边级结果不宜过度定位。</li>
    <li><strong>行为与眼动：</strong>目前没有可用的行为缺失审计或眼动记录，无法验证每个区块的跟踪投入。</li>
    <li><strong>语义与遮挡：</strong>自动车辆/行人检测尚未人工核对；可靠遮挡率需要目标轨迹或逐帧遮挡标注。</li>
    <li><strong>推断类型：</strong>同一87人数据同时产生旧探索发现和新全脑检验，不能称独立确认；未来需要更多视频或独立样本。</li>
  </ol>
  <div class="callout warning"><strong>仍建议用户补充：</strong>目标轨迹/遮挡标注；行为按键原始表及缺失原因；如存在则提供眼动数据；最初排除后4 run的QC阈值或记录；可选WM/CSF分割用于进一步去噪敏感性。</div>
</section>

<section id="next">
  <h2>7. 从现在开始的论文工作</h2>
  <ol>
    <li><strong>冻结结果：</strong>以<code>{html.escape(str(run_dir))}</code>为唯一正式结果目录，保存10,000次MVPA置换和searchlight完成状态后不再按显著性更换主模型。</li>
    <li><strong>重写结果：</strong>严格按“刺激边界 → 激活/MVPA/RSA → 全局FC → 边/NBS → 拓扑/TOST → 稳健性/QC”排序。正文只保留核心表图。</li>
    <li><strong>重写讨论：</strong>删除“稳定骨架”和“楔前叶—小脑VI已证实”；解释少量视觉增强边与分布式FC降低并存，并明确残余头动和刺激抽样限制。</li>
    <li><strong>补透明报告：</strong>完成行为缺失、语义检测人工QC和run排除依据；没有相应文件时在局限中明确写“不可验证”。</li>
    <li><strong>决定标题：</strong>“网络重组”可保留，但摘要必须把它定义为边/NBS层面的分布重配，而不是全局拓扑稳定或单一路径增强。</li>
  </ol>
</section>

<section id="references">
  <h2>8. 老师意见的合理性与方法依据</h2>
  <ul>
    <li>Graumann M, et al. (2022). <a href="https://www.nature.com/articles/s41562-022-01302-0">Environmental clutter affects visual object position representations</a>. 适合借鉴假设组织和跨刺激解码，不是FC或小脑结论的直接证据。</li>
    <li>Schaefer A, et al. (2018). <a href="https://doi.org/10.1093/cercor/bhx179">Local-global cortical parcellation</a>. 适合皮层网络分析，但单独不覆盖皮层下和小脑。</li>
    <li>Tian Y, et al. (2020). <a href="https://doi.org/10.1038/s41593-020-00711-6">Melbourne Subcortex Atlas</a>.</li>
    <li>Buckner RL, et al. (2011). <a href="https://doi.org/10.1152/jn.00339.2011">Cerebellar intrinsic connectivity networks</a>.</li>
    <li>Cole MW, et al. (2019). <a href="https://pubmed.ncbi.nlm.nih.gov/30597260/">Task activations produce spurious task-state functional connectivity increases</a>. 支持在背景FC前去除任务诱发均值。</li>
    <li>Ciric R, et al. (2017/2018). <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5483393/">Benchmarking confound regression strategies</a>. 支持24P、删失和QC-FC敏感性，但不存在适用于所有数据的单一“标准答案”。</li>
    <li>Zalesky A, et al. (2010). <a href="https://pubmed.ncbi.nlm.nih.gov/20600983/">Network-based statistic</a>. 用于控制连接成分层FWER。</li>
    <li>Lakens D. (2017). <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/">Equivalence tests</a>. 支持区分“不显著”与“足够小到可视为等效”。</li>
    <li>Nichols TE, et al. (2017). <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5685169/">COBIDAS reporting recommendations</a>. 支持透明报告任务、排除、缺失和软件参数。</li>
  </ul>
  <p class="small"><strong>AI辅助说明：</strong>本报告由AI辅助生成并由可执行代码从正式结果表读取数值。AI用于代码实现、审计、统计汇总和写作组织，不替代研究者对原始排除决策、行为缺失、人工语义标注、伦理披露和最终学术解释的责任。</p>
</section>
</main>
</body>
</html>"""
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    command.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return command


if __name__ == "__main__":
    args = parser().parse_args()
    generate(Path(args.run_dir), Path(args.output))
