"""Exploratory block-level FC models with video-level stimulus covariates."""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fmri_video_analysis.core.config import SESSION_VIDEOS
from fmri_video_analysis.core.statistics import (
    bh_fdr,
    cohens_dz,
    paired_sign_flip_pvalue,
)
from fmri_video_analysis.core.study_design import RUN_IDS, blocks_for_run

LOW_LEVEL_FEATURES = (
    "luminance_mean",
    "rms_contrast_mean",
    "entropy_mean",
    "edge_density_mean",
    "frame_difference_mean",
    "optical_flow_mean_mean",
    "global_motion_median_mean",
    "residual_flow_mean_mean",
)


def _zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / sd


def exact_paired_sign_flip(diff: np.ndarray) -> float:
    """Exhaustive two-sided sign-flip p-value for a small paired sample."""

    diff = np.asarray(diff, dtype=float)
    if diff.ndim != 1 or not len(diff) or not np.all(np.isfinite(diff)):
        raise ValueError("diff must be a finite, non-empty one-dimensional array.")
    observed = abs(float(diff.mean()))
    null = [
        abs(float(np.mean(diff * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(diff))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def connection_masks(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return prespecified upper-triangle masks for block FC summaries."""

    required = {"domain", "network"}
    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(f"Node table lacks columns: {sorted(missing)}")
    upper = np.triu_indices(len(nodes), 1)
    first, second = upper
    domains = nodes.domain.astype(str).to_numpy()
    networks = nodes.network.astype(str).to_numpy()

    def within(values: np.ndarray, label: str) -> np.ndarray:
        return (values[first] == label) & (values[second] == label)

    def between(values: np.ndarray, left: str, right: str) -> np.ndarray:
        return ((values[first] == left) & (values[second] == right)) | (
            (values[first] == right) & (values[second] == left)
        )

    visual = np.char.startswith(networks.astype(str), "Vis")
    dorsal = np.char.startswith(networks.astype(str), "DorsAttn")
    masks = {
        "global_all_edges": np.ones(len(first), dtype=bool),
        "cortex_within": within(domains, "Cortex"),
        "cortex_subcortex": between(domains, "Cortex", "Subcortex"),
        "cortex_cerebellum": between(domains, "Cortex", "Cerebellum"),
        "subcortex_within": within(domains, "Subcortex"),
        "cerebellum_within": within(domains, "Cerebellum"),
        "subcortex_cerebellum": between(
            domains, "Subcortex", "Cerebellum"
        ),
        "visual_within": visual[first] & visual[second],
        "dorsal_attention_within": dorsal[first] & dorsal[second],
        "visual_dorsal_attention": (visual[first] & dorsal[second])
        | (dorsal[first] & visual[second]),
    }
    empty = [name for name, mask in masks.items() if not np.any(mask)]
    if empty:
        raise RuntimeError(f"Empty block-FC connection families: {empty}")
    return masks


def _expected_block_table() -> pd.DataFrame:
    rows = []
    for session, run_id in enumerate(RUN_IDS, start=1):
        for block_index, (condition, onset, duration) in enumerate(
            blocks_for_run(run_id)
        ):
            rows.append(
                {
                    "block_index": (session - 1) * 3 + block_index,
                    "session": session,
                    "run_id": run_id,
                    "condition": condition,
                    "video_id": SESSION_VIDEOS[session][condition],
                    "onset_tr": onset,
                    "duration_tr": duration,
                }
            )
    return pd.DataFrame(rows)


def build_block_fc_table(
    run_dir: Path,
    block_matrices: np.ndarray,
    subject_ids: np.ndarray,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Create one row per subject/block/connection-family summary."""

    expected = _expected_block_table()
    if block_matrices.shape[:2] != (len(subject_ids), len(expected)):
        raise ValueError(
            "Unexpected block matrix shape: "
            f"{block_matrices.shape}; expected ({len(subject_ids)}, 12, nodes, nodes)."
        )
    upper = np.triu_indices(len(nodes), 1)
    masks = connection_masks(nodes)
    edge_values = block_matrices[:, :, upper[0], upper[1]]
    metric_values = {
        name: edge_values[..., mask].mean(axis=-1) for name, mask in masks.items()
    }

    rows = []
    for subject_index, subject_id in enumerate(subject_ids.astype(int)):
        qc_path = run_dir / "subjects" / f"sub-{subject_id:03d}" / "block_qc.csv"
        qc = pd.read_csv(qc_path, dtype={"run_id": str})
        qc["run_id"] = qc.run_id.str.zfill(4)
        observed = qc[["run_id", "condition", "onset_tr", "duration_tr"]].reset_index(
            drop=True
        )
        target = expected[["run_id", "condition", "onset_tr", "duration_tr"]]
        if not observed.equals(target):
            raise RuntimeError(f"Block order mismatch for subject {subject_id}.")
        for block_index, design in expected.iterrows():
            base = {
                "subject_id": subject_id,
                **design.to_dict(),
                "retained_trs": int(qc.loc[block_index, "retained_trs"]),
                "censored_trs": int(qc.loc[block_index, "censored_trs"]),
                "mean_fd_mm": float(qc.loc[block_index, "mean_fd_mm"]),
                "max_fd_mm": float(qc.loc[block_index, "max_fd_mm"]),
            }
            for metric, values in metric_values.items():
                rows.append(
                    {
                        **base,
                        "fc_metric": metric,
                        "mean_fisher_z": float(values[subject_index, block_index]),
                        "n_edges": int(masks[metric].sum()),
                    }
                )
    return pd.DataFrame(rows)


def stimulus_pca(stimulus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit two low-level PCs using the eight Simple/Complex videos only."""

    task = stimulus[stimulus.condition.isin(["Simple", "Complex"])].copy()
    if len(task) != 8 or task.video_id.nunique() != 8:
        raise ValueError("Expected four Simple and four Complex video rows.")
    missing = set(LOW_LEVEL_FEATURES).difference(task.columns)
    if missing:
        raise ValueError(f"Stimulus table lacks features: {sorted(missing)}")
    values = task.loc[:, LOW_LEVEL_FEATURES].astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Low-level stimulus features contain non-finite values.")
    scaled = StandardScaler().fit_transform(values)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(scaled)
    task["stimulus_pc1"] = scores[:, 0]
    task["stimulus_pc2"] = scores[:, 1]
    loadings = pd.DataFrame(
        {
            "feature": LOW_LEVEL_FEATURES,
            "pc1_loading": pca.components_[0],
            "pc2_loading": pca.components_[1],
        }
    )
    loadings["pc1_explained_variance_ratio"] = pca.explained_variance_ratio_[0]
    loadings["pc2_explained_variance_ratio"] = pca.explained_variance_ratio_[1]
    return task, loadings


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def fixed_video_condition_tests(
    task_table: pd.DataFrame, seed: int, n_permutations: int
) -> pd.DataFrame:
    """Subject-level inference with the eight observed task videos held fixed."""

    rows = []
    for metric, part in task_table.groupby("fc_metric", sort=False):
        wide = part.groupby(["subject_id", "condition"]).mean_fisher_z.mean().unstack()
        diff = (wide["Complex"] - wide["Simple"]).to_numpy()
        ci_low, ci_high = _bootstrap_mean_ci(diff, seed)
        rows.append(
            {
                "fc_metric": metric,
                "n_subjects": len(diff),
                "mean_complex_minus_simple": float(diff.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "cohens_dz": cohens_dz(diff),
                "p_subject_sign_flip": paired_sign_flip_pvalue(
                    diff, n_permutations, seed
                ),
                "inference_scope": "subjects; eight observed videos fixed",
            }
        )
    result = pd.DataFrame(rows)
    result["q_fdr_metric_family"] = bh_fdr(result.p_subject_sign_flip)
    return result


def video_pair_exact_tests(task_table: pd.DataFrame) -> pd.DataFrame:
    """Exact inference across the four prespecified session-matched video pairs."""

    video_means = (
        task_table.groupby(["fc_metric", "session", "condition"])
        .mean_fisher_z.mean()
        .unstack()
    )
    rows = []
    for metric in task_table.fc_metric.drop_duplicates():
        part = video_means.loc[metric]
        diff = (part["Complex"] - part["Simple"]).to_numpy()
        rows.append(
            {
                "fc_metric": metric,
                "n_video_pairs": len(diff),
                "mean_complex_minus_simple": float(diff.mean()),
                "cohens_dz_across_pairs": cohens_dz(diff),
                "p_exact_video_pair_sign_flip": exact_paired_sign_flip(diff),
                "minimum_attainable_two_sided_p": 2 / (2 ** len(diff)),
                "inference_scope": "four session-matched video pairs",
            }
        )
    result = pd.DataFrame(rows)
    result["q_fdr_metric_family"] = bh_fdr(result.p_exact_video_pair_sign_flip)
    return result


def _mixed_model_row(
    part: pd.DataFrame, metric: str, model_name: str, formula: str
) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = smf.mixedlm(
                formula,
                part,
                groups=np.ones(len(part), dtype=int),
                re_formula="0",
                vc_formula={
                    "subject": "0 + C(subject_id)",
                    "video": "0 + C(video_id)",
                },
            ).fit(reml=False, method="lbfgs", maxiter=2000, disp=False)
            return {
                "fc_metric": metric,
                "model": model_name,
                "n_observations": len(part),
                "n_subjects": part.subject_id.nunique(),
                "n_videos": part.video_id.nunique(),
                "condition_complex_beta": float(fit.params["condition_complex"]),
                "condition_complex_se": float(fit.bse["condition_complex"]),
                "condition_complex_p_asymptotic": float(
                    fit.pvalues["condition_complex"]
                ),
                "random_effects_structure": (
                    "crossed random intercepts for subject and video"
                ),
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "converged": bool(fit.converged),
                "warnings": " | ".join(str(item.message) for item in caught),
                "formula": formula,
            }
        except (ValueError, np.linalg.LinAlgError) as exc:
            return {
                "fc_metric": metric,
                "model": model_name,
                "n_observations": len(part),
                "n_subjects": part.subject_id.nunique(),
                "n_videos": part.video_id.nunique(),
                "condition_complex_beta": np.nan,
                "condition_complex_se": np.nan,
                "random_effects_structure": (
                    "crossed random intercepts for subject and video"
                ),
                "condition_complex_p_asymptotic": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "converged": False,
                "warnings": f"fit_failed: {exc}",
                "formula": formula,
            }


def mixed_effects_models(task_table: pd.DataFrame) -> pd.DataFrame:
    """Fit crossed subject/video exploratory models for each FC summary."""

    task_table = task_table.copy()
    task_table["condition_complex"] = (task_table.condition == "Complex").astype(int)
    task_table["z_mean_fd"] = _zscore(task_table.mean_fd_mm)
    task_table["z_censored_trs"] = _zscore(task_table.censored_trs)
    formulas = {
        "base_motion_adjusted": (
            "mean_fisher_z ~ condition_complex + z_mean_fd + z_censored_trs"
        ),
        "low_level_stimulus_adjusted": (
            "mean_fisher_z ~ condition_complex + stimulus_pc1 + stimulus_pc2 "
            "+ z_mean_fd + z_censored_trs"
        ),
    }
    rows = []
    for metric, part in task_table.groupby("fc_metric", sort=False):
        for model_name, formula in formulas.items():
            rows.append(_mixed_model_row(part, metric, model_name, formula))
    result = pd.DataFrame(rows)
    result["q_fdr_metric_family"] = np.nan
    for indices in result.groupby("model").groups.values():
        result.loc[indices, "q_fdr_metric_family"] = bh_fdr(
            result.loc[indices, "condition_complex_p_asymptotic"]
        )
    return result


def _plot_global_video_means(task_table: pd.DataFrame, output_dir: Path) -> None:
    global_rows = task_table[task_table.fc_metric == "global_all_edges"]
    video = (
        global_rows.groupby(
            ["session", "video_id", "condition", "stimulus_pc1"], as_index=False
        )
        .mean_fisher_z.agg(["mean", "sem"])
        .reset_index()
    )
    colors = {"Simple": "#2878B5", "Complex": "#D95F02"}
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for condition, part in video.groupby("condition"):
        axis.errorbar(
            part.stimulus_pc1,
            part["mean"],
            yerr=part["sem"],
            fmt="o",
            color=colors[condition],
            capsize=3,
            label=condition,
        )
        for row in part.itertuples():
            axis.annotate(row.video_id, (row.stimulus_pc1, row.mean), xytext=(4, 4),
                          textcoords="offset points", fontsize=8)
    axis.set_xlabel("Low-level stimulus PC1")
    axis.set_ylabel("Block mean whole-brain Fisher-z")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "block_global_fc_by_stimulus_pc1.png", dpi=300)
    figure.savefig(output_dir / "block_global_fc_by_stimulus_pc1.pdf")
    plt.close(figure)


def block_stimulus_analysis(
    run_dir: Path,
    stimulus_metrics: Path,
    n_permutations: int = 10000,
    seed: int = 20260731,
) -> dict:
    """Run the prespecified exploratory block-FC/stimulus analysis."""

    archive = np.load(
        run_dir / "subject_condition_connectivity.npz", allow_pickle=False
    )
    nodes = pd.read_csv(run_dir / "atlas/composite_249_nodes.tsv", sep="\t")
    output_dir = run_dir / "sensitivity" / "block_stimulus"
    output_dir.mkdir(parents=True, exist_ok=True)
    block_table = build_block_fc_table(
        run_dir,
        archive["block_matrices"].astype(np.float64),
        archive["subject_ids"].astype(int),
        nodes,
    )
    stimulus = pd.read_csv(stimulus_metrics, dtype={"video_id": str})
    stimulus["video_id"] = stimulus.video_id.str.zfill(3)
    scores, loadings = stimulus_pca(stimulus)
    task_table = block_table[block_table.condition.isin(["Simple", "Complex"])].merge(
        scores,
        on=["session", "condition", "video_id"],
        how="left",
        validate="many_to_one",
    )
    if task_table[["stimulus_pc1", "stimulus_pc2"]].isna().any().any():
        raise RuntimeError("Block/stimulus join produced missing PCA scores.")

    block_table.to_csv(output_dir / "block_fc_summaries.csv.gz", index=False)
    task_table.to_csv(output_dir / "task_block_fc_with_stimulus.csv.gz", index=False)
    scores.to_csv(output_dir / "stimulus_task_video_pca_scores.csv", index=False)
    loadings.to_csv(output_dir / "stimulus_pca_loadings.csv", index=False)
    fixed = fixed_video_condition_tests(task_table, seed, n_permutations)
    exact = video_pair_exact_tests(task_table)
    mixed = mixed_effects_models(task_table)
    fixed.to_csv(output_dir / "fixed_video_condition_tests.csv", index=False)
    exact.to_csv(output_dir / "video_pair_exact_tests.csv", index=False)
    mixed.to_csv(output_dir / "mixed_effects_results.csv", index=False)
    _plot_global_video_means(task_table, output_dir)

    global_fixed = fixed[fixed.fc_metric == "global_all_edges"].iloc[0]
    global_exact = exact[exact.fc_metric == "global_all_edges"].iloc[0]
    global_mixed = mixed[
        (mixed.fc_metric == "global_all_edges")
        & (mixed.model == "low_level_stimulus_adjusted")
    ].iloc[0]
    summary = {
        "status": "exploratory",
        "n_subjects": int(task_table.subject_id.nunique()),
        "n_task_videos": int(task_table.video_id.nunique()),
        "n_session_matched_video_pairs": int(task_table.session.nunique()),
        "n_fc_metrics": int(task_table.fc_metric.nunique()),
        "low_level_pca_features": list(LOW_LEVEL_FEATURES),
        "semantic_target_count_in_model": False,
        "semantic_exclusion_reason": "Human review of detector output is incomplete.",
        "global_fixed_video_mean_difference": float(
            global_fixed.mean_complex_minus_simple
        ),
        "global_fixed_video_p": float(global_fixed.p_subject_sign_flip),
        "global_random_video_exact_p": float(
            global_exact.p_exact_video_pair_sign_flip
        ),
        "global_stimulus_adjusted_beta": float(
            global_mixed.condition_complex_beta
        ),
        "global_stimulus_adjusted_asymptotic_p": float(
            global_mixed.condition_complex_p_asymptotic
        ),
        "interpretation_guardrail": (
            "Mixed-model p-values are exploratory because condition is represented by "
            "only four videos and four matched video pairs per condition; the exact "
            "video-pair test is the stimulus-generalisation check."
        ),
    }
    (output_dir / "block_stimulus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
