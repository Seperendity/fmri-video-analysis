"""Audit block order, run/session, motion, censoring and retained samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fmri_video_analysis.core.study_design import (
    BLOCK_ONSETS_TR,
    CONDITIONS,
    RUN_CONDITION_ORDER,
    RUN_IDS,
)

RUN_TO_SESSION = {run_id: index + 1 for index, run_id in enumerate(RUN_IDS)}
ONSET_TO_POSITION = {
    onset: index + 1 for index, onset in enumerate(BLOCK_ONSETS_TR)
}
QC_METRICS = ("mean_fd_mm", "censored_trs", "retained_trs")


def design_schedule() -> pd.DataFrame:
    """Return the frozen 12-block schedule with position and pair-order fields."""

    rows: list[dict[str, object]] = []
    for run_id in RUN_IDS:
        order = RUN_CONDITION_ORDER[run_id]
        simple_position = order.index("Simple") + 1
        complex_position = order.index("Complex") + 1
        for position, (condition, onset) in enumerate(
            zip(order, BLOCK_ONSETS_TR, strict=True), start=1
        ):
            rows.append(
                {
                    "session": RUN_TO_SESSION[run_id],
                    "run_id": run_id,
                    "block_position": position,
                    "onset_tr": onset,
                    "condition": condition,
                    "simple_complex_order": (
                        "Simple_before_Complex"
                        if simple_position < complex_position
                        else "Complex_before_Simple"
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_block_qc(run_dir: Path) -> pd.DataFrame:
    """Load and validate every participant's primary block-QC table."""

    rows: list[pd.DataFrame] = []
    for path in sorted((run_dir / "subjects").glob("sub-*/block_qc.csv")):
        table = pd.read_csv(path)
        table["subject_id"] = int(path.parent.name.split("-")[1])
        rows.append(table)
    if not rows:
        raise FileNotFoundError(f"No subject block-QC files under {run_dir}")
    table = pd.concat(rows, ignore_index=True)
    table["run_id"] = table["run_id"].astype(str).str.zfill(4)
    table["session"] = table["run_id"].map(RUN_TO_SESSION)
    table["block_position"] = table["onset_tr"].map(ONSET_TO_POSITION)
    expected_rows = table["subject_id"].nunique() * len(RUN_IDS) * len(CONDITIONS)
    if len(table) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} participant-block rows, found {len(table)}"
        )
    if table[["session", "block_position", *QC_METRICS]].isna().any().any():
        raise RuntimeError("Block-QC audit inputs contain missing values.")
    return table.sort_values(
        ["subject_id", "session", "block_position"]
    ).reset_index(drop=True)


def sign_flip_p(values: np.ndarray, signs: np.ndarray) -> float:
    """Return a two-sided plus-one participant-level sign-flip p value."""

    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = np.abs(signs[:, : len(values)] @ values / len(values))
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def bootstrap_mean_ci(
    values: np.ndarray, bootstrap_indices: np.ndarray
) -> tuple[float, float]:
    """Return a participant-bootstrap percentile interval for the mean."""

    values = np.asarray(values, dtype=float)
    draws = values[bootstrap_indices[:, : len(values)]].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted values with the original index preserved."""

    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    return pd.Series(restored, index=p_values.index)


def _participant_design(group: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = [
        np.ones(len(group)),
        (group["condition"] == "Complex").to_numpy(dtype=float),
        (group["condition"] == "Control").to_numpy(dtype=float),
    ]
    names = ["intercept", "complex_minus_simple", "control_minus_simple"]
    for run_id in RUN_IDS[1:]:
        columns.append((group["run_id"] == run_id).to_numpy(dtype=float))
        names.append(f"run_{run_id}_minus_0006")
    for position in (2, 3):
        columns.append(
            (group["block_position"] == position).to_numpy(dtype=float)
        )
        names.append(f"position_{position}_minus_1")
    matrix = np.column_stack(columns)
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise RuntimeError("The condition/run/position audit design is not full rank.")
    return matrix, names


def qc_fixed_effects(
    table: pd.DataFrame,
    *,
    signs: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> pd.DataFrame:
    """Estimate participant-wise condition, run and position coefficients."""

    records: list[dict[str, object]] = []
    for metric in QC_METRICS:
        coefficients = []
        names: list[str] | None = None
        for _, group in table.groupby("subject_id", sort=True):
            matrix, current_names = _participant_design(group)
            names = current_names
            coefficients.append(
                np.linalg.lstsq(
                    matrix, group[metric].to_numpy(dtype=float), rcond=None
                )[0]
            )
        coefficient_array = np.asarray(coefficients)
        assert names is not None
        for column_index, term in enumerate(names[1:], start=1):
            values = coefficient_array[:, column_index]
            low, high = bootstrap_mean_ci(values, bootstrap_indices)
            standard_deviation = float(values.std(ddof=1))
            records.append(
                {
                    "metric": metric,
                    "term": term,
                    "n_subjects": len(values),
                    "mean_coefficient": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "cohens_dz": (
                        float(values.mean() / standard_deviation)
                        if standard_deviation > 0
                        else np.nan
                    ),
                    "p_sign_flip": sign_flip_p(values, signs),
                    "model": "participant-wise OLS adjusted for condition, run and block position",
                }
            )
    result = pd.DataFrame(records)
    result["q_fdr_within_metric"] = result.groupby("metric")[
        "p_sign_flip"
    ].transform(bh_fdr)
    return result


def raw_condition_contrasts(
    table: pd.DataFrame,
    *,
    signs: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> pd.DataFrame:
    """Summarize unadjusted participant-level Complex-Simple QC differences."""

    rows = []
    for metric in QC_METRICS:
        means = table.groupby(["subject_id", "condition"])[metric].mean().unstack()
        values = (means["Complex"] - means["Simple"]).to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(values, bootstrap_indices)
        standard_deviation = float(values.std(ddof=1))
        rows.append(
            {
                "metric": metric,
                "contrast": "Complex_minus_Simple",
                "n_subjects": len(values),
                "simple_mean": float(means["Simple"].mean()),
                "complex_mean": float(means["Complex"].mean()),
                "mean_difference": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "cohens_dz": (
                    float(values.mean() / standard_deviation)
                    if standard_deviation > 0
                    else np.nan
                ),
                "p_sign_flip": sign_flip_p(values, signs),
            }
        )
    result = pd.DataFrame(rows)
    result["q_fdr_three_qc_metrics"] = bh_fdr(result["p_sign_flip"])
    return result


def block_fc_adjusted_models(
    run_dir: Path,
    *,
    signs: np.ndarray,
) -> pd.DataFrame:
    """Audit task-block FC after subject, session, position and motion adjustment."""

    path = (
        run_dir
        / "sensitivity"
        / "block_stimulus"
        / "task_block_fc_with_stimulus.csv.gz"
    )
    table = pd.read_csv(path)
    table = table.loc[table["condition"].isin(["Simple", "Complex"])].copy()
    table["condition_complex"] = (table["condition"] == "Complex").astype(float)
    table["block_position"] = table["onset_tr"].map(ONSET_TO_POSITION)
    rows = []
    for metric, group in table.groupby("fc_metric", sort=True):
        group = group.sort_values(
            ["subject_id", "session", "block_position"]
        ).reset_index(drop=True)
        nuisance = [np.ones(len(group))]
        for column in ("subject_id", "session", "block_position"):
            dummies = pd.get_dummies(
                group[column], drop_first=True, dtype=float
            ).to_numpy()
            nuisance.extend(dummies[:, index] for index in range(dummies.shape[1]))
        for column in ("mean_fd_mm", "censored_trs"):
            values = group[column].to_numpy(dtype=float)
            scale = values.std(ddof=0)
            nuisance.append((values - values.mean()) / (scale if scale > 0 else 1.0))
        nuisance_matrix = np.column_stack(nuisance)
        condition = group["condition_complex"].to_numpy(dtype=float)
        outcome = group["mean_fisher_z"].to_numpy(dtype=float)
        condition_residual = condition - nuisance_matrix @ np.linalg.lstsq(
            nuisance_matrix, condition, rcond=None
        )[0]
        denominator = float(condition_residual @ condition_residual)
        beta = float(condition_residual @ outcome / denominator)
        subject_ids = np.sort(group["subject_id"].unique())
        contributions = np.asarray(
            [
                condition_residual[group["subject_id"].to_numpy() == subject_id]
                @ outcome[group["subject_id"].to_numpy() == subject_id]
                / denominator
                for subject_id in subject_ids
            ]
        )
        null = signs[:, : len(subject_ids)] @ contributions
        p_value = float(
            (1 + np.sum(np.abs(null) >= abs(beta))) / (len(null) + 1)
        )
        rows.append(
            {
                "fc_metric": metric,
                "n_subjects": len(subject_ids),
                "n_task_blocks": len(group),
                "condition_complex_beta": beta,
                "p_cluster_sign_flip": p_value,
                "model": (
                    "participant fixed effects + session + block position + "
                    "block mean FD + censored TRs"
                ),
                "inference_scope": (
                    "diagnostic participant-cluster sign-flip; eight fixed videos"
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["q_fdr_10_fc_metrics"] = bh_fdr(result["p_cluster_sign_flip"])
    return result


def descriptive_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Return condition, run and position summaries for all QC endpoints."""

    rows = []
    for grouping in ("condition", "run_id", "block_position"):
        for level, group in table.groupby(grouping, sort=True):
            for metric in QC_METRICS:
                rows.append(
                    {
                        "grouping": grouping,
                        "level": level,
                        "metric": metric,
                        "n_subject_block_rows": len(group),
                        "mean": float(group[metric].mean()),
                        "sd": float(group[metric].std(ddof=1)),
                        "median": float(group[metric].median()),
                        "q25": float(group[metric].quantile(0.25)),
                        "q75": float(group[metric].quantile(0.75)),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    schedule: pd.DataFrame,
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
    block_fc: pd.DataFrame,
    *,
    seed: int,
    n_permutations: int,
) -> None:
    """Write a concise, manuscript-facing audit report."""

    position_counts = pd.crosstab(
        schedule["condition"], schedule["block_position"]
    )
    order_counts = schedule.drop_duplicates("run_id")[
        "simple_complex_order"
    ].value_counts()
    raw_lookup = raw.set_index("metric")
    adjusted_lookup = adjusted.loc[
        adjusted["term"] == "complex_minus_simple"
    ].set_index("metric")
    global_fc = block_fc.set_index("fc_metric").loc["global_all_edges"]

    def result_line(metric: str) -> str:
        unadjusted = raw_lookup.loc[metric]
        current = adjusted_lookup.loc[metric]
        return (
            f"- `{metric}`: unadjusted Complex-Simple "
            f"{unadjusted.mean_difference:.6f}, dz={unadjusted.cohens_dz:.3f}, "
            f"P={unadjusted.p_sign_flip:.4f}; run/position-adjusted "
            f"{current.mean_coefficient:.6f}, dz={current.cohens_dz:.3f}, "
            f"P={current.p_sign_flip:.4f}."
        )

    report = f"""# Block-position, session/run and motion audit

## Scope

- Primary run: `{output_dir.parents[1].name}`
- Participants: {int(raw.n_subjects.iloc[0])}
- Participant-block QC rows: {int(raw.n_subjects.iloc[0]) * 12}
- Random seed: {seed}
- Participant-level sign flips: {n_permutations:,}
- The audit is diagnostic. It does not turn eight fixed videos into a random stimulus sample.

## Design balance

All three conditions occurred once in every run. Simple preceded Complex in
{int(order_counts.get('Simple_before_Complex', 0))} runs and Complex preceded
Simple in {int(order_counts.get('Complex_before_Simple', 0))} runs, so their
pairwise order was balanced. Absolute block position was not perfectly balanced:
Simple occurred at positions {position_counts.loc['Simple'].tolist()}, whereas
Complex occurred at positions {position_counts.loc['Complex'].tolist()} across
positions 1, 2 and 3. Session and run are the same four-level factor, and each
session/run is tied to one fixed video triplet. Run effects therefore cannot be
separated from those video identities.

## Motion, censoring and retained samples

{result_line('mean_fd_mm')}
{result_line('censored_trs')}
{result_line('retained_trs')}

The adjusted participant-wise model included condition, run and absolute block
position. Retained TRs equal 30 minus censored TRs, so those two endpoints are
deterministic complements rather than independent findings.

## Block-level FC diagnostic

After participant fixed effects, session, absolute block position, block mean FD
and censored TRs were included, the whole-brain block-FC Complex-Simple
coefficient was {global_fc.condition_complex_beta:.6f} Fisher z
(cluster sign-flip P={global_fc.p_cluster_sign_flip:.4f},
q={global_fc.q_fdr_10_fc_metrics:.4f} across ten FC summaries). This diagnostic
does not support a global FC increase and is consistent with the existing
fixed-video block analysis. It does not directly re-test the condition-level NBS,
activation or MVPA endpoints.

## Interpretation

The available QC data do not show a material Complex-Simple difference in mean
FD, censored TRs or retained TRs, either before or after run and position
adjustment. Pairwise Simple/Complex order was balanced. Residual design limits
remain: absolute position was only partially counterbalanced, and session/run is
inseparable from the identity of the fixed videos. These facts should be stated
as design boundaries rather than described as fully removed confounds.
"""
    (output_dir / "audit_report.md").write_text(report, encoding="utf-8")


def run_audit(
    run_dir: Path,
    output_dir: Path,
    *,
    seed: int = 20260731,
    n_permutations: int = 10_000,
    n_bootstrap: int = 5_000,
) -> dict[str, object]:
    """Run the complete audit and write machine-readable artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    table = load_block_qc(run_dir)
    schedule = design_schedule()
    rng = np.random.default_rng(seed)
    n_subjects = table["subject_id"].nunique()
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_subjects))
    bootstrap_indices = rng.integers(
        0, n_subjects, size=(n_bootstrap, n_subjects)
    )
    raw = raw_condition_contrasts(
        table, signs=signs, bootstrap_indices=bootstrap_indices
    )
    adjusted = qc_fixed_effects(
        table, signs=signs, bootstrap_indices=bootstrap_indices
    )
    block_fc = block_fc_adjusted_models(run_dir, signs=signs)
    descriptive = descriptive_summary(table)

    schedule.to_csv(output_dir / "design_schedule.csv", index=False)
    table.to_csv(output_dir / "participant_block_qc.csv.gz", index=False)
    descriptive.to_csv(output_dir / "qc_descriptive_summary.csv", index=False)
    raw.to_csv(output_dir / "qc_complex_minus_simple.csv", index=False)
    adjusted.to_csv(output_dir / "qc_condition_run_position_models.csv", index=False)
    block_fc.to_csv(output_dir / "block_fc_adjusted_models.csv", index=False)
    write_report(
        output_dir,
        schedule,
        raw,
        adjusted,
        block_fc,
        seed=seed,
        n_permutations=n_permutations,
    )
    summary = {
        "status": "complete",
        "n_subjects": int(n_subjects),
        "n_participant_blocks": len(table),
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "simple_before_complex_runs": int(
            (schedule.drop_duplicates("run_id")["simple_complex_order"]
             == "Simple_before_Complex").sum()
        ),
        "complex_before_simple_runs": int(
            (schedule.drop_duplicates("run_id")["simple_complex_order"]
             == "Complex_before_Simple").sum()
        ),
        "session_run_alias": True,
        "session_video_identity_confounded": True,
        "output_files": sorted(
            {path.name for path in output_dir.iterdir()} | {"audit_summary.json"}
        ),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--n-bootstrap", type=int, default=5_000)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        args.run_dir / "reports" / "design_motion_audit"
    )
    summary = run_audit(
        args.run_dir,
        output_dir,
        seed=args.seed,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
