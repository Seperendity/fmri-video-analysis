"""Run the prespecified background-FC sensitivity variants."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fmri_video_analysis.core.config import PROJECT_ROOT, RESULTS_DIR
from fmri_video_analysis.core.environment import runtime_metadata
from fmri_video_analysis.core.provenance import atomic_write_json, write_stage_metadata
from fmri_video_analysis.core.validation import validate_subject_ids
from fmri_video_analysis.methods.connectivity.block_stimulus import (
    block_stimulus_analysis,
)
from fmri_video_analysis.methods.connectivity.sensitivity_fc import (
    RECOMPUTED_VARIANTS,
    SLICED_VARIANTS,
    block_reliability_analysis,
    create_sliced_variants,
    qc_fc_analysis,
    run_recomputed_variant,
    sensitivity_nbs_analysis,
    sensitivity_topology_analysis,
    summarize_variants,
)

DEFAULT_RUN_DIR = RESULTS_DIR
DEFAULT_TIMESERIES_ROOT = Path("/cy/data/fmri_video/fmrinii")


def parse_variants(values: list[str] | None) -> list[str]:
    available = list(RECOMPUTED_VARIANTS) + list(SLICED_VARIANTS)
    if not values or values == ["all"]:
        return available
    invalid = sorted(set(values) - set(available))
    if invalid:
        raise ValueError(f"Unknown variants: {invalid}; available={available}.")
    return values


def run(args) -> None:
    run_dir = Path(args.run_dir)
    stimulus_metrics = (
        Path(args.stimulus_metrics)
        if args.stimulus_metrics
        else run_dir / "stimulus/characterization/stimulus_metrics_combined.csv"
    )
    variants = parse_variants(args.variants)
    main = np.load(
        run_dir / "subject_condition_connectivity.npz", allow_pickle=False
    )
    subject_ids = main["subject_ids"].astype(int)
    validate_subject_ids(
        subject_ids,
        context=f"main background-FC archive {run_dir}",
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "variants": variants,
        "run_dir": str(run_dir.resolve()),
        "timeseries_root": str(Path(args.timeseries_root).resolve()),
        "stimulus_metrics": str(stimulus_metrics.resolve()),
        "n_jobs": args.n_jobs,
        "n_permutations": args.n_permutations,
        "runtime": runtime_metadata(),
    }
    sensitivity_dir = run_dir / "sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        sensitivity_dir / f"{args.stage}_run_metadata.json", metadata
    )
    write_stage_metadata(
        run_dir,
        f"sensitivity_fc_{args.stage}",
        {
            "resolved_inputs": metadata,
            "subject_ids": subject_ids.tolist(),
        },
        project_root=PROJECT_ROOT.parent,
    )

    if args.stage in {"extract", "all"}:
        for variant in variants:
            if variant not in RECOMPUTED_VARIANTS:
                continue
            path = run_recomputed_variant(
                run_dir,
                Path(args.timeseries_root),
                subject_ids,
                variant,
                n_jobs=args.n_jobs,
                force=args.force,
            )
            print(f"Sensitivity extraction complete: {variant} -> {path}.")
    if args.stage in {"slice", "all"} and any(
        variant in SLICED_VARIANTS for variant in variants
    ):
        outputs = create_sliced_variants(
            run_dir,
            main["matrices"],
            main["block_matrices"],
            subject_ids,
        )
        print(f"Sliced sensitivity variants complete: {outputs}.")
    if args.stage in {"summarize", "all"}:
        summary = summarize_variants(
            run_dir,
            variants,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
        print(f"Sensitivity summary complete: {len(summary)} rows.")
    if args.stage in {"qc-fc", "all"}:
        summary = qc_fc_analysis(run_dir)
        print(f"QC-FC complete: {summary}.")
    if args.stage in {"reliability", "all"}:
        summary = block_reliability_analysis(run_dir)
        print(f"Block reliability complete:\n{summary.to_string(index=False)}")
    if args.stage in {"block-stimulus", "all"}:
        summary = block_stimulus_analysis(
            run_dir,
            stimulus_metrics,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
        print(f"Block FC/stimulus analysis complete: {summary}.")
    if args.stage in {"nbs", "all"}:
        summary = sensitivity_nbs_analysis(
            run_dir,
            variants,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
        print(f"Sensitivity NBS complete: {summary}.")
    if args.stage in {"topology", "all"}:
        summary = sensitivity_topology_analysis(
            run_dir,
            variants,
            n_permutations=args.n_permutations,
            equivalence_bound_dz=args.equivalence_bound_dz,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )
        print(f"Sensitivity topology complete: {len(summary)} tests.")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Prespecified robustness analyses for background FC."
    )
    command.add_argument(
        "--stage",
        choices=[
            "extract",
            "slice",
            "summarize",
            "qc-fc",
            "reliability",
            "block-stimulus",
            "nbs",
            "topology",
            "all",
        ],
        default="all",
    )
    command.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    command.add_argument(
        "--timeseries-root", default=str(DEFAULT_TIMESERIES_ROOT)
    )
    command.add_argument("--variants", nargs="+", default=["all"])
    command.add_argument("--n-jobs", type=int, default=8)
    command.add_argument("--n-permutations", type=int, default=10000)
    command.add_argument("--stimulus-metrics")
    command.add_argument("--equivalence-bound-dz", type=float, default=0.2)
    command.add_argument("--seed", type=int, default=20260731)
    command.add_argument("--force", action="store_true")
    return command


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
