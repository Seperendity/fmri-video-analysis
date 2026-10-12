"""Command-line stages for group background-FC statistics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fmri_video_analysis.core.config import RESULTS_DIR
from fmri_video_analysis.core.environment import runtime_metadata
from fmri_video_analysis.methods.connectivity.group_fc import (
    edgewise_statistics,
    load_group_connectivity,
    nbs_analysis,
    predefined_metric_statistics,
    predefined_metric_values,
    topology_analysis,
)

DEFAULT_RUN_DIR = RESULTS_DIR


def run(args) -> None:
    run_dir = Path(args.run_dir)
    output_dir = run_dir / "group_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, subject_ids, nodes = load_group_connectivity(run_dir)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "run_dir": str(run_dir.resolve()),
        "n_subjects": len(subject_ids),
        "n_nodes": len(nodes),
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "runtime": runtime_metadata(),
    }
    (output_dir / f"{args.stage}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.stage in {"metrics", "all"}:
        values = predefined_metric_values(matrices, subject_ids, nodes)
        statistics = predefined_metric_statistics(
            values,
            output_dir / "predefined_metrics",
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        print(
            f"Predefined metrics complete: "
            f"{values['metric'].nunique()} metrics, {len(statistics)} tests.",
            flush=True,
        )
    if args.stage in {"edges", "all"}:
        summary = edgewise_statistics(
            matrices, nodes, output_dir / "edgewise"
        )
        print(f"Edgewise statistics complete: {summary}.", flush=True)
    if args.stage in {"nbs", "all"}:
        summary = nbs_analysis(
            matrices,
            nodes,
            output_dir / "nbs",
            n_permutations=args.n_permutations,
            thresholds=tuple(args.nbs_thresholds),
            seed=args.seed,
            batch_size=args.nbs_batch_size,
        )
        print(f"NBS complete: {summary}.", flush=True)
    if args.stage in {"topology", "all"}:
        _, statistics = topology_analysis(
            matrices,
            subject_ids,
            nodes,
            output_dir / "topology",
            n_permutations=args.n_permutations,
            equivalence_bound_dz=args.equivalence_bound_dz,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )
        print(
            f"Topology complete: {len(statistics)} tests.", flush=True
        )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Group statistics for the frozen background-FC run."
    )
    command.add_argument(
        "--stage",
        choices=["metrics", "edges", "nbs", "topology", "all"],
        default="all",
    )
    command.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    command.add_argument("--n-permutations", type=int, default=10000)
    command.add_argument("--n-bootstrap", type=int, default=5000)
    command.add_argument("--seed", type=int, default=20260731)
    command.add_argument("--n-jobs", type=int, default=1)
    command.add_argument(
        "--nbs-thresholds", nargs="+", type=float, default=[2.5, 3.1, 3.5]
    )
    command.add_argument("--nbs-batch-size", type=int, default=64)
    command.add_argument("--equivalence-bound-dz", type=float, default=0.2)
    return command


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
