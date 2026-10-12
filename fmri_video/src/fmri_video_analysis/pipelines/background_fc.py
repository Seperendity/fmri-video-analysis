"""Command-line pipeline for the frozen 249-node background-FC analysis."""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fmri_video_analysis.core.config import ATLAS_DIR, PROJECT_ROOT, RESULTS_DIR
from fmri_video_analysis.core.environment import runtime_metadata
from fmri_video_analysis.core.provenance import (
    CacheMismatchError,
    atomic_write_json,
    manifest_differences,
    write_stage_metadata,
)
from fmri_video_analysis.core.study_design import (
    INCLUDED_SUBJECT_IDS,
    PILOT_SUBJECT_IDS,
    RUN_IDS,
)
from fmri_video_analysis.methods.connectivity.background_fc import (
    assemble_connectivity,
    audit_timeseries_inputs,
    input_paths,
    process_subject,
)
from fmri_video_analysis.methods.connectivity.composite_atlas import (
    build_composite_atlas,
    load_composite_atlas,
)

DEFAULT_TIMESERIES_ROOT = Path("/cy/data/fmri_video/fmrinii")


def _git_metadata() -> dict:
    project_root = Path(__file__).resolve().parents[4]

    def command(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(command("status", "--porcelain")),
    }


def parse_subjects(values: list[str] | None) -> tuple[int, ...]:
    if not values or values == ["all"]:
        return INCLUDED_SUBJECT_IDS
    if values == ["pilot"]:
        return PILOT_SUBJECT_IDS
    try:
        subject_ids = tuple(int(value) for value in values)
    except ValueError as exc:
        raise ValueError("--subjects must be 'all', 'pilot', or integer IDs.") from exc
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("Duplicate subject IDs were supplied.")
    invalid = sorted(set(subject_ids) - set(INCLUDED_SUBJECT_IDS))
    if invalid:
        raise ValueError(
            f"Subjects are not in the frozen 87-subject sample: {invalid}"
        )
    return subject_ids


def write_run_metadata(args, subject_ids: tuple[int, ...], output_dir: Path) -> None:
    parent_run = getattr(args, "parent_run", None)
    scientific_config = {
        "schema_version": 2,
        "run_id": output_dir.resolve().name,
        "output_dir": str(output_dir.resolve()),
        "parent_run": (
            str(Path(parent_run).resolve()) if parent_run else None
        ),
        "timeseries_root": str(Path(args.timeseries_root).resolve()),
        "atlas_dir": str(Path(args.atlas_dir).resolve()),
        "subject_ids": list(subject_ids),
        "n_subjects": len(subject_ids),
        "run_ids": list(RUN_IDS),
        "parameters": {
            "fd_threshold_mm": args.fd_threshold,
            "head_radius_mm": args.head_radius_mm,
            "hrf_lag_trs": args.hrf_lag_trs,
            "fir_lags": args.fir_lags,
            "min_condition_trs": args.min_condition_trs,
            "high_pass_seconds": args.high_pass_seconds,
            "min_residual_dof": args.min_residual_dof,
        },
    }
    config_path = output_dir / "run_config.json"
    if config_path.is_file() and not args.force:
        observed = json.loads(config_path.read_text(encoding="utf-8"))
        comparable = {key: observed.get(key) for key in scientific_config}
        differences = manifest_differences(comparable, scientific_config)
        if differences:
            preview = "\n".join(f"- {item}" for item in differences[:30])
            raise CacheMismatchError(
                f"Run configuration mismatch for {config_path}:\n{preview}\n"
                "Use a new run directory or --force to replace this configuration."
            )
    payload = {
        **scientific_config,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_metadata(),
        "git": _git_metadata(),
        "hash_inputs": args.hash_inputs,
    }
    atomic_write_json(config_path, payload)
    write_stage_metadata(
        output_dir,
        f"background_fc_{args.stage}",
        {
            "resolved_inputs": {
                "timeseries_root": scientific_config["timeseries_root"],
                "atlas_dir": scientific_config["atlas_dir"],
            },
            "parameters": scientific_config["parameters"],
            "subject_ids": list(subject_ids),
            "hash_inputs": args.hash_inputs,
        },
        project_root=PROJECT_ROOT.parent,
        parent_run=Path(parent_run) if parent_run else None,
    )


def run_audit(args, subject_ids: tuple[int, ...], output_dir: Path) -> pd.DataFrame:
    manifest, errors = audit_timeseries_inputs(
        Path(args.timeseries_root),
        subject_ids,
        hash_inputs=args.hash_inputs,
    )
    manifest_path = output_dir / "input_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "n_subjects": int(manifest["subject_id"].nunique()),
        "n_runs": len(manifest),
        "expected_subjects": len(subject_ids),
        "expected_runs": len(subject_ids) * len(RUN_IDS),
        "hash_mode": "full_sha256" if args.hash_inputs else "header_sha256",
        "passed": not errors
        and len(manifest) == len(subject_ids) * len(RUN_IDS),
        "errors": errors,
    }
    atomic_write_json(output_dir / "input_audit.json", summary)
    if errors:
        raise RuntimeError(
            f"Input audit failed with {len(errors)} errors. See "
            f"{output_dir / 'input_audit.json'}."
        )
    print(
        f"Input audit passed: {len(subject_ids)} subjects, {len(manifest)} runs.",
        flush=True,
    )
    return manifest


def run_atlas(args, subject_ids: tuple[int, ...], output_dir: Path) -> Path:
    reference, _ = input_paths(
        Path(args.timeseries_root), subject_ids[0], RUN_IDS[0]
    )
    atlas_output = output_dir / "atlas"
    atlas_path, nodes, qc = build_composite_atlas(
        reference,
        Path(args.atlas_dir),
        atlas_output,
        small_roi_warning_voxels=args.small_roi_warning_voxels,
    )
    print(
        f"Atlas passed: {len(nodes)} nodes, "
        f"{qc['edge_count']} undirected edges, reference={reference}.",
        flush=True,
    )
    return atlas_path


def run_extract(args, subject_ids: tuple[int, ...], output_dir: Path) -> Path:
    atlas_path, _ = load_composite_atlas(output_dir / "atlas")
    failures = []
    qc_rows = []
    for index, subject_id in enumerate(subject_ids, 1):
        try:
            qc = process_subject(
                subject_id,
                Path(args.timeseries_root),
                atlas_path,
                output_dir,
                fd_threshold=args.fd_threshold,
                head_radius_mm=args.head_radius_mm,
                high_pass_seconds=args.high_pass_seconds,
                fir_lags=args.fir_lags,
                hrf_lag_trs=args.hrf_lag_trs,
                min_condition_trs=args.min_condition_trs,
                min_residual_dof=args.min_residual_dof,
                hash_inputs=args.hash_inputs,
                force=args.force,
            )
            qc_rows.append(qc)
            print(
                f"Subject {subject_id} passed ({index}/{len(subject_ids)}): "
                f"TRs={qc['condition_retained_trs']}, "
                f"FD spikes={sum(qc['fd_spikes_by_run'].values())}, "
                f"DoF={qc['ols']['residual_dof']}.",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-subject failures
            failure = {
                "subject_id": subject_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"Subject {subject_id} FAILED: {exc}", flush=True)
    atomic_write_json(output_dir / "extraction_failures.json", failures)
    if failures:
        raise RuntimeError(
            f"{len(failures)} subject(s) failed extraction; passing outputs "
            "were retained for targeted reruns."
        )
    group_path = assemble_connectivity(subject_ids, output_dir)
    print(
        f"Extraction passed: {len(qc_rows)} subjects assembled at {group_path}.",
        flush=True,
    )
    return group_path


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_ids = parse_subjects(args.subjects)
    write_run_metadata(args, subject_ids, output_dir)

    if args.stage in {"audit", "all"}:
        run_audit(args, subject_ids, output_dir)
    if args.stage in {"atlas", "all"}:
        run_atlas(args, subject_ids, output_dir)
    if args.stage in {"extract", "all"}:
        run_extract(args, subject_ids, output_dir)
    if args.stage == "assemble":
        group_path = assemble_connectivity(subject_ids, output_dir)
        print(f"Assembled {group_path}.", flush=True)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Schaefer-200 + Tian S2 + Buckner-17 background functional "
            "connectivity pipeline."
        )
    )
    command.add_argument(
        "--stage",
        choices=["audit", "atlas", "extract", "assemble", "all"],
        default="all",
    )
    command.add_argument(
        "--timeseries-root", default=str(DEFAULT_TIMESERIES_ROOT)
    )
    command.add_argument("--atlas-dir", default=str(ATLAS_DIR))
    command.add_argument(
        "--output-dir", default=str(RESULTS_DIR / "connectivity/background_fc")
    )
    command.add_argument(
        "--parent-run",
        help="Optional immutable parent run recorded in corrected-run provenance.",
    )
    command.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        help="'all', 'pilot', or explicit included subject IDs.",
    )
    command.add_argument("--fd-threshold", type=float, default=0.5)
    command.add_argument("--head-radius-mm", type=float, default=50.0)
    command.add_argument("--hrf-lag-trs", type=int, default=2)
    command.add_argument("--fir-lags", type=int, default=35)
    command.add_argument("--min-condition-trs", type=int, default=90)
    command.add_argument("--high-pass-seconds", type=float, default=128.0)
    command.add_argument("--min-residual-dof", type=int, default=100)
    command.add_argument("--small-roi-warning-voxels", type=int, default=10)
    command.add_argument("--hash-inputs", action="store_true")
    command.add_argument("--force", action="store_true")
    return command


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
