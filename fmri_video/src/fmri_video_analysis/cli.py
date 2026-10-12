"""Safe operational entry points for maintained fMRI analyses."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fmri_video_analysis.core.config import OUTPUTS_DIR, PROJECT_ROOT
from fmri_video_analysis.core.environment import runtime_metadata
from fmri_video_analysis.core.provenance import (
    atomic_write_json,
    git_metadata,
    stable_hash,
)

OPERATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    module: str
    output_flag: str
    pilot_args: tuple[str, ...] | None


WORKFLOWS = {
    "revision": WorkflowSpec(
        name="revision",
        module="fmri_video_analysis.pipelines.revision",
        output_flag="--output-dir",
        pilot_args=("--analysis", "audit"),
    ),
    "background-fc": WorkflowSpec(
        name="background-fc",
        module="fmri_video_analysis.pipelines.background_fc",
        output_flag="--output-dir",
        pilot_args=("--stage", "all", "--subjects", "pilot"),
    ),
    "group-fc": WorkflowSpec(
        name="group-fc",
        module="fmri_video_analysis.pipelines.group_fc",
        output_flag="--run-dir",
        pilot_args=None,
    ),
    "sensitivity-fc": WorkflowSpec(
        name="sensitivity-fc",
        module="fmri_video_analysis.pipelines.sensitivity_fc",
        output_flag="--run-dir",
        pilot_args=None,
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pipeline_tokens(values: Sequence[str]) -> list[str]:
    tokens = list(values)
    if tokens[:1] == ["--"]:
        return tokens[1:]
    return tokens


def _parse_pipeline_args(spec: WorkflowSpec, tokens: Sequence[str]) -> argparse.Namespace:
    module = importlib.import_module(spec.module)
    parser_factory = getattr(module, "parser", None)
    if parser_factory is None:
        raise RuntimeError(f"{spec.module} does not expose parser().")
    try:
        return parser_factory().parse_args(list(tokens))
    except SystemExit as exc:
        raise ValueError(
            f"Invalid arguments for workflow {spec.name!r}."
        ) from exc


def _append_enforced(
    tokens: Sequence[str], spec: WorkflowSpec, run_dir: Path, extras: Sequence[str]
) -> list[str]:
    return [
        *tokens,
        *extras,
        spec.output_flag,
        str(run_dir),
    ]


def _run_root(path: Path, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    output_runs = (Path(outputs_dir) / "runs").resolve()
    candidate = Path(path).expanduser().resolve()
    try:
        relative = candidate.relative_to(output_runs)
    except ValueError as exc:
        raise ValueError(
            f"Run path must be under {output_runs}, got {candidate}."
        ) from exc
    if not relative.parts:
        raise ValueError(f"Run path must include a run ID below {output_runs}.")
    return output_runs / relative.parts[0]


def _nearest_existing(path: Path) -> Path:
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _path_record(name: str, path: Path, *, required: bool = True) -> dict[str, Any]:
    expanded = Path(path).expanduser()
    exists = expanded.exists()
    resolved = expanded.resolve()
    kind_ok = expanded.is_file() or expanded.is_dir() if exists else False
    passed = (exists and kind_ok) or not required
    return {
        "name": name,
        "path": str(resolved),
        "required": required,
        "exists": exists,
        "kind": (
            "directory"
            if exists and expanded.is_dir()
            else "file"
            if exists and expanded.is_file()
            else "missing"
        ),
        "passed": passed,
    }


def _required_inputs(
    workflow: str, namespace: argparse.Namespace
) -> list[tuple[str, Path, bool]]:
    if workflow == "revision":
        analysis = namespace.analysis
        return [
            ("first_level_dir", Path(namespace.first_level_dir), True),
            (
                "stimulus_frames_dir",
                Path(namespace.stimulus_frames_dir),
                analysis in {"audit", "stimulus", "all-available", "all"},
            ),
            (
                "original_mvpa_results_dir",
                Path(namespace.original_mvpa_results_dir),
                analysis
                in {
                    "audit",
                    "mvpa-validation",
                    "beta-connectivity",
                    "posthoc-roi",
                    "supplementary",
                    "all-available",
                    "all",
                },
            ),
            (
                "timeseries_root",
                Path(namespace.timeseries_root)
                if namespace.timeseries_root
                else Path("."),
                analysis in {"connectivity", "topology", "all"},
            ),
        ]
    if workflow == "background-fc":
        return [
            ("timeseries_root", Path(namespace.timeseries_root), True),
            ("atlas_dir", Path(namespace.atlas_dir), True),
        ]
    if workflow == "group-fc":
        run_dir = Path(namespace.run_dir)
        return [
            ("run_dir", run_dir, True),
            (
                "subject_condition_connectivity",
                run_dir / "subject_condition_connectivity.npz",
                True,
            ),
        ]
    if workflow == "sensitivity-fc":
        run_dir = Path(namespace.run_dir)
        stage = namespace.stage
        stimulus = (
            Path(namespace.stimulus_metrics)
            if namespace.stimulus_metrics
            else run_dir / "stimulus/characterization/stimulus_metrics_combined.csv"
        )
        return [
            ("run_dir", run_dir, True),
            (
                "subject_condition_connectivity",
                run_dir / "subject_condition_connectivity.npz",
                True,
            ),
            (
                "timeseries_root",
                Path(namespace.timeseries_root),
                stage in {"extract", "all"},
            ),
            (
                "stimulus_metrics",
                stimulus,
                stage in {"block-stimulus", "all"},
            ),
        ]
    raise ValueError(f"Unknown workflow {workflow!r}.")


def build_preflight_report(
    workflow: str,
    pipeline_args: Sequence[str],
    *,
    min_free_gb: float = 20.0,
    require_gpu: bool = False,
    outputs_dir: Path = OUTPUTS_DIR,
) -> dict[str, Any]:
    spec = WORKFLOWS[workflow]
    tokens = _pipeline_tokens(pipeline_args)
    namespace = _parse_pipeline_args(spec, tokens)
    run_path = Path(getattr(namespace, spec.output_flag[2:].replace("-", "_")))
    run_root = _run_root(run_path, outputs_dir)

    checks: list[dict[str, Any]] = []
    for name, path, required in _required_inputs(workflow, namespace):
        checks.append(_path_record(name, path, required=required))

    existing = _nearest_existing(run_root)
    usage = shutil.disk_usage(existing)
    free_gb = usage.free / (1024**3)
    checks.append(
        {
            "name": "free_disk_gb",
            "path": str(existing.resolve()),
            "value": free_gb,
            "minimum": min_free_gb,
            "passed": free_gb >= min_free_gb,
        }
    )

    gpu = {"required": require_gpu, "available": False, "count": 0}
    try:
        import torch

        gpu["available"] = bool(torch.cuda.is_available())
        gpu["count"] = int(torch.cuda.device_count())
    except (ImportError, RuntimeError):
        pass
    checks.append(
        {
            "name": "gpu",
            **gpu,
            "passed": gpu["available"] or not require_gpu,
        }
    )

    output_exists = run_root.exists()
    output_nonempty = output_exists and any(run_root.iterdir())
    checks.append(
        {
            "name": "run_output",
            "path": str(run_root),
            "exists": output_exists,
            "nonempty": output_nonempty,
            "passed": True,
            "warning": (
                "Run directory already contains files; existing cache contracts apply."
                if output_nonempty
                else None
            ),
        }
    )

    report = {
        "schema_version": OPERATION_SCHEMA_VERSION,
        "kind": "preflight",
        "created_utc": utc_now(),
        "workflow": workflow,
        "pipeline_module": spec.module,
        "pipeline_args": tokens,
        "resolved_arguments": vars(namespace),
        "run_dir": str(run_root),
        "checks": checks,
        "runtime": runtime_metadata(),
        "git": git_metadata(PROJECT_ROOT.parent),
    }
    report["passed"] = all(bool(check["passed"]) for check in checks)
    report["signature_sha256"] = stable_hash(report)
    return report


def build_pilot_command(
    workflow: str,
    run_id: str,
    pipeline_args: Sequence[str],
    *,
    outputs_dir: Path = OUTPUTS_DIR,
) -> tuple[list[str], Path]:
    spec = WORKFLOWS[workflow]
    if spec.pilot_args is None:
        raise ValueError(
            f"Workflow {workflow!r} has no scientifically bounded pilot. "
            "Use preflight and a purpose-built run directory instead."
        )
    run_dir = (Path(outputs_dir) / "runs" / run_id).resolve()
    _run_root(run_dir, outputs_dir)
    tokens = _append_enforced(
        _pipeline_tokens(pipeline_args),
        spec,
        run_dir,
        spec.pilot_args,
    )
    _parse_pipeline_args(spec, tokens)
    return [sys.executable, "-m", spec.module, *tokens], run_dir


def _operation_environment(run_id: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "FMRI_VIDEO_RUN_ID": run_id,
            "FMRI_VIDEO_HEADLESS": "1",
            "MPLBACKEND": "Agg",
            "PYTHONNOUSERSITE": "1",
        }
    )
    environment.pop("DISPLAY", None)
    return environment


def _manifest_payload(
    *,
    kind: str,
    workflow: str,
    command: Sequence[str],
    run_dir: Path,
    preflight: dict[str, Any],
    parent_manifest: Path | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": OPERATION_SCHEMA_VERSION,
        "kind": kind,
        "workflow": workflow,
        "created_utc": utc_now(),
        "status": "planned",
        "command": list(command),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "preflight": preflight,
        "runtime": runtime_metadata(),
        "git": git_metadata(PROJECT_ROOT.parent),
        "parent_manifest": (
            str(Path(parent_manifest).resolve()) if parent_manifest else None
        ),
    }
    payload["signature_sha256"] = stable_hash(payload)
    return payload


def execute_operation(
    manifest: dict[str, Any],
    *,
    dry_run: bool = False,
) -> int:
    if dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    run_dir = Path(manifest["run_dir"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to start in non-empty run directory: {run_dir}"
        )
    operation_dir = run_dir / "metadata/operations"
    operation_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = operation_dir / "execution_manifest.json"
    log_path = operation_dir / f"{manifest['kind']}.log"

    manifest["status"] = "running"
    manifest["started_utc"] = utc_now()
    manifest["log_path"] = str(log_path)
    manifest["signature_sha256"] = stable_hash(
        {key: value for key, value in manifest.items() if key != "signature_sha256"}
    )
    atomic_write_json(manifest_path, manifest)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            manifest["command"],
            cwd=PROJECT_ROOT,
            env=_operation_environment(manifest["run_id"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()

    manifest["status"] = "completed" if return_code == 0 else "failed"
    manifest["return_code"] = return_code
    manifest["finished_utc"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    return return_code


def _replace_output_path(
    command: Sequence[str], output_flag: str, new_run_dir: Path
) -> list[str]:
    updated = list(command)
    positions = [index for index, token in enumerate(updated) if token == output_flag]
    if not positions:
        raise ValueError(f"Recorded command has no {output_flag} option.")
    index = positions[-1]
    if index + 1 >= len(updated):
        raise ValueError(f"Recorded command ends after {output_flag}.")
    updated[index + 1] = str(new_run_dir)
    return updated


def build_reproduction(
    source_manifest: Path,
    run_id: str,
    *,
    allow_code_drift: bool = False,
    outputs_dir: Path = OUTPUTS_DIR,
) -> dict[str, Any]:
    source_path = Path(source_manifest).expanduser().resolve(strict=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema_version") != OPERATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported operation manifest schema: {source.get('schema_version')!r}."
        )
    workflow = source["workflow"]
    spec = WORKFLOWS[workflow]
    current_git = git_metadata(PROJECT_ROOT.parent)
    if current_git != source.get("git") and not allow_code_drift:
        raise RuntimeError(
            "Current Git state differs from the recorded operation. "
            "Use --allow-code-drift only for an intentional non-identical rerun."
        )

    run_dir = (Path(outputs_dir) / "runs" / run_id).resolve()
    _run_root(run_dir, outputs_dir)
    command = _replace_output_path(source["command"], spec.output_flag, run_dir)
    module_index = command.index(spec.module)
    pipeline_args = command[module_index + 1 :]
    preflight = build_preflight_report(
        workflow,
        pipeline_args,
        outputs_dir=outputs_dir,
    )
    if not preflight["passed"]:
        raise RuntimeError("Reproduction preflight failed.")
    return _manifest_payload(
        kind="reproduce",
        workflow=workflow,
        command=command,
        run_dir=run_dir,
        preflight=preflight,
        parent_manifest=source_path,
    )


def _write_or_print_report(report: dict[str, Any], path: str | None) -> None:
    if path:
        atomic_write_json(Path(path), report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="fmri-video",
        description="Preflight, pilot, and reproduce maintained fMRI workflows.",
    )
    subcommands = command.add_subparsers(dest="operation", required=True)

    preflight = subcommands.add_parser(
        "preflight", help="Validate inputs, output boundaries, disk, and resources."
    )
    preflight.add_argument("workflow", choices=sorted(WORKFLOWS))
    preflight.add_argument("--run-id")
    preflight.add_argument("--min-free-gb", type=float, default=20.0)
    preflight.add_argument("--require-gpu", action="store_true")
    preflight.add_argument("--report")

    pilot = subcommands.add_parser(
        "pilot", help="Run a workflow-specific bounded pilot with full provenance."
    )
    pilot.add_argument("workflow", choices=sorted(WORKFLOWS))
    pilot.add_argument("--run-id", required=True)
    pilot.add_argument("--min-free-gb", type=float, default=20.0)
    pilot.add_argument("--require-gpu", action="store_true")
    pilot.add_argument("--dry-run", action="store_true")

    reproduce = subcommands.add_parser(
        "reproduce", help="Rerun a recorded operation into a new immutable run ID."
    )
    reproduce.add_argument("manifest")
    reproduce.add_argument("--run-id", required=True)
    reproduce.add_argument("--allow-code-drift", action="store_true")
    reproduce.add_argument("--dry-run", action="store_true")
    return command


def run_cli(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_args:
        separator = raw_args.index("--")
        operation_args = raw_args[:separator]
        pipeline_args = raw_args[separator + 1 :]
    else:
        operation_args = raw_args
        pipeline_args = []
    args = parser().parse_args(operation_args)
    if args.operation == "reproduce" and pipeline_args:
        raise ValueError("reproduce does not accept pipeline arguments after --.")
    args.pipeline_args = pipeline_args
    if args.operation == "preflight":
        spec = WORKFLOWS[args.workflow]
        tokens = _pipeline_tokens(args.pipeline_args)
        if args.run_id and spec.output_flag == "--output-dir":
            run_dir = OUTPUTS_DIR / "runs" / args.run_id
            tokens = [*tokens, spec.output_flag, str(run_dir)]
        report = build_preflight_report(
            args.workflow,
            tokens,
            min_free_gb=args.min_free_gb,
            require_gpu=args.require_gpu,
        )
        _write_or_print_report(report, args.report)
        return 0 if report["passed"] else 2

    if args.operation == "pilot":
        command, run_dir = build_pilot_command(
            args.workflow,
            args.run_id,
            args.pipeline_args,
        )
        module_index = command.index(WORKFLOWS[args.workflow].module)
        preflight = build_preflight_report(
            args.workflow,
            command[module_index + 1 :],
            min_free_gb=args.min_free_gb,
            require_gpu=args.require_gpu,
        )
        if not preflight["passed"]:
            _write_or_print_report(preflight, None)
            return 2
        manifest = _manifest_payload(
            kind="pilot",
            workflow=args.workflow,
            command=command,
            run_dir=run_dir,
            preflight=preflight,
        )
        return execute_operation(manifest, dry_run=args.dry_run)

    manifest = build_reproduction(
        Path(args.manifest),
        args.run_id,
        allow_code_drift=args.allow_code_drift,
    )
    return execute_operation(manifest, dry_run=args.dry_run)


def main() -> None:
    raise SystemExit(run_cli())


def preflight_main() -> None:
    raise SystemExit(run_cli(["preflight", *sys.argv[1:]]))


def pilot_main() -> None:
    raise SystemExit(run_cli(["pilot", *sys.argv[1:]]))


def reproduce_main() -> None:
    raise SystemExit(run_cli(["reproduce", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
