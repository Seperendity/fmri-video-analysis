from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fmri_video_analysis import cli
from fmri_video_analysis.core.provenance import git_metadata


def _background_args(tmp_path: Path, outputs_dir: Path) -> list[str]:
    timeseries = tmp_path / "timeseries"
    atlas = tmp_path / "atlases"
    timeseries.mkdir()
    atlas.mkdir()
    return [
        "--timeseries-root",
        str(timeseries),
        "--atlas-dir",
        str(atlas),
        "--output-dir",
        str(outputs_dir / "runs/run_test"),
    ]


def test_run_root_rejects_paths_outside_run_tree(tmp_path):
    outputs_dir = tmp_path / "outputs"
    with pytest.raises(ValueError, match="must be under"):
        cli._run_root(tmp_path / "elsewhere", outputs_dir)


def test_background_pilot_is_forced_to_frozen_subset(tmp_path):
    outputs_dir = tmp_path / "outputs"
    command, run_dir = cli.build_pilot_command(
        "background-fc",
        "pilot_test",
        ["--subjects", "1", "--stage", "audit"],
        outputs_dir=outputs_dir,
    )

    module_index = command.index(cli.WORKFLOWS["background-fc"].module)
    namespace = cli._parse_pipeline_args(
        cli.WORKFLOWS["background-fc"],
        command[module_index + 1 :],
    )
    assert namespace.subjects == ["pilot"]
    assert namespace.stage == "all"
    assert Path(namespace.output_dir) == run_dir
    assert run_dir == (outputs_dir / "runs/pilot_test").resolve()


def test_group_workflow_has_no_implicit_pilot(tmp_path):
    with pytest.raises(ValueError, match="no scientifically bounded pilot"):
        cli.build_pilot_command(
            "group-fc",
            "pilot_group",
            [],
            outputs_dir=tmp_path / "outputs",
        )


def test_preflight_checks_inputs_disk_and_run_boundary(tmp_path):
    outputs_dir = tmp_path / "outputs"
    args = _background_args(tmp_path, outputs_dir)
    report = cli.build_preflight_report(
        "background-fc",
        args,
        min_free_gb=0,
        outputs_dir=outputs_dir,
    )

    assert report["passed"]
    assert report["resolved_arguments"]["stage"] == "all"
    assert report["run_dir"] == str((outputs_dir / "runs/run_test").resolve())
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["timeseries_root"]["passed"]
    assert checks["atlas_dir"]["passed"]
    assert checks["free_disk_gb"]["passed"]


def test_reproduction_remaps_output_and_records_parent(tmp_path):
    outputs_dir = tmp_path / "outputs"
    pipeline_args = _background_args(tmp_path, outputs_dir)
    source_run = outputs_dir / "runs/run_test"
    command = [
        sys.executable,
        "-m",
        cli.WORKFLOWS["background-fc"].module,
        *pipeline_args,
    ]
    source = {
        "schema_version": cli.OPERATION_SCHEMA_VERSION,
        "kind": "pilot",
        "workflow": "background-fc",
        "command": command,
        "run_id": source_run.name,
        "run_dir": str(source_run),
        "git": git_metadata(cli.PROJECT_ROOT.parent),
    }
    manifest_path = tmp_path / "execution_manifest.json"
    manifest_path.write_text(json.dumps(source), encoding="utf-8")

    reproduced = cli.build_reproduction(
        manifest_path,
        "run_reproduced",
        outputs_dir=outputs_dir,
    )

    new_run = (outputs_dir / "runs/run_reproduced").resolve()
    assert reproduced["run_dir"] == str(new_run)
    assert reproduced["parent_manifest"] == str(manifest_path.resolve())
    output_index = len(reproduced["command"]) - 1
    assert reproduced["command"][output_index] == str(new_run)


def test_execute_operation_writes_observable_log(tmp_path):
    run_dir = tmp_path / "outputs/runs/pilot_exec"
    manifest = {
        "schema_version": cli.OPERATION_SCHEMA_VERSION,
        "kind": "pilot",
        "workflow": "revision",
        "command": [sys.executable, "-c", "print('pilot-ok')"],
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "status": "planned",
    }

    assert cli.execute_operation(manifest) == 0
    operation_dir = run_dir / "metadata/operations"
    recorded = json.loads(
        (operation_dir / "execution_manifest.json").read_text(encoding="utf-8")
    )
    assert recorded["status"] == "completed"
    assert recorded["return_code"] == 0
    assert "pilot-ok" in (operation_dir / "pilot.log").read_text(encoding="utf-8")


def test_run_cli_splits_operation_and_pipeline_arguments(monkeypatch, tmp_path):
    outputs_dir = tmp_path / "outputs"
    timeseries = tmp_path / "timeseries"
    atlas = tmp_path / "atlases"
    timeseries.mkdir()
    atlas.mkdir()
    captured = {}

    def fake_report(workflow, pipeline_args, **kwargs):
        captured["workflow"] = workflow
        captured["pipeline_args"] = pipeline_args
        return {"passed": True}

    monkeypatch.setattr(cli, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(cli, "build_preflight_report", fake_report)
    monkeypatch.setattr(cli, "_write_or_print_report", lambda report, path: None)

    result = cli.run_cli(
        [
            "preflight",
            "background-fc",
            "--run-id",
            "run_cli_test",
            "--min-free-gb",
            "0",
            "--",
            "--timeseries-root",
            str(timeseries),
            "--atlas-dir",
            str(atlas),
        ]
    )

    assert result == 0
    assert captured["workflow"] == "background-fc"
    assert captured["pipeline_args"][-2:] == [
        "--output-dir",
        str(outputs_dir / "runs/run_cli_test"),
    ]
