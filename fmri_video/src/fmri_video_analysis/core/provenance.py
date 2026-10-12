"""Atomic provenance and strict cache contracts for maintained analyses."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fmri_video_analysis.core.environment import runtime_metadata

CACHE_SCHEMA_VERSION = 2
RUN_SCHEMA_VERSION = 2


class CacheMismatchError(RuntimeError):
    """Raised when an existing artifact does not match its cache contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_fingerprint(path: Path, *, full_hash: bool) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": None if full_hash else stat.st_mtime_ns,
        "sha256": sha256_file(resolved) if full_hash else None,
    }


def atomic_write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def git_metadata(project_root: Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()

    def command(*args: str) -> tuple[int, str]:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *args],
            check=False,
            capture_output=True,
            text=False,
        )
        return completed.returncode, completed.stdout.decode(
            "utf-8", errors="replace"
        ).strip()

    commit_status, commit = command("rev-parse", "HEAD")
    branch_status, branch = command("branch", "--show-current")
    status_code, status = command("status", "--porcelain=v1", "--untracked-files=all")
    diff = subprocess.run(
        ["git", "-C", str(project_root), "diff", "--no-ext-diff", "--binary", "HEAD"],
        check=False,
        capture_output=True,
    )
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=False,
        capture_output=True,
    )
    untracked_digest = hashlib.sha256()
    untracked_count = 0
    if untracked.returncode == 0:
        for encoded_path in sorted(
            value for value in untracked.stdout.split(b"\x00") if value
        ):
            relative = encoded_path.decode("utf-8", errors="surrogateescape")
            candidate = project_root / relative
            untracked_digest.update(encoded_path)
            untracked_digest.update(b"\x00")
            if candidate.is_symlink():
                untracked_digest.update(candidate.readlink().as_posix().encode("utf-8"))
            elif candidate.is_file():
                untracked_digest.update(sha256_file(candidate).encode("ascii"))
            else:
                untracked_digest.update(b"<missing-or-non-file>")
            untracked_count += 1
    return {
        "commit": commit if commit_status == 0 else None,
        "branch": branch if branch_status == 0 else None,
        "dirty": bool(status) if status_code == 0 else None,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest()
        if status_code == 0
        else None,
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest()
        if diff.returncode == 0
        else None,
        "untracked_count": untracked_count if untracked.returncode == 0 else None,
        "untracked_content_sha256": (
            untracked_digest.hexdigest() if untracked.returncode == 0 else None
        ),
    }


def cache_manifest(
    *,
    stage: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
    code: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage": stage,
        "parameters": parameters,
        "inputs": inputs,
        "code": code or {},
    }
    payload["signature_sha256"] = stable_hash(payload)
    return payload


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(payload):
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(payload[key], child))
        return flattened
    if isinstance(payload, list):
        flattened = {}
        for index, value in enumerate(payload):
            flattened.update(_flatten(value, f"{prefix}[{index}]"))
        return flattened
    return {prefix: payload}


def manifest_differences(observed: dict, expected: dict) -> list[str]:
    observed_flat = _flatten(observed)
    expected_flat = _flatten(expected)
    differences = []
    for key in sorted(set(observed_flat) | set(expected_flat)):
        old = observed_flat.get(key, "<missing>")
        new = expected_flat.get(key, "<missing>")
        if old != new:
            differences.append(f"{key}: cached={old!r}, requested={new!r}")
    return differences


def require_matching_cache(
    manifest_path: Path,
    expected: dict,
    *,
    artifacts_exist: bool,
    force: bool,
) -> bool:
    """Return True only when an existing cache is safe to reuse."""

    manifest_path = Path(manifest_path)
    if force:
        return False
    if not artifacts_exist:
        return False
    if not manifest_path.is_file():
        raise CacheMismatchError(
            f"Existing artifacts have no cache manifest: {manifest_path}. "
            "Use --force to recompute them under the current contract."
        )
    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    differences = manifest_differences(observed, expected)
    if differences:
        preview = "\n".join(f"- {item}" for item in differences[:30])
        raise CacheMismatchError(
            f"Cache configuration mismatch for {manifest_path}:\n{preview}\n"
            "Use a new run directory or pass --force to recompute."
        )
    return True


def write_stage_metadata(
    run_dir: Path,
    stage: str,
    payload: dict[str, Any],
    *,
    project_root: Path,
    parent_run: Path | None = None,
) -> Path:
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "metadata/run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != run_dir.name:
            raise RuntimeError(
                f"Run manifest ID {manifest.get('run_id')!r} does not match {run_dir.name!r}."
            )
    else:
        manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_dir.name,
            "output_dir": str(run_dir),
            "parent_run": str(Path(parent_run).resolve()) if parent_run else None,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stages": {},
        }
    stage_payload = {
        **payload,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_metadata(),
        "git": git_metadata(project_root),
    }
    manifest["stages"][stage] = stage_payload
    atomic_write_json(run_dir / f"metadata/stages/{stage}.json", stage_payload)
    atomic_write_json(manifest_path, manifest)
    return manifest_path
