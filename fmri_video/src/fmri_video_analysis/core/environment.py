"""Runtime isolation and provenance for the maintained analysis package."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import site
import subprocess
import sys
from pathlib import Path

EXPECTED_CONDA_ENV = "video"
TRACKED_PACKAGES = (
    "fmri-video-analysis",
    "matplotlib",
    "networkx",
    "nibabel",
    "nilearn",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "torch",
)


def require_video_environment() -> None:
    """Refuse to import maintained analysis code outside the video environment."""

    prefix_name = Path(sys.prefix).name
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if prefix_name != EXPECTED_CONDA_ENV or conda_env not in {
        None,
        EXPECTED_CONDA_ENV,
    }:
        raise RuntimeError(
            "fmri_video_analysis must run in the dedicated 'video' Conda "
            f"environment; CONDA_DEFAULT_ENV={conda_env!r}, sys.prefix={sys.prefix!r}. "
            "Use 'conda run -n video ...'."
        )
    if site.ENABLE_USER_SITE:
        raise RuntimeError(
            "User site-packages are enabled. Run with PYTHONNOUSERSITE=1 or use "
            "the configured video Conda environment."
        )

    os.environ.setdefault("MPLBACKEND", "Agg")
    # The server exports a stale remote DISPLAY. Inkscape 0.92, imported by
    # pycortex, fails even for --version while that variable is present.
    if os.environ.get("FMRI_VIDEO_HEADLESS", "1") == "1":
        os.environ.pop("DISPLAY", None)


def _resolve_executable(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is not None:
        return executable
    roots = (
        os.environ.get("FSLDIR"),
        os.environ.get("FREESURFER_HOME"),
    )
    return next(
        (
            str(candidate)
            for root in roots
            if root
            for candidate in [Path(root) / "bin" / command]
            if candidate.is_file()
        ),
        None,
    )


def _command_version(command: str, *args: str) -> str | None:
    executable = _resolve_executable(command)
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    version_lines = (completed.stdout or completed.stderr).strip().splitlines()
    return version_lines[0] if version_lines else None


def _read_version_file(path: str) -> str | None:
    version_path = Path(path)
    if not version_path.is_file():
        return None
    return version_path.read_text(encoding="utf-8").strip()


def runtime_metadata() -> dict:
    """Return auditable environment metadata for an analysis run."""

    packages = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "user_site_enabled": site.ENABLE_USER_SITE,
        "packages": packages,
        "external_tools": {
            "fsl_version": _read_version_file("/usr/local/fsl/etc/fslversion"),
            "randomise": _resolve_executable("randomise"),
            "fslmaths": _resolve_executable("fslmaths"),
            "freesurfer_version": _read_version_file(
                "/usr/local/freesurfer/build-stamp.txt"
            ),
            "mri_convert": _resolve_executable("mri_convert"),
            "pandoc": _command_version("pandoc", "--version"),
            "ffmpeg": _command_version("ffmpeg", "-version"),
        },
    }
