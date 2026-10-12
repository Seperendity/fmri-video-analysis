#!/usr/bin/env python3
"""Run the normalized original MVPA pipeline without installation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fmri_video_analysis.pipelines.original_mvpa import main  # noqa: E402


if __name__ == "__main__":
    main()
