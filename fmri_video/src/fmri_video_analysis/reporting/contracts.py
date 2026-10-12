"""Input-layout and formal-inference contracts for generated reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LAYOUTS = ("normalized", "legacy")
FORMAL_STRUCTURED_PERMUTATIONS = 10_000


def result_dir(
    root: Path,
    normalized: str,
    legacy: str,
    *,
    layout: str,
) -> Path:
    """Resolve one module under one explicitly selected layout."""

    if layout not in LAYOUTS:
        raise ValueError(f"Unknown results layout {layout!r}; expected {LAYOUTS}.")
    relative = normalized if layout == "normalized" else legacy
    selected = Path(root) / relative
    if not selected.exists():
        raise FileNotFoundError(
            f"Missing {layout} report input: {selected}. "
            "A report cannot mix normalized and legacy module layouts."
        )
    return selected


def require_formal_mvpa(
    metrics_or_path: dict[str, Any] | Path,
    *,
    minimum_permutations: int = FORMAL_STRUCTURED_PERMUTATIONS,
) -> dict[str, Any]:
    """Reject manuscript-facing MVPA unless structured inference is complete."""

    if isinstance(metrics_or_path, dict):
        metrics = metrics_or_path
        source = "MVPA metrics"
    else:
        path = Path(metrics_or_path)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    completed = int(metrics.get("structured_permutations_completed", 0) or 0)
    p_value = metrics.get("structured_label_permutation_p")
    formal = bool(metrics.get("formal_inference_complete", False))
    if not formal or completed < minimum_permutations or p_value is None:
        raise RuntimeError(
            f"Formal MVPA inference is incomplete in {source}: "
            f"completed={completed}, required={minimum_permutations}, "
            f"structured_p={p_value!r}, formal_inference_complete={formal}. "
            "Resume the structured permutation run before generating a formal report."
        )
    return metrics
