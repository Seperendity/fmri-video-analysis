"""Frozen design constants for the four-run video fMRI analysis."""

from __future__ import annotations

CONDITIONS = ("Simple", "Complex", "Control")
RUN_IDS = ("0006", "0007", "0008", "0009")
EXCLUDED_SUBJECT_IDS = frozenset(
    {5, 7, 16, 31, 34, 38, 53, 62, 66, 67, 68, 73, 83}
)
INCLUDED_SUBJECT_IDS = tuple(
    subject_id
    for subject_id in range(1, 101)
    if subject_id not in EXCLUDED_SUBJECT_IDS
)

TR_SECONDS = 2.0
N_VOLUMES_PER_RUN = 130
BLOCK_ONSETS_TR = (10, 50, 90)
BLOCK_DURATION_TR = 30

# Each tuple is ordered by the three fixed onsets above.
RUN_CONDITION_ORDER = {
    "0006": ("Simple", "Complex", "Control"),
    "0007": ("Complex", "Control", "Simple"),
    "0008": ("Control", "Simple", "Complex"),
    "0009": ("Complex", "Control", "Simple"),
}

PILOT_SUBJECT_IDS = (23, 20, 81, 87)


def blocks_for_run(run_id: str) -> tuple[tuple[str, int, int], ...]:
    """Return (condition, onset TR, duration TR) for one run."""

    try:
        condition_order = RUN_CONDITION_ORDER[run_id]
    except KeyError as exc:
        raise ValueError(f"Unknown run ID: {run_id}") from exc
    return tuple(
        (condition, onset, BLOCK_DURATION_TR)
        for condition, onset in zip(condition_order, BLOCK_ONSETS_TR)
    )
