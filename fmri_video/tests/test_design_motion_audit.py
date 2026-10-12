from __future__ import annotations

import numpy as np
import pandas as pd
from fmri_video_analysis.reporting.design_motion_audit import (
    _participant_design,
    design_schedule,
)


def test_design_schedule_balances_pair_order_but_not_absolute_position():
    schedule = design_schedule()
    runs = schedule.drop_duplicates("run_id")
    assert runs.simple_complex_order.value_counts().to_dict() == {
        "Simple_before_Complex": 2,
        "Complex_before_Simple": 2,
    }
    counts = pd.crosstab(schedule.condition, schedule.block_position)
    assert counts.loc["Simple"].tolist() == [1, 1, 2]
    assert counts.loc["Complex"].tolist() == [2, 1, 1]
    assert counts.loc["Control"].tolist() == [1, 2, 1]


def test_participant_design_is_full_rank_and_recovers_condition_effect():
    table = design_schedule().copy()
    matrix, names = _participant_design(table)
    assert np.linalg.matrix_rank(matrix) == matrix.shape[1]
    coefficients = np.asarray([0.2, 0.3, -0.1, 0.01, 0.02, 0.03, 0.04, 0.05])
    outcome = matrix @ coefficients
    fitted = np.linalg.lstsq(matrix, outcome, rcond=None)[0]
    assert names[1] == "complex_minus_simple"
    assert np.allclose(fitted, coefficients)
