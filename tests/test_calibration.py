import pytest

from wc_predictor.calibration import CalibrationTargets, calibrate_poisson_model
from wc_predictor.probabilities import poisson_score_matrix


def test_calibration_approximately_reproduces_poisson_1x2_targets() -> None:
    source = poisson_score_matrix(1.65, 0.95, max_goals=10)
    outcomes = source.outcome_probabilities()
    result = calibrate_poisson_model(CalibrationTargets(**outcomes), max_goals=10)
    assert result.lambda_a == pytest.approx(1.65, abs=0.05)
    assert result.lambda_b == pytest.approx(0.95, abs=0.05)


def test_over_under_constraint_moves_total_goals_probability_towards_target() -> None:
    targets = CalibrationTargets(a_win=0.45, draw=0.28, b_win=0.27)
    without_total = calibrate_poisson_model(targets)
    with_total = calibrate_poisson_model(CalibrationTargets(0.45, 0.28, 0.27, over_2_5=0.75))
    assert abs(with_total.model_probabilities["over_2_5"] - 0.75) < abs(
        without_total.model_probabilities["over_2_5"] - 0.75
    )


def test_invalid_calibration_targets_raise() -> None:
    with pytest.raises(ValueError):
        CalibrationTargets(0.5, 0.3, 0.3)

