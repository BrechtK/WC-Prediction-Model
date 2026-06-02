import pytest
from scipy.optimize import OptimizeResult

from wc_predictor.calibration import CALIBRATION_STARTING_POINTS, CalibrationTargets, calibrate_poisson_model
from wc_predictor.probabilities import poisson_market_probabilities, poisson_score_matrix


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


def test_btts_constraint_moves_btts_probability_towards_target() -> None:
    targets = CalibrationTargets(a_win=0.45, draw=0.28, b_win=0.27)
    without_btts = calibrate_poisson_model(targets)
    with_btts = calibrate_poisson_model(CalibrationTargets(0.45, 0.28, 0.27, btts_yes=0.75))
    assert abs(with_btts.model_probabilities["btts_yes"] - 0.75) < abs(
        without_btts.model_probabilities["btts_yes"] - 0.75
    )


def test_calibration_uses_full_distribution_even_when_score_grid_is_small() -> None:
    source = poisson_market_probabilities(4.5, 3.8)
    result = calibrate_poisson_model(
        CalibrationTargets(
            source["a_win"],
            source["draw"],
            source["b_win"],
            source["over_2_5"],
            source["btts_yes"],
        ),
        max_goals=2,
    )
    assert result.lambda_a == pytest.approx(4.5, abs=0.05)
    assert result.lambda_b == pytest.approx(3.8, abs=0.05)
    assert result.score_matrix.tail_probability > 0.50
    assert result.model_probabilities["over_2_5"] == pytest.approx(source["over_2_5"], abs=1e-4)


def test_calibrated_lambdas_respect_positive_bounds() -> None:
    result = calibrate_poisson_model(CalibrationTargets(0.98, 0.01, 0.01))
    assert 0.02 <= result.lambda_a <= 7.0
    assert 0.02 <= result.lambda_b <= 7.0
    assert "lambda_a_near_bound" in result.warnings


def test_calibration_handles_extreme_favourite_market() -> None:
    result = calibrate_poisson_model(CalibrationTargets(0.995, 0.004, 0.001))

    assert result.success
    assert result.model_probabilities["a_win"] > 0.98
    assert "lambda_a_near_bound" in result.warnings or "lambda_b_near_bound" in result.warnings


def test_calibration_raises_clear_error_if_all_starts_fail(monkeypatch) -> None:
    calls = []

    def failed_minimize(*args, **kwargs):
        calls.append(kwargs["x0"])
        return OptimizeResult(success=False, message="forced failure", fun=float("inf"), x=kwargs["x0"])

    monkeypatch.setattr("wc_predictor.calibration.minimize", failed_minimize)

    with pytest.raises(RuntimeError, match="Poisson calibration failed for all starting points"):
        calibrate_poisson_model(CalibrationTargets(0.45, 0.30, 0.25))
    assert len(calls) == len(CALIBRATION_STARTING_POINTS)


def test_calibration_requires_at_least_one_starting_point() -> None:
    with pytest.raises(ValueError, match="At least one Poisson calibration starting point"):
        calibrate_poisson_model(CalibrationTargets(0.45, 0.30, 0.25), starting_points=())


def test_invalid_calibration_targets_raise() -> None:
    with pytest.raises(ValueError):
        CalibrationTargets(0.5, 0.3, 0.3)
