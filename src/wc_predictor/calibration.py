"""Calibration of independent-Poisson lambdas to market-implied targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from wc_predictor.config import CalibrationWeights
from wc_predictor.probabilities import ScoreProbabilityMatrix, poisson_market_probabilities, poisson_score_matrix

LAMBDA_BOUNDS = ((0.02, 7.0), (0.02, 7.0))
LAMBDA_BOUND_WARNING_TOLERANCE = 0.05
CALIBRATION_STARTING_POINTS = (
    (1.35, 1.05),
    (0.40, 0.20),
    (2.50, 0.50),
    (0.50, 2.50),
    (3.00, 0.30),
    (0.30, 3.00),
)
SINGLE_START_CALIBRATION_POINTS = (CALIBRATION_STARTING_POINTS[0],)


@dataclass(frozen=True)
class CalibrationTargets:
    """Market-implied targets used to fit a scoreline model."""

    a_win: float
    draw: float
    b_win: float
    over_2_5: float | None = None
    btts_yes: float | None = None

    def __post_init__(self) -> None:
        one_x_two = np.asarray([self.a_win, self.draw, self.b_win], dtype=float)
        if not np.all(np.isfinite(one_x_two)) or np.any(one_x_two <= 0):
            raise ValueError("1X2 targets must be finite and positive")
        if not np.isclose(one_x_two.sum(), 1.0, atol=1e-6):
            raise ValueError("1X2 targets must sum to one")
        for name in ("over_2_5", "btts_yes"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0 < value < 1):
                raise ValueError(f"{name} target must lie strictly between zero and one")


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted lambdas, model diagnostics, and calibrated matrix."""

    lambda_a: float
    lambda_b: float
    score_matrix: ScoreProbabilityMatrix
    target_probabilities: dict[str, float]
    model_probabilities: dict[str, float]
    loss: float
    success: bool
    warnings: tuple[str, ...] = ()


def calibrate_poisson_model(
    targets: CalibrationTargets,
    max_goals: int = 8,
    weights: CalibrationWeights | None = None,
    renormalise: bool = True,
    poor_fit_threshold: float = 0.01,
    starting_points: Sequence[tuple[float, float]] | None = None,
) -> CalibrationResult:
    """Fit positive independent-Poisson lambdas to market-implied probabilities."""

    weights = weights or CalibrationWeights()
    starting_points = tuple(CALIBRATION_STARTING_POINTS if starting_points is None else starting_points)
    if not starting_points:
        raise ValueError("At least one Poisson calibration starting point is required")

    def objective(lambdas: np.ndarray) -> float:
        outcomes = poisson_market_probabilities(float(lambdas[0]), float(lambdas[1]))
        loss = weights.one_x_two * (
            (outcomes["a_win"] - targets.a_win) ** 2
            + (outcomes["draw"] - targets.draw) ** 2
            + (outcomes["b_win"] - targets.b_win) ** 2
        )
        if targets.over_2_5 is not None:
            loss += weights.over_under_2_5 * (outcomes["over_2_5"] - targets.over_2_5) ** 2
        if targets.btts_yes is not None:
            loss += weights.btts * (outcomes["btts_yes"] - targets.btts_yes) ** 2
        return float(loss)

    successful_fits = []
    failure_messages: list[str] = []
    for starting_point in starting_points:
        initial = np.asarray(starting_point, dtype=float)
        if initial.shape != (2,) or not np.all(np.isfinite(initial)):
            raise ValueError("Poisson calibration starting points must contain two finite lambdas")
        try:
            fitted = minimize(
                objective,
                x0=initial,
                method="L-BFGS-B",
                bounds=LAMBDA_BOUNDS,
            )
        except Exception as error:  # pragma: no cover - scipy failures are uncommon, but must be surfaced clearly
            failure_messages.append(str(error))
            continue
        if fitted.success and np.isfinite(fitted.fun) and np.all(np.isfinite(fitted.x)):
            successful_fits.append(fitted)
        else:
            failure_messages.append(str(fitted.message))
    if not successful_fits:
        detail = "; ".join(failure_messages) or "no optimiser result"
        raise RuntimeError(f"Poisson calibration failed for all starting points: {detail}")
    fitted = min(successful_fits, key=lambda result: float(result.fun))
    lambda_a, lambda_b = (float(value) for value in fitted.x)
    matrix = poisson_score_matrix(lambda_a, lambda_b, max_goals, renormalise)
    model_probabilities = poisson_market_probabilities(lambda_a, lambda_b)
    target_probabilities = {
        "a_win": targets.a_win,
        "draw": targets.draw,
        "b_win": targets.b_win,
    }
    if targets.over_2_5 is not None:
        target_probabilities["over_2_5"] = targets.over_2_5
    if targets.btts_yes is not None:
        target_probabilities["btts_yes"] = targets.btts_yes

    warnings: list[str] = []
    loss = objective(fitted.x)
    if loss > poor_fit_threshold:
        warnings.append(f"Calibration loss {loss:.6f} exceeds threshold {poor_fit_threshold:.6f}")
    if matrix.tail_probability > 0.01:
        warnings.append(f"Score grid omits {matrix.tail_probability:.2%} raw tail probability before normalisation")
    for label, value, bounds in (
        ("lambda_a", lambda_a, LAMBDA_BOUNDS[0]),
        ("lambda_b", lambda_b, LAMBDA_BOUNDS[1]),
    ):
        if value - bounds[0] <= LAMBDA_BOUND_WARNING_TOLERANCE or bounds[1] - value <= LAMBDA_BOUND_WARNING_TOLERANCE:
            warnings.append(f"{label}_near_bound")
    return CalibrationResult(
        lambda_a,
        lambda_b,
        matrix,
        target_probabilities,
        model_probabilities,
        loss,
        True,
        tuple(warnings),
    )
