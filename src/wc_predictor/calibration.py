"""Calibration of independent-Poisson lambdas to market-implied targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from wc_predictor.config import CalibrationWeights
from wc_predictor.probabilities import ScoreProbabilityMatrix, poisson_score_matrix


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
) -> CalibrationResult:
    """Fit positive independent-Poisson lambdas to market-implied probabilities."""

    weights = weights or CalibrationWeights()

    def objective(lambdas: np.ndarray) -> float:
        matrix = poisson_score_matrix(float(lambdas[0]), float(lambdas[1]), max_goals, renormalise)
        outcomes = matrix.outcome_probabilities()
        loss = weights.one_x_two * (
            (outcomes["a_win"] - targets.a_win) ** 2
            + (outcomes["draw"] - targets.draw) ** 2
            + (outcomes["b_win"] - targets.b_win) ** 2
        )
        if targets.over_2_5 is not None:
            loss += weights.over_under_2_5 * (matrix.over_2_5_probability() - targets.over_2_5) ** 2
        if targets.btts_yes is not None:
            loss += weights.btts * (matrix.btts_yes_probability() - targets.btts_yes) ** 2
        return float(loss)

    fitted = minimize(
        objective,
        x0=np.asarray([1.35, 1.05]),
        method="L-BFGS-B",
        bounds=((0.02, 7.0), (0.02, 7.0)),
    )
    lambda_a, lambda_b = (float(value) for value in fitted.x)
    matrix = poisson_score_matrix(lambda_a, lambda_b, max_goals, renormalise)
    model_probabilities = matrix.outcome_probabilities()
    model_probabilities["over_2_5"] = matrix.over_2_5_probability()
    model_probabilities["btts_yes"] = matrix.btts_yes_probability()
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
    if not fitted.success:
        warnings.append(f"Optimisation did not converge: {fitted.message}")
    if loss > poor_fit_threshold:
        warnings.append(f"Calibration loss {loss:.6f} exceeds threshold {poor_fit_threshold:.6f}")
    if matrix.tail_probability > 0.01:
        warnings.append(f"Score grid omits {matrix.tail_probability:.2%} raw tail probability before normalisation")
    return CalibrationResult(
        lambda_a,
        lambda_b,
        matrix,
        target_probabilities,
        model_probabilities,
        loss,
        bool(fitted.success),
        tuple(warnings),
    )

