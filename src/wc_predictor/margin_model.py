"""Diagnostic Skellam margin model fitted to Asian-handicap lines.

Asian handicap prices the goal-difference (margin ``X - Y``) distribution
directly. This module fits a Skellam distribution to the fair cover
probabilities of the supplied half-goal Asian-handicap lines and compares the
implied margin distribution with the calibrated independent-Poisson margins.

It is strictly diagnostic: it never changes the default EV recommendation. It
fits only push-free half-goal lines so cover probabilities map cleanly onto
``P(margin >= threshold)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skellam

MIN_LINES_FOR_SKELLAM_FIT = 2
_SKELLAM_BOUNDS = ((0.02, 8.0), (0.02, 8.0))


@dataclass(frozen=True)
class SkellamMarginModel:
    """Fitted Skellam margin model and diagnostics."""

    margin_model_type: str
    fitted_margin_parameters: dict[str, float]
    margin_fit_error: float
    margin_distribution_comparison: str
    recommendation_if_margin_adjusted: str
    warnings: str
    lines_used: int


def _cover_threshold(handicap: float) -> int | None:
    """Return the inclusive margin threshold a team-A half handicap covers.

    Team A covers when ``X - Y + handicap > 0`` i.e. ``margin > -handicap``.
    For a half-goal handicap this is the integer ``floor(-handicap) + 1``.
    """

    value = float(handicap)
    if not np.isfinite(value):
        return None
    if not np.isclose(abs(value) % 1, 0.5):
        return None
    return int(math.floor(-value) + 1)


def _margin_distribution(mu1: float, mu2: float, lower: int, upper: int) -> dict[int, float]:
    return {margin: float(skellam.pmf(margin, mu1, mu2)) for margin in range(lower, upper + 1)}


def _poisson_margin_distribution(lambda_a: float, lambda_b: float, lower: int, upper: int) -> dict[int, float]:
    return _margin_distribution(lambda_a, lambda_b, lower, upper)


def fit_skellam_margin_model(
    asian_handicap_rows: pd.DataFrame | None,
    *,
    lambda_a: float,
    lambda_b: float,
    comparison_range: int = 4,
) -> SkellamMarginModel | None:
    """Fit a Skellam margin model to half-goal Asian-handicap cover probabilities.

    Returns ``None`` when no Asian-handicap data is supplied. Returns a model
    with a skip warning when too few usable half-goal lines are present.
    """

    if asian_handicap_rows is None or asian_handicap_rows.empty:
        return None
    if "handicap" not in asian_handicap_rows or "fair_team_a" not in asian_handicap_rows:
        return None

    observations: list[tuple[int, float]] = []
    for _, row in asian_handicap_rows.iterrows():
        threshold = _cover_threshold(row["handicap"])
        if threshold is None:
            continue
        fair_team_a = float(row["fair_team_a"])
        if not np.isfinite(fair_team_a) or not 0 < fair_team_a < 1:
            continue
        observations.append((threshold, fair_team_a))

    if len(observations) < MIN_LINES_FOR_SKELLAM_FIT:
        return SkellamMarginModel(
            margin_model_type="skellam",
            fitted_margin_parameters={},
            margin_fit_error=float("nan"),
            margin_distribution_comparison="",
            recommendation_if_margin_adjusted="",
            warnings=f"insufficient_half_goal_lines:{len(observations)}",
            lines_used=len(observations),
        )

    thresholds = np.array([threshold for threshold, _ in observations], dtype=float)
    targets = np.array([fair for _, fair in observations], dtype=float)

    def objective(parameters: np.ndarray) -> float:
        mu1, mu2 = float(parameters[0]), float(parameters[1])
        predicted = np.array([skellam.sf(int(t) - 1, mu1, mu2) for t in thresholds], dtype=float)
        return float(np.mean((predicted - targets) ** 2))

    best = None
    for start in ((lambda_a, lambda_b), (1.3, 1.1), (2.0, 0.6)):
        try:
            fitted = minimize(objective, x0=np.asarray(start, dtype=float), method="L-BFGS-B", bounds=_SKELLAM_BOUNDS)
        except Exception:  # pragma: no cover - scipy failures are rare but must not crash live runs
            continue
        if fitted.success and np.all(np.isfinite(fitted.x)) and (best is None or fitted.fun < best.fun):
            best = fitted
    if best is None:
        return SkellamMarginModel(
            margin_model_type="skellam",
            fitted_margin_parameters={},
            margin_fit_error=float("nan"),
            margin_distribution_comparison="",
            recommendation_if_margin_adjusted="",
            warnings="skellam_fit_failed",
            lines_used=len(observations),
        )

    mu1, mu2 = float(best.x[0]), float(best.x[1])
    fit_error = float(math.sqrt(objective(best.x)))
    skellam_margins = _margin_distribution(mu1, mu2, -comparison_range, comparison_range)
    poisson_margins = _poisson_margin_distribution(lambda_a, lambda_b, -comparison_range, comparison_range)
    comparison = "; ".join(
        f"{margin:+d}: skellam={skellam_margins[margin]:.4f} poisson={poisson_margins[margin]:.4f}"
        for margin in range(-comparison_range, comparison_range + 1)
    )
    return SkellamMarginModel(
        margin_model_type="skellam",
        fitted_margin_parameters={"mu1": mu1, "mu2": mu2, "implied_mean_margin": mu1 - mu2},
        margin_fit_error=fit_error,
        margin_distribution_comparison=comparison,
        recommendation_if_margin_adjusted="",
        warnings="",
        lines_used=len(observations),
    )
