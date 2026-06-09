"""Asian handicap settlement and fair expected-profit pricing."""

from __future__ import annotations

import math

import numpy as np

VALID_HANDICAP_SIDES = {"team_a", "team_b"}


def asian_handicap_components(handicap: float) -> tuple[tuple[float, float], ...]:
    """Return half-stake components for an integer, half, or quarter handicap."""

    value = float(handicap)
    if not np.isfinite(value):
        raise ValueError("Asian handicap line must be finite")
    sign = -1.0 if value < 0 else 1.0
    absolute = abs(value)
    integer = math.floor(absolute)
    remainder = absolute - integer
    if np.isclose(remainder, 0.0):
        components = ((float(integer), 1.0),)
    elif np.isclose(remainder, 0.5):
        components = ((float(integer) + 0.5, 1.0),)
    elif np.isclose(remainder, 0.25):
        components = ((float(integer), 0.5), (float(integer) + 0.5, 0.5))
    elif np.isclose(remainder, 0.75):
        components = ((float(integer) + 0.5, 0.5), (float(integer) + 1.0, 0.5))
    else:
        raise ValueError("Asian handicap line must be integer, half-goal, or quarter-goal")
    return tuple((sign * line, stake) for line, stake in components)


def asian_handicap_component_profit(
    score_a: int | float,
    score_b: int | float,
    handicap: float,
    side: str,
    decimal_odds: float,
) -> float:
    """Return profit for one unit stake on a single handicap component."""

    if side not in VALID_HANDICAP_SIDES:
        raise ValueError("Asian handicap side must be 'team_a' or 'team_b'")
    odds = float(decimal_odds)
    if not np.isfinite(odds) or odds <= 1.0:
        raise ValueError("Decimal odds must be finite and greater than 1")
    adjusted_margin = float(score_a) - float(score_b) + float(handicap)
    if side == "team_b":
        adjusted_margin = -adjusted_margin
    if adjusted_margin > 0:
        return odds - 1.0
    if np.isclose(adjusted_margin, 0.0):
        return 0.0
    return -1.0


def asian_handicap_profit(
    score_a: int | float,
    score_b: int | float,
    handicap: float,
    side: str,
    decimal_odds: float,
) -> float:
    """Return settled profit for a full unit stake on an Asian handicap."""

    return float(
        sum(
            stake * asian_handicap_component_profit(score_a, score_b, component, side, decimal_odds)
            for component, stake in asian_handicap_components(handicap)
        )
    )


def asian_handicap_profit_vector(
    shape: tuple[int, int],
    handicap: float,
    side: str,
    decimal_odds: float,
) -> np.ndarray:
    """Return per-scoreline Asian-handicap profit over a score matrix shape."""

    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Score matrix shape must be two positive dimensions")
    if side not in VALID_HANDICAP_SIDES:
        raise ValueError("Asian handicap side must be 'team_a' or 'team_b'")
    odds = float(decimal_odds)
    if not np.isfinite(odds) or odds <= 1.0:
        raise ValueError("Decimal odds must be finite and greater than 1")
    scores_a, scores_b = np.indices(shape)
    margin = scores_a - scores_b
    profit = np.zeros(shape, dtype=float)
    for component, stake in asian_handicap_components(handicap):
        adjusted = margin + component
        if side == "team_b":
            adjusted = -adjusted
        component_profit = np.where(
            adjusted > 0,
            odds - 1.0,
            np.where(np.isclose(adjusted, 0.0), 0.0, -1.0),
        )
        profit += stake * component_profit
    return profit


def asian_handicap_expected_profit(
    probabilities: np.ndarray,
    handicap: float,
    side: str,
    decimal_odds: float,
) -> float:
    """Return expected profit for a unit stake against a score probability matrix."""

    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("Probabilities must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0) or matrix.sum() <= 0:
        raise ValueError("Probabilities must be finite, non-negative, and positive in total")
    normalised = matrix / matrix.sum()
    return float((normalised * asian_handicap_profit_vector(normalised.shape, handicap, side, decimal_odds)).sum())
