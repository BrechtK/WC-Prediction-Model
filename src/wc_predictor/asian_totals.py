"""Asian total-goals settlement and fair expected-profit pricing."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


VALID_TOTAL_SIDES = {"over", "under"}


def asian_total_components(line: float) -> tuple[tuple[float, float], ...]:
    """Return half-stake components for a half, integer, or quarter total line."""

    value = float(line)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Total-goals line must be finite and non-negative")
    integer = math.floor(value)
    remainder = value - integer
    if np.isclose(remainder, 0.0):
        return ((float(integer), 1.0),)
    if np.isclose(remainder, 0.5):
        return ((float(integer) + 0.5, 1.0),)
    if np.isclose(remainder, 0.25):
        return ((float(integer), 0.5), (float(integer) + 0.5, 0.5))
    if np.isclose(remainder, 0.75):
        return ((float(integer) + 0.5, 0.5), (float(integer) + 1.0, 0.5))
    raise ValueError("Asian total-goals line must be integer, half-goal, or quarter-goal")


def asian_total_component_profit(total_goals: int | float, line: float, side: str, decimal_odds: float) -> float:
    """Return profit for one unit stake on a single integer or half-goal component."""

    if side not in VALID_TOTAL_SIDES:
        raise ValueError("Total-goals side must be 'over' or 'under'")
    odds = float(decimal_odds)
    if not np.isfinite(odds) or odds <= 1.0:
        raise ValueError("Decimal odds must be finite and greater than 1")
    total = float(total_goals)
    if side == "over":
        if total > line:
            return odds - 1.0
        if np.isclose(total, line):
            return 0.0
        return -1.0
    if total < line:
        return odds - 1.0
    if np.isclose(total, line):
        return 0.0
    return -1.0


def asian_total_profit(total_goals: int | float, line: float, side: str, decimal_odds: float) -> float:
    """Return settled profit for a full unit stake on a total-goals line."""

    return float(
        sum(
            stake * asian_total_component_profit(total_goals, component_line, side, decimal_odds)
            for component_line, stake in asian_total_components(line)
        )
    )


def asian_total_profit_vector(
    shape: tuple[int, int],
    line: float,
    side: str,
    decimal_odds: float,
) -> np.ndarray:
    """Return per-scoreline Asian-total profit over a score matrix shape."""

    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Score matrix shape must be two positive dimensions")
    scores_a, scores_b = np.indices(shape)
    totals = scores_a + scores_b
    profit = np.zeros(shape, dtype=float)
    for component_line, stake in asian_total_components(line):
        if side == "over":
            component = np.where(
                totals > component_line,
                decimal_odds - 1.0,
                np.where(np.isclose(totals, component_line), 0.0, -1.0),
            )
        elif side == "under":
            component = np.where(
                totals < component_line,
                decimal_odds - 1.0,
                np.where(np.isclose(totals, component_line), 0.0, -1.0),
            )
        else:
            raise ValueError("Total-goals side must be 'over' or 'under'")
        profit += stake * component
    return profit


def asian_total_expected_profit(
    probabilities: np.ndarray,
    line: float,
    side: str,
    decimal_odds: float,
) -> float:
    """Return expected profit for a unit stake against a score probability matrix."""

    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("Probabilities must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    total_mass = float(matrix.sum())
    if total_mass <= 0:
        raise ValueError("Probabilities must contain positive mass")
    normalised = matrix / total_mass
    return float((normalised * asian_total_profit_vector(normalised.shape, line, side, decimal_odds)).sum())


def asian_total_expected_profit_from_distribution(
    total_goal_probabilities: Iterable[float],
    line: float,
    side: str,
    decimal_odds: float,
) -> float:
    """Return expected Asian-total profit from a total-goals probability vector."""

    probabilities = np.asarray(list(total_goal_probabilities), dtype=float)
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("Total-goals probabilities must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("Total-goals probabilities must be finite, non-negative, and positive in total")
    probabilities = probabilities / probabilities.sum()
    profits = np.array(
        [asian_total_profit(total_goals, line, side, decimal_odds) for total_goals in range(len(probabilities))],
        dtype=float,
    )
    return float(np.dot(probabilities, profits))
