"""Scoreline probability matrix representation and probability summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson, skellam


@dataclass(frozen=True)
class ScoreProbabilityMatrix:
    """A finite scoreline grid with explicit diagnostic tail mass."""

    probabilities: np.ndarray
    tail_probability: float = 0.0
    renormalised: bool = True
    lambda_a: float | None = None
    lambda_b: float | None = None

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=float)
        if probabilities.ndim != 2 or min(probabilities.shape) == 0:
            raise ValueError("Score probabilities must be a non-empty two-dimensional matrix")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("Score probabilities must be finite and non-negative")
        grid_probability = float(probabilities.sum())
        if grid_probability <= 0:
            raise ValueError("Score probabilities must contain positive mass")
        if not np.isfinite(self.tail_probability) or not 0 <= self.tail_probability <= 1:
            raise ValueError("Tail probability must lie between zero and one")
        if self.renormalised and not np.isclose(grid_probability, 1.0, atol=1e-9):
            raise ValueError("A renormalised score matrix must sum to one")
        if not self.renormalised and not np.isclose(grid_probability + self.tail_probability, 1.0, atol=1e-9):
            raise ValueError("A raw score matrix plus its tail probability must sum to one")
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def grid_probability(self) -> float:
        """Probability represented by the score grid after optional normalisation."""

        return float(self.probabilities.sum())

    def outcome_probabilities(self) -> dict[str, float]:
        """Return model-implied A-win, draw, and B-win probabilities."""

        return {
            "a_win": float(np.tril(self.probabilities, k=-1).sum()),
            "draw": float(np.trace(self.probabilities)),
            "b_win": float(np.triu(self.probabilities, k=1).sum()),
        }

    def over_2_5_probability(self) -> float:
        """Return the probability of at least three total goals."""

        scores_a, scores_b = np.indices(self.probabilities.shape)
        return float(self.probabilities[scores_a + scores_b >= 3].sum())

    def btts_yes_probability(self) -> float:
        """Return the probability that both teams score."""

        return float(self.probabilities[1:, 1:].sum())

    def exact_score_probability(self, goals_a: int, goals_b: int) -> float:
        """Return scoreline probability, or zero outside the finite grid."""

        if goals_a < 0 or goals_b < 0:
            return 0.0
        if goals_a >= self.probabilities.shape[0] or goals_b >= self.probabilities.shape[1]:
            return 0.0
        return float(self.probabilities[goals_a, goals_b])

    def goal_difference_probability(self, difference: int) -> float:
        """Return probability mass having the supplied A-minus-B goal difference."""

        scores_a, scores_b = np.indices(self.probabilities.shape)
        return float(self.probabilities[scores_a - scores_b == difference].sum())

    def result_probability(self, sign: int) -> float:
        """Return A-win, draw, or B-win probability for sign 1, 0, or -1."""

        if sign not in {-1, 0, 1}:
            raise ValueError("Result sign must be -1, 0, or 1")
        outcomes = self.outcome_probabilities()
        return outcomes[{1: "a_win", 0: "draw", -1: "b_win"}[sign]]

    def most_likely_scoreline(self) -> tuple[int, int]:
        """Return the highest-probability scoreline in the finite grid."""

        return tuple(int(value) for value in np.unravel_index(np.argmax(self.probabilities), self.probabilities.shape))


def poisson_score_matrix(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 8,
    renormalise: bool = True,
) -> ScoreProbabilityMatrix:
    """Build an independent-Poisson score grid from 0-0 through max_goals-max_goals."""

    if not np.isfinite(lambda_a) or not np.isfinite(lambda_b) or lambda_a <= 0 or lambda_b <= 0:
        raise ValueError("Poisson lambdas must be finite and positive")
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")
    scores = np.arange(max_goals + 1)
    probabilities = np.outer(poisson.pmf(scores, lambda_a), poisson.pmf(scores, lambda_b))
    represented_mass = float(probabilities.sum())
    tail_probability = max(0.0, 1.0 - represented_mass)
    if renormalise:
        probabilities = probabilities / represented_mass
    return ScoreProbabilityMatrix(probabilities, tail_probability, renormalise, lambda_a, lambda_b)


def poisson_market_probabilities(lambda_a: float, lambda_b: float) -> dict[str, float]:
    """Return full-distribution market summaries without finite-grid truncation."""

    if not np.isfinite(lambda_a) or not np.isfinite(lambda_b) or lambda_a <= 0 or lambda_b <= 0:
        raise ValueError("Poisson lambdas must be finite and positive")
    return {
        "a_win": float(skellam.sf(0, lambda_a, lambda_b)),
        "draw": float(skellam.pmf(0, lambda_a, lambda_b)),
        "b_win": float(skellam.cdf(-1, lambda_a, lambda_b)),
        "over_2_5": float(poisson.sf(2, lambda_a + lambda_b)),
        "btts_yes": float((1.0 - np.exp(-lambda_a)) * (1.0 - np.exp(-lambda_b))),
    }
