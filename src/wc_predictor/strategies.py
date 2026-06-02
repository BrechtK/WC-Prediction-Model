"""Reusable prediction strategy interfaces for later backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wc_predictor.optimiser import optimise_group_prediction
from wc_predictor.probabilities import ScoreProbabilityMatrix


class PredictionStrategy(Protocol):
    """Common interface for strategies evaluated by future backtests."""

    name: str

    def predict(self, matrix: ScoreProbabilityMatrix) -> tuple[int, int]:
        """Return a score prediction from a scoreline matrix."""


@dataclass(frozen=True)
class FixedScoreStrategy:
    """Always predict one configured scoreline."""

    predicted_score: tuple[int, int]
    name: str = "fixed_score"

    def predict(self, matrix: ScoreProbabilityMatrix) -> tuple[int, int]:
        del matrix
        return self.predicted_score


@dataclass(frozen=True)
class MostLikelyScoreStrategy:
    """Predict the raw modal scoreline."""

    name: str = "most_likely_scoreline"

    def predict(self, matrix: ScoreProbabilityMatrix) -> tuple[int, int]:
        return matrix.most_likely_scoreline()


@dataclass(frozen=True)
class ExpectedPointsOptimalStrategy:
    """Predict the group-stage score that maximises expected pool points."""

    max_candidate_goals: int = 5
    name: str = "expected_points_optimal"

    def predict(self, matrix: ScoreProbabilityMatrix) -> tuple[int, int]:
        return optimise_group_prediction(matrix, self.max_candidate_goals).best.predicted_score


@dataclass(frozen=True)
class FavouriteScoreStrategy:
    """Predict a home/away win when that outcome is the clear market favourite."""

    winning_goals: int
    draw_score: tuple[int, int] = (1, 1)
    name: str = "favourite_score"

    def __post_init__(self) -> None:
        if self.winning_goals <= 0:
            raise ValueError("winning_goals must be positive")

    def predict(self, matrix: ScoreProbabilityMatrix) -> tuple[int, int]:
        """Return winning_goals-0, 0-winning_goals, or a draw when no win is clearly favoured."""

        outcomes = matrix.outcome_probabilities()
        home, draw, away = outcomes["a_win"], outcomes["draw"], outcomes["b_win"]
        if home > max(draw, away):
            return self.winning_goals, 0
        if away > max(home, draw):
            return 0, self.winning_goals
        return self.draw_score
