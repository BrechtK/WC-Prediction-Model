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

