"""Score-model interfaces and the transparent Version 1 Poisson baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from wc_predictor.probabilities import ScoreProbabilityMatrix, poisson_score_matrix


@dataclass(frozen=True)
class Match:
    """Minimal match metadata shared by score models and workflows."""

    match_id: str
    stage: str
    team_a: str
    team_b: str


class ScoreModel(ABC):
    """Common interface for market-implied and future challenger models."""

    @abstractmethod
    def predict_score_matrix(self, match: Match, max_goals: int) -> ScoreProbabilityMatrix:
        """Predict a full scoreline matrix for a match."""


@dataclass(frozen=True)
class IndependentPoissonScoreModel(ScoreModel):
    """Independent Poisson model with pre-calibrated expected goals."""

    lambda_a: float
    lambda_b: float
    renormalise: bool = True

    def predict_score_matrix(self, match: Match, max_goals: int) -> ScoreProbabilityMatrix:
        """Generate the finite score grid; match metadata is accepted for interface consistency."""

        del match
        return poisson_score_matrix(self.lambda_a, self.lambda_b, max_goals, self.renormalise)


class ChallengerScoreModel(ScoreModel):
    """Extension point for future xG/Elo/ML challengers; intentionally unimplemented."""

    def predict_score_matrix(self, match: Match, max_goals: int) -> ScoreProbabilityMatrix:
        raise NotImplementedError("Challenger models are intentionally outside Version 1")

