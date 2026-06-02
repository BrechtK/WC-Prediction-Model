"""Score-model interfaces, the Poisson baseline, and optional challengers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

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


class ChallengerScoreModel(ScoreModel, ABC):
    """Common interface for optional score-matrix challengers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable report key for this challenger."""


@dataclass(frozen=True)
class DixonColesScoreModel(ChallengerScoreModel):
    """Independent Poisson with the Dixon-Coles low-score dependence correction."""

    lambda_a: float
    lambda_b: float
    rho: float = 0.0
    renormalise: bool = True

    @property
    def name(self) -> str:
        return "dixon_coles"

    def predict_score_matrix(self, match: Match, max_goals: int) -> ScoreProbabilityMatrix:
        """Apply the four-cell Dixon-Coles correction to a finite Poisson grid."""

        del match
        if not np.isfinite(self.rho):
            raise ValueError("Dixon-Coles rho must be finite")
        raw_poisson = poisson_score_matrix(self.lambda_a, self.lambda_b, max_goals, renormalise=False)
        probabilities = raw_poisson.probabilities.copy()
        corrections = {
            (0, 0): 1.0 - self.lambda_a * self.lambda_b * self.rho,
            (1, 0): 1.0 + self.lambda_b * self.rho,
            (0, 1): 1.0 + self.lambda_a * self.rho,
            (1, 1): 1.0 - self.rho,
        }
        if any(not np.isfinite(value) or value < 0 for value in corrections.values()):
            raise ValueError("Dixon-Coles rho produces a negative low-score correction")
        for (score_a, score_b), correction in corrections.items():
            if score_a <= max_goals and score_b <= max_goals:
                probabilities[score_a, score_b] *= correction
        represented_mass = float(probabilities.sum())
        tail_probability = max(0.0, 1.0 - represented_mass)
        if self.renormalise:
            probabilities = probabilities / represented_mass
        return ScoreProbabilityMatrix(
            probabilities,
            tail_probability=tail_probability,
            renormalised=self.renormalise,
            lambda_a=self.lambda_a,
            lambda_b=self.lambda_b,
        )


def build_challenger_score_matrices(
    match: Match,
    max_goals: int,
    models: Iterable[ChallengerScoreModel],
) -> dict[str, ScoreProbabilityMatrix]:
    """Build named challenger matrices through one model-agnostic interface."""

    matrices: dict[str, ScoreProbabilityMatrix] = {}
    for model in models:
        if model.name in matrices:
            raise ValueError(f"Duplicate challenger model name: {model.name}")
        matrices[model.name] = model.predict_score_matrix(match, max_goals)
    return matrices
