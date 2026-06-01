"""Expected-points optimisation over scoreline predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wc_predictor.config import KnockoutScoringConfig
from wc_predictor.probabilities import ScoreProbabilityMatrix
from wc_predictor.scoring_rules import score_group_prediction
from wc_predictor.utils import goal_difference, result_sign


@dataclass(frozen=True)
class GroupPredictionEvaluation:
    """Expected-value diagnostics for one group-stage predicted score."""

    predicted_score: tuple[int, int]
    expected_points: float
    exact_score_probability: float
    correct_goal_difference_probability: float
    correct_result_probability: float
    participation_only_probability: float


@dataclass(frozen=True)
class GroupPredictionRecommendation:
    """EV-optimal group-stage prediction with ranked alternatives."""

    best: GroupPredictionEvaluation
    alternatives: tuple[GroupPredictionEvaluation, ...]
    most_likely_scoreline: tuple[int, int]
    differs_from_most_likely: bool


@dataclass(frozen=True)
class KnockoutPredictionEvaluation:
    """Expected-value diagnostics for a knockout score and qualifier prediction."""

    predicted_score: tuple[int, int]
    predicted_qualifier: str
    expected_points: float
    qualifier_probability: float
    exact_score_probability: float
    correct_goal_difference_probability: float


@dataclass(frozen=True)
class KnockoutPredictionRecommendation:
    """EV-optimal knockout score and qualifier prediction."""

    best: KnockoutPredictionEvaluation
    alternatives: tuple[KnockoutPredictionEvaluation, ...]
    most_likely_scoreline: tuple[int, int]
    differs_from_most_likely: bool


def evaluate_group_prediction(
    matrix: ScoreProbabilityMatrix, pred_a: int, pred_b: int
) -> GroupPredictionEvaluation:
    """Calculate expected pool points and probability diagnostics for one prediction."""

    expected_points = 0.0
    for actual_a, actual_b in np.ndindex(matrix.probabilities.shape):
        expected_points += (
            matrix.probabilities[actual_a, actual_b]
            * score_group_prediction(pred_a, pred_b, actual_a, actual_b)
        )
    difference = goal_difference(pred_a, pred_b)
    sign = result_sign(pred_a, pred_b)
    correct_result_probability = matrix.result_probability(sign)
    return GroupPredictionEvaluation(
        (pred_a, pred_b),
        float(expected_points),
        matrix.exact_score_probability(pred_a, pred_b),
        matrix.goal_difference_probability(difference),
        correct_result_probability,
        1.0 - correct_result_probability,
    )


def optimise_group_prediction(
    matrix: ScoreProbabilityMatrix,
    max_candidate_goals: int = 5,
    top_n: int = 5,
) -> GroupPredictionRecommendation:
    """Rank all candidate group-stage scores by expected competition points."""

    if max_candidate_goals < 0:
        raise ValueError("max_candidate_goals must be non-negative")
    evaluations = [
        evaluate_group_prediction(matrix, pred_a, pred_b)
        for pred_a in range(max_candidate_goals + 1)
        for pred_b in range(max_candidate_goals + 1)
    ]
    evaluations.sort(key=lambda item: (-item.expected_points, item.predicted_score))
    best = evaluations[0]
    most_likely = matrix.most_likely_scoreline()
    return GroupPredictionRecommendation(
        best,
        tuple(evaluations[1 : top_n + 1]),
        most_likely,
        best.predicted_score != most_likely,
    )


def evaluate_knockout_prediction(
    matrix: ScoreProbabilityMatrix,
    pred_a: int,
    pred_b: int,
    predicted_qualifier: str,
    qualifier_probabilities: dict[str, float],
    config: KnockoutScoringConfig | None = None,
) -> KnockoutPredictionEvaluation:
    """Calculate additive knockout EV using score and qualifier components separately."""

    config = config or KnockoutScoringConfig()
    if predicted_qualifier not in qualifier_probabilities:
        raise ValueError(f"No qualification probability for {predicted_qualifier!r}")
    exact_probability = matrix.exact_score_probability(pred_a, pred_b)
    difference_probability = matrix.goal_difference_probability(goal_difference(pred_a, pred_b))
    qualifier_probability = qualifier_probabilities[predicted_qualifier]
    if config.additive:
        expected_points = (
            config.participation_points
            + config.qualifier_points * qualifier_probability
            + config.exact_score_points * exact_probability
            + config.goal_difference_points * difference_probability
        )
    else:
        raise NotImplementedError("Non-additive knockout EV requires joint component probabilities")
    return KnockoutPredictionEvaluation(
        (pred_a, pred_b),
        predicted_qualifier,
        float(expected_points),
        qualifier_probability,
        exact_probability,
        difference_probability,
    )


def optimise_knockout_prediction(
    matrix: ScoreProbabilityMatrix,
    qualifier_probabilities: dict[str, float],
    max_candidate_goals: int = 5,
    top_n: int = 5,
    config: KnockoutScoringConfig | None = None,
) -> KnockoutPredictionRecommendation:
    """Rank knockout score and qualifier combinations by expected pool points."""

    evaluations = [
        evaluate_knockout_prediction(matrix, pred_a, pred_b, qualifier, qualifier_probabilities, config)
        for pred_a in range(max_candidate_goals + 1)
        for pred_b in range(max_candidate_goals + 1)
        for qualifier in qualifier_probabilities
    ]
    evaluations.sort(key=lambda item: (-item.expected_points, item.predicted_score, item.predicted_qualifier))
    best = evaluations[0]
    most_likely = matrix.most_likely_scoreline()
    return KnockoutPredictionRecommendation(
        best,
        tuple(evaluations[1 : top_n + 1]),
        most_likely,
        best.predicted_score != most_likely,
    )
