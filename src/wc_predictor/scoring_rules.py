"""Pure group-stage and configurable knockout-stage scoring functions."""

from __future__ import annotations

from wc_predictor.config import KnockoutScoringConfig
from wc_predictor.utils import goal_difference, result_sign


def score_group_prediction(pred_a: int, pred_b: int, actual_a: int, actual_b: int) -> int:
    """Score a group-stage prediction according to the private-pool rules."""

    if (pred_a, pred_b) == (actual_a, actual_b):
        return 10
    if goal_difference(pred_a, pred_b) == goal_difference(actual_a, actual_b):
        return 7
    if result_sign(pred_a, pred_b) == result_sign(actual_a, actual_b):
        return 5
    return 1


def score_knockout_prediction(
    pred_a: int,
    pred_b: int,
    predicted_qualifier: str,
    actual_a: int,
    actual_b: int,
    actual_qualifier: str,
    config: KnockoutScoringConfig | None = None,
) -> int:
    """Score a knockout prediction under the configured interpretation."""

    config = config or KnockoutScoringConfig()
    points = config.participation_points
    qualifier_correct = predicted_qualifier == actual_qualifier
    exact_score = (pred_a, pred_b) == (actual_a, actual_b)
    correct_difference = goal_difference(pred_a, pred_b) == goal_difference(actual_a, actual_b)
    if config.knockout_scoring_mode in {"additive", "unverified"}:
        return (
            points
            + config.qualifier_points * qualifier_correct
            + config.exact_score_points * exact_score
            + config.goal_difference_points * correct_difference
        )
    if config.knockout_scoring_mode == "hierarchical":
        if qualifier_correct:
            points += config.qualifier_points
        if exact_score:
            return points + config.exact_score_points
        if correct_difference:
            return points + config.goal_difference_points
        return points
    if qualifier_correct:
        points += config.qualifier_points
    if exact_score:
        return points + config.exact_score_points
    if correct_difference:
        return points + config.goal_difference_points
    return points
