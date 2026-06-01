import numpy as np
import pytest

from wc_predictor.optimiser import (
    evaluate_group_prediction,
    evaluate_knockout_prediction,
    optimise_group_prediction,
)
from wc_predictor.probabilities import ScoreProbabilityMatrix


def test_expected_points_are_calculated_for_manual_matrix() -> None:
    probabilities = np.zeros((4, 2))
    probabilities[1, 0] = 0.30
    probabilities[2, 0] = 0.25
    probabilities[3, 1] = 0.25
    probabilities[0, 0] = 0.20
    matrix = ScoreProbabilityMatrix(probabilities)
    assert evaluate_group_prediction(matrix, 1, 0).expected_points == pytest.approx(5.7)


def test_group_ev_diagnostics_reconcile_with_scoring_formula() -> None:
    matrix = ScoreProbabilityMatrix(np.array([[0.20, 0.05], [0.30, 0.10], [0.25, 0.10]]))
    evaluation = evaluate_group_prediction(matrix, 1, 0)
    expected = (
        1
        + 4 * evaluation.correct_result_probability
        + 2 * evaluation.correct_goal_difference_probability
        + 3 * evaluation.exact_score_probability
    )
    assert evaluation.expected_points == pytest.approx(expected)
    assert evaluation.participation_only_probability == pytest.approx(1 - evaluation.correct_result_probability)


def test_ev_optimal_prediction_can_differ_from_modal_scoreline() -> None:
    probabilities = np.zeros((4, 2))
    probabilities[1, 0] = 0.30
    probabilities[2, 0] = 0.25
    probabilities[3, 1] = 0.25
    probabilities[0, 0] = 0.20
    recommendation = optimise_group_prediction(ScoreProbabilityMatrix(probabilities), max_candidate_goals=3)
    assert recommendation.most_likely_scoreline == (1, 0)
    assert recommendation.best.predicted_score == (2, 0)
    assert recommendation.differs_from_most_likely
    assert len(recommendation.alternatives) == 5


def test_knockout_ev_reconciles_with_additive_scoring_formula() -> None:
    matrix = ScoreProbabilityMatrix(np.array([[0.20, 0.05], [0.30, 0.25], [0.10, 0.10]]))
    evaluation = evaluate_knockout_prediction(matrix, 1, 0, "A", {"A": 0.65, "B": 0.35})
    assert evaluation.expected_points == pytest.approx(
        1
        + 10 * evaluation.qualifier_probability
        + 6 * evaluation.exact_score_probability
        + 4 * evaluation.correct_goal_difference_probability
    )


def test_ev_evaluation_rejects_raw_truncated_matrix() -> None:
    matrix = ScoreProbabilityMatrix(np.array([[0.30, 0.10], [0.20, 0.10]]), tail_probability=0.30, renormalised=False)
    with pytest.raises(ValueError, match="renormalised"):
        evaluate_group_prediction(matrix, 1, 0)


def test_knockout_ev_rejects_invalid_qualifier_probabilities() -> None:
    matrix = ScoreProbabilityMatrix(np.array([[0.25, 0.25], [0.25, 0.25]]))
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_knockout_prediction(matrix, 1, 0, "A", {"A": 0.70, "B": 0.40})
