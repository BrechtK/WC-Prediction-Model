import numpy as np
import pytest

from wc_predictor.optimiser import evaluate_group_prediction, optimise_group_prediction
from wc_predictor.probabilities import ScoreProbabilityMatrix


def test_expected_points_are_calculated_for_manual_matrix() -> None:
    probabilities = np.zeros((4, 2))
    probabilities[1, 0] = 0.30
    probabilities[2, 0] = 0.25
    probabilities[3, 1] = 0.25
    probabilities[0, 0] = 0.20
    matrix = ScoreProbabilityMatrix(probabilities)
    assert evaluate_group_prediction(matrix, 1, 0).expected_points == pytest.approx(5.7)


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

