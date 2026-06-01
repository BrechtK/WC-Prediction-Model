import numpy as np
import pytest

from wc_predictor.probabilities import ScoreProbabilityMatrix, poisson_market_probabilities, poisson_score_matrix


@pytest.mark.parametrize("lambda_a,lambda_b", [(0.05, 0.10), (1.4, 1.1), (5.0, 4.0)])
def test_poisson_matrix_is_non_negative_and_reports_tail(lambda_a: float, lambda_b: float) -> None:
    matrix = poisson_score_matrix(lambda_a, lambda_b, max_goals=8, renormalise=True)
    assert (matrix.probabilities >= 0).all()
    assert matrix.grid_probability == pytest.approx(1.0)
    assert 0 <= matrix.tail_probability < 1


def test_non_renormalised_grid_plus_tail_sums_to_one() -> None:
    matrix = poisson_score_matrix(2.0, 1.5, max_goals=3, renormalise=False)
    assert matrix.grid_probability + matrix.tail_probability == pytest.approx(1.0)


def test_invalid_lambda_raises() -> None:
    with pytest.raises(ValueError):
        poisson_score_matrix(0.0, 1.0)


def test_full_poisson_market_probabilities_are_not_truncated() -> None:
    probabilities = poisson_market_probabilities(4.5, 3.8)
    assert probabilities["a_win"] + probabilities["draw"] + probabilities["b_win"] == pytest.approx(1.0)
    assert probabilities["over_2_5"] > 0.95


def test_score_matrix_rejects_inconsistent_tail_semantics() -> None:
    with pytest.raises(ValueError, match="renormalised"):
        ScoreProbabilityMatrix(np.array([[0.40, 0.20], [0.10, 0.10]]))
    with pytest.raises(ValueError, match="tail probability"):
        ScoreProbabilityMatrix(np.array([[0.40, 0.20], [0.10, 0.10]]), tail_probability=0.10, renormalised=False)
