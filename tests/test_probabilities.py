import pytest

from wc_predictor.probabilities import poisson_score_matrix


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

