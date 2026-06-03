import numpy as np
import pytest

from wc_predictor.score_models import (
    DixonColesScoreModel,
    IndependentPoissonScoreModel,
    Match,
    build_challenger_score_matrices,
)


MATCH = Match("M1", "group", "Alpha", "Beta")


def test_dixon_coles_rho_zero_reproduces_independent_poisson() -> None:
    baseline = IndependentPoissonScoreModel(1.4, 1.1).predict_score_matrix(MATCH, max_goals=8)
    challenger = DixonColesScoreModel(1.4, 1.1, rho=0.0).predict_score_matrix(MATCH, max_goals=8)

    assert np.allclose(challenger.probabilities, baseline.probabilities)
    assert challenger.tail_probability == pytest.approx(baseline.tail_probability)


def test_dixon_coles_nonzero_rho_changes_each_low_score_probability() -> None:
    baseline = IndependentPoissonScoreModel(1.4, 1.1).predict_score_matrix(MATCH, max_goals=8)
    challenger = DixonColesScoreModel(1.4, 1.1, rho=-0.08).predict_score_matrix(MATCH, max_goals=8)

    for scoreline in ((0, 0), (1, 0), (0, 1), (1, 1)):
        assert challenger.probabilities[scoreline] != pytest.approx(baseline.probabilities[scoreline])


def test_dixon_coles_probability_matrix_remains_normalised() -> None:
    challenger = DixonColesScoreModel(1.4, 1.1, rho=-0.08).predict_score_matrix(MATCH, max_goals=8)

    assert challenger.grid_probability == pytest.approx(1.0)
    assert (challenger.probabilities >= 0).all()


def test_challenger_matrix_interface_returns_named_matrices() -> None:
    matrices = build_challenger_score_matrices(
        MATCH,
        max_goals=8,
        models=(DixonColesScoreModel(1.4, 1.1),),
    )

    assert set(matrices) == {"dixon_coles"}
