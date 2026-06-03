import numpy as np
import pytest

from wc_predictor.score_models import (
    DixonColesScoreModel,
    IndependentPoissonScoreModel,
    Match,
    build_challenger_score_matrices,
    estimate_dixon_coles_rho_from_market,
)
from wc_predictor.probabilities import ScoreProbabilityMatrix


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


def test_estimated_dixon_coles_rho_detects_synthetic_low_score_inflation() -> None:
    baseline = IndependentPoissonScoreModel(1.2, 1.0).predict_score_matrix(MATCH, max_goals=8)
    market = baseline.probabilities.copy()
    market[0, 0] *= 1.20
    market[1, 1] *= 1.20
    market[1, 0] *= 0.85
    market[0, 1] *= 0.85
    market = market / market.sum()

    estimate = estimate_dixon_coles_rho_from_market(
        lambda_a=1.2,
        lambda_b=1.0,
        market_matrix=ScoreProbabilityMatrix(market),
        max_goals=8,
        fallback_rho=0.0,
    )

    assert estimate.source == "market_estimated"
    assert estimate.rho < 0
    assert -0.20 <= estimate.rho <= 0.20


def test_estimated_dixon_coles_rho_falls_back_without_market_data() -> None:
    estimate = estimate_dixon_coles_rho_from_market(
        lambda_a=1.2,
        lambda_b=1.0,
        market_matrix=None,
        max_goals=8,
        fallback_rho=0.03,
    )

    assert estimate.rho == pytest.approx(0.03)
    assert estimate.source == "fallback"
    assert "insufficient correct-score market data" in estimate.warning
