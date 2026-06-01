from wc_predictor.probabilities import poisson_score_matrix
from wc_predictor.strategies import (
    ExpectedPointsOptimalStrategy,
    FavouriteScoreStrategy,
    FixedScoreStrategy,
    MostLikelyScoreStrategy,
)


def test_strategy_interface_has_working_baselines() -> None:
    matrix = poisson_score_matrix(1.4, 1.0)
    assert FixedScoreStrategy((1, 1)).predict(matrix) == (1, 1)
    assert isinstance(MostLikelyScoreStrategy().predict(matrix), tuple)
    assert isinstance(ExpectedPointsOptimalStrategy().predict(matrix), tuple)


def test_favourite_strategy_predicts_home_away_or_draw() -> None:
    strategy = FavouriteScoreStrategy(2)
    assert strategy.predict((0.50, 0.30, 0.20)) == (2, 0)
    assert strategy.predict((0.20, 0.30, 0.50)) == (0, 2)
    assert strategy.predict((0.30, 0.40, 0.30)) == (1, 1)
    assert strategy.predict((0.40, 0.20, 0.40)) == (1, 1)
