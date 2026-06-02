import numpy as np

from wc_predictor.probabilities import ScoreProbabilityMatrix, poisson_score_matrix
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
    assert strategy.predict(ScoreProbabilityMatrix(np.array([[0.30, 0.20], [0.50, 0.00]]))) == (2, 0)
    assert strategy.predict(ScoreProbabilityMatrix(np.array([[0.30, 0.50], [0.20, 0.00]]))) == (0, 2)
    assert strategy.predict(ScoreProbabilityMatrix(np.array([[0.40, 0.30], [0.30, 0.00]]))) == (1, 1)
    assert strategy.predict(ScoreProbabilityMatrix(np.array([[0.20, 0.40], [0.40, 0.00]]))) == (1, 1)
