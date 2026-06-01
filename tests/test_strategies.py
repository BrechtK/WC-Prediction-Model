from wc_predictor.probabilities import poisson_score_matrix
from wc_predictor.strategies import ExpectedPointsOptimalStrategy, FixedScoreStrategy, MostLikelyScoreStrategy


def test_strategy_interface_has_working_baselines() -> None:
    matrix = poisson_score_matrix(1.4, 1.0)
    assert FixedScoreStrategy((1, 1)).predict(matrix) == (1, 1)
    assert isinstance(MostLikelyScoreStrategy().predict(matrix), tuple)
    assert isinstance(ExpectedPointsOptimalStrategy().predict(matrix), tuple)

