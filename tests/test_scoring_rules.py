import pytest

from wc_predictor.config import KnockoutScoringConfig
from wc_predictor.scoring_rules import score_group_prediction, score_knockout_prediction


@pytest.mark.parametrize(
    ("prediction", "actual", "expected"),
    [
        ((2, 1), (2, 1), 10),
        ((3, 1), (2, 0), 7),
        ((1, 0), (3, 0), 5),
        ((0, 1), (3, 0), 1),
        ((0, 0), (0, 0), 10),
        ((1, 1), (2, 2), 7),
        ((1, 1), (1, 0), 1),
    ],
)
def test_group_stage_scoring(prediction: tuple[int, int], actual: tuple[int, int], expected: int) -> None:
    assert score_group_prediction(*prediction, *actual) == expected


def test_knockout_components_are_additive_and_qualifier_is_separate() -> None:
    config = KnockoutScoringConfig()
    assert score_knockout_prediction(1, 1, "A", 1, 1, "A", config) == 21
    assert score_knockout_prediction(1, 1, "B", 1, 1, "A", config) == 11
    assert score_knockout_prediction(2, 1, "A", 3, 2, "A", config) == 15
    assert score_knockout_prediction(0, 1, "B", 2, 0, "A", config) == 1


def test_non_additive_knockout_scoring_fails_at_config_construction() -> None:
    with pytest.raises(ValueError, match="Non-additive knockout scoring is not implemented. Use additive=True."):
        KnockoutScoringConfig(additive=False)
