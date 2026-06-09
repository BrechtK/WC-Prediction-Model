import pytest

from wc_predictor.config import GroupScoringConfig, KnockoutScoringConfig
from wc_predictor.scoring_rules import DEFAULT_GROUP_SCORING, score_group_prediction, score_knockout_prediction


def test_group_scoring_config_preserves_current_points() -> None:
    assert DEFAULT_GROUP_SCORING == GroupScoringConfig(
        exact_score_points=10,
        goal_difference_points=7,
        result_points=5,
        participation_points=1,
    )
    assert DEFAULT_GROUP_SCORING.exact_increment == 3
    assert DEFAULT_GROUP_SCORING.goal_difference_increment == 2
    assert DEFAULT_GROUP_SCORING.result_increment == 4
    assert DEFAULT_GROUP_SCORING.draw_increment == 6


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
    config = KnockoutScoringConfig(knockout_scoring_mode="additive")
    assert score_knockout_prediction(1, 1, "A", 1, 1, "A", config) == 21
    assert score_knockout_prediction(1, 1, "B", 1, 1, "A", config) == 11
    assert score_knockout_prediction(2, 1, "A", 3, 2, "A", config) == 15
    assert score_knockout_prediction(0, 1, "B", 2, 0, "A", config) == 1


def test_unverified_knockout_mode_preserves_existing_additive_scoring() -> None:
    assert score_knockout_prediction(1, 1, "A", 1, 1, "A", KnockoutScoringConfig()) == 21


def test_hierarchical_knockout_scoring_does_not_stack_exact_and_goal_difference() -> None:
    config = KnockoutScoringConfig(knockout_scoring_mode="hierarchical")

    assert score_knockout_prediction(1, 1, "A", 1, 1, "A", config) == 17
    assert score_knockout_prediction(2, 1, "A", 3, 2, "A", config) == 15


def test_invalid_knockout_scoring_mode_fails_at_config_construction() -> None:
    with pytest.raises(ValueError, match="knockout_scoring_mode"):
        KnockoutScoringConfig(knockout_scoring_mode="unknown")
