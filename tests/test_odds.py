import numpy as np
import pandas as pd
import pytest

from wc_predictor.odds import (
    aggregate_bookmaker_probabilities,
    decimal_odds_to_implied_probabilities,
    process_bookmaker_odds,
    process_correct_score_odds,
)


def test_decimal_odds_convert_to_raw_implied_probabilities() -> None:
    assert np.allclose(decimal_odds_to_implied_probabilities([2.0, 4.0]), [0.5, 0.25])


@pytest.mark.parametrize("odds", [[1.0, 2.0], [0.0, 2.0], [np.nan, 2.0]])
def test_invalid_decimal_odds_raise(odds: list[float]) -> None:
    with pytest.raises(ValueError):
        decimal_odds_to_implied_probabilities(odds)


def test_missing_optional_market_is_ignored_cleanly() -> None:
    odds = pd.DataFrame(
        [{"match_id": "M1", "bookmaker": "Book", "odds_a_win": 2.0, "odds_draw": 3.5, "odds_b_win": 4.0}]
    )
    processed = process_bookmaker_odds(odds)
    assert {"fair_a_win", "fair_draw", "fair_b_win"}.issubset(processed.columns)
    assert "fair_btts_yes" not in processed.columns


def test_incomplete_optional_market_creates_warning() -> None:
    odds = pd.DataFrame(
        [
            {
                "match_id": "M1",
                "bookmaker": "Book",
                "odds_a_win": 2.0,
                "odds_draw": 3.5,
                "odds_b_win": 4.0,
                "odds_over_2_5": 2.0,
                "odds_under_2_5": np.nan,
            }
        ]
    )
    processed = process_bookmaker_odds(odds)
    assert "Ignored incomplete over_under_2_5 market" in processed.loc[0, "warnings"]
    assert "fair_over_2_5" not in processed.columns


def test_bookmaker_aggregation_uses_complete_market_vectors() -> None:
    probabilities = pd.DataFrame(
        [
            {
                "match_id": "M1",
                "bookmaker": "BookA",
                "fair_a_win": 0.50,
                "fair_draw": 0.30,
                "fair_b_win": 0.20,
                "fair_over_2_5": 0.60,
                "fair_under_2_5": 0.40,
                "warnings": "",
            },
            {
                "match_id": "M1",
                "bookmaker": "BookB",
                "fair_a_win": 0.40,
                "fair_draw": 0.35,
                "fair_b_win": 0.25,
                "fair_over_2_5": np.nan,
                "fair_under_2_5": 0.45,
                "warnings": "Ignored incomplete over_under_2_5 market",
            },
        ]
    )
    aggregated = aggregate_bookmaker_probabilities(probabilities)
    assert aggregated.loc[0, "fair_a_win"] == pytest.approx(0.45)
    assert aggregated.loc[0, "fair_draw"] == pytest.approx(0.325)
    assert aggregated.loc[0, "fair_b_win"] == pytest.approx(0.225)
    assert aggregated.loc[0, "fair_over_2_5"] == pytest.approx(0.60)
    assert aggregated.loc[0, "fair_under_2_5"] == pytest.approx(0.40)


def test_negative_bookmaker_weight_is_rejected() -> None:
    probabilities = pd.DataFrame(
        [
            {"match_id": "M1", "bookmaker": "BookA", "fair_a_win": 0.5, "fair_draw": 0.3, "fair_b_win": 0.2, "warnings": ""},
            {"match_id": "M1", "bookmaker": "BookB", "fair_a_win": 0.4, "fair_draw": 0.35, "fair_b_win": 0.25, "warnings": ""},
        ]
    )
    with pytest.raises(ValueError, match="non-negative"):
        aggregate_bookmaker_probabilities(probabilities, "weighted", {"BookA": 1.0, "BookB": -0.5})


def test_correct_score_odds_are_margin_adjusted() -> None:
    odds = pd.DataFrame(
        [
            {"match_id": "M1", "bookmaker": "Book", "score_a": 0, "score_b": 0, "decimal_odds": 2.0},
            {"match_id": "M1", "bookmaker": "Book", "score_a": 1, "score_b": 0, "decimal_odds": 4.0},
        ]
    )
    processed = process_correct_score_odds(odds)
    assert processed["fair_score_probability"].sum() == pytest.approx(1.0)
    assert processed["correct_score_overround"].unique().tolist() == pytest.approx([0.75])
    assert (processed["number_of_scorelines"] == 2).all()
    assert processed["common_scoreline_coverage"].unique().tolist() == pytest.approx([0.25])
    assert not processed["has_other_bucket"].any()
    assert processed["suspicious_overround_warning"].str.contains("below").all()
