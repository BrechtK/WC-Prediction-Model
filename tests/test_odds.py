import numpy as np
import pandas as pd
import pytest

from wc_predictor.odds import (
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


def test_correct_score_odds_are_margin_adjusted() -> None:
    odds = pd.DataFrame(
        [
            {"match_id": "M1", "bookmaker": "Book", "score_a": 0, "score_b": 0, "decimal_odds": 2.0},
            {"match_id": "M1", "bookmaker": "Book", "score_a": 1, "score_b": 0, "decimal_odds": 4.0},
        ]
    )
    processed = process_correct_score_odds(odds)
    assert processed["fair_score_probability"].sum() == pytest.approx(1.0)

