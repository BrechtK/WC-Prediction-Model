from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wc_predictor.overround_diagnostics import (
    build_overround_diagnostics,
    filter_overround_observations,
    implied_overround,
    load_overround_observations,
    summarise_overrounds,
)


def test_one_x_two_overround_calculation() -> None:
    assert implied_overround([2.0, 3.5, 4.0]) == pytest.approx((1 / 2.0) + (1 / 3.5) + (1 / 4.0))


def test_two_way_overround_calculation() -> None:
    assert implied_overround([1.91, 1.91]) == pytest.approx((1 / 1.91) * 2)


def test_invalid_odds_are_excluded_from_filtered_sample() -> None:
    observations = pd.DataFrame(
        [
            {
                "year": 2014,
                "market_type": "btts",
                "match_id": "M001",
                "bookmaker": "Book",
                "line_key": "",
                "overround": float("nan"),
                "source_file": "synthetic",
                "quoted_scorelines": pd.NA,
                "valid_bookmakers_on_line": pd.NA,
            }
        ]
    )

    filtered = filter_overround_observations(observations)

    assert filtered.iloc[0]["filter_status"] == "excluded"
    assert filtered.iloc[0]["filter_reason"] == "invalid_or_missing_odds"


def test_underround_rows_are_excluded_from_filtered_sample() -> None:
    observations = pd.DataFrame(
        [
            {
                "year": 2014,
                "market_type": "total_goals",
                "match_id": "M001",
                "bookmaker": "Book",
                "line_key": "2.5",
                "overround": 0.98,
                "source_file": "synthetic",
                "quoted_scorelines": pd.NA,
                "valid_bookmakers_on_line": pd.NA,
            }
        ]
    )

    filtered = filter_overround_observations(observations)

    assert filtered.iloc[0]["filter_status"] == "excluded"
    assert filtered.iloc[0]["filter_reason"] == "underround_below_1"


def test_asian_handicap_single_valid_bookmaker_line_is_not_clean_multi_bookmaker(tmp_path: Path) -> None:
    source_root = tmp_path / "cache"
    year_dir = source_root / "wc2014"
    year_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "match_id": "M001",
                "bookmaker": "Book A",
                "handicap": 0.5,
                "odds_team_a": 1.91,
                "odds_team_b": 1.91,
            }
        ]
    ).to_csv(year_dir / "asian_handicap_odds.csv", index=False)

    observations = filter_overround_observations(load_overround_observations(source_root, years=[2014]))

    row = observations.iloc[0]
    assert row["market_type"] == "asian_handicap"
    assert row["valid_bookmakers_on_line"] == 1
    assert row["filter_status"] == "excluded"
    assert row["filter_reason"] == "asian_handicap_line_fewer_than_2_valid_bookmakers"


def test_summary_statistics_on_tiny_fixture() -> None:
    observations = pd.DataFrame(
        [
            {"year": 2014, "market_type": "1x2", "overround": 1.02},
            {"year": 2014, "market_type": "1x2", "overround": 1.04},
            {"year": 2014, "market_type": "1x2", "overround": 1.06},
        ]
    )

    summary = summarise_overrounds(observations, sample="raw")
    row = summary.iloc[0]

    assert row["n_observations"] == 3
    assert row["mean_overround"] == pytest.approx(1.04)
    assert row["median_overround"] == pytest.approx(1.04)
    assert row["iqr_overround"] == pytest.approx(0.02)
    assert row["mean_margin_percentage_points"] == pytest.approx(4.0)


def test_build_diagnostics_writes_pairwise_ready_filtered_summary(tmp_path: Path) -> None:
    source_root = tmp_path / "cache"
    for year, odds in [(2014, (2.0, 3.5, 4.0)), (2018, (2.1, 3.4, 3.9)), (2022, (2.2, 3.3, 3.8))]:
        year_dir = source_root / f"wc{year}"
        year_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "match_id": "M001",
                    "bookmaker": "Book",
                    "odds_a_win": odds[0],
                    "odds_draw": odds[1],
                    "odds_b_win": odds[2],
                }
            ]
        ).to_csv(year_dir / "core_odds.csv", index=False)

    result = build_overround_diagnostics(source_root)

    assert set(result.filtered_summary["year"]) == {2014, 2018, 2022}
    assert set(result.pairwise_year_differences["market_type"]) == {"1x2"}
    assert pd.notna(result.pairwise_year_differences.iloc[0]["diff_2022_minus_2014"])
