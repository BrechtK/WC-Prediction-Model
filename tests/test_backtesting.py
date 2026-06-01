from pathlib import Path

import pandas as pd
import pytest

from wc_predictor.backtesting import (
    BatchBacktestRunner,
    BatchBacktestSettings,
    BacktestRunner,
    BacktestSettings,
    FootballDataCSVLoader,
)
from wc_predictor.strategies import PredictionStrategy


EXAMPLES = Path("data/examples")
BATCH_EXAMPLES = EXAMPLES / "historical_batch"


def test_football_data_loader_maps_average_odds_and_results() -> None:
    data = FootballDataCSVLoader(EXAMPLES / "example_historical_matches.csv").load()
    assert len(data.matches) == 6
    assert data.matches.loc[0, ["team_a", "team_b"]].tolist() == ["Alpha FC", "Beta FC"]
    assert data.results.loc[0, ["team_a_goals_90", "team_b_goals_90"]].tolist() == [2.0, 0.0]
    assert len(data.odds) == 5
    assert data.odds.loc[0, "bookmaker"] == "Average"
    assert data.odds.loc[0, "odds_a_win"] == pytest.approx(1.80)
    assert pd.isna(data.odds.loc[1, "odds_over_2_5"])


def test_football_data_loader_falls_back_to_bookmaker_columns(tmp_path: Path) -> None:
    history = tmp_path / "bookmaker_history.csv"
    pd.DataFrame(
        [
            {
                "Date": "2025-01-01",
                "HomeTeam": "Alpha",
                "AwayTeam": "Beta",
                "FTHG": 1,
                "FTAG": 0,
                "B365H": 1.80,
                "B365D": 3.50,
                "B365A": 4.50,
                "B365>2.5": 1.95,
                "B365<2.5": 1.85,
            }
        ]
    ).to_csv(history, index=False)
    data = FootballDataCSVLoader(history).load()
    assert data.odds.loc[0, "bookmaker"] == "Bet365"
    assert data.odds.loc[0, "odds_under_2_5"] == pytest.approx(1.85)


def test_backtest_runner_calculates_summary_metrics_and_skip_reasons(tmp_path: Path) -> None:
    output = tmp_path / "backtest_results.csv"
    report = BacktestRunner(
        FootballDataCSVLoader(EXAMPLES / "example_historical_matches.csv"),
        settings=BacktestSettings(output),
    ).run()
    summary = report.summary.set_index("strategy")

    assert output.exists()
    assert len(summary) == 7
    assert summary.loc["always_1_1", "matches_used"] == 5
    assert summary.loc["always_1_1", "skipped_matches"] == 1
    assert summary.loc["always_1_1", "total_points"] == 20
    assert summary.loc["always_1_1", "average_realised_points"] == pytest.approx(4.0)
    assert summary.loc["always_1_1", "exact_score_hit_rate"] == pytest.approx(0.2)
    assert summary.loc["always_1_1", "correct_goal_difference_hit_rate"] == pytest.approx(0.4)
    assert summary.loc["always_1_1", "correct_result_hit_rate"] == pytest.approx(0.4)
    assert summary.loc["always_1_1", "points_variance"] == pytest.approx(14.4)
    assert summary.loc["ev_optimal_1x2", "matches_used"] == 4
    assert summary.loc["ev_optimal_1x2", "skipped_matches"] == 2
    assert "missing or invalid 1X2 odds: 1" in summary.loc["ev_optimal_1x2", "skip_reasons"]
    assert "missing or invalid full-time result: 1" in summary.loc["ev_optimal_1x2", "skip_reasons"]


def test_over_under_strategy_falls_back_to_1x2_when_optional_market_is_missing(tmp_path: Path) -> None:
    report = BacktestRunner(
        FootballDataCSVLoader(EXAMPLES / "example_historical_matches.csv"),
        settings=BacktestSettings(tmp_path / "summary.csv"),
    ).run(export=False)
    predictions = report.predictions[report.predictions["strategy"] == "ev_optimal_1x2_over_under"]
    assert len(predictions) == 4
    assert predictions["used_over_under_2_5"].sum() == 3
    assert not predictions.loc[predictions["match_id"] == "H000002", "used_over_under_2_5"].item()


def test_batch_backtest_runs_folder_and_exports_detailed_aggregate_and_skips(tmp_path: Path) -> None:
    settings = BatchBacktestSettings(
        detailed_output_path=tmp_path / "detailed.csv",
        aggregate_output_path=tmp_path / "aggregate.csv",
        skipped_output_path=tmp_path / "skipped.csv",
    )
    report = BatchBacktestRunner(BATCH_EXAMPLES, settings=settings).run()
    detailed = report.detailed_summary.set_index(["source_file", "strategy"])
    aggregate = report.aggregate_summary.set_index("strategy")

    assert settings.detailed_output_path.exists()
    assert settings.aggregate_output_path.exists()
    assert settings.skipped_output_path.exists()
    assert len(detailed) == 14
    assert set(report.detailed_summary["source_file"]) == {"season_a.csv", "season_b.csv"}
    assert detailed.loc[("season_a.csv", "always_1_1"), "matches_used"] == 3
    assert detailed.loc[("season_b.csv", "always_1_1"), "matches_used"] == 3
    assert aggregate.loc["always_1_1", "source_file"] == "ALL_FILES"
    assert aggregate.loc["always_1_1", "matches_used"] == 6
    assert aggregate.loc["always_1_1", "total_points"] == 27
    assert aggregate.loc["always_1_1", "average_realised_points"] == pytest.approx(4.5)

    skipped = report.skipped_by_file_reason.set_index(["source_file", "reason"])
    assert skipped.loc[("season_a.csv", "missing or invalid 1X2 odds"), "skipped_matches"] == 1
    assert skipped.loc[("season_a.csv", "missing or invalid full-time result"), "skipped_matches"] == 1
    assert "ev_optimal_1x2" in skipped.loc[
        ("season_a.csv", "missing or invalid 1X2 odds"), "affected_strategies"
    ]


def test_batch_backtest_accepts_single_csv_and_includes_source_file(tmp_path: Path) -> None:
    settings = BatchBacktestSettings(
        detailed_output_path=tmp_path / "detailed.csv",
        aggregate_output_path=tmp_path / "aggregate.csv",
        skipped_output_path=tmp_path / "skipped.csv",
    )
    report = BatchBacktestRunner(EXAMPLES / "example_historical_matches.csv", settings=settings).run()
    assert set(report.detailed_summary["source_file"]) == {"example_historical_matches.csv"}
    assert set(report.predictions["source_file"]) == {"example_historical_matches.csv"}
    assert set(report.skipped["source_file"]) == {"example_historical_matches.csv"}


def test_strategy_interface_exists() -> None:
    assert PredictionStrategy.__doc__
