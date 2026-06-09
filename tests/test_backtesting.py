from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wc_predictor.backtest_cli import run_and_print_backtest
from wc_predictor.backtesting import (
    BatchBacktestRunner,
    BatchBacktestSettings,
    BacktestRunner,
    BacktestSettings,
    FootballDataCSVLoader,
    FootballDataWorldCupXLSXLoader,
    HistoricalWorldCupCSVLoader,
    WORLD_CUP_GROUP_STAGE_WINDOWS,
    WorldCupResearchBacktestRunner,
    WorldCupResearchBacktestSettings,
)
from wc_predictor.odds import process_bookmaker_odds
from wc_predictor.calibration import SINGLE_START_CALIBRATION_POINTS
from wc_predictor.strategies import PredictionStrategy
from wc_predictor.utils import favourite_strength_bucket


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


def _world_cup_xlsx_rows(year: int, count: int = 48) -> list[dict[str, object]]:
    start, end = (pd.Timestamp(value) for value in WORLD_CUP_GROUP_STAGE_WINDOWS[year])
    days = pd.date_range(start, end, freq="D")
    rows: list[dict[str, object]] = []
    for index in range(count):
        rows.append(
            {
                "Competition": f"World Cup {year}",
                "Home": f"Team {year} A{index:02d}",
                "Away": f"Team {year} B{index:02d}",
                "Date": days[index % len(days)].strftime("%d/%m/%Y"),
                "Time": "20:00",
                "HGFT": index % 4,
                "AGFT": (index + 1) % 3,
                "H-Avg": "1,80" if index != 0 else pd.NA,
                "D-Avg": "3,40" if index != 0 else pd.NA,
                "A-Avg": "4,80" if index != 0 else pd.NA,
                "bet365-H": "1.85",
                "bet365-D": "3.30",
                "bet365-A": "4.60",
                "Betfair_Exch-H": "1.90",
                "Betfair_Exch-D": "3.50",
                "Betfair_Exch-A": "4.70",
            }
        )
    rows.append(
        {
            "Competition": f"World Cup {year}",
            "Home": "Knockout A",
            "Away": "Knockout B",
            "Date": (end + pd.Timedelta(days=1)).strftime("%d/%m/%Y"),
            "HGFT": 1,
            "AGFT": 0,
            "H-Avg": "1.80",
            "D-Avg": "3.40",
            "A-Avg": "4.80",
        }
    )
    return rows


def _write_world_cup_xlsx(path: Path, years: tuple[int, ...], count: int = 48) -> None:
    with pd.ExcelWriter(path) as writer:
        for year in years:
            pd.DataFrame(_world_cup_xlsx_rows(year, count=count)).to_excel(
                writer,
                sheet_name=f"World Cup {year}",
                index=False,
            )


def test_football_data_world_cup_xlsx_loader_filters_group_stage_and_validates_odds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook = tmp_path / "world_cup.xlsx"
    _write_world_cup_xlsx(workbook, (2010, 2014, 2018, 2022))

    data = FootballDataWorldCupXLSXLoader(workbook).load()

    assert len(data.matches) == 192
    assert data.matches.groupby("year")["match_id"].nunique().to_dict() == {
        2010: 48,
        2014: 48,
        2018: 48,
        2022: 48,
    }
    assert data.results[["team_a_goals_90", "team_b_goals_90"]].ge(0).all().all()
    assert data.odds[["odds_a_win", "odds_draw", "odds_b_win"]].gt(1.0).all().all()
    assert "Bet365" in set(data.odds["odds_source"])

    processed = process_bookmaker_odds(data.odds)
    fair_sum = processed[["fair_a_win", "fair_draw", "fair_b_win"]].sum(axis=1)
    assert np.allclose(fair_sum, 1.0)
    assert "used Bet365 fallback odds" in capsys.readouterr().out


def test_world_cup_xlsx_backtest_exports_yearly_and_probabilistic_summaries(tmp_path: Path) -> None:
    workbook = tmp_path / "world_cup_small.xlsx"
    _write_world_cup_xlsx(workbook, (2010, 2014), count=2)
    settings = WorldCupResearchBacktestSettings(
        summary_output_path=tmp_path / "summary.csv",
        predictions_output_path=tmp_path / "predictions.csv",
        skipped_output_path=tmp_path / "skipped.csv",
        yearly_summary_output_path=tmp_path / "summary_by_year.csv",
        probabilistic_summary_output_path=tmp_path / "probabilistic_summary.csv",
        excel_output_path=tmp_path / "backtest.xlsx",
        margin_methods=("normalised_inverse_odds",),
    )

    report = WorldCupResearchBacktestRunner(
        FootballDataWorldCupXLSXLoader(workbook, years=(2010, 2014)),
        settings=settings,
        fast=True,
    ).run(export=True)

    assert report.predictions["match_id"].nunique() == 4
    assert set(report.yearly_summary["year"]) == {2010, 2014}
    assert not report.probabilistic_summary.empty
    assert (tmp_path / "summary_by_year.csv").exists()
    assert (tmp_path / "probabilistic_summary.csv").exists()


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
    summary = report.summary.set_index("strategy")
    assert len(predictions) == 4
    assert predictions["used_over_under_2_5"].sum() == 3
    assert not predictions.loc[predictions["match_id"] == "H000002", "used_over_under_2_5"].item()
    assert summary.loc["ev_optimal_1x2_over_under", "fraction_used_over_under_2_5"] == pytest.approx(0.75)


def test_batch_backtest_runs_folder_and_exports_detailed_aggregate_and_skips(tmp_path: Path) -> None:
    settings = BatchBacktestSettings(
        detailed_output_path=tmp_path / "detailed.csv",
        aggregate_output_path=tmp_path / "aggregate.csv",
        skipped_output_path=tmp_path / "skipped.csv",
        favourite_strength_output_path=tmp_path / "favourite_strength.csv",
    )
    report = BatchBacktestRunner(BATCH_EXAMPLES, settings=settings).run()
    detailed = report.detailed_summary.set_index(["source_file", "strategy"])
    aggregate = report.aggregate_summary.set_index("strategy")

    assert settings.detailed_output_path.exists()
    assert settings.aggregate_output_path.exists()
    assert settings.skipped_output_path.exists()
    assert settings.favourite_strength_output_path.exists()
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
        favourite_strength_output_path=tmp_path / "favourite_strength.csv",
    )
    report = BatchBacktestRunner(EXAMPLES / "example_historical_matches.csv", settings=settings).run()
    assert set(report.detailed_summary["source_file"]) == {"example_historical_matches.csv"}
    assert set(report.predictions["source_file"]) == {"example_historical_matches.csv"}
    assert set(report.skipped["source_file"]) == {"example_historical_matches.csv"}


def test_batch_backtest_recurses_into_nested_folders_and_uses_relative_source_paths(tmp_path: Path) -> None:
    history = tmp_path / "history"
    belgium = history / "Belgium"
    england = history / "England"
    belgium.mkdir(parents=True)
    england.mkdir(parents=True)
    copyfile(BATCH_EXAMPLES / "season_a.csv", belgium / "season_a.csv")
    copyfile(BATCH_EXAMPLES / "season_b.csv", england / "season_b.csv")

    report = BatchBacktestRunner(history).run(export=False)
    assert set(report.detailed_summary["source_file"]) == {
        "Belgium/season_a.csv",
        "England/season_b.csv",
    }


def test_fast_mode_uses_single_start_and_calibrates_each_required_matrix_once(monkeypatch) -> None:
    from wc_predictor import backtesting

    original = backtesting.calibrate_poisson_model
    starting_points = []

    def wrapped(*args, **kwargs):
        starting_points.append(kwargs["starting_points"])
        return original(*args, **kwargs)

    monkeypatch.setattr(backtesting, "calibrate_poisson_model", wrapped)

    BacktestRunner(
        FootballDataCSVLoader(EXAMPLES / "example_historical_matches.csv"),
        fast=True,
        max_matches=1,
    ).run(export=False)

    assert starting_points == [SINGLE_START_CALIBRATION_POINTS, SINGLE_START_CALIBRATION_POINTS]


def test_normal_mode_keeps_robust_calibration_and_existing_results() -> None:
    loader = FootballDataCSVLoader(EXAMPLES / "example_historical_matches.csv")
    baseline = BacktestRunner(loader).run(export=False)
    explicit_normal = BacktestRunner(loader, fast=False).run(export=False)

    assert_frame_equal(explicit_normal.summary, baseline.summary)
    assert_frame_equal(explicit_normal.predictions, baseline.predictions)
    assert_frame_equal(explicit_normal.skipped, baseline.skipped)


def test_batch_backtest_max_files_limits_smoke_test_scope() -> None:
    report = BatchBacktestRunner(BATCH_EXAMPLES, max_files=1).run(export=False)

    assert set(report.detailed_summary["source_file"]) == {"season_a.csv"}


def test_batch_backtest_max_matches_limits_total_smoke_test_scope() -> None:
    report = BatchBacktestRunner(BATCH_EXAMPLES, max_matches=2).run(export=False)

    assert report.predictions[["source_file", "match_id"]].drop_duplicates().shape[0] == 2
    assert set(report.detailed_summary["source_file"]) == {"season_a.csv"}


def test_batch_backtest_skips_non_football_data_csv_files(tmp_path: Path) -> None:
    history = tmp_path / "history"
    history.mkdir()
    copyfile(BATCH_EXAMPLES / "season_b.csv", history / "season.csv")
    pd.DataFrame([{"match_id": "M1", "player": "Example"}]).to_csv(history / "unrelated.csv", index=False)

    report = BatchBacktestRunner(history).run(export=False)

    assert set(report.detailed_summary["source_file"]) == {"season.csv"}
    assert report.skipped_files.loc[0, "source_file"] == "unrelated.csv"
    assert "skipped non-Football-Data CSV" in report.skipped_files.loc[0, "reason"]


def test_batch_backtest_handles_folder_with_only_non_football_data_csv_files(tmp_path: Path) -> None:
    history = tmp_path / "history"
    history.mkdir()
    pd.DataFrame([{"match_id": "M1", "player": "Example"}]).to_csv(history / "unrelated.csv", index=False)

    report = BatchBacktestRunner(history).run(export=False)

    assert report.detailed_summary.empty
    assert report.predictions.empty
    assert report.skipped_files["source_file"].tolist() == ["unrelated.csv"]


def test_shared_backtest_cli_helper_runs_and_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = run_and_print_backtest(
        input_path=BATCH_EXAMPLES,
        detailed_output_path=tmp_path / "detailed.csv",
        aggregate_output_path=tmp_path / "aggregate.csv",
        skipped_output_path=tmp_path / "skipped.csv",
        favourite_strength_output_path=tmp_path / "favourite_strength.csv",
    )
    output = capsys.readouterr().out
    assert "[Backtest file 1/2]" in output
    assert "matches in file:" in output
    assert "elapsed:" in output
    assert "Backtest Scope" in output
    assert "- Files processed: 2" in output
    assert len(report.detailed_summary) == 14
    assert (tmp_path / "detailed.csv").exists()
    assert (tmp_path / "aggregate.csv").exists()
    assert (tmp_path / "skipped.csv").exists()
    assert (tmp_path / "favourite_strength.csv").exists()


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.44, "balanced"),
        (0.45, "slight_favourite"),
        (0.54, "slight_favourite"),
        (0.55, "clear_favourite"),
        (0.64, "clear_favourite"),
        (0.65, "strong_favourite"),
        (0.74, "strong_favourite"),
        (0.75, "huge_favourite"),
        (0.84, "huge_favourite"),
        (0.85, "extreme_favourite"),
        (0.95, "extreme_favourite"),
    ],
)
def test_favourite_strength_bucket_assignment(probability: float, expected: str) -> None:
    assert favourite_strength_bucket(probability) == expected


def test_favourite_strength_bucket_summary_aggregates_strategy_points() -> None:
    rows = []
    points = {
        "always_1_1": (1, 1),
        "always_0_0": (2, 2),
        "favourite_1_0": (3, 5),
        "favourite_2_0": (4, 4),
        "most_likely_poisson": (2, 4),
        "ev_optimal_1x2": (5, 7),
        "ev_optimal_1x2_over_under": (6, 8),
    }
    for strategy, realised_points in points.items():
        for match_id, points_for_match in zip(("M1", "M2"), realised_points, strict=True):
            rows.append(
                {
                    "source_file": "season.csv",
                    "match_id": match_id,
                    "strategy": strategy,
                    "realised_points": points_for_match,
                    "favourite_bucket": "strong_favourite",
                }
            )
    summary = BatchBacktestRunner._favourite_strength_summary(pd.DataFrame(rows)).set_index("favourite_bucket")
    strong = summary.loc["strong_favourite"]
    assert strong["matches"] == 2
    assert strong["average_points_favourite_1_0"] == pytest.approx(4.0)
    assert strong["best_strategy"] == "ev_optimal_1x2_over_under"
    assert strong["best_ev_gap_vs_favourite_1_0"] == pytest.approx(3.0)
    assert strong["best_ev_gap_vs_most_likely_poisson"] == pytest.approx(4.0)
    assert summary.loc["extreme_favourite", "matches"] == 0


def test_strategy_interface_exists() -> None:
    assert PredictionStrategy.__doc__


def test_historical_world_cup_loader_and_research_backtest_run_on_synthetic_data(tmp_path: Path) -> None:
    history = tmp_path / "world_cup_matches.csv"
    pd.DataFrame(
        [
            {
                "match_id": "WC1",
                "date": "2022-11-20",
                "tournament": "World Cup 2022",
                "stage": "group stage",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
                "actual_score_a": 2,
                "actual_score_b": 0,
                "bookmaker": "Historical",
                "odds_a_win": 1.80,
                "odds_draw": 3.60,
                "odds_b_win": 5.00,
                "odds_btts_yes": 2.05,
                "odds_btts_no": 1.75,
                "odds_over_2_5": 1.95,
                "odds_under_2_5": 1.85,
            },
            {
                "match_id": "WC2",
                "date": "2022-11-21",
                "tournament": "World Cup 2022",
                "stage": "group stage",
                "group": "B",
                "team_a": "Gamma",
                "team_b": "Delta",
                "actual_score_a": 1,
                "actual_score_b": 1,
                "bookmaker": "Historical",
                "odds_a_win": 2.40,
                "odds_draw": 3.10,
                "odds_b_win": 3.20,
            },
        ]
    ).to_csv(history, index=False)

    loaded = HistoricalWorldCupCSVLoader(history).load()
    assert loaded.results.loc[0, "team_a_goals_90"] == 2

    report = WorldCupResearchBacktestRunner(HistoricalWorldCupCSVLoader(history)).run(export=False)

    assert not report.summary.empty
    assert "rank_by_average_points" in report.summary
    assert not report.predictions.empty
    assert report.skipped["reason"].astype(str).str.contains("blend-weight validation skipped").any()
