from pathlib import Path

from wc_predictor.backtesting import BatchBacktestRunner, BatchBacktestSettings
from wc_predictor.market_data import load_odds
from wc_predictor.reporting import format_backtest_console_summary, format_model_inspection_report
from wc_predictor.workflow import run_prediction_workflow


EXAMPLES = Path("data/examples")
BATCH_EXAMPLES = EXAMPLES / "historical_batch"


def test_model_inspection_report_contains_manual_audit_fields() -> None:
    workflow = run_prediction_workflow(load_odds(EXAMPLES / "example_odds.csv"))
    report = format_model_inspection_report(workflow.match_report)
    assert "M001 | group stage | Belgium vs Canada" in report
    assert "Raw 1X2 by bookmaker:" in report
    assert "Fair 1X2 by bookmaker:" in report
    assert "Aggregated fair 1X2:" in report
    assert "Favourite strength: p_fav=" in report
    assert "Calibrated lambdas:" in report
    assert "Model-implied 1X2:" in report
    assert "Calibration error:" in report
    assert "Score-matrix tail mass:" in report
    assert "Most likely scoreline:" in report
    assert "EV-optimal prediction:" in report
    assert "Top 5 EV predictions:" in report
    assert report.count("overround=") == 6


def test_match_report_retains_raw_and_top_five_ev_columns() -> None:
    workflow = run_prediction_workflow(load_odds(EXAMPLES / "example_odds.csv"))
    assert {"bookmaker_raw_1x2", "top_5_ev_predictions"}.issubset(workflow.match_report.columns)
    assert workflow.match_report.loc[0, "top_5_ev_predictions"].count(";") == 4


def test_backtest_console_summary_is_compact_and_interpretable(tmp_path: Path) -> None:
    settings = BatchBacktestSettings(
        detailed_output_path=tmp_path / "detailed.csv",
        aggregate_output_path=tmp_path / "aggregate.csv",
        skipped_output_path=tmp_path / "skipped.csv",
        favourite_strength_output_path=tmp_path / "favourite_strength.csv",
    )
    report = BatchBacktestRunner(BATCH_EXAMPLES, settings=settings).run()
    summary = format_backtest_console_summary(report, BATCH_EXAMPLES, settings)

    assert "Backtest Scope" in summary
    assert "- Files processed: 2" in summary
    assert "- Total matches used: 6" in summary
    assert "- Total skipped matches: 2" in summary
    assert "Overall Strategy Ranking" in summary
    assert "vs fav" in summary
    assert "vs modal" in summary
    assert "Key Conclusions" in summary
    assert "Favourite-Strength Analysis" in summary
    assert "slight_favourite" in summary
    assert "- Best strategy overall:" in summary
    assert "- Results support the project hypothesis:" in summary
    assert "Per-File Winners" in summary
    assert "First-Place Counts" in summary
    assert "- always_0_0: 1" in summary
    assert "- always_1_1: 1" in summary
    assert "Skipped Matches" in summary
    assert "missing or invalid 1X2 odds" in summary
    assert "File Outputs" in summary
    assert "Verbose Per-File Strategy Results" not in summary


def test_backtest_console_summary_reports_no_skips_and_verbose_tables(tmp_path: Path) -> None:
    settings = BatchBacktestSettings(
        detailed_output_path=tmp_path / "detailed.csv",
        aggregate_output_path=tmp_path / "aggregate.csv",
        skipped_output_path=tmp_path / "skipped.csv",
        favourite_strength_output_path=tmp_path / "favourite_strength.csv",
    )
    source = BATCH_EXAMPLES / "season_b.csv"
    report = BatchBacktestRunner(source, settings=settings).run()
    summary = format_backtest_console_summary(report, source, settings, verbose=True)

    assert "No skipped matches." in summary
    assert "Verbose Per-File Strategy Results" in summary
    assert "Verbose Aggregate Strategy Results" in summary
    assert "Verbose Skipped Matches By File And Reason" in summary
    assert "Verbose Favourite-Strength Results" in summary
