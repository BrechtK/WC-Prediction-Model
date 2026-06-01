from pathlib import Path

from wc_predictor.market_data import load_odds
from wc_predictor.reporting import format_model_inspection_report
from wc_predictor.workflow import run_prediction_workflow


EXAMPLES = Path("data/examples")


def test_model_inspection_report_contains_manual_audit_fields() -> None:
    workflow = run_prediction_workflow(load_odds(EXAMPLES / "example_odds.csv"))
    report = format_model_inspection_report(workflow.match_report)
    assert "M001 | group stage | Belgium vs Canada" in report
    assert "Raw 1X2 by bookmaker:" in report
    assert "Fair 1X2 by bookmaker:" in report
    assert "Aggregated fair 1X2:" in report
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
