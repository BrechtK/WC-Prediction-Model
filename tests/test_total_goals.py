from __future__ import annotations

import pandas as pd
import pytest

from wc_predictor.calibration import CalibrationTargets, calibrate_poisson_model, poisson_over_total_probability
from wc_predictor.odds import aggregate_total_goals_probabilities, process_total_goals_odds
from wc_predictor.workflow import run_prediction_workflow


def _total_goals_odds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": "M1", "bookmaker": "Book A", "line": 1.5, "odds_over": 1.50, "odds_under": 2.70},
            {"match_id": "M1", "bookmaker": "Book B", "line": 1.5, "odds_over": 1.55, "odds_under": 2.60},
            {"match_id": "M1", "bookmaker": "Book A", "line": 0.5, "odds_over": 1.08, "odds_under": 8.50},
            {"match_id": "M1", "bookmaker": "Book B", "line": 0.5, "odds_over": 1.10, "odds_under": 8.00},
            {"match_id": "M1", "bookmaker": "Book A", "line": 2.0, "odds_over": 1.85, "odds_under": 2.05},
            {"match_id": "M1", "bookmaker": "Book B", "line": 2.0, "odds_over": 1.90, "odds_under": 2.00},
            {"match_id": "M1", "bookmaker": "Book A", "line": 2.25, "odds_over": 2.00, "odds_under": 1.90},
            {"match_id": "M1", "bookmaker": "Book B", "line": 2.25, "odds_over": 2.05, "odds_under": 1.85},
            {"match_id": "M1", "bookmaker": "Book A", "line": 2.5, "odds_over": 2.20, "odds_under": 1.75},
            {"match_id": "M1", "bookmaker": "Book B", "line": 2.5, "odds_over": 2.10, "odds_under": 1.80},
            {"match_id": "M1", "bookmaker": "Book A", "line": 3.5, "odds_over": 3.50, "odds_under": 1.35},
            {"match_id": "M1", "bookmaker": "Book B", "line": 3.5, "odds_over": 3.60, "odds_under": 1.32},
        ]
    )


def _core_odds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "M1",
                "date": "2026-06-10",
                "stage": "group",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
                "bookmaker": "Book A",
                "odds_a_win": 2.10,
                "odds_draw": 3.30,
                "odds_b_win": 3.70,
                "odds_over_2_5": 2.20,
                "odds_under_2_5": 1.75,
            }
        ]
    )


def test_total_goals_margin_is_removed_per_bookmaker_line_before_aggregation() -> None:
    processed = process_total_goals_odds(_total_goals_odds())
    first = processed.iloc[0]
    expected_over = (1 / 1.50) / ((1 / 1.50) + (1 / 2.70))

    assert first["fair_over"] == pytest.approx(expected_over)
    assert first["fair_over"] + first["fair_under"] == pytest.approx(1.0)
    assert first["total_goals_overround"] == pytest.approx((1 / 1.50) + (1 / 2.70))
    assert set(processed.loc[processed["used_for_calibration"], "line"]) == {0.5, 1.5, 2.5, 3.5}

    aggregated = aggregate_total_goals_probabilities(processed)
    line = aggregated.loc[aggregated["line"] == 1.5].iloc[0]
    expected_book_b = (1 / 1.55) / ((1 / 1.55) + (1 / 2.60))
    assert line["fair_over"] == pytest.approx((expected_over + expected_book_b) / 2)
    assert line["bookmakers_count"] == 2


def test_multi_line_total_goals_constraints_move_poisson_fit_towards_targets() -> None:
    baseline = calibrate_poisson_model(CalibrationTargets(0.45, 0.28, 0.27))
    targets = ((1.5, 0.80), (2.5, 0.60), (3.5, 0.40))
    fitted = calibrate_poisson_model(CalibrationTargets(0.45, 0.28, 0.27, total_goals_over=targets))

    baseline_error = sum((poisson_over_total_probability(baseline.lambda_a, baseline.lambda_b, line) - target) ** 2 for line, target in targets)
    fitted_error = sum((poisson_over_total_probability(fitted.lambda_a, fitted.lambda_b, line) - target) ** 2 for line, target in targets)
    assert fitted_error < baseline_error


def test_world_cup_workflow_uses_all_half_goal_lines_and_reports_skipped_asian_lines(monkeypatch) -> None:
    captured_targets = []
    original_calibrate = calibrate_poisson_model

    def capture_targets(targets, *args, **kwargs):
        captured_targets.append(targets)
        return original_calibrate(targets, *args, **kwargs)

    monkeypatch.setattr("wc_predictor.workflow.calibrate_poisson_model", capture_targets)

    report = run_prediction_workflow(_core_odds(), total_goals_odds=_total_goals_odds()).match_report.iloc[0]

    assert tuple(line for line, _ in captured_targets[0].total_goals_over) == (0.5, 1.5, 2.5, 3.5)
    assert report["total_goals_lines_available"] == "0.5; 1.5; 2; 2.25; 2.5; 3.5"
    assert report["total_goals_lines_used_for_calibration"] == "0.5; 1.5; 2.5; 3.5"
    assert report["total_goals_lines_skipped_for_calibration"] == "2; 2.25"
    assert report["total_goals_lines_skipped"] == "2; 2.25"
    assert report["over_under_2_5_used"]
    assert report["multi_line_totals_used"]
    assert pd.notna(report["total_goals_line_fit_error"])
    assert "0.5: market_over=" in report["total_goals_line_diagnostics"]
    assert "1.5: market_over=" in report["total_goals_line_diagnostics"]
    assert "2.5: market_over=" in report["total_goals_line_diagnostics"]
    assert "3.5: market_over=" in report["total_goals_line_diagnostics"]
    assert "model_over=" in report["total_goals_line_diagnostics"]
    assert "error=" in report["total_goals_line_diagnostics"]
    assert "skipped from default Poisson calibration" in report["warnings"]


def test_world_cup_workflow_preserves_legacy_two_point_five_only_path() -> None:
    report = run_prediction_workflow(_core_odds()).match_report.iloc[0]

    assert report["total_goals_lines_available"] == ""
    assert report["total_goals_lines_used_for_calibration"] == ""
    assert report["total_goals_line_diagnostics"] == ""
    assert report["over_under_2_5_used"]
    assert not report["multi_line_totals_used"]


def test_invalid_asian_line_is_not_accepted_as_direct_calibration_target() -> None:
    with pytest.raises(ValueError, match="half-goal lines only"):
        CalibrationTargets(0.45, 0.28, 0.27, total_goals_over=((2.25, 0.50),))
