from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc_predictor.asian_handicap import (
    asian_handicap_components,
    asian_handicap_expected_profit,
    asian_handicap_profit,
    asian_handicap_profit_vector,
)
from wc_predictor.market_consistent import (
    MarketConsistentWeights,
    fit_market_consistent_matrix,
    select_asian_handicap_constraints,
)
from wc_predictor.odds import aggregate_asian_handicap_probabilities, process_asian_handicap_odds
from wc_predictor.oddsportal_asian_handicap import parse_oddsportal_asian_handicap_text
from wc_predictor.probabilities import poisson_score_matrix
from wc_predictor.config import ProjectConfig
from wc_predictor.workflow import run_prediction_workflow


def test_half_handicap_and_integer_push_settlement() -> None:
    assert asian_handicap_components(-0.5) == ((-0.5, 1.0),)
    assert asian_handicap_profit(1, 0, -0.5, "team_a", 1.90) == pytest.approx(0.90)
    assert asian_handicap_profit(0, 0, -0.5, "team_a", 1.90) == pytest.approx(-1.0)
    assert asian_handicap_profit(1, 0, -1.0, "team_a", 1.90) == pytest.approx(0.0)
    assert asian_handicap_profit(2, 0, -1.0, "team_a", 1.90) == pytest.approx(0.90)


def test_quarter_handicap_half_win_loss_and_push_cases() -> None:
    assert asian_handicap_components(-3.25) == ((-3.0, 0.5), (-3.5, 0.5))
    assert asian_handicap_components(2.25) == ((2.0, 0.5), (2.5, 0.5))
    assert asian_handicap_profit(4, 0, -3.25, "team_a", 2.0) == pytest.approx(1.0)
    assert asian_handicap_profit(3, 0, -3.25, "team_a", 2.0) == pytest.approx(-0.5)
    assert asian_handicap_profit(1, 3, 2.25, "team_a", 2.0) == pytest.approx(0.5)
    assert asian_handicap_profit(0, 3, 2.25, "team_a", 2.0) == pytest.approx(-1.0)


def test_team_a_team_b_symmetry() -> None:
    shape = (6, 6)
    fair_a = 0.55
    fair_b = 1.0 - fair_a
    team_a = asian_handicap_profit_vector(shape, -1.25, "team_a", 1.0 / fair_a).reshape(-1)
    team_b = asian_handicap_profit_vector(shape, -1.25, "team_b", 1.0 / fair_b).reshape(-1)

    assert team_b == pytest.approx((-fair_a / fair_b) * team_a)


def test_expected_profit_from_distribution() -> None:
    matrix = np.zeros((3, 3))
    matrix[1, 0] = 0.5
    matrix[0, 1] = 0.5

    assert asian_handicap_expected_profit(matrix, 0.0, "team_a", 2.0) == pytest.approx(0.0)


def test_parser_handles_oddsportal_paste_duplicates_and_missing_odds() -> None:
    text = """
Asian Handicap -0.5
Bet365
1.91
1.95
Unibet
1.90
-
Book A -1.25 2.10 1.75
Book A -1.25 2.10 1.75
"""

    result = parse_oddsportal_asian_handicap_text(text, "M001", source_file="M001_asian_handicap.txt")

    assert len(result.odds) == 2
    assert set(result.odds["handicap"]) == {-0.5, -1.25}
    warnings = result.report.iloc[0]["warnings"]
    assert "missing_odds:Unibet:-0.5" in warnings
    assert "deduplicated_identical_row:Book A:-1.25" in warnings


def test_handicap_processing_and_aggregation() -> None:
    raw = pd.DataFrame(
        [
            {"match_id": "M001", "bookmaker": "A", "handicap": -0.5, "odds_team_a": 1.91, "odds_team_b": 1.95},
            {"match_id": "M001", "bookmaker": "B", "handicap": -0.5, "odds_team_a": 1.90, "odds_team_b": 1.96},
        ]
    )

    processed = process_asian_handicap_odds(raw)
    aggregated = aggregate_asian_handicap_probabilities(processed)

    assert len(processed) == 2
    assert aggregated.iloc[0]["bookmakers_count"] == 2
    assert aggregated.iloc[0]["line_kind"] == "half_goal"
    assert aggregated.iloc[0]["fair_team_a"] + aggregated.iloc[0]["fair_team_b"] == pytest.approx(1.0)


def test_parser_stops_before_page_history_after_last_handicap_line() -> None:
    text = """
Asian Handicap +5
Book A
+5
2.10
1.80

AI Match Predictions
Germany
5.20
4.38
1.72
Previous Matches: USA
Paraguay
2.00
3.35
4.00
"""

    result = parse_oddsportal_asian_handicap_text(text, "M004", source_file="M004_asian_handicap.txt")
    processed = process_asian_handicap_odds(result.odds)
    aggregated = aggregate_asian_handicap_probabilities(processed)

    assert len(result.odds) == 1
    assert result.odds.iloc[0]["bookmaker"] == "Book A"
    assert aggregated.iloc[0]["handicap"] == pytest.approx(5.0)
    assert aggregated.iloc[0]["bookmakers_count"] == 1
    assert "Germany" not in set(result.odds["bookmaker"])
    assert "stopped_at_non_market_section:AI Match Predictions" in result.report.iloc[0]["warnings"]


def test_underround_asian_handicap_line_is_not_selected_for_market_consistent() -> None:
    handicap = pd.DataFrame(
        [
            {
                "match_id": "M004",
                "handicap": 5.0,
                "fair_team_a": 0.578,
                "fair_team_b": 0.422,
                "fair_odds_team_a": 1 / 0.578,
                "fair_odds_team_b": 1 / 0.422,
                "bookmakers_count": 7,
                "average_asian_handicap_overround": 0.673146705,
                "line_kind": "integer_asian",
                "warnings": "Overround 0.6731 is below 1.0000",
            }
        ]
    )

    selected = select_asian_handicap_constraints(
        handicap,
        (9, 9),
        {"a_win": 0.60, "draw": 0.25, "b_win": 0.15},
    )

    row = selected.iloc[0]
    assert row["selected_for_market_consistent"] == "no"
    assert row["selection_weight"] == pytest.approx(0.0)
    assert "asian_handicap_overround_below_1" in row["skipped_reason"]


def test_market_consistent_handicap_constraints_reduce_fit_error_and_shift_margin_distribution() -> None:
    prior = poisson_score_matrix(1.0, 1.0, max_goals=6)
    handicap = pd.DataFrame(
        [
            {
                "match_id": "M001",
                "handicap": -1.5,
                "fair_team_a": 0.60,
                "fair_team_b": 0.40,
                "fair_odds_team_a": 1 / 0.60,
                "fair_odds_team_b": 1 / 0.40,
                "bookmakers_count": 4,
                "average_asian_handicap_overround": 1.03,
                "line_kind": "half_goal",
                "warnings": "",
            }
        ]
    )
    market = prior.outcome_probabilities()
    result = fit_market_consistent_matrix(
        prior,
        {"a_win": market["a_win"], "draw": market["draw"], "b_win": market["b_win"]},
        asian_handicap=handicap,
        weights=MarketConsistentWeights(one_x_two=0.0, liquid_total_goals=0.0, btts=0.0, correct_score=0.0, asian_handicap=500.0),
    )

    before = asian_handicap_expected_profit(prior.probabilities, -1.5, "team_a", 1 / 0.60)
    after = asian_handicap_expected_profit(result.matrix.probabilities, -1.5, "team_a", 1 / 0.60)
    assert abs(after) < abs(before)
    assert result.diagnostics["market_consistent_asian_handicap_constraint_count"] == 1
    assert result.diagnostics["market_consistent_asian_handicap_lines_used"] == "-1.5 half_goal"


def _many_extreme_favourite_handicap_lines() -> pd.DataFrame:
    rows = []
    probabilities = {
        -0.5: 0.97,
        -1.5: 0.91,
        -2.5: 0.82,
        -3.0: 0.72,
        -3.25: 0.66,
        -3.5: 0.58,
        -3.75: 0.52,
        -4.0: 0.47,
        -4.25: 0.41,
        -4.5: 0.35,
        -5.0: 0.22,
        -6.5: 0.11,
        -7.5: 0.05,
        -7.75: 0.03,
    }
    for handicap, fair_team_a in probabilities.items():
        rows.append(
            {
                "match_id": "MEXT",
                "handicap": handicap,
                "fair_team_a": fair_team_a,
                "fair_team_b": 1.0 - fair_team_a,
                "fair_odds_team_a": 1.0 / fair_team_a,
                "fair_odds_team_b": 1.0 / (1.0 - fair_team_a),
                "bookmakers_count": 5,
                "average_asian_handicap_overround": 1.04,
                "line_kind": "integer_asian" if float(handicap).is_integer() else "quarter_asian" if abs(handicap) % 1 in {0.25, 0.75} else "half_goal",
                "warnings": "",
            }
        )
    return pd.DataFrame(rows)


def test_selection_keeps_near_money_lines_and_skips_extremes() -> None:
    selected = select_asian_handicap_constraints(
        _many_extreme_favourite_handicap_lines(),
        (9, 9),
        {"a_win": 0.90, "draw": 0.07, "b_win": 0.03},
    )

    selected_lines = set(selected.loc[selected["selected_for_market_consistent"].eq("yes"), "handicap"])
    assert {-3.5, -3.75, -4.0, -4.25, -4.5}.issubset(selected_lines)
    assert -0.5 not in selected_lines
    assert -7.5 not in selected_lines
    assert len(selected_lines) <= 7


def test_selection_skips_grid_boundary_sensitive_lines() -> None:
    selected = select_asian_handicap_constraints(
        _many_extreme_favourite_handicap_lines(),
        (9, 9),
        {"a_win": 0.90, "draw": 0.07, "b_win": 0.03},
    )
    boundary = selected[selected["handicap"].eq(-7.5)].iloc[0]

    assert boundary["handicap_line_near_grid_boundary"] == "yes"
    assert boundary["skipped_due_to_grid_boundary"] == "yes"
    assert boundary["selected_for_market_consistent"] == "no"


def test_market_consistent_uses_selected_handicap_constraints_not_all_lines() -> None:
    prior = poisson_score_matrix(3.2, 0.25, max_goals=8)
    handicap = _many_extreme_favourite_handicap_lines()
    market = prior.outcome_probabilities()

    result = fit_market_consistent_matrix(
        prior,
        {"a_win": market["a_win"], "draw": market["draw"], "b_win": market["b_win"]},
        asian_handicap=handicap,
        weights=MarketConsistentWeights(one_x_two=0.0, liquid_total_goals=0.0, btts=0.0, correct_score=0.0, asian_handicap=250.0),
    )

    assert result.diagnostics["market_consistent_asian_handicap_lines_available"] == len(handicap)
    assert result.diagnostics["market_consistent_asian_handicap_constraint_count"] <= 7
    assert result.diagnostics["market_consistent_asian_handicap_lines_selected"] <= 7
    assert "-7.5" not in result.diagnostics["market_consistent_asian_handicap_lines_used"]
    assert pd.notna(result.diagnostics["market_consistent_asian_handicap_fit_error_all"])


def test_many_deep_handicap_lines_do_not_create_artificial_boundary_mass() -> None:
    prior = poisson_score_matrix(3.2, 0.25, max_goals=8)
    handicap = _many_extreme_favourite_handicap_lines()
    market = prior.outcome_probabilities()

    result = fit_market_consistent_matrix(
        prior,
        {"a_win": market["a_win"], "draw": market["draw"], "b_win": market["b_win"]},
        asian_handicap=handicap,
        weights=MarketConsistentWeights(one_x_two=0.0, liquid_total_goals=0.0, btts=0.0, correct_score=0.0, asian_handicap=250.0),
    )

    assert result.matrix.probabilities[8, 0] < 0.04
    assert result.diagnostics["market_consistent_status"] != "failed_severe"


def _orientation_core_odds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "MORIENT",
                "date": "2026-06-14",
                "stage": "group",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
                "bookmaker": "Book",
                "odds_a_win": 1.40,
                "odds_draw": 4.80,
                "odds_b_win": 8.00,
            }
        ]
    )


def test_asian_handicap_margin_model_diagnostic_is_reported_when_enabled() -> None:
    asian_handicap = pd.DataFrame(
        [
            {"match_id": "MORIENT", "bookmaker": "Book", "handicap": handicap, "odds_team_a": 1.91, "odds_team_b": 1.91}
            for handicap in (-1.5, -0.5, 0.5)
        ]
    )

    workflow = run_prediction_workflow(
        _orientation_core_odds(),
        config=ProjectConfig(
            enable_margin_method_comparison=False,
            enable_market_consistent_challenger=False,
            enable_asian_handicap_margin_model=True,
        ),
        asian_handicap_odds=asian_handicap,
    )

    report = workflow.match_report.iloc[0]
    assert report["asian_handicap_margin_model_type"] == "skellam"
    assert "skellam=" in report["asian_handicap_margin_distribution_comparison"]
    assert pd.notna(report["asian_handicap_margin_fit_error"])


def test_asian_handicap_margin_model_diagnostic_off_by_default() -> None:
    asian_handicap = pd.DataFrame(
        [
            {"match_id": "MORIENT", "bookmaker": "Book", "handicap": handicap, "odds_team_a": 1.91, "odds_team_b": 1.91}
            for handicap in (-1.5, -0.5, 0.5)
        ]
    )

    workflow = run_prediction_workflow(
        _orientation_core_odds(),
        config=ProjectConfig(enable_margin_method_comparison=False, enable_market_consistent_challenger=False),
        asian_handicap_odds=asian_handicap,
    )

    assert workflow.match_report.iloc[0]["asian_handicap_margin_model_type"] == ""


def test_asian_handicap_orientation_guard_accepts_normal_ladder() -> None:
    asian_handicap = pd.DataFrame(
        [
            {
                "match_id": "MORIENT",
                "bookmaker": "Book",
                "handicap": -1.5,
                "odds_team_a": 1.91,
                "odds_team_b": 1.91,
            }
        ]
    )

    workflow = run_prediction_workflow(
        _orientation_core_odds(),
        config=ProjectConfig(enable_margin_method_comparison=False, enable_market_consistent_challenger=False),
        asian_handicap_odds=asian_handicap,
    )

    report = workflow.match_report.iloc[0]
    diagnostics = workflow.aggregated_asian_handicap_probabilities.iloc[0]
    assert "asian_handicap_orientation_suspicious" not in report["warning_flags"]
    assert report["asian_handicap_orientation_suspicious"] == "no"
    assert diagnostics["asian_handicap_orientation_suspicious"] == "no"


def test_asian_handicap_orientation_guard_warns_on_reversed_ladder() -> None:
    asian_handicap = pd.DataFrame(
        [
            {
                "match_id": "MORIENT",
                "bookmaker": "Book",
                "handicap": 1.5,
                "odds_team_a": 1.91,
                "odds_team_b": 1.91,
            }
        ]
    )

    workflow = run_prediction_workflow(
        _orientation_core_odds(),
        config=ProjectConfig(enable_margin_method_comparison=False, enable_market_consistent_challenger=False),
        asian_handicap_odds=asian_handicap,
    )

    report = workflow.match_report.iloc[0]
    dashboard = workflow.final_decision_dashboard.iloc[0]
    diagnostics = workflow.aggregated_asian_handicap_probabilities.iloc[0]
    assert "asian_handicap_orientation_suspicious" in report["warning_flags"]
    assert report["asian_handicap_orientation_suspicious"] == "yes"
    assert diagnostics["asian_handicap_orientation_suspicious"] == "yes"
    assert "asian_handicap_orientation_suspicious" in dashboard["risk_notes"]
