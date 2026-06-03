import json
from pathlib import Path
import runpy
import sys

import pandas as pd
import pytest
from openpyxl import load_workbook

from wc_predictor.config import ProjectConfig
from wc_predictor.reporting import (
    format_world_cup_console_summary,
    format_world_cup_model_risk_summary,
    format_world_cup_prediction_diagnostics,
)
from wc_predictor.world_cup import (
    WORLD_CUP_ODDS_MISSING_MESSAGE,
    WorldCupPredictionSettings,
    create_world_cup_odds_file,
    resolve_world_cup_odds_input,
    run_world_cup_predictions,
)
from wc_predictor.workflow import _btts_warning_flags, _decision_aid, _ev_explanation


EXAMPLES = Path("data/examples")
TEMPLATES = Path("data/templates")
RUN_SCRIPT = Path("scripts/run_world_cup_predictions.py").resolve()
CREATE_SCRIPT = Path("scripts/create_world_cup_odds_file.py").resolve()
ODDS_COLUMNS = [
    "match_id",
    "date",
    "stage",
    "group",
    "team_a",
    "team_b",
    "bookmaker",
    "odds_a_win",
    "odds_draw",
    "odds_b_win",
    "odds_over_2_5",
    "odds_under_2_5",
    "odds_btts_yes",
    "odds_btts_no",
    "odds_a_qualifies",
    "odds_b_qualifies",
    "odds_timestamp",
    "odds_source_url",
    "source_quality",
    "notes",
]


def _settings(tmp_path: Path) -> WorldCupPredictionSettings:
    return WorldCupPredictionSettings(
        input_path=EXAMPLES / "example_world_cup_odds.csv",
        csv_output_path=tmp_path / "world_cup_recommendations.csv",
        xlsx_output_path=tmp_path / "world_cup_recommendations.xlsx",
        submission_xlsx_output_path=tmp_path / "world_cup_submission_sheet.xlsx",
    )


def test_world_cup_templates_have_expected_columns() -> None:
    csv_odds = pd.read_csv(TEMPLATES / "world_cup_odds_template.csv")
    xlsx_odds = pd.read_excel(TEMPLATES / "world_cup_odds_template.xlsx")
    predictions = pd.read_csv(TEMPLATES / "friend_predictions_template.csv")

    assert csv_odds.columns.tolist() == ODDS_COLUMNS
    assert xlsx_odds.columns.tolist() == ODDS_COLUMNS
    assert predictions.columns.tolist() == [
        "match_id",
        "player",
        "predicted_team_a_goals",
        "predicted_team_b_goals",
        "predicted_qualifier",
    ]


def test_world_cup_workflow_exports_real_tournament_recommendations(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    workflow = run_world_cup_predictions(settings)

    assert settings.csv_output_path.exists()
    assert settings.xlsx_output_path.exists()
    assert settings.submission_xlsx_output_path.exists()
    assert len(workflow.match_report) == 2
    assert workflow.match_report.loc[0, "group"] == "Group A"
    assert workflow.match_report.loc[1, "recommended_qualifier"] in {"Gamma", "Delta"}
    assert {
        "market_a_win",
        "market_draw",
        "market_b_win",
        "lambda_a",
        "lambda_b",
        "favourite_bucket",
        "most_likely_scoreline",
        "recommended_score",
        "top_5_ev_predictions",
        "odds_timestamp_min",
        "odds_timestamp_max",
        "bookmakers_used",
        "number_of_bookmakers",
        "has_over_under",
        "has_btts",
        "has_qualification_odds",
        "favourite_probability",
        "ev_gap_best_vs_second",
        "ev_gap_best_vs_modal",
        "baseline_poisson_recommended_score",
        "baseline_poisson_ev_gap_best_vs_second",
        "correct_score_blended_recommended_score",
        "correct_score_blended_ev_gap_best_vs_second",
        "final_live_recommended_score",
        "final_live_ev_gap_best_vs_second",
        "model_recommendations_agree",
        "model_disagreement_warning",
        "dixon_coles_rho",
        "dixon_coles_recommended_score",
        "dixon_coles_ev_gap_best_vs_second",
        "dixon_coles_top_5_ev_predictions",
        "dixon_coles_changes_recommendation",
        "estimated_most_crowded_public_score",
        "estimated_most_crowded_public_pick_share",
        "public_strategy_score",
        "public_strategy_ev_cost",
        "public_strategy_public_pick_share",
        "public_strategy_leverage_score",
        "public_strategy_mode",
        "public_strategy_reason",
        "friend_strategy_score",
        "warning_flags",
        "recommendation_confidence",
        "manual_review_flag",
        "close_alternatives",
        "ev_gap_to_second",
        "ev_gap_to_third",
        "decision_note",
        "market_fair_btts_yes_probability",
        "market_fair_btts_no_probability",
        "model_implied_btts_yes_probability",
        "model_implied_btts_no_probability",
        "btts_fit_error",
        "btts_market_available",
        "model_probability_team_a_clean_sheet",
        "model_probability_team_b_clean_sheet",
        "model_probability_no_btts",
        "model_probability_btts",
        "expected_total_goals",
        "expected_team_a_goals",
        "expected_team_b_goals",
        "probability_total_goals_0",
        "probability_total_goals_1",
        "probability_total_goals_2",
        "probability_total_goals_3",
        "probability_total_goals_4_plus",
        "top_5_ev_decomposition",
        "top_5_ev_decomposition_json",
        "top_10_ev_decomposition",
        "top_10_ev_decomposition_json",
        "ev_explanation",
        "extreme_favourite_audit_triggered",
        "normal_grid_recommendation",
        "normal_grid_poisson_recommendation",
        "larger_grid_recommendation",
        "larger_grid_poisson_recommendation",
        "recommendation_changes_with_larger_grid",
        "larger_grid_poisson_changes_recommendation",
        "larger_grid_differs_from_live_recommendation",
        "normal_grid_tail_mass",
        "larger_grid_tail_mass",
        "ev_favourite_3_0_normal_grid",
        "ev_favourite_4_0_normal_grid",
        "ev_favourite_5_0_normal_grid",
        "ev_favourite_3_0_larger_grid",
        "ev_favourite_4_0_larger_grid",
        "ev_favourite_5_0_larger_grid",
    }.issubset(workflow.match_report.columns)
    assert workflow.match_report["recommended_score"].equals(workflow.match_report["final_live_recommended_score"])
    assert workflow.match_report["baseline_poisson_recommended_score"].equals(
        workflow.match_report["dixon_coles_recommended_score"]
    )
    assert len(pd.read_csv(settings.csv_output_path)) == 2
    assert len(pd.read_excel(settings.xlsx_output_path)) == 2


def test_btts_audit_uses_aggregated_market_fair_probability(tmp_path: Path) -> None:
    report = run_world_cup_predictions(_settings(tmp_path)).match_report.set_index("match_id")
    yes_one = (1 / 1.95) / ((1 / 1.95) + (1 / 1.87))
    yes_two = (1 / 1.98) / ((1 / 1.98) + (1 / 1.85))
    expected_yes = (yes_one + yes_two) / 2

    assert report.loc["WC001", "market_fair_btts_yes_probability"] == pytest.approx(expected_yes)
    assert report.loc["WC001", "market_fair_btts_no_probability"] == pytest.approx(1 - expected_yes)
    assert report.loc["WC001", "btts_market_available"] == "yes"


def test_model_implied_btts_audit_is_computed_from_score_matrix(tmp_path: Path) -> None:
    workflow = run_world_cup_predictions(_settings(tmp_path))
    report = workflow.match_report.set_index("match_id")
    matrix = workflow.score_matrices["WC001"]

    assert report.loc["WC001", "model_implied_btts_yes_probability"] == pytest.approx(
        matrix.btts_yes_probability()
    )
    assert report.loc["WC001", "model_implied_btts_no_probability"] == pytest.approx(
        1 - matrix.btts_yes_probability()
    )
    assert report.loc["WC001", "model_probability_btts"] == pytest.approx(matrix.btts_yes_probability())
    assert report.loc["WC001", "model_probability_no_btts"] == pytest.approx(1 - matrix.btts_yes_probability())


def test_ev_decomposition_components_sum_to_total_ev(tmp_path: Path) -> None:
    report = run_world_cup_predictions(_settings(tmp_path)).match_report.set_index("match_id")
    records = json.loads(report.loc["WC001", "top_10_ev_decomposition_json"])

    assert len(records) == 10
    for record in records:
        component_total = (
            record["participation_component"]
            + record["result_component"]
            + record["goal_difference_component"]
            + record["exact_score_component"]
        )
        assert component_total == pytest.approx(record["total_expected_points"])
        assert record["total_expected_points_from_components"] == pytest.approx(record["total_expected_points"])


def test_ev_explanation_uses_probability_weighted_margin_wording() -> None:
    explanation = _ev_explanation(
        [
            {
                "predicted_score": "3-0",
                "result_component": 3.6,
                "goal_difference_component": 0.50,
                "exact_score_component": 0.20,
                "participation_component": 1.0,
            },
            {
                "predicted_score": "4-0",
                "result_component": 3.6,
                "goal_difference_component": 0.40,
                "exact_score_component": 0.20,
                "participation_component": 1.0,
            },
        ]
    )

    assert "expected value to the +3 margin than the +4 margin" in explanation
    assert "payoff is higher" not in explanation


def test_decision_aid_flags_clustered_recommendations_for_manual_review() -> None:
    aid = _decision_aid(
        [
            {"predicted_score": "3-0", "total_expected_points": 5.00},
            {"predicted_score": "4-0", "total_expected_points": 4.96},
            {"predicted_score": "5-0", "total_expected_points": 4.94},
        ],
        high_score_cluster=True,
        high_score_cluster_alternatives="3-0, 4-0, 5-0",
    )

    assert aid["recommendation_confidence"] == "low / clustered"
    assert aid["manual_review_flag"] == "yes"
    assert aid["ev_gap_to_second"] == pytest.approx(0.04)
    assert aid["ev_gap_to_third"] == pytest.approx(0.06)
    assert aid["close_alternatives"] == "4-0, 5-0, 3-0"
    assert "High-score cluster" in aid["decision_note"]


def test_decision_aid_marks_clear_ev_lead_as_high_confidence() -> None:
    aid = _decision_aid(
        [
            {"predicted_score": "2-0", "total_expected_points": 5.00},
            {"predicted_score": "1-0", "total_expected_points": 4.80},
            {"predicted_score": "3-0", "total_expected_points": 4.70},
        ],
        high_score_cluster=False,
        high_score_cluster_alternatives="",
    )

    assert aid["recommendation_confidence"] == "high"
    assert aid["manual_review_flag"] == "no"
    assert aid["close_alternatives"] == ""
    assert aid["ev_gap_to_second"] == pytest.approx(0.20)
    assert aid["decision_note"] == "No manual review signal."


def test_btts_warning_appears_for_recommendation_conflicting_with_strong_market_signal() -> None:
    flags = _btts_warning_flags(
        has_btts=True,
        market_btts_yes=0.60,
        market_btts_no=0.40,
        model_btts_yes=0.58,
        recommended_score=(1, 0),
    )

    assert "recommended_no_btts_against_strong_btts_yes_market" in flags


def test_top_ten_ev_decomposition_is_written_to_excel(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    run_world_cup_predictions(settings)
    workbook = load_workbook(settings.xlsx_output_path)
    recommendation_headers = [cell.value for cell in workbook["recommendations"][1]]
    diagnostics_headers = [cell.value for cell in workbook["ev_decomposition"][1]]

    assert "top_10_ev_decomposition" in recommendation_headers
    assert "recommendation_confidence" in recommendation_headers
    assert "manual_review_flag" in recommendation_headers
    assert "decision_note" in recommendation_headers
    assert "ev_decomposition" in workbook.sheetnames
    assert "predicted_score" in diagnostics_headers
    assert "exact_score_component" in diagnostics_headers
    assert workbook["ev_decomposition"].max_row == 21


def test_extreme_favourite_audit_and_larger_grid_sensitivity_trigger(tmp_path: Path) -> None:
    input_path = tmp_path / "extreme.csv"
    pd.DataFrame(
        [
            {
                "match_id": "EXTREME",
                "date": "2026-06-12",
                "stage": "group stage",
                "group": "A",
                "team_a": "Favourite",
                "team_b": "Longshot",
                "bookmaker": "MarketOne",
                "odds_a_win": 1.08,
                "odds_draw": 12.0,
                "odds_b_win": 35.0,
                "odds_btts_yes": 2.30,
                "odds_btts_no": 1.65,
            }
        ]
    ).to_csv(input_path, index=False)
    settings = WorldCupPredictionSettings(
        input_path=input_path,
        csv_output_path=tmp_path / "recommendations.csv",
        xlsx_output_path=tmp_path / "recommendations.xlsx",
        submission_xlsx_output_path=tmp_path / "submission.xlsx",
    )

    report = run_world_cup_predictions(settings).match_report.iloc[0]

    assert report["favourite_probability"] > 0.75
    assert bool(report["extreme_favourite_audit_triggered"])
    assert report["top_clean_sheet_scores"]
    assert report["top_favourite_margin_probabilities"]
    assert report["current_final_recommendation"] == report["recommended_score"]
    assert report["normal_grid_poisson_recommendation"] == report["baseline_poisson_recommended_score"]
    assert report["larger_grid_recommendation"]
    assert report["larger_grid_poisson_recommendation"] == report["larger_grid_recommendation"]
    assert pd.notna(report["larger_grid_tail_mass"])
    assert report["larger_grid_tail_mass"] < report["normal_grid_tail_mass"]
    assert pd.notna(report["ev_favourite_3_0_normal_grid"])
    assert pd.notna(report["ev_favourite_3_0_larger_grid"])


def test_larger_grid_sensitivity_can_change_poisson_recommendation_on_synthetic_extreme_favourite(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "extreme_small_grid.csv"
    pd.DataFrame(
        [
            {
                "match_id": "EXTREME_GRID",
                "date": "2026-06-12",
                "stage": "group stage",
                "group": "A",
                "team_a": "Favourite",
                "team_b": "Longshot",
                "bookmaker": "MarketOne",
                "odds_a_win": 1.04,
                "odds_draw": 17.0,
                "odds_b_win": 55.0,
                "odds_btts_yes": 2.60,
                "odds_btts_no": 1.55,
            }
        ]
    ).to_csv(input_path, index=False)
    settings = WorldCupPredictionSettings(
        input_path=input_path,
        csv_output_path=tmp_path / "recommendations.csv",
        xlsx_output_path=tmp_path / "recommendations.xlsx",
        submission_xlsx_output_path=tmp_path / "submission.xlsx",
    )

    report = run_world_cup_predictions(
        settings,
        ProjectConfig(max_goals_score_matrix=2, max_candidate_goals=5),
    ).match_report.iloc[0]

    assert bool(report["extreme_favourite_audit_triggered"])
    assert bool(report["larger_grid_poisson_changes_recommendation"])
    assert report["normal_grid_poisson_recommendation"] != report["larger_grid_poisson_recommendation"]
    assert report["larger_grid_tail_mass"] < report["normal_grid_tail_mass"]
    assert "larger_grid_sensitivity" in report["warning_flags"]


def test_world_cup_workflow_optionally_uses_correct_score_blend(tmp_path: Path) -> None:
    settings = WorldCupPredictionSettings(
        input_path=EXAMPLES / "example_world_cup_odds.csv",
        correct_score_input_path=EXAMPLES / "example_world_cup_correct_score_odds.csv",
        csv_output_path=tmp_path / "recommendations.csv",
        xlsx_output_path=tmp_path / "recommendations.xlsx",
        submission_xlsx_output_path=tmp_path / "submission.xlsx",
    )

    workflow = run_world_cup_predictions(
        settings,
        ProjectConfig(correct_score_poisson_weight=0.5, min_scorelines_for_blend=6),
    )
    report = workflow.match_report.set_index("match_id")

    assert report["has_correct_score_market"].all()
    assert (report["correct_score_poisson_weight"] == 0.5).all()
    assert report["correct_score_market_top_10"].str.len().gt(0).all()
    assert report["correct_score_blended_top_10"].str.len().gt(0).all()
    assert report["correct_score_kl_divergence"].gt(0).all()
    assert (report["correct_score_scorelines_count"] == 6).all()
    assert (report["correct_score_bookmakers_count"] == 2).all()
    assert report["correct_score_sparse_warning"].str.contains("fewer_than_10_scorelines").all()
    assert report["top_10_market_scorelines"].equals(report["correct_score_market_top_10"])
    assert report["top_10_blended_scorelines"].equals(report["correct_score_blended_top_10"])
    assert report["kl_divergence"].equals(report["correct_score_kl_divergence"])
    assert (report["selected_blend_weight"] == 0.5).all()
    assert report["correct_score_blend_applied"].all()
    assert (report["correct_score_aggregation_method"] == "mean").all()
    assert (report["number_of_correct_score_bookmakers"] == 2).all()
    assert report["average_correct_score_overround"].gt(0).all()
    assert report["max_correct_score_overround"].gt(0).all()
    assert report["outlier_count"].eq(0).all()
    assert report["scoreline_coverage_warning"].str.contains("missing_common_scorelines").all()
    assert report["correct_score_bookmaker_diagnostics"].str.contains("overround=").all()


def test_world_cup_workflow_optionally_loads_total_goals_ladder(tmp_path: Path) -> None:
    total_goals_path = tmp_path / "total_goals.csv"
    pd.DataFrame(
        [
            {"match_id": "WC001", "bookmaker": "MarketOne", "line": 1.5, "odds_over": 1.40, "odds_under": 3.00},
            {"match_id": "WC001", "bookmaker": "MarketTwo", "line": 1.5, "odds_over": 1.45, "odds_under": 2.90},
            {"match_id": "WC001", "bookmaker": "MarketOne", "line": 2.5, "odds_over": 2.00, "odds_under": 1.80},
            {"match_id": "WC001", "bookmaker": "MarketTwo", "line": 2.5, "odds_over": 2.10, "odds_under": 1.75},
        ]
    ).to_csv(total_goals_path, index=False)
    baseline = _settings(tmp_path)
    settings = WorldCupPredictionSettings(
        input_path=baseline.input_path,
        total_goals_input_path=total_goals_path,
        csv_output_path=baseline.csv_output_path,
        xlsx_output_path=baseline.xlsx_output_path,
        submission_xlsx_output_path=baseline.submission_xlsx_output_path,
    )

    report = run_world_cup_predictions(settings).match_report.set_index("match_id")

    assert report.loc["WC001", "total_goals_lines_available"] == "1.5; 2.5"
    assert bool(report.loc["WC001", "multi_line_totals_used"])
    assert report.loc["WC002", "total_goals_lines_available"] == ""


def test_world_cup_excel_export_is_formatted_for_manual_review(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    run_world_cup_predictions(settings)
    worksheet = load_workbook(settings.xlsx_output_path)["recommendations"]
    headers = [cell.value for cell in worksheet[1]]

    assert worksheet.freeze_panes == "A2"
    assert headers[:19] == [
        "match_id",
        "date",
        "stage",
        "group",
        "team_a",
        "team_b",
        "recommended_score",
        "market_a",
        "market_draw",
        "market_b",
        "favourite_probability",
        "favourite_bucket",
        "lambda_a",
        "lambda_b",
        "recommended_qualifier",
        "best_expected_points",
        "most_likely_scoreline",
        "ev_optimal_differs",
        "top_5_ev_predictions",
    ]
    column = {name: index + 1 for index, name in enumerate(headers)}
    assert worksheet.cell(2, column["market_a"]).number_format == "0.00%"
    assert worksheet.cell(2, column["favourite_probability"]).number_format == "0.00%"
    assert worksheet.cell(2, column["lambda_a"]).number_format == "0.0000"
    assert worksheet.cell(2, column["calibration_loss"]).number_format == "0.0000"
    assert worksheet.cell(2, column["best_expected_points"]).number_format == "0.000"
    assert worksheet.cell(2, column["estimated_most_crowded_public_pick_share"]).number_format == "0.00%"
    assert worksheet.cell(2, column["public_strategy_ev_cost"]).number_format == "0.000"
    assert worksheet.cell(2, column["top_5_ev_predictions"]).alignment.wrap_text
    assert worksheet.cell(2, column["public_strategy_reason"]).alignment.wrap_text
    assert worksheet.cell(2, column["warnings"]).alignment.wrap_text
    assert worksheet.column_dimensions["A"].width >= len("match_id")


def test_world_cup_submission_sheet_contains_only_entry_columns_and_formatting(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    run_world_cup_predictions(settings)
    worksheet = load_workbook(settings.submission_xlsx_output_path)["submission"]
    headers = [cell.value for cell in worksheet[1]]
    column = {name: index + 1 for index, name in enumerate(headers)}

    assert headers == [
        "match_id",
        "date",
        "stage",
        "group",
        "team_a",
        "team_b",
        "recommended_score",
        "recommended_qualifier",
        "best_expected_points",
        "recommendation_confidence",
        "manual_review_flag",
        "close_alternatives",
        "ev_gap_to_second",
        "ev_gap_to_third",
        "decision_note",
        "favourite_bucket",
        "top_3_alternatives",
        "notes",
    ]
    assert worksheet.freeze_panes == "A2"
    assert worksheet.cell(2, column["best_expected_points"]).number_format == "0.000"
    assert worksheet.cell(2, column["ev_gap_to_second"]).number_format == "0.000"
    assert worksheet.cell(2, column["top_3_alternatives"]).alignment.wrap_text
    assert worksheet.cell(2, column["decision_note"]).alignment.wrap_text
    assert worksheet.cell(2, column["notes"]).alignment.wrap_text
    assert worksheet.cell(2, column["top_3_alternatives"]).value.count(";") == 2
    assert worksheet.cell(2, column["match_id"]).value == "WC001"
    assert worksheet.cell(3, column["match_id"]).value == "WC002"
    assert worksheet.column_dimensions["A"].width >= len("match_id")


def test_create_world_cup_odds_file_copies_template(tmp_path: Path) -> None:
    destination = tmp_path / "raw" / "world_cup_odds.xlsx"
    path, changed = create_world_cup_odds_file(destination)

    assert changed
    assert path == destination
    assert destination.read_bytes() == (TEMPLATES / "world_cup_odds_template.xlsx").read_bytes()


def test_create_world_cup_odds_file_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "world_cup_odds.xlsx"
    destination.write_text("manual odds", encoding="utf-8")

    _, changed = create_world_cup_odds_file(destination)

    assert not changed
    assert destination.read_text(encoding="utf-8") == "manual odds"

    _, changed = create_world_cup_odds_file(destination, overwrite=True)

    assert changed
    assert destination.read_bytes() == (TEMPLATES / "world_cup_odds_template.xlsx").read_bytes()


def test_world_cup_workflow_loads_xlsx_odds_input(tmp_path: Path) -> None:
    input_path = tmp_path / "world_cup_odds.xlsx"
    pd.read_csv(EXAMPLES / "example_world_cup_odds.csv").to_excel(input_path, index=False)
    settings = WorldCupPredictionSettings(
        input_path=input_path,
        csv_output_path=tmp_path / "recommendations.csv",
        xlsx_output_path=tmp_path / "recommendations.xlsx",
        submission_xlsx_output_path=tmp_path / "submission.xlsx",
    )

    workflow = run_world_cup_predictions(settings)

    assert len(workflow.match_report) == 2
    assert workflow.match_report.loc[0, "match_id"] == "WC001"


def test_optional_odds_source_metadata_is_summarised_when_present(tmp_path: Path) -> None:
    input_path = tmp_path / "world_cup_odds.xlsx"
    odds = pd.read_csv(EXAMPLES / "example_world_cup_odds.csv")
    odds["odds_timestamp"] = [
        "2026-06-01T09:00:00Z",
        "2026-06-01T11:00:00Z",
        "2026-06-02T09:00:00Z",
        "2026-06-02T11:00:00Z",
    ]
    odds["odds_source_url"] = ["https://one.example", "https://two.example"] * 2
    odds["source_quality"] = ["major_bookmaker", "sharp"] * 2
    odds.to_excel(input_path, index=False)
    settings = WorldCupPredictionSettings(
        input_path=input_path,
        csv_output_path=tmp_path / "recommendations.csv",
        xlsx_output_path=tmp_path / "recommendations.xlsx",
        submission_xlsx_output_path=tmp_path / "submission.xlsx",
    )

    report = run_world_cup_predictions(settings).match_report.set_index("match_id")

    assert report.loc["WC001", "odds_timestamp_min"] == "2026-06-01T09:00:00+00:00"
    assert report.loc["WC001", "odds_timestamp_max"] == "2026-06-01T11:00:00+00:00"
    assert report.loc["WC001", "odds_source_urls"] == "https://one.example; https://two.example"
    assert report.loc["WC001", "source_qualities"] == "major_bookmaker; sharp"
    assert report.loc["WC001", "bookmakers_used"] == "MarketOne; MarketTwo"
    assert report.loc["WC001", "number_of_bookmakers"] == 2


def test_world_cup_warning_flags_cover_model_risk_conditions(tmp_path: Path) -> None:
    input_path = tmp_path / "risky_odds.csv"
    pd.DataFrame(
        [
            {
                "match_id": "RISKY",
                "date": "2026-07-01",
                "stage": "round of 32",
                "group": "",
                "team_a": "Favourite",
                "team_b": "Longshot",
                "bookmaker": "OnlyBook",
                "odds_a_win": 1.03,
                "odds_draw": 20.0,
                "odds_b_win": 50.0,
                "odds_timestamp": "2000-01-01T00:00:00Z",
            }
        ]
    ).to_csv(input_path, index=False)
    settings = WorldCupPredictionSettings(
        input_path=input_path,
        csv_output_path=tmp_path / "recommendations.csv",
        xlsx_output_path=tmp_path / "recommendations.xlsx",
        submission_xlsx_output_path=tmp_path / "submission.xlsx",
    )

    report = run_world_cup_predictions(
        settings,
        ProjectConfig(max_goals_score_matrix=0, poor_calibration_loss_threshold=-1.0),
    ).match_report.iloc[0]
    flags = set(report["warning_flags"].split("; "))

    assert {
        "only_one_bookmaker",
        "no_over_under",
        "no_btts",
        "high_calibration_error",
        "high_tail_mass",
        "stale_odds_timestamp",
        "extreme_favourite",
        "knockout_missing_qualification_odds",
    }.issubset(flags)


def test_world_cup_model_risk_summary_reports_compact_counts(tmp_path: Path) -> None:
    report = run_world_cup_predictions(_settings(tmp_path)).match_report.copy()
    report.loc[0, "warning_flags"] = "only_one_bookmaker; no_over_under; no_btts; extreme_favourite"
    report.loc[1, "warning_flags"] = "high_calibration_error; knockout_missing_qualification_odds"

    summary = format_world_cup_model_risk_summary(report)

    assert "Model-Risk Summary" in summary
    assert "- Matches: 2" in summary
    assert "- Matches with only one bookmaker: 1" in summary
    assert "- Matches without over/under odds: 1" in summary
    assert "- Matches without BTTS odds: 1" in summary
    assert "- Extreme favourites: 1" in summary
    assert "- Matches with high calibration error: 1" in summary
    assert "- Knockout matches missing qualification odds: 1" in summary


def test_model_risk_documentation_exists() -> None:
    assert Path("docs/stylised_facts.md").exists()
    assert Path("docs/model_roadmap.md").exists()


def test_default_world_cup_input_prefers_xlsx_when_both_exist(tmp_path: Path, monkeypatch) -> None:
    prepared = tmp_path / "input" / "prepared"
    prepared.mkdir(parents=True)
    (prepared / "world_cup_odds.csv").touch()
    (prepared / "world_cup_odds.xlsx").touch()
    monkeypatch.chdir(tmp_path)

    assert resolve_world_cup_odds_input() == Path("input/prepared/world_cup_odds.xlsx")


def test_world_cup_console_summary_contains_manual_inspection_fields(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    workflow = run_world_cup_predictions(settings)
    summary = format_world_cup_console_summary(
        workflow.match_report,
        settings.input_path,
        settings.csv_output_path,
        settings.xlsx_output_path,
        settings.submission_xlsx_output_path,
    )

    assert "World Cup Predictions" in summary
    assert "WC001 | Alpha vs Beta" in summary
    assert "Stage/group: group stage / Group A" in summary
    assert "Fair 1X2:" in summary
    assert "Calibrated lambdas:" in summary
    assert "Favourite strength: p_fav=" in summary
    assert "Modal scoreline:" in summary
    assert "EV-optimal scoreline:" in summary
    assert "Top 5 EV scorelines:" in summary
    assert "Public-ranking strategy:" in summary
    assert "Estimated most crowded public score:" in summary
    assert "Public strategy score:" in summary
    assert "CSV recommendations:" in summary
    assert "Excel recommendations:" in summary
    assert "Submission sheet:" in summary
    assert "Prediction Output Diagnostics" in summary
    assert "Recommended Score Counts" in summary
    assert "Favourite-Strength Bucket Counts" in summary
    assert "- Average lambda_a:" in summary
    assert "- Average lambda_b:" in summary
    assert "- EV-optimal score differs from modal scoreline:" in summary
    assert "Top 10 Matches By Favourite Probability" in summary
    assert "Model-Risk Summary" in summary
    assert "market_a=" in summary
    assert "market_draw=" in summary
    assert "market_b=" in summary


def test_world_cup_prediction_diagnostics_show_only_ten_strongest_favourites(tmp_path: Path) -> None:
    report = run_world_cup_predictions(_settings(tmp_path)).match_report
    rows = []
    for index in range(11):
        row = report.iloc[0].copy()
        row["match_id"] = f"M{index:02d}"
        row["favourite_probability"] = 0.40 + index / 100
        rows.append(row)

    diagnostics = format_world_cup_prediction_diagnostics(pd.DataFrame(rows))

    assert "M10 | Alpha vs Beta | p_fav=50.00%" in diagnostics
    assert "M01 | Alpha vs Beta | p_fav=41.00%" in diagnostics
    assert "M00 | Alpha vs Beta" not in diagnostics
    assert diagnostics.index("M10 | Alpha vs Beta") < diagnostics.index("M09 | Alpha vs Beta")


def test_world_cup_prediction_script_runs_with_overrides(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_world_cup_predictions.py",
            "--odds",
            str(settings.input_path),
            "--csv-output",
            str(settings.csv_output_path),
            "--xlsx-output",
            str(settings.xlsx_output_path),
            "--submission-xlsx-output",
            str(settings.submission_xlsx_output_path),
        ],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    assert settings.csv_output_path.exists()
    assert settings.xlsx_output_path.exists()
    assert settings.submission_xlsx_output_path.exists()
    assert "WC002 | Gamma vs Delta" in capsys.readouterr().out


def test_world_cup_prediction_script_prints_friendly_missing_file_message(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_world_cup_predictions.py"])

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    assert capsys.readouterr().out.strip() == WORLD_CUP_ODDS_MISSING_MESSAGE


def test_create_world_cup_odds_file_script_runs_with_overrides(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    destination = tmp_path / "world_cup_odds.xlsx"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_world_cup_odds_file.py",
            "--output",
            str(destination),
            "--template",
            str(TEMPLATES.resolve() / "world_cup_odds_template.xlsx"),
        ],
    )

    runpy.run_path(str(CREATE_SCRIPT), run_name="__main__")

    assert destination.exists()
    assert "Fill in" in capsys.readouterr().out
