from pathlib import Path
import runpy
import sys

import pandas as pd

from wc_predictor.reporting import format_world_cup_console_summary
from wc_predictor.world_cup import WorldCupPredictionSettings, run_world_cup_predictions


EXAMPLES = Path("data/examples")
TEMPLATES = Path("data/templates")


def _settings(tmp_path: Path) -> WorldCupPredictionSettings:
    return WorldCupPredictionSettings(
        input_path=EXAMPLES / "example_world_cup_odds.csv",
        csv_output_path=tmp_path / "world_cup_recommendations.csv",
        xlsx_output_path=tmp_path / "world_cup_recommendations.xlsx",
    )


def test_world_cup_templates_have_expected_columns() -> None:
    odds = pd.read_csv(TEMPLATES / "world_cup_odds_template.csv")
    predictions = pd.read_csv(TEMPLATES / "friend_predictions_template.csv")

    assert odds.columns.tolist() == [
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
    ]
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
    }.issubset(workflow.match_report.columns)
    assert len(pd.read_csv(settings.csv_output_path)) == 2
    assert len(pd.read_excel(settings.xlsx_output_path)) == 2


def test_world_cup_console_summary_contains_manual_inspection_fields(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    workflow = run_world_cup_predictions(settings)
    summary = format_world_cup_console_summary(
        workflow.match_report,
        settings.input_path,
        settings.csv_output_path,
        settings.xlsx_output_path,
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
    assert "CSV recommendations:" in summary
    assert "Excel recommendations:" in summary


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
        ],
    )

    runpy.run_path("scripts/run_world_cup_predictions.py", run_name="__main__")

    assert settings.csv_output_path.exists()
    assert settings.xlsx_output_path.exists()
    assert "WC002 | Gamma vs Delta" in capsys.readouterr().out
