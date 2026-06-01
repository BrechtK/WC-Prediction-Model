from pathlib import Path
import runpy
import sys

import pandas as pd
from openpyxl import load_workbook

from wc_predictor.reporting import format_world_cup_console_summary, format_world_cup_prediction_diagnostics
from wc_predictor.world_cup import (
    WORLD_CUP_ODDS_MISSING_MESSAGE,
    WorldCupPredictionSettings,
    create_world_cup_odds_file,
    resolve_world_cup_odds_input,
    run_world_cup_predictions,
)


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
    "notes",
]


def _settings(tmp_path: Path) -> WorldCupPredictionSettings:
    return WorldCupPredictionSettings(
        input_path=EXAMPLES / "example_world_cup_odds.csv",
        csv_output_path=tmp_path / "world_cup_recommendations.csv",
        xlsx_output_path=tmp_path / "world_cup_recommendations.xlsx",
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
        "market_a",
        "market_draw",
        "market_b",
        "favourite_probability",
        "favourite_bucket",
        "lambda_a",
        "lambda_b",
        "recommended_score",
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
    assert worksheet.cell(2, column["top_5_ev_predictions"]).alignment.wrap_text
    assert worksheet.cell(2, column["warnings"]).alignment.wrap_text
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
    )

    workflow = run_world_cup_predictions(settings)

    assert len(workflow.match_report) == 2
    assert workflow.match_report.loc[0, "match_id"] == "WC001"


def test_default_world_cup_input_prefers_xlsx_when_both_exist(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "world_cup_odds.csv").touch()
    (raw / "world_cup_odds.xlsx").touch()
    monkeypatch.chdir(tmp_path)

    assert resolve_world_cup_odds_input() == Path("data/raw/world_cup_odds.xlsx")


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
    assert "Prediction Output Diagnostics" in summary
    assert "Recommended Score Counts" in summary
    assert "Favourite-Strength Bucket Counts" in summary
    assert "- Average lambda_a:" in summary
    assert "- Average lambda_b:" in summary
    assert "- EV-optimal score differs from modal scoreline:" in summary
    assert "Top 10 Matches By Favourite Probability" in summary
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
        ],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    assert settings.csv_output_path.exists()
    assert settings.xlsx_output_path.exists()
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
