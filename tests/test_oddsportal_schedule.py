from __future__ import annotations

from pathlib import Path
import runpy
import sys

import pandas as pd

from wc_predictor.oddsportal_schedule import (
    REPORT_COLUMNS,
    SCHEDULE_COLUMNS,
    format_oddsportal_schedule_parse_summary,
    parse_oddsportal_schedule_file,
    parse_oddsportal_schedule_text,
    prepare_schedule_metadata,
)

FIXTURES = Path("tests/fixtures/oddsportal_schedule_pastes")
SCHEDULE_FIXTURE = FIXTURES / "schedule_three_fixtures.txt"
RUN_SCRIPT = Path("scripts/parse_oddsportal_schedule.py").resolve()


def test_schedule_parser_assigns_sequential_ids_and_preserves_fixture_metadata() -> None:
    result = parse_oddsportal_schedule_text(
        SCHEDULE_FIXTURE.read_text(encoding="utf-8"),
        existing_metadata_path=None,
    )
    schedule = result.schedule

    assert schedule.columns.tolist() == SCHEDULE_COLUMNS
    assert schedule["match_id"].tolist() == ["M001", "M002", "M003"]
    assert schedule["date"].tolist() == ["2026-06-11", "2026-06-12", "2026-06-13"]
    assert schedule["time"].tolist() == ["21:00", "04:00", "01:30"]
    assert schedule.loc[0, ["team_a", "team_b"]].tolist() == ["Mexico", "South Africa"]
    assert schedule.loc[1, ["team_a", "team_b"]].tolist() == ["South Korea", "Czech Republic"]
    assert schedule.loc[2, ["team_a", "team_b"]].tolist() == ["Bosnia & Herzegovina", "D.R. Congo"]
    assert schedule.loc[0, "schedule_odds_a_win"] == 1.46
    assert schedule.loc[0, "schedule_odds_draw"] == 4.55
    assert schedule.loc[0, "schedule_odds_b_win"] == 8.70
    assert schedule["source_quality"].eq("oddsportal_schedule_paste").all()


def test_schedule_parser_stores_missing_odds_and_reports_diagnostics() -> None:
    result = parse_oddsportal_schedule_text(
        SCHEDULE_FIXTURE.read_text(encoding="utf-8"),
        existing_metadata_path=None,
    )
    schedule = result.schedule
    report = result.report.iloc[0]

    assert result.report.columns.tolist() == REPORT_COLUMNS
    assert schedule.loc[2, ["schedule_odds_a_win", "schedule_odds_draw", "schedule_odds_b_win"]].isna().all()
    assert schedule.loc[2, "notes"] == "missing_schedule_1x2_odds"
    assert report["fixtures_found"] == 3
    assert report["dates_found"] == 3
    assert report["fixtures_with_missing_odds"] == 1
    assert report["duplicate_fixtures"] == 0
    assert report["warnings"] == "fixtures_with_missing_odds:M003"


def test_schedule_parser_accepts_star_placeholders_for_missing_odds() -> None:
    result = parse_oddsportal_schedule_text(
        "\n".join(
            [
                "11 Jun 2026",
                "21:00",
                "Cape Verde",
                "Cape Verde",
                "-",
                "Ivory Coast",
                "Ivory Coast",
                "*",
                "*",
                "*",
            ]
        ),
        existing_metadata_path=None,
    )

    assert result.schedule.loc[0, ["schedule_odds_a_win", "schedule_odds_draw", "schedule_odds_b_win"]].isna().all()
    assert result.schedule.loc[0, "notes"] == "missing_schedule_1x2_odds"


def test_schedule_parser_joins_existing_group_metadata_by_match_id(tmp_path: Path) -> None:
    metadata = tmp_path / "world_cup_odds.xlsx"
    pd.DataFrame(
        [
            {"match_id": "M001", "stage": "group stage", "group": "Group A"},
            {"match_id": "M002", "stage": "group stage", "group": "Group B"},
        ]
    ).to_excel(metadata, index=False)

    result = parse_oddsportal_schedule_text(
        SCHEDULE_FIXTURE.read_text(encoding="utf-8"),
        existing_metadata_path=metadata,
    )

    assert result.schedule.loc[0, "stage"] == "group stage"
    assert result.schedule.loc[0, "group"] == "Group A"
    assert result.schedule.loc[1, "group"] == "Group B"
    assert pd.isna(result.schedule.loc[2, "group"])


def test_schedule_file_parser_exports_csv_report_and_summary(tmp_path: Path) -> None:
    output = tmp_path / "schedule.csv"
    report = tmp_path / "report.csv"

    result = parse_oddsportal_schedule_file(
        SCHEDULE_FIXTURE,
        output,
        report,
        existing_metadata_path=None,
    )
    summary = format_oddsportal_schedule_parse_summary(result, output, report)

    assert output.exists()
    assert report.exists()
    assert "Schedule Paste Parse" in summary
    assert "- Fixtures parsed: 3" in summary
    assert "- First fixture: M001 | Mexico vs South Africa" in summary
    assert "- Last fixture: M003 | Bosnia & Herzegovina vs D.R. Congo" in summary
    assert "- Missing odds fixtures: 1" in summary


def test_prepare_schedule_metadata_prefers_schedule_csv_then_falls_back(tmp_path: Path) -> None:
    schedule_input = tmp_path / "schedule.txt"
    schedule_output = tmp_path / "schedule.csv"
    report = tmp_path / "report.csv"
    fallback = tmp_path / "world_cup_odds.xlsx"
    schedule_input.write_text(SCHEDULE_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    pd.DataFrame([{"match_id": "OLD"}]).to_excel(fallback, index=False)

    parsed = prepare_schedule_metadata(
        schedule_input,
        schedule_output,
        report,
        existing_metadata_path=fallback,
    )
    loaded = prepare_schedule_metadata(
        tmp_path / "missing.txt",
        schedule_output,
        report,
        existing_metadata_path=fallback,
    )
    fallback_only = prepare_schedule_metadata(
        tmp_path / "missing.txt",
        tmp_path / "missing.csv",
        report,
        existing_metadata_path=fallback,
    )

    assert parsed.metadata_path == schedule_output
    assert parsed.parse_result is not None
    assert loaded.metadata_path == schedule_output
    assert loaded.parse_result is None
    assert fallback_only.metadata_path == fallback


def test_schedule_parser_script_runs_with_overrides(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "schedule.csv"
    report = tmp_path / "report.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_oddsportal_schedule.py",
            "--input",
            str(SCHEDULE_FIXTURE.resolve()),
            "--output",
            str(output),
            "--report-output",
            str(report),
            "--existing-metadata",
            str(tmp_path / "missing.xlsx"),
        ],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    assert output.exists()
    assert report.exists()
    assert "Schedule Paste Parse" in capsys.readouterr().out
