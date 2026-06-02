from pathlib import Path
import runpy
import sys

import pandas as pd

from wc_predictor.oddsportal import (
    OUTPUT_COLUMNS,
    format_oddsportal_parse_summary,
    infer_match_id_from_filename,
    parse_oddsportal_correct_score_folder,
    parse_oddsportal_correct_score_text,
)


FIXTURES = Path("tests/fixtures/oddsportal_pastes")
RUN_SCRIPT = Path("scripts/parse_oddsportal_correct_scores.py").resolve()


def test_match_id_is_inferred_from_oddsportal_paste_filename() -> None:
    assert infer_match_id_from_filename("M001_correct_score.txt") == "M001"


def test_oddsportal_paste_parser_extracts_bookmaker_rows_and_ignores_noise(tmp_path: Path) -> None:
    output = tmp_path / "world_cup_correct_score_odds.csv"
    report_output = tmp_path / "oddsportal_correct_score_parse_report.csv"

    result = parse_oddsportal_correct_score_folder(FIXTURES, output, report_output)
    odds = result.odds
    report = result.report.set_index("scoreline")

    assert output.exists()
    assert report_output.exists()
    assert odds.columns.tolist() == OUTPUT_COLUMNS
    assert set(odds["match_id"]) == {"MTEST"}
    assert set(zip(odds["score_a"], odds["score_b"])) == {(1, 0), (2, 1)}
    assert set(odds["bookmaker"]) == {"888sport", "bet365", "BetInAsia", "Betsson", "GGBET", "N1 Bet"}
    assert len(odds) == 8
    assert not odds["bookmaker"].str.contains("best value|Bookmakers|claim bonus", case=False).any()
    assert not (odds["decimal_odds"] == 6.0).any()
    assert not (odds["decimal_odds"] == 5.99).any()
    assert (odds["decimal_odds"] == 5.50).sum() == 2
    assert report.loc["1-0", "bookmakers_found"] == 6
    assert report.loc["1-0", "odds_rows_found"] == 6
    assert "missing_odds:N1 Bet" in report.loc["2-1", "warnings"]
    assert "suspicious_odds:GGBET=501" in report.loc["2-1", "warnings"]
    assert "match_fewer_than_10_scorelines:2" in report.loc["2-1", "warnings"]
    assert len(pd.read_csv(report_output)) == 2


def test_oddsportal_parse_summary_is_compact(tmp_path: Path) -> None:
    result = parse_oddsportal_correct_score_folder(
        FIXTURES,
        tmp_path / "correct_scores.csv",
        tmp_path / "parse_report.csv",
    )

    summary = format_oddsportal_parse_summary(result)

    assert "- Files parsed: 1" in summary
    assert "- Total scorelines found: 2" in summary
    assert "- Total bookmaker odds rows extracted: 8" in summary
    assert "- Scorelines with missing odds: 1" in summary
    assert "- Scorelines with fewer than 3 bookmakers: 1" in summary
    assert "- Matches with sparse correct-score coverage: 1" in summary


def test_oddsportal_parser_script_runs_with_overrides(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "correct_scores.csv"
    report_output = tmp_path / "parse_report.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_oddsportal_correct_scores.py",
            "--input-folder",
            str(FIXTURES.resolve()),
            "--output",
            str(output),
            "--report-output",
            str(report_output),
        ],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    assert output.exists()
    assert report_output.exists()
    assert "OddsPortal Correct-Score Paste Parse" in capsys.readouterr().out


def test_other_bucket_is_detected_without_attaching_its_odds_to_numeric_scoreline() -> None:
    result = parse_oddsportal_correct_score_text(
        "1:0\nBook A\n5.50\nOther\nBook A\n2.00\n",
        "MOTHER",
    )

    assert len(result.odds) == 1
    assert bool(result.odds.loc[0, "has_other_bucket"])
    assert result.odds.loc[0, "decimal_odds"] == 5.50


def test_only_exact_odds_rows_are_deduplicated() -> None:
    result = parse_oddsportal_correct_score_text(
        "1:0\nBook A\n5.50\nBook A\n5.50\nBook A\n5.60\nBook B\n5.50\n",
        "MDEDUP",
    )

    assert list(zip(result.odds["bookmaker"], result.odds["decimal_odds"])) == [
        ("Book A", 5.50),
        ("Book A", 5.60),
        ("Book B", 5.50),
    ]
    assert "deduplicated_identical_row:Book A" in result.report.loc[0, "warnings"]
    assert "conflicting_duplicate_odds:Book A=5.5/5.6" in result.report.loc[0, "warnings"]


def test_empty_paste_reports_no_scorelines_without_inflating_summary_count(tmp_path: Path) -> None:
    input_folder = tmp_path / "pastes"
    input_folder.mkdir()
    (input_folder / "MEMPTY_correct_score.txt").write_text("", encoding="utf-8")
    result = parse_oddsportal_correct_score_folder(
        input_folder,
        tmp_path / "correct_scores.csv",
        tmp_path / "parse_report.csv",
    )

    summary = format_oddsportal_parse_summary(result)

    assert "- Total scorelines found: 0" in summary
    assert "- Scorelines with fewer than 3 bookmakers: 0" in summary
    assert "no_scorelines_found" in result.report.loc[0, "warnings"]
