from __future__ import annotations

from pathlib import Path
import runpy
import sys

import pandas as pd

from wc_predictor.oddsportal_core import (
    OUTPUT_COLUMNS,
    TOTAL_GOALS_OUTPUT_COLUMNS,
    format_oddsportal_core_parse_summary,
    infer_match_and_market_from_filename,
    parse_oddsportal_core_market_text,
    parse_oddsportal_core_odds_folder,
)


FIXTURES = Path("tests/fixtures/oddsportal_core_pastes")
RUN_SCRIPT = Path("scripts/parse_oddsportal_core_odds.py").resolve()


def test_match_id_and_market_are_inferred_from_oddsportal_core_filename() -> None:
    assert infer_match_and_market_from_filename("M001_1x2.txt") == ("M001", "1x2")
    assert infer_match_and_market_from_filename("M001_btts.txt") == ("M001", "btts")
    assert infer_match_and_market_from_filename("M001_over_under.txt") == ("M001", "over_under")


def test_core_folder_parser_extracts_merges_and_joins_metadata(tmp_path: Path) -> None:
    output = tmp_path / "world_cup_odds_from_pastes.csv"
    report_output = tmp_path / "parse_report.csv"
    total_goals_output = tmp_path / "total_goals.csv"
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "match_id": "MCORE",
                "date": "2026-06-12",
                "stage": "group",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
                "bookmaker": "bet365",
                "odds_a_qualifies": 1.25,
                "odds_b_qualifies": 3.75,
            }
        ]
    ).to_csv(metadata, index=False)

    result = parse_oddsportal_core_odds_folder(
        FIXTURES,
        output,
        report_output,
        total_goals_output_path=total_goals_output,
        metadata_odds_path=metadata,
        odds_timestamp="2026-06-02T10:00:00Z",
        odds_source_url="https://example.test/match",
    )
    odds = result.odds.set_index(["match_id", "bookmaker"])

    assert output.exists()
    assert report_output.exists()
    assert total_goals_output.exists()
    assert result.odds.columns.tolist() == OUTPUT_COLUMNS
    assert set(odds.loc["MCORE"].index) == {"888sport", "bet365"}
    assert odds.loc[("MCORE", "888sport"), "odds_a_win"] == 1.40
    assert odds.loc[("MCORE", "bet365"), "odds_draw"] == 4.33
    assert odds.loc[("MCORE", "888sport"), "odds_btts_yes"] == 2.15
    assert odds.loc[("MCORE", "bet365"), "odds_under_2_5"] == 1.73
    assert odds.loc[("MCORE", "bet365"), "team_a"] == "Alpha"
    assert odds.loc[("MCORE", "bet365"), "odds_a_qualifies"] == 1.25
    assert odds.loc[("MCORE", "888sport"), "source_quality"] == "oddsportal_paste"
    assert odds.loc[("MCORE", "888sport"), "notes"] == "parsed_from_oddsportal_paste"
    assert odds.loc[("MCORE", "888sport"), "odds_timestamp"] == "2026-06-02T10:00:00Z"
    assert len(pd.read_csv(output)) == 3
    assert pd.read_csv(total_goals_output).columns.tolist() == TOTAL_GOALS_OUTPUT_COLUMNS


def test_over_under_parser_preserves_wide_two_point_five_and_extracts_total_goals_ladder() -> None:
    result = parse_oddsportal_core_market_text(
        (FIXTURES / "MCORE_over_under.txt").read_text(encoding="utf-8"),
        "MCORE",
        "over_under",
    )
    odds = result.odds.set_index("bookmaker")

    assert set(odds.index) == {"888sport", "bet365"}
    assert odds.loc["888sport", "odds_over_2_5"] == 2.00
    assert odds.loc["888sport", "odds_under_2_5"] == 1.75
    ladder = result.total_goals_odds
    assert set(ladder["line"]) == {0.5, 1.5, 2.0, 2.25, 2.5, 3.0, 3.5}
    assert len(ladder) == 14
    assert set(ladder["bookmaker"]) == {"888sport", "bet365"}
    assert set(ladder.loc[ladder["line"] == 3.5, "bookmaker"]) == {"888sport", "bet365"}
    assert result.report.loc[0, "total_goals_lines_found"] == 7
    assert result.report.loc[0, "bookmakers_per_total_goals_line"] == "0.5:2; 1.5:2; 2:2; 2.25:2; 2.5:2; 3:2; 3.5:2"
    assert bool(result.report.loc[0, "has_over_under_2_5"])
    assert "betting_exchange_section_ignored" in result.report.loc[0, "warnings"]


def test_missing_market_and_sparse_bookmaker_warnings_are_reported(tmp_path: Path) -> None:
    result = parse_oddsportal_core_odds_folder(
        FIXTURES,
        tmp_path / "odds.csv",
        tmp_path / "report.csv",
        total_goals_output_path=tmp_path / "total_goals.csv",
        metadata_odds_path=None,
    )
    report = result.report.set_index("source_file")
    warnings = report.loc["MMISSING_1x2.txt", "warnings"]

    assert "fewer_than_2_bookmakers:1" in warnings
    assert "missing_over_under_2_5_market" in warnings
    assert "missing_btts_market" in warnings


def test_suspicious_prices_and_missing_two_point_five_block_are_reported() -> None:
    suspicious = parse_oddsportal_core_market_text(
        "Bookmakers\n1\nX\n2\nPayout\nBook A\nBook A\n0.99\n3.00\n600.00\n",
        "MRISK",
        "1x2",
    )
    missing_total = parse_oddsportal_core_market_text(
        "Over/Under +3.5\nBookmakers\nTotal\nOver\nUnder\nBook A\nBook A\n+3.5\n3.00\n1.40\n",
        "MRISK",
        "over_under",
    )

    assert "suspicious_odds_le_1:Book A=0.99" in suspicious.report.loc[0, "warnings"]
    assert "suspicious_odds_very_high:Book A=600" in suspicious.report.loc[0, "warnings"]
    assert "no_2_5_over_under_block_found" in missing_total.report.loc[0, "warnings"]
    assert missing_total.odds.empty
    assert set(missing_total.total_goals_odds["line"]) == {3.5}


def test_over_under_parser_accepts_leading_decimal_total_and_warns_on_sparse_coverage() -> None:
    result = parse_oddsportal_core_market_text(
        "Over/Under +.5\nBookmakers\nTotal\nOver\nUnder\nBook A\nBook A\n+.5\n1.05\n9.00\n",
        "MLEADING",
        "over_under",
    )

    assert result.total_goals_odds.loc[0, "line"] == 0.5
    assert "total_goals_line_fewer_than_2_bookmakers:0.5=1" in result.report.loc[0, "warnings"]


def test_folder_reports_missing_over_under_when_file_has_no_two_point_five_rows(tmp_path: Path) -> None:
    input_folder = tmp_path / "pastes"
    input_folder.mkdir()
    (input_folder / "MNO25_1x2.txt").write_text(
        "Bookmakers\nBook A\nBook A\n1.80\n3.50\n4.50\n",
        encoding="utf-8",
    )
    (input_folder / "MNO25_over_under.txt").write_text(
        "Over/Under +3.5\nBookmakers\nTotal\nOver\nUnder\nBook A\nBook A\n+3.5\n3.00\n1.40\n",
        encoding="utf-8",
    )

    result = parse_oddsportal_core_odds_folder(
        input_folder,
        tmp_path / "odds.csv",
        tmp_path / "report.csv",
        total_goals_output_path=tmp_path / "total_goals.csv",
        metadata_odds_path=None,
    )
    over_under_warnings = result.report.set_index("market_type").loc["over_under", "warnings"]

    assert "no_2_5_over_under_block_found" in over_under_warnings
    assert "missing_over_under_2_5_market" in over_under_warnings


def test_core_parse_summary_is_compact(tmp_path: Path) -> None:
    result = parse_oddsportal_core_odds_folder(
        FIXTURES,
        tmp_path / "odds.csv",
        tmp_path / "report.csv",
        total_goals_output_path=tmp_path / "total_goals.csv",
        metadata_odds_path=None,
    )

    summary = format_oddsportal_core_parse_summary(result)

    assert "- Files parsed: 4" in summary
    assert "- Matches parsed: 2" in summary
    assert "- 1X2 rows extracted: 3" in summary
    assert "- BTTS rows extracted: 2" in summary
    assert "- O/U 2.5 rows extracted: 2" in summary
    assert "- Total-goals ladder rows extracted: 14" in summary
    assert "- MCORE: none" in summary
    assert "- MMISSING: btts, over_under" in summary


def test_core_parser_script_runs_with_overrides(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "odds.csv"
    report_output = tmp_path / "report.csv"
    total_goals_output = tmp_path / "total_goals.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_oddsportal_core_odds.py",
            "--input-folder",
            str(FIXTURES.resolve()),
            "--output",
            str(output),
            "--report-output",
            str(report_output),
            "--total-goals-output",
            str(total_goals_output),
            "--metadata-odds",
            str(tmp_path / "missing_metadata.xlsx"),
        ],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    assert output.exists()
    assert report_output.exists()
    assert total_goals_output.exists()
    assert "OddsPortal Core Odds Paste Parse" in capsys.readouterr().out
