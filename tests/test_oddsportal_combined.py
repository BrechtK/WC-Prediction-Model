from __future__ import annotations

from pathlib import Path
import runpy
import sys

import pandas as pd
import pytest

from wc_predictor.oddsportal_combined import (
    CombinedOddsPortalPasteError,
    format_combined_oddsportal_split_summary,
    infer_match_id_from_combined_filename,
    split_combined_oddsportal_pastes,
)

RUN_SCRIPT = Path("scripts/split_oddsportal_combined_pastes.py").resolve()


def _write_combined(path: Path, sections: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n\n".join(f"### {header}\n{content}" for header, content in sections),
        encoding="utf-8",
    )


def test_complete_combined_paste_is_split_into_four_existing_parser_inputs(tmp_path: Path) -> None:
    input_folder = tmp_path / "combined"
    output_folder = tmp_path / "pastes"
    _write_combined(
        input_folder / "M001_all_odds.txt",
        [
            ("1X2", "one x two"),
            ("OVER_UNDER", "totals"),
            ("BTTS", "both teams"),
            ("CORRECT_SCORE", "correct scores"),
        ],
    )

    result = split_combined_oddsportal_pastes(input_folder, output_folder)

    assert infer_match_id_from_combined_filename("M001_all_odds.txt") == "M001"
    assert result.files_processed == 1
    assert result.sections_extracted == 4
    assert result.files_written == 4
    assert (output_folder / "M001_1x2.txt").read_text(encoding="utf-8") == "one x two\n"
    assert (output_folder / "M001_over_under.txt").read_text(encoding="utf-8") == "totals\n"
    assert (output_folder / "M001_btts.txt").read_text(encoding="utf-8") == "both teams\n"
    assert (output_folder / "M001_correct_score.txt").read_text(encoding="utf-8") == "correct scores\n"


def test_missing_optional_section_warns_but_succeeds(tmp_path: Path) -> None:
    input_folder = tmp_path / "combined"
    output_folder = tmp_path / "pastes"
    _write_combined(input_folder / "M001_all_odds.txt", [("1X2", "one x two")])

    result = split_combined_oddsportal_pastes(input_folder, output_folder)

    assert result.files_written == 1
    assert "M001: missing_optional_section:over_under" in result.warnings
    assert "M001: missing_optional_section:btts" in result.warnings
    assert "M001: missing_optional_section:correct_score" in result.warnings


def test_missing_required_one_x_two_section_fails_clearly(tmp_path: Path) -> None:
    input_folder = tmp_path / "combined"
    _write_combined(input_folder / "M001_all_odds.txt", [("BTTS", "both teams")])

    with pytest.raises(CombinedOddsPortalPasteError, match=r"missing required ### 1X2 section"):
        split_combined_oddsportal_pastes(input_folder, tmp_path / "pastes")


def test_common_section_aliases_are_supported(tmp_path: Path) -> None:
    input_folder = tmp_path / "combined"
    output_folder = tmp_path / "pastes"
    _write_combined(
        input_folder / "M001_all_odds.txt",
        [
            ("FULL_TIME_RESULT", "one x two"),
            ("O/U", "totals"),
            ("BOTH TEAMS TO SCORE", "both teams"),
            ("CORRECT SCORE", "correct scores"),
        ],
    )

    result = split_combined_oddsportal_pastes(input_folder, output_folder)

    assert result.sections_extracted == 4
    assert not result.warnings
    assert (output_folder / "M001_1x2.txt").exists()
    assert (output_folder / "M001_over_under.txt").exists()
    assert (output_folder / "M001_btts.txt").exists()
    assert (output_folder / "M001_correct_score.txt").exists()


@pytest.mark.parametrize(
    "header,expected_suffix",
    [
        ("MATCH_ODDS", "1x2"),
        ("FULL_TIME_RESULT", "1x2"),
        ("OVER UNDER", "over_under"),
        ("O/U", "over_under"),
        ("BOTH_TEAMS_TO_SCORE", "btts"),
        ("BOTH TEAMS TO SCORE", "btts"),
        ("CORRECT SCORE", "correct_score"),
    ],
)
def test_each_common_alias_maps_to_expected_section(
    tmp_path: Path,
    header: str,
    expected_suffix: str,
) -> None:
    input_folder = tmp_path / "combined"
    output_folder = tmp_path / "pastes"
    sections = [("1X2", "one x two")]
    if expected_suffix != "1x2":
        sections.append((header, "alias content"))
    else:
        sections = [(header, "alias content")]
    _write_combined(input_folder / "M001_all_odds.txt", sections)

    split_combined_oddsportal_pastes(input_folder, output_folder)

    assert (output_folder / f"M001_{expected_suffix}.txt").exists()


def test_existing_split_file_is_not_overwritten_by_default(tmp_path: Path) -> None:
    input_folder = tmp_path / "combined"
    output_folder = tmp_path / "pastes"
    output_folder.mkdir()
    destination = output_folder / "M001_1x2.txt"
    destination.write_text("manual\n", encoding="utf-8")
    _write_combined(input_folder / "M001_all_odds.txt", [("1X2", "replacement")])

    result = split_combined_oddsportal_pastes(input_folder, output_folder)

    assert destination.read_text(encoding="utf-8") == "manual\n"
    assert "M001: existing_split_file_skipped:M001_1x2.txt" in result.warnings


def test_overwrite_replaces_existing_split_file(tmp_path: Path) -> None:
    input_folder = tmp_path / "combined"
    output_folder = tmp_path / "pastes"
    output_folder.mkdir()
    destination = output_folder / "M001_1x2.txt"
    destination.write_text("manual\n", encoding="utf-8")
    _write_combined(input_folder / "M001_all_odds.txt", [("MATCH_ODDS", "replacement")])

    result = split_combined_oddsportal_pastes(input_folder, output_folder, overwrite=True)

    assert result.files_written == 1
    assert destination.read_text(encoding="utf-8") == "replacement\n"


def test_unknown_and_empty_sections_warn(tmp_path: Path) -> None:
    input_folder = tmp_path / "combined"
    _write_combined(
        input_folder / "M001_all_odds.txt",
        [("1X2", ""), ("MYSTERY", "ignored")],
    )

    result = split_combined_oddsportal_pastes(input_folder, tmp_path / "pastes")

    assert "M001: empty_section:1x2" in result.warnings
    assert "M001: unknown_section_marker:MYSTERY" in result.warnings


def test_splitter_script_runs_with_overrides(tmp_path: Path, monkeypatch, capsys) -> None:
    input_folder = tmp_path / "combined"
    output_folder = tmp_path / "pastes"
    _write_combined(input_folder / "M001_all_odds.txt", [("1X2", "one x two")])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_oddsportal_combined_pastes.py",
            "--input-folder",
            str(input_folder),
            "--output-folder",
            str(output_folder),
        ],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    assert "Combined OddsPortal Paste Split" in output
    assert "- Files processed: 1" in output
    assert "- Sections extracted: 1" in output
    assert "- Files written: 1" in output
    assert (output_folder / "M001_1x2.txt").exists()


def test_split_summary_reports_warnings() -> None:
    result = split_combined_oddsportal_pastes("does-not-exist")

    assert format_combined_oddsportal_split_summary(result).endswith("  - none")


def test_schedule_validation_rejects_unknown_combined_match_id(tmp_path: Path) -> None:
    input_folder = tmp_path / "combined"
    _write_combined(input_folder / "M999_all_odds.txt", [("1X2", "one x two")])
    schedule = pd.DataFrame([{"match_id": "M001", "team_a": "Mexico", "team_b": "South Africa"}])

    with pytest.raises(
        CombinedOddsPortalPasteError,
        match=r"M999_all_odds\.txt was found, but M999 is not present in the parsed schedule\.",
    ):
        split_combined_oddsportal_pastes(input_folder, tmp_path / "pastes", schedule=schedule)


def test_schedule_validation_warns_on_detectable_team_mismatch(tmp_path: Path) -> None:
    input_folder = tmp_path / "combined"
    _write_combined(
        input_folder / "M001_all_odds.txt",
        [("1X2", "South Korea\nSouth Korea\n-\nCzech Republic\nCzech Republic")],
    )
    schedule = pd.DataFrame(
        [
            {"match_id": "M001", "team_a": "Mexico", "team_b": "South Africa"},
            {"match_id": "M002", "team_a": "South Korea", "team_b": "Czech Republic"},
        ]
    )

    result = split_combined_oddsportal_pastes(input_folder, tmp_path / "pastes", schedule=schedule)

    assert any("combined_paste_team_mismatch:M001_all_odds.txt" in warning for warning in result.warnings)
