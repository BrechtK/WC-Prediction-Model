"""Split combined per-match OddsPortal pastes into the existing parser inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

DEFAULT_COMBINED_INPUT_FOLDER = Path("data/raw/oddsportal_combined_pastes")
DEFAULT_SPLIT_OUTPUT_FOLDER = Path("data/raw/oddsportal_pastes")

SECTION_FILENAMES = {
    "1x2": "1x2",
    "over_under": "over_under",
    "btts": "btts",
    "correct_score": "correct_score",
}
OPTIONAL_SECTIONS = ("over_under", "btts", "correct_score")
_FILENAME_PATTERN = re.compile(r"^(?P<match_id>.+)_all_odds\.txt$", re.IGNORECASE)
_SECTION_PATTERN = re.compile(r"^\s*###\s*(?P<header>.*?)\s*$")
_SECTION_ALIASES = {
    "1X2": "1x2",
    "MATCH ODDS": "1x2",
    "FULL TIME RESULT": "1x2",
    "OVER UNDER": "over_under",
    "O/U": "over_under",
    "BTTS": "btts",
    "BOTH TEAMS TO SCORE": "btts",
    "CORRECT SCORE": "correct_score",
}


class CombinedOddsPortalPasteError(ValueError):
    """Raised when one combined paste cannot be split safely."""


@dataclass(frozen=True)
class CombinedOddsPortalFileSplit:
    """Split outputs and warnings for one combined paste file."""

    source_file: Path
    match_id: str
    sections_extracted: tuple[str, ...]
    files_written: tuple[Path, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CombinedOddsPortalSplitResult:
    """Summary of all combined paste files considered in one run."""

    files: tuple[CombinedOddsPortalFileSplit, ...]
    validation_warnings: tuple[str, ...] = ()

    @property
    def files_processed(self) -> int:
        return len(self.files)

    @property
    def sections_extracted(self) -> int:
        return sum(len(file.sections_extracted) for file in self.files)

    @property
    def files_written(self) -> int:
        return sum(len(file.files_written) for file in self.files)

    @property
    def warnings(self) -> tuple[str, ...]:
        file_warnings = tuple(
            f"{file.match_id}: {warning}"
            for file in self.files
            for warning in file.warnings
        )
        return (*self.validation_warnings, *file_warnings)


def infer_match_id_from_combined_filename(filename: str) -> str:
    """Return the match ID prefix from one *_all_odds.txt filename."""

    match = _FILENAME_PATTERN.fullmatch(Path(filename).name)
    if not match:
        raise CombinedOddsPortalPasteError(
            f"Combined OddsPortal paste filename must end with _all_odds.txt: {filename}"
        )
    return match.group("match_id")


def _normalise_section_header(header: str) -> str:
    return " ".join(header.strip().replace("_", " ").split()).upper()


def _canonical_section(header: str) -> str | None:
    return _SECTION_ALIASES.get(_normalise_section_header(header))


def _append_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _normalise_team_name(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


def validate_combined_pastes_against_schedule(
    input_folder: str | Path,
    schedule: pd.DataFrame,
) -> tuple[str, ...]:
    """Require known match IDs and warn on detectable team-name contradictions."""

    required = {"match_id", "team_a", "team_b"}
    missing = required - set(schedule.columns)
    if missing:
        raise CombinedOddsPortalPasteError(f"Parsed schedule is missing required columns: {sorted(missing)}")
    schedule = schedule.copy()
    schedule["match_id"] = schedule["match_id"].astype(str)
    if schedule["match_id"].duplicated().any():
        raise CombinedOddsPortalPasteError("Parsed schedule contains duplicate match IDs")
    fixtures = schedule.set_index("match_id")[["team_a", "team_b"]]
    known_teams = {
        str(team): _normalise_team_name(team)
        for team in pd.concat([schedule["team_a"], schedule["team_b"]]).dropna().unique()
    }
    warnings: list[str] = []
    for path in sorted(Path(input_folder).glob("*_all_odds.txt")):
        match_id = infer_match_id_from_combined_filename(path.name)
        if match_id not in fixtures.index:
            raise CombinedOddsPortalPasteError(
                f"{path.name} was found, but {match_id} is not present in the parsed schedule."
            )
        text = _normalise_team_name(path.read_text(encoding="utf-8-sig"))
        detected = sorted(team for team, normalised in known_teams.items() if normalised and normalised in text)
        expected = {str(fixtures.loc[match_id, "team_a"]), str(fixtures.loc[match_id, "team_b"])}
        unexpected = sorted(set(detected) - expected)
        if unexpected:
            warnings.append(
                f"{match_id}: combined_paste_team_mismatch:{path.name}:"
                f"expected={' vs '.join(sorted(expected))}:found={','.join(detected)}"
            )
    return tuple(warnings)


def split_combined_oddsportal_paste_file(
    source_file: str | Path,
    output_folder: str | Path = DEFAULT_SPLIT_OUTPUT_FOLDER,
    *,
    overwrite: bool = False,
) -> CombinedOddsPortalFileSplit:
    """Split one marked combined paste into the existing four parser inputs."""

    source_file = Path(source_file)
    output_folder = Path(output_folder)
    match_id = infer_match_id_from_combined_filename(source_file.name)
    sections: dict[str, list[str]] = {}
    warnings: list[str] = []
    current_section: str | None = None
    for raw_line in source_file.read_text(encoding="utf-8").splitlines():
        marker = _SECTION_PATTERN.fullmatch(raw_line)
        if marker:
            header = marker.group("header")
            canonical = _canonical_section(header)
            if canonical is None:
                _append_warning(warnings, f"unknown_section_marker:{header.strip()}")
                current_section = None
                continue
            if canonical in sections:
                _append_warning(warnings, f"duplicate_section_marker:{canonical}")
            sections.setdefault(canonical, [])
            current_section = canonical
            continue
        if current_section is not None:
            sections[current_section].append(raw_line)

    if "1x2" not in sections:
        raise CombinedOddsPortalPasteError(
            f"Combined OddsPortal paste {source_file.name} is missing required ### 1X2 section"
        )
    for section in OPTIONAL_SECTIONS:
        if section not in sections:
            _append_warning(warnings, f"missing_optional_section:{section}")
    for section, lines in sections.items():
        if not "\n".join(lines).strip():
            _append_warning(warnings, f"empty_section:{section}")

    files_written: list[Path] = []
    for section, lines in sections.items():
        destination = output_folder / f"{match_id}_{SECTION_FILENAMES[section]}.txt"
        if destination.exists() and not overwrite:
            _append_warning(warnings, f"existing_split_file_skipped:{destination.name}")
            continue
        output_folder.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        files_written.append(destination)
    return CombinedOddsPortalFileSplit(
        source_file,
        match_id,
        tuple(section for section in SECTION_FILENAMES if section in sections),
        tuple(files_written),
        tuple(warnings),
    )


def split_combined_oddsportal_pastes(
    input_folder: str | Path = DEFAULT_COMBINED_INPUT_FOLDER,
    output_folder: str | Path = DEFAULT_SPLIT_OUTPUT_FOLDER,
    *,
    overwrite: bool = False,
    schedule: pd.DataFrame | None = None,
) -> CombinedOddsPortalSplitResult:
    """Split every recognised combined paste in a folder."""

    input_folder = Path(input_folder)
    files = (
        tuple(sorted(input_folder.glob("*_all_odds.txt")))
        if input_folder.exists()
        else ()
    )
    validation_warnings = (
        validate_combined_pastes_against_schedule(input_folder, schedule)
        if schedule is not None and not schedule.empty
        else ()
    )
    return CombinedOddsPortalSplitResult(
        tuple(
            split_combined_oddsportal_paste_file(path, output_folder, overwrite=overwrite)
            for path in files
        ),
        validation_warnings,
    )


def format_combined_oddsportal_split_summary(result: CombinedOddsPortalSplitResult) -> str:
    """Render a concise combined-paste preprocessing summary."""

    warnings = result.warnings
    return "\n".join(
        [
            "Combined OddsPortal Paste Split",
            f"- Files processed: {result.files_processed}",
            f"- Sections extracted: {result.sections_extracted}",
            f"- Files written: {result.files_written}",
            "- Warnings:",
            *(f"  - {warning}" for warning in warnings),
            *(["  - none"] if not warnings else []),
        ]
    )
