"""Parse a manually pasted fixture schedule into sequential live match metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from wc_predictor.market_data import load_tabular_data
from wc_predictor.utils import ensure_parent_directory

DEFAULT_SCHEDULE_INPUT_PATH = Path("data/raw/oddsportal_schedule.txt")
DEFAULT_SCHEDULE_OUTPUT_PATH = Path("data/raw/world_cup_schedule_from_paste.csv")
DEFAULT_SCHEDULE_REPORT_PATH = Path("data/processed/oddsportal_schedule_parse_report.csv")
DEFAULT_EXISTING_METADATA_PATH = Path("data/raw/world_cup_odds.xlsx")

SCHEDULE_COLUMNS = [
    "match_id",
    "date",
    "time",
    "stage",
    "group",
    "team_a",
    "team_b",
    "schedule_odds_a_win",
    "schedule_odds_draw",
    "schedule_odds_b_win",
    "source_quality",
    "notes",
]
REPORT_COLUMNS = [
    "source_file",
    "fixtures_found",
    "dates_found",
    "fixtures_with_missing_odds",
    "duplicate_fixtures",
    "warnings",
]

_DATE_PATTERN = re.compile(r"^\s*(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})\s*$")
_TIME_PATTERN = re.compile(r"^\s*([01]?\d|2[0-3]):([0-5]\d)\s*$")
_ODDS_PATTERN = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*$")
_MISSING_ODDS_PATTERN = re.compile(r"^\s*(?:\*|-)\s*$")
_HEADER_LINES = {"1", "X", "2"}


class OddsPortalScheduleParseError(ValueError):
    """Raised when a pasted schedule cannot produce fixture metadata."""


@dataclass(frozen=True)
class OddsPortalScheduleParseResult:
    """Parsed schedule rows and one compact file-level diagnostic row."""

    schedule: pd.DataFrame
    report: pd.DataFrame


@dataclass(frozen=True)
class PreparedScheduleMetadata:
    """Resolved metadata source for a live run and any newly parsed schedule."""

    metadata_path: Path | None
    schedule_path: Path | None
    schedule: pd.DataFrame
    parse_result: OddsPortalScheduleParseResult | None


def _normalise(line: str) -> str:
    return " ".join(line.strip().split())


def _parse_date(line: str) -> str | None:
    match = _DATE_PATTERN.fullmatch(line)
    if not match:
        return None
    try:
        parsed = datetime.strptime(" ".join(match.groups()), "%d %b %Y")
    except ValueError:
        return None
    return parsed.date().isoformat()


def _parse_time(line: str) -> str | None:
    match = _TIME_PATTERN.fullmatch(line)
    return f"{int(match.group(1)):02d}:{match.group(2)}" if match else None


def _is_dash_separator(line: str) -> bool:
    normalised = _normalise(line)
    if normalised in {"-", "–", "—"}:
        return True
    return "â" in normalised and any(character in normalised for character in {"€", "“", "™"})


def _parse_odd(line: str) -> float | None:
    match = _ODDS_PATTERN.fullmatch(line)
    return float(match.group(1).replace(",", ".")) if match else None


def _odds_values(lines: list[str]) -> tuple[float | None, float | None, float | None]:
    values: list[float | None] = []
    for line in lines:
        tokens = line.split() if line and all(token in {"*", "-"} for token in line.split()) else [line]
        for token in tokens:
            if _MISSING_ODDS_PATTERN.fullmatch(token):
                values.append(None)
            else:
                value = _parse_odd(token)
                if value is not None:
                    values.append(value)
            if len(values) == 3:
                return tuple(values)  # type: ignore[return-value]
    return tuple([*values, *([None] * (3 - len(values)))])  # type: ignore[return-value]


def _duplicate_pair_fixture(lines: list[str]) -> tuple[str, str] | None:
    candidates = [
        line
        for line in lines
        if line not in _HEADER_LINES and _parse_odd(line) is None and not _MISSING_ODDS_PATTERN.fullmatch(line)
    ]
    for index in range(len(candidates) - 3):
        if candidates[index] == candidates[index + 1] and candidates[index + 2] == candidates[index + 3]:
            return candidates[index], candidates[index + 2]
    return None


def _fixture_teams(lines: list[str]) -> tuple[str, str] | None:
    for index, line in enumerate(lines):
        if not _is_dash_separator(line):
            continue
        before = [candidate for candidate in lines[:index] if candidate and candidate not in _HEADER_LINES]
        after = [candidate for candidate in lines[index + 1 :] if candidate and candidate not in _HEADER_LINES]
        if before and after:
            return before[-1], after[0]
    return _duplicate_pair_fixture(lines)


def _existing_groups(existing_metadata_path: str | Path | None) -> dict[str, dict[str, object]]:
    if existing_metadata_path is None or not Path(existing_metadata_path).exists():
        return {}
    metadata = load_tabular_data(existing_metadata_path)
    if "match_id" not in metadata:
        return {}
    available = [column for column in ("stage", "group") if column in metadata]
    if not available:
        return {}
    rows = metadata[["match_id", *available]].drop_duplicates("match_id")
    return {
        str(row["match_id"]): {column: row[column] for column in available}
        for _, row in rows.iterrows()
    }


def _with_existing_stage_and_group(
    schedule: pd.DataFrame,
    existing_metadata_path: str | Path | None,
) -> pd.DataFrame:
    existing = _existing_groups(existing_metadata_path)
    schedule = schedule.copy()
    for index, row in schedule.iterrows():
        metadata = existing.get(str(row["match_id"]), {})
        stage = metadata.get("stage")
        group = metadata.get("group")
        if pd.notna(stage) and str(stage).strip():
            schedule.at[index, "stage"] = stage
        if pd.notna(group) and str(group).strip():
            schedule.at[index, "group"] = group
    return schedule


def parse_oddsportal_schedule_text(
    text: str,
    *,
    source_file: str = "oddsportal_schedule.txt",
    existing_metadata_path: str | Path | None = DEFAULT_EXISTING_METADATA_PATH,
) -> OddsPortalScheduleParseResult:
    """Parse pasted schedule text and assign sequential M001-style match IDs."""

    lines = [_normalise(line) for line in text.splitlines() if _normalise(line)]
    current_date: str | None = None
    dates: list[str] = []
    fixture_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    index = 0
    while index < len(lines):
        parsed_date = _parse_date(lines[index])
        if parsed_date is not None:
            current_date = parsed_date
            if parsed_date not in dates:
                dates.append(parsed_date)
            index += 1
            continue
        kickoff = _parse_time(lines[index])
        if kickoff is None:
            index += 1
            continue
        block_end = index + 1
        while block_end < len(lines):
            if _parse_date(lines[block_end]) is not None or _parse_time(lines[block_end]) is not None:
                break
            block_end += 1
        block = lines[index + 1 : block_end]
        teams = _fixture_teams(block)
        if current_date is None:
            warnings.append(f"fixture_without_date:{kickoff}")
        elif teams is None:
            warnings.append(f"fixture_without_team_pair:{current_date}T{kickoff}")
        else:
            odds = _odds_values(block)
            missing_odds = any(value is None for value in odds)
            notes = "missing_schedule_1x2_odds" if missing_odds else ""
            fixture_rows.append(
                {
                    "date": current_date,
                    "time": kickoff,
                    "stage": "group",
                    "group": pd.NA,
                    "team_a": teams[0],
                    "team_b": teams[1],
                    "schedule_odds_a_win": odds[0],
                    "schedule_odds_draw": odds[1],
                    "schedule_odds_b_win": odds[2],
                    "source_quality": "oddsportal_schedule_paste",
                    "notes": notes,
                }
            )
        index = block_end
    if not fixture_rows:
        raise OddsPortalScheduleParseError("No fixtures could be parsed from the OddsPortal schedule paste")

    schedule = pd.DataFrame(fixture_rows)
    schedule.insert(0, "match_id", [f"M{number:03d}" for number in range(1, len(schedule) + 1)])
    schedule = _with_existing_stage_and_group(schedule, existing_metadata_path)
    schedule = schedule[SCHEDULE_COLUMNS]
    duplicate_count = int(schedule.duplicated(["date", "time", "team_a", "team_b"]).sum())
    if duplicate_count:
        warnings.append(f"duplicate_fixtures:{duplicate_count}")
    missing_odds_count = int(schedule[["schedule_odds_a_win", "schedule_odds_draw", "schedule_odds_b_win"]].isna().any(axis=1).sum())
    if missing_odds_count:
        missing_ids = schedule.loc[
            schedule[["schedule_odds_a_win", "schedule_odds_draw", "schedule_odds_b_win"]].isna().any(axis=1),
            "match_id",
        ]
        warnings.append("fixtures_with_missing_odds:" + ",".join(missing_ids.astype(str)))
    report = pd.DataFrame(
        [
            {
                "source_file": source_file,
                "fixtures_found": len(schedule),
                "dates_found": len(dates),
                "fixtures_with_missing_odds": missing_odds_count,
                "duplicate_fixtures": duplicate_count,
                "warnings": "; ".join(dict.fromkeys(warnings)),
            }
        ],
        columns=REPORT_COLUMNS,
    )
    return OddsPortalScheduleParseResult(schedule, report)


def parse_oddsportal_schedule_file(
    input_path: str | Path = DEFAULT_SCHEDULE_INPUT_PATH,
    output_path: str | Path = DEFAULT_SCHEDULE_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_SCHEDULE_REPORT_PATH,
    *,
    existing_metadata_path: str | Path | None = DEFAULT_EXISTING_METADATA_PATH,
) -> OddsPortalScheduleParseResult:
    """Parse one schedule paste file and export its metadata and diagnostics."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    result = parse_oddsportal_schedule_text(
        input_path.read_text(encoding="utf-8-sig"),
        source_file=input_path.name,
        existing_metadata_path=existing_metadata_path,
    )
    ensure_parent_directory(output_path)
    ensure_parent_directory(report_path)
    result.schedule.to_csv(output_path, index=False)
    result.report.to_csv(report_path, index=False)
    return result


def prepare_schedule_metadata(
    input_path: str | Path | None = DEFAULT_SCHEDULE_INPUT_PATH,
    output_path: str | Path = DEFAULT_SCHEDULE_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_SCHEDULE_REPORT_PATH,
    *,
    existing_metadata_path: str | Path | None = DEFAULT_EXISTING_METADATA_PATH,
    skip_parse: bool = False,
) -> PreparedScheduleMetadata:
    """Prefer parsed schedule metadata, falling back to the existing workbook."""

    input_path = Path(input_path) if input_path is not None else None
    output_path = Path(output_path)
    if input_path is not None and input_path.exists() and not skip_parse:
        parse_result = parse_oddsportal_schedule_file(
            input_path,
            output_path,
            report_path,
            existing_metadata_path=existing_metadata_path,
        )
        return PreparedScheduleMetadata(output_path, output_path, parse_result.schedule, parse_result)
    if output_path.exists():
        return PreparedScheduleMetadata(output_path, output_path, load_tabular_data(output_path), None)
    fallback = Path(existing_metadata_path) if existing_metadata_path is not None else None
    return PreparedScheduleMetadata(fallback, None, pd.DataFrame(columns=SCHEDULE_COLUMNS), None)


def format_oddsportal_schedule_parse_summary(
    result: OddsPortalScheduleParseResult,
    output_path: str | Path = DEFAULT_SCHEDULE_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_SCHEDULE_REPORT_PATH,
) -> str:
    """Render a compact schedule parsing summary."""

    schedule = result.schedule
    first = schedule.iloc[0]
    last = schedule.iloc[-1]
    missing_odds = int(result.report.loc[0, "fixtures_with_missing_odds"])
    return "\n".join(
        [
            "Schedule Paste Parse",
            f"- Fixtures parsed: {len(schedule)}",
            f"- First fixture: {first['match_id']} | {first['team_a']} vs {first['team_b']}",
            f"- Last fixture: {last['match_id']} | {last['team_a']} vs {last['team_b']}",
            f"- Missing odds fixtures: {missing_odds}",
            f"- Output path: {Path(output_path)}",
            f"- Parse report path: {Path(report_path)}",
        ]
    )
