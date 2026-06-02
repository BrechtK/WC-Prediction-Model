"""Parse manually pasted OddsPortal core-market text into model-ready rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from wc_predictor.market_data import load_tabular_data
from wc_predictor.utils import ensure_parent_directory

DEFAULT_INPUT_FOLDER = Path("data/raw/oddsportal_pastes")
DEFAULT_OUTPUT_PATH = Path("data/raw/world_cup_odds_from_pastes.csv")
DEFAULT_TOTAL_GOALS_OUTPUT_PATH = Path("data/raw/world_cup_total_goals_odds_from_pastes.csv")
DEFAULT_REPORT_PATH = Path("data/processed/oddsportal_core_odds_parse_report.csv")
DEFAULT_METADATA_ODDS_PATH = Path("data/raw/world_cup_odds.xlsx")

OUTPUT_COLUMNS = [
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
REPORT_COLUMNS = [
    "source_file",
    "match_id",
    "market_type",
    "bookmakers_found",
    "rows_extracted",
    "total_goals_lines_found",
    "bookmakers_per_total_goals_line",
    "has_over_under_2_5",
    "warnings",
]
TOTAL_GOALS_OUTPUT_COLUMNS = [
    "match_id",
    "bookmaker",
    "line",
    "odds_over",
    "odds_under",
    "market_kind",
    "source_quality",
    "notes",
]
MARKET_COLUMNS = {
    "1x2": ("odds_a_win", "odds_draw", "odds_b_win"),
    "btts": ("odds_btts_yes", "odds_btts_no"),
    "over_under": ("odds_over_2_5", "odds_under_2_5"),
}
EXPECTED_MARKETS = tuple(MARKET_COLUMNS)
MATCH_METADATA_COLUMNS = ("date", "stage", "group", "team_a", "team_b")
OPTIONAL_BOOKMAKER_COLUMNS = ("odds_a_qualifies", "odds_b_qualifies")
SUSPICIOUS_ODDS_HIGH = 500.0

_FILENAME_PATTERN = re.compile(r"^(?P<match_id>.+)_(?P<market_type>1x2|btts|over_under)\.txt$", re.IGNORECASE)
_NUMBER_PATTERN = r"(?:\d+(?:[.,]\d+)?|[.,]\d+)"
_ODDS_PATTERN = re.compile(rf"^\s*({_NUMBER_PATTERN})\s*$")
_PERCENT_PATTERN = re.compile(rf"^\s*{_NUMBER_PATTERN}%\s*$")
_TOTAL_PATTERN = re.compile(rf"^\s*\+?({_NUMBER_PATTERN})\s*$")
_ROW_TOTAL_PATTERN = re.compile(rf"^\s*\+({_NUMBER_PATTERN})\s*$")
_OVER_UNDER_BLOCK_PATTERN = re.compile(rf"^\s*over\s*/?\s*under\s+\+?({_NUMBER_PATTERN})\s*$", re.IGNORECASE)
_NOISE_FRAGMENTS = (
    "bookmakers",
    "claim bonus",
    "my coupon",
    "user predictions",
    "oddsalert",
    "notify me when odds reach",
    "looking for the best value",
    "is offering the highest payout",
    "you must be logged-in",
    "you must be logged in",
)
_EXCHANGE_FRAGMENTS = (
    "betting exchange",
    "betfair exchange",
)
_HEADER_LINES = {
    "x",
    "yes",
    "no",
    "payout",
    "total",
    "over",
    "under",
    "back",
    "lay",
    "odds",
}


def _append_warning(warnings: list[str], warning: str) -> None:
    if warning and warning not in warnings:
        warnings.append(warning)


def _normalise(line: str) -> str:
    return " ".join(line.strip().split())


def _is_noise(line: str) -> bool:
    normalised = _normalise(line).lower()
    return normalised in _HEADER_LINES or any(fragment in normalised for fragment in _NOISE_FRAGMENTS)


def _is_exchange_heading(line: str) -> bool:
    normalised = _normalise(line).lower()
    return any(fragment in normalised for fragment in _EXCHANGE_FRAGMENTS)


def _looks_like_bookmaker(line: str) -> bool:
    if not line or _is_noise(line) or _is_exchange_heading(line):
        return False
    if _PERCENT_PATTERN.fullmatch(line) or _ODDS_PATTERN.fullmatch(line) or _TOTAL_PATTERN.fullmatch(line):
        return False
    if any(character in line for character in "?!"):
        return False
    return len(line.split()) <= 5


def _decimal_odds(line: str) -> float | None:
    match = _ODDS_PATTERN.fullmatch(line)
    return float(match.group(1).replace(",", ".")) if match else None


@dataclass
class _MarketRows:
    """Bookmaker rows and diagnostics extracted from one pasted table."""

    market_type: str
    expected_values: int
    rows: list[tuple[str, tuple[float, ...]]] = field(default_factory=list)
    total_goals_rows: list[tuple[str, float, float, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bookmaker_labels: set[str] = field(default_factory=set)
    current_bookmaker: str | None = None
    current_values: list[float] = field(default_factory=list)
    current_total: float | None = None
    row_complete: bool = False

    def start_bookmaker(self, bookmaker: str) -> None:
        self.bookmaker_labels.add(bookmaker)
        if bookmaker == self.current_bookmaker and not self.current_values and not self.row_complete:
            return
        self.finish_row()
        self.current_bookmaker = bookmaker

    def add_total(self, total: float) -> None:
        self.current_total = total

    def add_odds(self, value: float) -> None:
        if self.current_bookmaker is None or self.row_complete:
            return
        self.current_values.append(value)
        if len(self.current_values) == self.expected_values:
            if self.market_type == "over_under" and self.current_total is not None:
                self.total_goals_rows.append(
                    (self.current_bookmaker, self.current_total, self.current_values[0], self.current_values[1])
                )
            if self.market_type != "over_under" or (
                self.current_total is not None and np.isclose(self.current_total, 2.5)
            ):
                self.rows.append((self.current_bookmaker, tuple(self.current_values)))
            for odds in self.current_values:
                if odds <= 1:
                    _append_warning(self.warnings, f"suspicious_odds_le_1:{self.current_bookmaker}={odds:g}")
                elif odds > SUSPICIOUS_ODDS_HIGH:
                    _append_warning(self.warnings, f"suspicious_odds_very_high:{self.current_bookmaker}={odds:g}")
            self.row_complete = True

    def finish_row(self) -> None:
        if self.current_bookmaker is not None and self.current_values and not self.row_complete:
            _append_warning(self.warnings, f"incomplete_odds_row:{self.current_bookmaker}")
        self.current_bookmaker = None
        self.current_values = []
        self.current_total = None
        self.row_complete = False


@dataclass(frozen=True)
class OddsPortalCoreParseResult:
    """Model-ready core odds rows and file-level parser diagnostics."""

    odds: pd.DataFrame
    total_goals_odds: pd.DataFrame
    report: pd.DataFrame


def infer_match_and_market_from_filename(path: str | Path) -> tuple[str, str]:
    """Infer match ID and core-market type from one named paste file."""

    name = Path(path).name
    match = _FILENAME_PATTERN.fullmatch(name)
    if not match:
        raise ValueError("OddsPortal core paste filename must end with '_1x2.txt', '_btts.txt', or '_over_under.txt'")
    return match.group("match_id"), match.group("market_type").lower()


def parse_oddsportal_core_market_text(
    text: str,
    match_id: str,
    market_type: str,
    *,
    source_file: str = "",
) -> OddsPortalCoreParseResult:
    """Parse one manually copied OddsPortal 1X2, BTTS, or over/under table."""

    if market_type not in MARKET_COLUMNS:
        raise ValueError(f"Unsupported OddsPortal core market type: {market_type}")
    parsed = _MarketRows(market_type, len(MARKET_COLUMNS[market_type]))
    active_total: float | None = None
    target_total_seen = False
    inside_exchange_section = False

    for raw_line in text.splitlines():
        line = _normalise(raw_line)
        if not line:
            continue
        if market_type == "over_under":
            block_match = _OVER_UNDER_BLOCK_PATTERN.fullmatch(line)
            if block_match:
                parsed.finish_row()
                active_total = float(block_match.group(1).replace(",", "."))
                inside_exchange_section = False
                if active_total == 2.5:
                    target_total_seen = True
                continue
        if _is_exchange_heading(line):
            inside_exchange_section = True
            _append_warning(parsed.warnings, "betting_exchange_section_ignored")
            parsed.finish_row()
            continue
        if inside_exchange_section:
            continue
        if _PERCENT_PATTERN.fullmatch(line) or _is_noise(line):
            continue
        if market_type == "over_under" and parsed.current_bookmaker is not None:
            total_match = _ROW_TOTAL_PATTERN.fullmatch(line)
            if total_match and not parsed.current_values:
                parsed.add_total(float(total_match.group(1).replace(",", ".")))
                if parsed.current_total == 2.5:
                    target_total_seen = True
                continue
        odds = _decimal_odds(line)
        if odds is not None:
            if market_type != "over_under" or parsed.current_total is not None or active_total == 2.5:
                if market_type == "over_under" and parsed.current_total is None:
                    parsed.add_total(active_total or float("nan"))
                parsed.add_odds(odds)
            continue
        if _looks_like_bookmaker(line):
            parsed.start_bookmaker(line)
    parsed.finish_row()

    unique_rows = list(dict.fromkeys(parsed.rows))
    unique_total_goals_rows = list(dict.fromkeys(parsed.total_goals_rows))
    if len({bookmaker for bookmaker, _ in unique_rows}) < 2:
        _append_warning(parsed.warnings, f"fewer_than_2_bookmakers:{len({bookmaker for bookmaker, _ in unique_rows})}")
    if market_type == "over_under" and not target_total_seen:
        _append_warning(parsed.warnings, "no_2_5_over_under_block_found")
    totals_line_counts: dict[float, int] = {}
    for _, line, _, _ in unique_total_goals_rows:
        totals_line_counts[line] = totals_line_counts.get(line, 0) + 1
    for line, count in sorted(totals_line_counts.items()):
        if count < 2:
            _append_warning(parsed.warnings, f"total_goals_line_fewer_than_2_bookmakers:{line:g}={count}")

    odds_rows = []
    for bookmaker, values in unique_rows:
        row: dict[str, object] = {"match_id": match_id, "bookmaker": bookmaker}
        row.update(dict(zip(MARKET_COLUMNS[market_type], values, strict=True)))
        odds_rows.append(row)
    total_goals_rows = [
        {
            "match_id": match_id,
            "bookmaker": bookmaker,
            "line": line,
            "odds_over": odds_over,
            "odds_under": odds_under,
            "market_kind": "coherent_bookmaker",
            "source_quality": "oddsportal_paste",
            "notes": "parsed_from_oddsportal_paste",
        }
        for bookmaker, line, odds_over, odds_under in unique_total_goals_rows
    ]
    report = pd.DataFrame(
        [
            {
                "source_file": source_file,
                "match_id": match_id,
                "market_type": market_type,
                "bookmakers_found": len(parsed.bookmaker_labels),
                "rows_extracted": len(unique_rows),
                "total_goals_lines_found": len(totals_line_counts),
                "bookmakers_per_total_goals_line": "; ".join(
                    f"{line:g}:{count}" for line, count in sorted(totals_line_counts.items())
                ),
                "has_over_under_2_5": any(np.isclose(line, 2.5) for line in totals_line_counts),
                "warnings": "; ".join(parsed.warnings),
            }
        ],
        columns=REPORT_COLUMNS,
    )
    odds_columns = ["match_id", "bookmaker", *MARKET_COLUMNS[market_type]]
    return OddsPortalCoreParseResult(
        pd.DataFrame(odds_rows, columns=odds_columns),
        pd.DataFrame(total_goals_rows, columns=TOTAL_GOALS_OUTPUT_COLUMNS),
        report,
    )


def _first_non_blank(values: pd.Series) -> object:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return value
    return pd.NA


def _load_metadata(metadata_odds_path: str | Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metadata_odds_path is None or not Path(metadata_odds_path).exists():
        return pd.DataFrame(columns=["match_id", *MATCH_METADATA_COLUMNS]), pd.DataFrame(
            columns=["match_id", "bookmaker", *OPTIONAL_BOOKMAKER_COLUMNS]
        )
    metadata = load_tabular_data(metadata_odds_path)
    if "match_id" not in metadata:
        return pd.DataFrame(columns=["match_id", *MATCH_METADATA_COLUMNS]), pd.DataFrame(
            columns=["match_id", "bookmaker", *OPTIONAL_BOOKMAKER_COLUMNS]
        )
    metadata = metadata.copy()
    metadata["match_id"] = metadata["match_id"].astype(str)
    for column in (*MATCH_METADATA_COLUMNS, "bookmaker", *OPTIONAL_BOOKMAKER_COLUMNS):
        if column not in metadata:
            metadata[column] = pd.NA
    match_metadata = (
        metadata.groupby("match_id", as_index=False)[list(MATCH_METADATA_COLUMNS)]
        .agg(_first_non_blank)
    )
    bookmaker_metadata = (
        metadata.groupby(["match_id", "bookmaker"], as_index=False)[list(OPTIONAL_BOOKMAKER_COLUMNS)]
        .agg(_first_non_blank)
    )
    return match_metadata, bookmaker_metadata


def _merge_market_rows(
    market_rows: pd.DataFrame,
    *,
    metadata_odds_path: str | Path | None,
    odds_timestamp: str,
    odds_source_url: str,
) -> pd.DataFrame:
    market_odds_columns = [column for columns in MARKET_COLUMNS.values() for column in columns]
    if market_rows.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    for column in market_odds_columns:
        if column not in market_rows:
            market_rows[column] = np.nan
    merged = (
        market_rows.groupby(["match_id", "bookmaker"], as_index=False)[market_odds_columns]
        .agg(_first_non_blank)
    )
    match_metadata, bookmaker_metadata = _load_metadata(metadata_odds_path)
    merged = merged.merge(match_metadata, on="match_id", how="left")
    merged = merged.merge(bookmaker_metadata, on=["match_id", "bookmaker"], how="left")
    merged["odds_timestamp"] = odds_timestamp
    merged["odds_source_url"] = odds_source_url
    merged["source_quality"] = "oddsportal_paste"
    merged["notes"] = "parsed_from_oddsportal_paste"
    for column in OUTPUT_COLUMNS:
        if column not in merged:
            merged[column] = pd.NA
    return merged[OUTPUT_COLUMNS].sort_values(["match_id", "bookmaker"], kind="stable").reset_index(drop=True)


def _apply_missing_market_warnings(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return report
    report = report.copy()
    for match_id, rows in report.groupby("match_id", sort=False):
        present = set(rows.loc[rows["rows_extracted"] > 0, "market_type"])
        missing = [
            warning
            for market_type, warning in (
                ("1x2", "missing_1x2_market"),
                ("over_under", "missing_over_under_2_5_market"),
                ("btts", "missing_btts_market"),
            )
            if market_type not in present
        ]
        if missing:
            indexes = rows.index
            report.loc[indexes, "warnings"] = report.loc[indexes, "warnings"].map(
                lambda value: "; ".join(filter(None, [str(value), *missing]))
            )
    return report


def parse_oddsportal_core_odds_folder(
    input_folder: str | Path = DEFAULT_INPUT_FOLDER,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    *,
    total_goals_output_path: str | Path = DEFAULT_TOTAL_GOALS_OUTPUT_PATH,
    metadata_odds_path: str | Path | None = DEFAULT_METADATA_ODDS_PATH,
    odds_timestamp: str = "",
    odds_source_url: str = "",
    match_ids: Iterable[str] | None = None,
) -> OddsPortalCoreParseResult:
    """Parse every named core-market paste file, merge bookmakers, and export CSVs."""

    input_folder = Path(input_folder)
    output_path = Path(output_path)
    report_path = Path(report_path)
    total_goals_output_path = Path(total_goals_output_path)
    input_folder.mkdir(parents=True, exist_ok=True)
    results: list[OddsPortalCoreParseResult] = []
    selected_match_ids = {str(match_id) for match_id in match_ids} if match_ids is not None else None
    paths = sorted(
        path
        for path in input_folder.glob("*.txt")
        if _FILENAME_PATTERN.fullmatch(path.name)
        and (
            selected_match_ids is None
            or infer_match_and_market_from_filename(path)[0] in selected_match_ids
        )
    )
    for path in paths:
        match_id, market_type = infer_match_and_market_from_filename(path)
        results.append(
            parse_oddsportal_core_market_text(
                path.read_text(encoding="utf-8-sig"),
                match_id,
                market_type,
                source_file=path.name,
            )
        )
    market_rows = pd.concat([result.odds for result in results], ignore_index=True) if results else pd.DataFrame()
    total_goals_odds = (
        pd.concat([result.total_goals_odds for result in results], ignore_index=True)
        if results
        else pd.DataFrame(columns=TOTAL_GOALS_OUTPUT_COLUMNS)
    )
    odds = _merge_market_rows(
        market_rows,
        metadata_odds_path=metadata_odds_path,
        odds_timestamp=odds_timestamp,
        odds_source_url=odds_source_url,
    )
    report = (
        pd.concat([result.report for result in results], ignore_index=True)
        if results
        else pd.DataFrame(columns=REPORT_COLUMNS)
    )
    report = _apply_missing_market_warnings(report)
    ensure_parent_directory(output_path)
    ensure_parent_directory(total_goals_output_path)
    ensure_parent_directory(report_path)
    odds.to_csv(output_path, index=False)
    total_goals_odds.to_csv(total_goals_output_path, index=False)
    report.to_csv(report_path, index=False)
    return OddsPortalCoreParseResult(odds, total_goals_odds, report)


def format_oddsportal_core_parse_summary(
    result: OddsPortalCoreParseResult,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    total_goals_output_path: str | Path = DEFAULT_TOTAL_GOALS_OUTPUT_PATH,
) -> str:
    """Render compact extraction and missing-market diagnostics."""

    report = result.report
    files_parsed = report["source_file"].nunique() if not report.empty else 0
    matches = sorted(report["match_id"].astype(str).unique()) if not report.empty else []
    counts = (
        report.groupby("market_type")["rows_extracted"].sum().to_dict()
        if not report.empty
        else {}
    )
    bookmaker_counts = (
        result.odds.groupby("match_id")["bookmaker"].nunique().to_dict()
        if not result.odds.empty
        else {}
    )
    missing_lines = []
    for match_id, rows in report.groupby("match_id", sort=True):
        present = set(rows.loc[rows["rows_extracted"] > 0, "market_type"])
        missing = [market for market in EXPECTED_MARKETS if market not in present]
        missing_lines.append(f"- {match_id}: {', '.join(missing) if missing else 'none'}")
    return "\n".join(
        [
            "OddsPortal Core Odds Paste Parse",
            f"- Files parsed: {files_parsed}",
            f"- Matches parsed: {len(matches)}",
            f"- 1X2 rows extracted: {counts.get('1x2', 0)}",
            f"- BTTS rows extracted: {counts.get('btts', 0)}",
            f"- O/U 2.5 rows extracted: {counts.get('over_under', 0)}",
            f"- Total-goals ladder rows extracted: {len(result.total_goals_odds)}",
            "- Bookmakers per match: "
            + (", ".join(f"{match_id}={count}" for match_id, count in sorted(bookmaker_counts.items())) or "none"),
            "Missing markets by match:",
            *(missing_lines or ["- none"]),
            f"- Output path: {Path(output_path)}",
            f"- Total-goals output path: {Path(total_goals_output_path)}",
        ]
    )
