"""Parse manually pasted OddsPortal Asian-handicap text into model-ready rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from wc_predictor.paths import (
    CACHE_ASIAN_HANDICAP_ODDS_PATH,
    CACHE_SPLIT_PASTES_DIR,
    OUTPUT_ASIAN_HANDICAP_PARSE_REPORT_PATH,
)
from wc_predictor.utils import ensure_parent_directory

DEFAULT_INPUT_FOLDER = CACHE_SPLIT_PASTES_DIR
DEFAULT_OUTPUT_PATH = CACHE_ASIAN_HANDICAP_ODDS_PATH
DEFAULT_REPORT_PATH = OUTPUT_ASIAN_HANDICAP_PARSE_REPORT_PATH

OUTPUT_COLUMNS = [
    "match_id",
    "bookmaker",
    "handicap",
    "odds_team_a",
    "odds_team_b",
    "market_kind",
    "source_quality",
    "notes",
]
REPORT_COLUMNS = [
    "source_file",
    "match_id",
    "lines_found",
    "bookmakers_found",
    "rows_extracted",
    "bookmakers_per_line",
    "warnings",
]

_FILENAME_PATTERN = re.compile(r"^(?P<match_id>.+)_asian_handicap\.txt$", re.IGNORECASE)
_NUMBER_PATTERN = r"(?:[+-]?\d+(?:[.,]\d+)?|[+-]?[.,]\d+)"
_ODDS_PATTERN = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*$")
_SIGNED_LINE_PATTERN = re.compile(rf"^\s*([+-]\s*\d+(?:[.,]\d+)?)\s*$")
_HANDICAP_HEADER_PATTERN = re.compile(rf"asian\s+handicap\s+({_NUMBER_PATTERN})", re.IGNORECASE)
_INLINE_ROW_PATTERN = re.compile(
    rf"^\s*(?P<bookmaker>.+?)\s+(?P<line>[+-]\s*\d+(?:[.,]\d+)?)\s+"
    rf"(?P<odds_a>\d+(?:[.,]\d+)?)\s+(?P<odds_b>\d+(?:[.,]\d+)?)\s*$"
)
_PERCENT_PATTERN = re.compile(r"^\s*\d+(?:[.,]\d+)?%\s*$")
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
    "betting exchange",
)
_TERMINAL_SECTION_FRAGMENTS = (
    "ai match predictions",
    "previous matches:",
    "h2h results",
    "odds, predictions and h2h results",
)
_HEADER_LINES = {"handicap", "asian handicap", "home", "away", "1", "2", "odds", "payout", "back", "lay"}


def _append_warning(warnings: list[str], warning: str) -> None:
    if warning and warning not in warnings:
        warnings.append(warning)


def _normalise(line: str) -> str:
    return " ".join(line.strip().split())


def _parse_number(value: str) -> float:
    return float(value.replace(" ", "").replace(",", "."))


def _is_noise(line: str) -> bool:
    normalised = _normalise(line).lower()
    return normalised in _HEADER_LINES or any(fragment in normalised for fragment in _NOISE_FRAGMENTS)


def _is_terminal_non_market_section(line: str) -> bool:
    normalised = _normalise(line).lower()
    return any(fragment in normalised for fragment in _TERMINAL_SECTION_FRAGMENTS)


def _looks_like_bookmaker(line: str) -> bool:
    if not line or _is_noise(line):
        return False
    if _PERCENT_PATTERN.fullmatch(line) or _ODDS_PATTERN.fullmatch(line) or _SIGNED_LINE_PATTERN.fullmatch(line):
        return False
    if any(character in line for character in "?!"):
        return False
    return len(line.split()) <= 6


def _line_kind(handicap: float) -> str:
    remainder = abs(float(handicap)) % 1
    if np.isclose(remainder, 0.5):
        return "half_goal"
    if np.isclose(remainder, 0.0):
        return "integer_asian"
    if np.isclose(remainder, 0.25) or np.isclose(remainder, 0.75):
        return "quarter_asian"
    return "unsupported"


@dataclass
class _AsianHandicapRows:
    rows: list[tuple[str, float, float, float]] = field(default_factory=list)
    exact_rows: set[tuple[str, float, float, float]] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    bookmaker_labels: set[str] = field(default_factory=set)
    current_bookmaker: str | None = None
    current_handicap: float | None = None
    current_values: list[float] = field(default_factory=list)

    def start_bookmaker(self, bookmaker: str) -> None:
        self.finish_row()
        self.current_bookmaker = bookmaker
        self.bookmaker_labels.add(bookmaker)

    def set_handicap(self, handicap: float) -> None:
        if _line_kind(handicap) == "unsupported":
            _append_warning(self.warnings, f"unsupported_handicap_line:{handicap:g}")
        self.current_handicap = handicap

    def add_odds(self, odds: float) -> None:
        if self.current_bookmaker is None:
            return
        self.current_values.append(odds)
        if len(self.current_values) == 2:
            self.finish_row()

    def add_inline_row(self, bookmaker: str, handicap: float, odds_a: float, odds_b: float) -> None:
        self.bookmaker_labels.add(bookmaker)
        self._store_row(bookmaker, handicap, odds_a, odds_b)

    def _store_row(self, bookmaker: str, handicap: float, odds_a: float, odds_b: float) -> None:
        if odds_a <= 1.0 or odds_b <= 1.0:
            _append_warning(self.warnings, f"invalid_odds:{bookmaker}:{handicap:g}")
            return
        exact = (bookmaker, float(handicap), float(odds_a), float(odds_b))
        if exact in self.exact_rows:
            _append_warning(self.warnings, f"deduplicated_identical_row:{bookmaker}:{handicap:g}")
            return
        previous = [
            row for row in self.rows if row[0] == bookmaker and np.isclose(row[1], handicap)
        ]
        if previous:
            _append_warning(self.warnings, f"duplicate_bookmaker_line:{bookmaker}:{handicap:g}")
        self.rows.append(exact)
        self.exact_rows.add(exact)

    def finish_row(self) -> None:
        if self.current_bookmaker is not None and self.current_values:
            if self.current_handicap is None:
                _append_warning(self.warnings, f"missing_handicap_line:{self.current_bookmaker}")
            elif len(self.current_values) == 1:
                _append_warning(self.warnings, f"missing_odds:{self.current_bookmaker}:{self.current_handicap:g}")
            elif len(self.current_values) >= 2:
                self._store_row(
                    self.current_bookmaker,
                    float(self.current_handicap),
                    float(self.current_values[0]),
                    float(self.current_values[1]),
                )
        self.current_bookmaker = None
        self.current_handicap = None
        self.current_values = []


@dataclass(frozen=True)
class OddsPortalAsianHandicapParseResult:
    """Model-ready Asian-handicap rows and file-level parser diagnostics."""

    odds: pd.DataFrame
    report: pd.DataFrame


def infer_match_id_from_filename(path: str | Path) -> str:
    """Infer the match ID prefix from an ``*_asian_handicap.txt`` filename."""

    name = Path(path).name
    match = _FILENAME_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"OddsPortal Asian-handicap paste filename must end with '_asian_handicap.txt': {name}")
    return match.group("match_id")


def parse_oddsportal_asian_handicap_text(
    text: str,
    match_id: str,
    *,
    source_file: str = "",
) -> OddsPortalAsianHandicapParseResult:
    """Parse one manually copied OddsPortal Asian-handicap market."""

    parsed = _AsianHandicapRows()
    active_handicap: float | None = None
    for raw_line in text.splitlines():
        line = _normalise(raw_line)
        if not line:
            continue
        if _is_terminal_non_market_section(line):
            parsed.finish_row()
            _append_warning(parsed.warnings, f"stopped_at_non_market_section:{line}")
            break
        header = _HANDICAP_HEADER_PATTERN.search(line)
        if header:
            parsed.finish_row()
            active_handicap = _parse_number(header.group(1))
            continue
        inline = _INLINE_ROW_PATTERN.fullmatch(line)
        if inline and _looks_like_bookmaker(inline.group("bookmaker")):
            parsed.finish_row()
            parsed.add_inline_row(
                _normalise(inline.group("bookmaker")),
                _parse_number(inline.group("line")),
                _parse_number(inline.group("odds_a")),
                _parse_number(inline.group("odds_b")),
            )
            continue
        if _PERCENT_PATTERN.fullmatch(line) or _is_noise(line):
            continue
        signed_line = _SIGNED_LINE_PATTERN.fullmatch(line)
        if signed_line and parsed.current_bookmaker is not None and not parsed.current_values:
            parsed.set_handicap(_parse_number(signed_line.group(1)))
            continue
        odds = _ODDS_PATTERN.fullmatch(line)
        if odds:
            if parsed.current_bookmaker is not None:
                if parsed.current_handicap is None and active_handicap is not None:
                    parsed.set_handicap(active_handicap)
                parsed.add_odds(_parse_number(odds.group(1)))
            continue
        if line == "-":
            if parsed.current_bookmaker is not None:
                handicap_label = (
                    f":{parsed.current_handicap:g}" if parsed.current_handicap is not None else ""
                )
                _append_warning(parsed.warnings, f"missing_odds:{parsed.current_bookmaker}{handicap_label}")
            continue
        if _looks_like_bookmaker(line):
            parsed.start_bookmaker(line)
    parsed.finish_row()

    line_counts: dict[float, int] = {}
    for _, handicap, _, _ in parsed.rows:
        line_counts[handicap] = line_counts.get(handicap, 0) + 1
    for handicap, count in sorted(line_counts.items()):
        if count < 2:
            _append_warning(parsed.warnings, f"handicap_line_fewer_than_2_bookmakers:{handicap:g}={count}")
    if not parsed.rows:
        _append_warning(parsed.warnings, "no_asian_handicap_rows_found")

    odds_rows = [
        {
            "match_id": match_id,
            "bookmaker": bookmaker,
            "handicap": handicap,
            "odds_team_a": odds_a,
            "odds_team_b": odds_b,
            "market_kind": "coherent_bookmaker",
            "source_quality": "oddsportal_paste",
            "notes": "parsed_from_oddsportal_paste",
        }
        for bookmaker, handicap, odds_a, odds_b in parsed.rows
    ]
    report = pd.DataFrame(
        [
            {
                "source_file": source_file,
                "match_id": match_id,
                "lines_found": len(line_counts),
                "bookmakers_found": len(parsed.bookmaker_labels),
                "rows_extracted": len(parsed.rows),
                "bookmakers_per_line": "; ".join(
                    f"{line:g}:{count}" for line, count in sorted(line_counts.items())
                ),
                "warnings": "; ".join(parsed.warnings),
            }
        ],
        columns=REPORT_COLUMNS,
    )
    return OddsPortalAsianHandicapParseResult(
        pd.DataFrame(odds_rows, columns=OUTPUT_COLUMNS).drop_duplicates(ignore_index=True),
        report,
    )


def parse_oddsportal_asian_handicap_folder(
    input_folder: str | Path = DEFAULT_INPUT_FOLDER,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    *,
    match_ids: Iterable[str] | None = None,
) -> OddsPortalAsianHandicapParseResult:
    """Parse every named Asian-handicap paste file and export model-ready CSVs."""

    input_folder = Path(input_folder)
    output_path = Path(output_path)
    report_path = Path(report_path)
    input_folder.mkdir(parents=True, exist_ok=True)
    selected_match_ids = {str(match_id) for match_id in match_ids} if match_ids is not None else None
    paths = sorted(
        path
        for path in input_folder.glob("*_asian_handicap.txt")
        if selected_match_ids is None or infer_match_id_from_filename(path) in selected_match_ids
    )
    results = [
        parse_oddsportal_asian_handicap_text(
            path.read_text(encoding="utf-8-sig"),
            infer_match_id_from_filename(path),
            source_file=path.name,
        )
        for path in paths
    ]
    odds = (
        pd.concat([result.odds for result in results], ignore_index=True)
        if results
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    report = (
        pd.concat([result.report for result in results], ignore_index=True)
        if results
        else pd.DataFrame(columns=REPORT_COLUMNS)
    )
    ensure_parent_directory(output_path)
    ensure_parent_directory(report_path)
    odds.to_csv(output_path, index=False)
    report.to_csv(report_path, index=False)
    return OddsPortalAsianHandicapParseResult(odds.drop_duplicates(ignore_index=True), report)
