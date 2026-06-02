"""Parse manually pasted OddsPortal correct-score text into model-ready rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from collections.abc import Iterable

import pandas as pd

from wc_predictor.utils import ensure_parent_directory

DEFAULT_INPUT_FOLDER = Path("data/raw/oddsportal_pastes")
DEFAULT_OUTPUT_PATH = Path("data/raw/world_cup_correct_score_odds.csv")
DEFAULT_REPORT_PATH = Path("data/processed/oddsportal_correct_score_parse_report.csv")

OUTPUT_COLUMNS = [
    "match_id",
    "bookmaker",
    "score_a",
    "score_b",
    "decimal_odds",
    "market_kind",
    "source_quality",
    "odds_timestamp",
    "odds_source_url",
    "has_other_bucket",
    "notes",
]
REPORT_COLUMNS = [
    "source_file",
    "match_id",
    "scoreline",
    "bookmakers_found",
    "odds_rows_found",
    "warnings",
]
COMMON_SCORELINES = {
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 2),
}

SUSPICIOUS_ODDS_LOW = 1.05
SUSPICIOUS_ODDS_HIGH = 500.0

_SCORELINE_PATTERN = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")
_ODDS_PATTERN = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*$")
_FILENAME_PATTERN = re.compile(r"^(?P<match_id>.+)_correct_score\.txt$", re.IGNORECASE)
_OTHER_BUCKET_PATTERN = re.compile(r"^(?:any\s+)?other(?:\s+score(?:line)?)?$", re.IGNORECASE)
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


def _append_warning(warnings: list[str], warning: str) -> None:
    """Append a warning once while preserving its first-seen order."""

    if warning and warning not in warnings:
        warnings.append(warning)


def _scoreline_label(scoreline: tuple[int, int]) -> str:
    return f"{scoreline[0]}-{scoreline[1]}"


def _is_noise(line: str) -> bool:
    normalised = " ".join(line.lower().split())
    return normalised == "odds" or any(fragment in normalised for fragment in _NOISE_FRAGMENTS)


def _looks_like_bookmaker(line: str) -> bool:
    """Return whether an unrecognised line is plausible as a bookmaker label."""

    if not line or _is_noise(line) or _OTHER_BUCKET_PATTERN.fullmatch(line):
        return False
    if re.fullmatch(r"\d+(?:[.,]\d+)?%", line):
        return False
    if _SCORELINE_PATTERN.fullmatch(line) or _ODDS_PATTERN.fullmatch(line) or line == "-":
        return False
    if any(character in line for character in "?!"):
        return False
    return len(line.split()) <= 4


@dataclass
class _ParsedScoreline:
    score_a: int
    score_b: int
    bookmaker_labels: set[str] = field(default_factory=set)
    pending_bookmakers: list[str] = field(default_factory=list)
    odds_rows: list[tuple[str, float]] = field(default_factory=list)
    exact_odds_rows: set[tuple[str, float]] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    @property
    def scoreline(self) -> tuple[int, int]:
        return self.score_a, self.score_b

    def add_bookmaker(self, bookmaker: str) -> None:
        self.bookmaker_labels.add(bookmaker)
        if self.pending_bookmakers and self.pending_bookmakers[-1] == bookmaker:
            return
        self.pending_bookmakers.append(bookmaker)

    def add_missing_odds(self) -> None:
        if not self.pending_bookmakers:
            _append_warning(self.warnings, "missing_odds_without_bookmaker")
            return
        bookmaker = self.pending_bookmakers.pop(0)
        _append_warning(self.warnings, f"missing_odds:{bookmaker}")

    def add_odds(self, decimal_odds: float) -> None:
        if not self.pending_bookmakers:
            # The initial block-level count and displayed best price land here.
            return
        bookmaker = self.pending_bookmakers.pop(0)
        if decimal_odds <= 1.0:
            _append_warning(self.warnings, f"invalid_odds:{bookmaker}={decimal_odds:g}")
            return
        if decimal_odds < SUSPICIOUS_ODDS_LOW or decimal_odds > SUSPICIOUS_ODDS_HIGH:
            _append_warning(self.warnings, f"suspicious_odds:{bookmaker}={decimal_odds:g}")
        exact_row = bookmaker, decimal_odds
        if exact_row in self.exact_odds_rows:
            _append_warning(self.warnings, f"deduplicated_identical_row:{bookmaker}")
            return
        previous_values = [odds for row_bookmaker, odds in self.odds_rows if row_bookmaker == bookmaker]
        if previous_values:
            _append_warning(
                self.warnings,
                f"conflicting_duplicate_odds:{bookmaker}={previous_values[-1]:g}/{decimal_odds:g}",
            )
        self.odds_rows.append(exact_row)
        self.exact_odds_rows.add(exact_row)

    def finish(self) -> None:
        if self.pending_bookmakers:
            pending = ",".join(dict.fromkeys(self.pending_bookmakers))
            _append_warning(self.warnings, f"bookmaker_labels_without_odds:{pending}")
            self.pending_bookmakers.clear()


@dataclass(frozen=True)
class OddsPortalParseResult:
    """Model-ready correct-score rows and scoreline-level parser diagnostics."""

    odds: pd.DataFrame
    report: pd.DataFrame


def infer_match_id_from_filename(path: str | Path) -> str:
    """Infer the match ID prefix from an ``*_correct_score.txt`` filename."""

    name = Path(path).name
    match = _FILENAME_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"OddsPortal paste filename must end with '_correct_score.txt': {name}")
    return match.group("match_id")


def _parse_blocks(text: str) -> tuple[list[_ParsedScoreline], bool]:
    """Extract scoreline blocks while leaving summary-only odds unpaired."""

    blocks: list[_ParsedScoreline] = []
    current: _ParsedScoreline | None = None
    has_other_bucket = False
    inside_best_value_footer = False

    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        scoreline_match = _SCORELINE_PATTERN.fullmatch(line)
        if scoreline_match:
            if current is not None:
                current.finish()
            current = _ParsedScoreline(int(scoreline_match.group(1)), int(scoreline_match.group(2)))
            blocks.append(current)
            inside_best_value_footer = False
            continue
        if _OTHER_BUCKET_PATTERN.fullmatch(line):
            if current is not None:
                current.finish()
            current = None
            has_other_bucket = True
            continue
        if current is None:
            continue
        if "looking for the best value" in line.lower() or "is offering the highest payout" in line.lower():
            inside_best_value_footer = True
            continue
        if inside_best_value_footer or _is_noise(line):
            continue
        if line == "-":
            current.add_missing_odds()
            continue
        odds_match = _ODDS_PATTERN.fullmatch(line)
        if odds_match:
            current.add_odds(float(odds_match.group(1).replace(",", ".")))
            continue
        if _looks_like_bookmaker(line):
            current.add_bookmaker(line)

    if current is not None:
        current.finish()
    return blocks, has_other_bucket


def _apply_quality_gates(blocks: list[_ParsedScoreline]) -> None:
    """Attach match- and scoreline-level coverage warnings."""

    parsed_scorelines = {block.scoreline for block in blocks}
    all_bookmakers = {bookmaker for block in blocks for bookmaker, _ in block.odds_rows}
    match_warnings: list[str] = []
    if len(parsed_scorelines) < 10:
        match_warnings.append(f"match_fewer_than_10_scorelines:{len(parsed_scorelines)}")
    missing_common = sorted(COMMON_SCORELINES - parsed_scorelines)
    if missing_common:
        labels = ",".join(_scoreline_label(scoreline) for scoreline in missing_common)
        match_warnings.append(f"missing_common_scorelines:{labels}")
    if len(all_bookmakers) == 1:
        match_warnings.append("match_only_one_bookmaker")

    for block in blocks:
        if len(block.odds_rows) < 2:
            _append_warning(block.warnings, f"scoreline_fewer_than_2_bookmaker_odds:{len(block.odds_rows)}")
        for warning in match_warnings:
            _append_warning(block.warnings, warning)


def parse_oddsportal_correct_score_text(
    text: str,
    match_id: str,
    *,
    source_file: str = "",
    odds_timestamp: str = "",
    odds_source_url: str = "",
    has_other_bucket: bool = False,
    notes: str = "",
) -> OddsPortalParseResult:
    """Parse one manually copied OddsPortal correct-score market."""

    blocks, detected_other_bucket = _parse_blocks(text)
    _apply_quality_gates(blocks)
    resolved_other_bucket = bool(has_other_bucket or detected_other_bucket)
    odds_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    for block in blocks:
        for bookmaker, decimal_odds in block.odds_rows:
            odds_rows.append(
                {
                    "match_id": match_id,
                    "bookmaker": bookmaker,
                    "score_a": block.score_a,
                    "score_b": block.score_b,
                    "decimal_odds": decimal_odds,
                    "market_kind": "coherent_bookmaker",
                    "source_quality": "oddsportal_paste",
                    "odds_timestamp": odds_timestamp,
                    "odds_source_url": odds_source_url,
                    "has_other_bucket": resolved_other_bucket,
                    "notes": notes,
                }
            )
        report_rows.append(
            {
                "source_file": source_file,
                "match_id": match_id,
                "scoreline": _scoreline_label(block.scoreline),
                "bookmakers_found": len(block.bookmaker_labels),
                "odds_rows_found": len(block.odds_rows),
                "warnings": "; ".join(block.warnings),
            }
        )
    if not blocks:
        common = ",".join(_scoreline_label(scoreline) for scoreline in sorted(COMMON_SCORELINES))
        report_rows.append(
            {
                "source_file": source_file,
                "match_id": match_id,
                "scoreline": "",
                "bookmakers_found": 0,
                "odds_rows_found": 0,
                "warnings": (
                    "no_scorelines_found; match_fewer_than_10_scorelines:0; "
                    f"missing_common_scorelines:{common}"
                ),
            }
        )
    return OddsPortalParseResult(
        pd.DataFrame(odds_rows, columns=OUTPUT_COLUMNS).drop_duplicates(ignore_index=True),
        pd.DataFrame(report_rows, columns=REPORT_COLUMNS),
    )


def parse_oddsportal_correct_score_folder(
    input_folder: str | Path = DEFAULT_INPUT_FOLDER,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    *,
    odds_timestamp: str = "",
    odds_source_url: str = "",
    has_other_bucket: bool = False,
    notes: str = "",
    match_ids: Iterable[str] | None = None,
) -> OddsPortalParseResult:
    """Parse all named paste files in a folder and export model-ready CSVs."""

    input_folder = Path(input_folder)
    output_path = Path(output_path)
    report_path = Path(report_path)
    input_folder.mkdir(parents=True, exist_ok=True)
    results: list[OddsPortalParseResult] = []
    selected_match_ids = {str(match_id) for match_id in match_ids} if match_ids is not None else None
    paths = sorted(
        path
        for path in input_folder.glob("*_correct_score.txt")
        if selected_match_ids is None or infer_match_id_from_filename(path) in selected_match_ids
    )
    for path in paths:
        results.append(
            parse_oddsportal_correct_score_text(
                path.read_text(encoding="utf-8-sig"),
                infer_match_id_from_filename(path),
                source_file=path.name,
                odds_timestamp=odds_timestamp,
                odds_source_url=odds_source_url,
                has_other_bucket=has_other_bucket,
                notes=notes,
            )
        )
    odds = (
        pd.concat([result.odds for result in results], ignore_index=True)
        if results
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    odds = odds.drop_duplicates(ignore_index=True)
    report = (
        pd.concat([result.report for result in results], ignore_index=True)
        if results
        else pd.DataFrame(columns=REPORT_COLUMNS)
    )
    ensure_parent_directory(output_path)
    ensure_parent_directory(report_path)
    odds.to_csv(output_path, index=False)
    report.to_csv(report_path, index=False)
    return OddsPortalParseResult(odds, report)


def format_oddsportal_parse_summary(
    result: OddsPortalParseResult,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> str:
    """Render a compact manual-review summary for the paste parser."""

    report = result.report
    warnings = report["warnings"].fillna("") if "warnings" in report else pd.Series(dtype=str)
    files_parsed = report["source_file"].nunique() if not report.empty else 0
    scoreline_rows = report["scoreline"].fillna("").ne("") if not report.empty else pd.Series(dtype=bool)
    scorelines_found = int(scoreline_rows.sum()) if not report.empty else 0
    missing_odds = int(warnings.str.contains(r"(?:^|; )missing_odds:", regex=True).sum())
    fewer_than_three = int((scoreline_rows & (report["odds_rows_found"] < 3)).sum()) if not report.empty else 0
    sparse_matches = (
        int(
            report.loc[
                warnings.str.contains(r"(?:^|; )match_fewer_than_10_scorelines:", regex=True),
                "match_id",
            ].nunique()
        )
        if not report.empty
        else 0
    )
    return "\n".join(
        [
            "OddsPortal Correct-Score Paste Parse",
            f"- Files parsed: {files_parsed}",
            f"- Total scorelines found: {scorelines_found}",
            f"- Total bookmaker odds rows extracted: {len(result.odds)}",
            f"- Scorelines with missing odds: {missing_odds}",
            f"- Scorelines with fewer than 3 bookmakers: {fewer_than_three}",
            f"- Matches with sparse correct-score coverage: {sparse_matches}",
            f"- Output path: {Path(output_path)}",
        ]
    )
