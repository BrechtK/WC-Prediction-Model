"""Bookmaker overround diagnostics for historical World Cup market data."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_YEARS = (2014, 2018, 2022)
DEFAULT_SOURCE_ROOT = Path("cache") / "calibration_weight_search"

TWO_WAY_MAX_OVERROUND = 1.30
THREE_WAY_MAX_OVERROUND = 1.50
CORRECT_SCORE_MAX_OVERROUND = 5.00
CORRECT_SCORE_MIN_QUOTED_SCORELINES = 8
ASIAN_HANDICAP_MIN_VALID_BOOKMAKERS_PER_LINE = 2

SUMMARY_COLUMNS = [
    "sample",
    "year",
    "market_type",
    "n_observations",
    "mean_overround",
    "median_overround",
    "std_overround",
    "iqr_overround",
    "p10_overround",
    "p90_overround",
    "min_overround",
    "max_overround",
    "mean_margin_percentage_points",
]
PAIRWISE_COLUMNS = [
    "market_type",
    "available_years",
    "mean_overround_2014",
    "mean_overround_2018",
    "mean_overround_2022",
    "diff_2018_minus_2014",
    "diff_2022_minus_2018",
    "diff_2022_minus_2014",
    "max_minus_min_mean_overround",
    "max_minus_min_margin_percentage_points",
]
AUDIT_COLUMNS = ["year", "market_type", "filter_reason", "n_observations"]


@dataclass(frozen=True)
class OverroundDiagnosticResult:
    """All tables produced by the bookmaker-margin stability diagnostic."""

    observations: pd.DataFrame
    raw_summary: pd.DataFrame
    filtered_summary: pd.DataFrame
    pairwise_year_differences: pd.DataFrame
    filter_audit: pd.DataFrame


def implied_overround(odds: Iterable[object]) -> float:
    """Return sum of inverse decimal odds for one complete market row."""

    values = pd.to_numeric(pd.Series(list(odds)), errors="coerce").astype(float)
    if values.isna().any() or not np.all(np.isfinite(values)) or np.any(values <= 1.0):
        return float("nan")
    return float((1.0 / values).sum())


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _observation(
    *,
    year: int,
    market_type: str,
    match_id: object,
    bookmaker: object,
    line_key: object,
    overround: float,
    source_file: Path,
    quoted_scorelines: int | None = None,
    valid_bookmakers_on_line: int | None = None,
) -> dict[str, object]:
    return {
        "year": int(year),
        "market_type": market_type,
        "match_id": str(match_id),
        "bookmaker": str(bookmaker),
        "line_key": "" if pd.isna(line_key) else str(line_key),
        "overround": overround,
        "source_file": str(source_file),
        "quoted_scorelines": pd.NA if quoted_scorelines is None else int(quoted_scorelines),
        "valid_bookmakers_on_line": (
            pd.NA if valid_bookmakers_on_line is None else int(valid_bookmakers_on_line)
        ),
    }


def _core_market_observations(year: int, path: Path) -> list[dict[str, object]]:
    frame = _read_csv_if_present(path)
    if frame.empty:
        return []
    rows: list[dict[str, object]] = []
    market_columns = {
        "1x2": ("odds_a_win", "odds_draw", "odds_b_win"),
        "btts": ("odds_btts_yes", "odds_btts_no"),
    }
    for market_type, columns in market_columns.items():
        if not set(columns).issubset(frame.columns):
            continue
        for _, row in frame.iterrows():
            rows.append(
                _observation(
                    year=year,
                    market_type=market_type,
                    match_id=row.get("match_id", ""),
                    bookmaker=row.get("bookmaker", ""),
                    line_key="",
                    overround=implied_overround(row[list(columns)]),
                    source_file=path,
                )
            )
    return rows


def _total_goals_observations(year: int, path: Path) -> list[dict[str, object]]:
    frame = _read_csv_if_present(path)
    if frame.empty or not {"match_id", "bookmaker", "line", "odds_over", "odds_under"}.issubset(frame.columns):
        return []
    return [
        _observation(
            year=year,
            market_type="total_goals",
            match_id=row.get("match_id", ""),
            bookmaker=row.get("bookmaker", ""),
            line_key=row.get("line", ""),
            overround=implied_overround([row.get("odds_over"), row.get("odds_under")]),
            source_file=path,
        )
        for _, row in frame.iterrows()
    ]


def _asian_handicap_observations(year: int, path: Path) -> list[dict[str, object]]:
    frame = _read_csv_if_present(path)
    required = {"match_id", "bookmaker", "handicap", "odds_team_a", "odds_team_b"}
    if frame.empty or not required.issubset(frame.columns):
        return []

    prepared = frame.copy()
    prepared["_overround"] = [
        implied_overround([row.get("odds_team_a"), row.get("odds_team_b")])
        for _, row in prepared.iterrows()
    ]
    valid_pairs = prepared[np.isfinite(pd.to_numeric(prepared["_overround"], errors="coerce"))].copy()
    line_counts = (
        valid_pairs.groupby(["match_id", "handicap"], dropna=False)["bookmaker"].nunique().to_dict()
        if not valid_pairs.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    for _, row in prepared.iterrows():
        line_key = row.get("handicap", "")
        rows.append(
            _observation(
                year=year,
                market_type="asian_handicap",
                match_id=row.get("match_id", ""),
                bookmaker=row.get("bookmaker", ""),
                line_key=line_key,
                overround=float(row["_overround"]) if pd.notna(row["_overround"]) else float("nan"),
                source_file=path,
                valid_bookmakers_on_line=line_counts.get((row.get("match_id"), row.get("handicap")), 0),
            )
        )
    return rows


def _correct_score_observations(year: int, path: Path) -> list[dict[str, object]]:
    frame = _read_csv_if_present(path)
    required = {"match_id", "bookmaker", "score_a", "score_b", "decimal_odds"}
    if frame.empty or not required.issubset(frame.columns):
        return []
    rows: list[dict[str, object]] = []
    for (match_id, bookmaker), group in frame.groupby(["match_id", "bookmaker"], sort=False):
        scorelines = group[["score_a", "score_b"]].drop_duplicates()
        rows.append(
            _observation(
                year=year,
                market_type="correct_score",
                match_id=match_id,
                bookmaker=bookmaker,
                line_key="quoted_scorelines",
                overround=implied_overround(group["decimal_odds"]),
                source_file=path,
                quoted_scorelines=len(scorelines),
            )
        )
    return rows


def load_overround_observations(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    years: Iterable[int] = DEFAULT_YEARS,
) -> pd.DataFrame:
    """Load processed historical market CSVs and compute raw overround observations."""

    root = Path(source_root)
    rows: list[dict[str, object]] = []
    for year in years:
        year_dir = root / f"wc{int(year)}"
        rows.extend(_core_market_observations(int(year), year_dir / "core_odds.csv"))
        rows.extend(_total_goals_observations(int(year), year_dir / "total_goals_odds.csv"))
        rows.extend(_asian_handicap_observations(int(year), year_dir / "asian_handicap_odds.csv"))
        rows.extend(_correct_score_observations(int(year), year_dir / "correct_score_odds.csv"))
    columns = [
        "year",
        "market_type",
        "match_id",
        "bookmaker",
        "line_key",
        "overround",
        "source_file",
        "quoted_scorelines",
        "valid_bookmakers_on_line",
    ]
    return pd.DataFrame(rows, columns=columns)


def _int_or_zero(value: object) -> int:
    if pd.isna(value):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def filter_overround_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Annotate raw observations with conservative quality-filter decisions."""

    if observations.empty:
        annotated = observations.copy()
        annotated["filter_status"] = pd.Series(dtype=object)
        annotated["filter_reason"] = pd.Series(dtype=object)
        return annotated

    annotated = observations.copy()
    statuses: list[str] = []
    reasons: list[str] = []
    for _, row in annotated.iterrows():
        overround = row.get("overround", np.nan)
        market_type = str(row.get("market_type", ""))
        status = "included"
        reason = "included"
        if pd.isna(overround) or not np.isfinite(float(overround)):
            status = "excluded"
            reason = "invalid_or_missing_odds"
        elif float(overround) < 1.0:
            status = "excluded"
            reason = "underround_below_1"
        else:
            maximum = (
                THREE_WAY_MAX_OVERROUND
                if market_type == "1x2"
                else CORRECT_SCORE_MAX_OVERROUND
                if market_type == "correct_score"
                else TWO_WAY_MAX_OVERROUND
            )
            if float(overround) > maximum:
                status = "excluded"
                reason = f"overround_above_{maximum:.2f}"
            elif (
                market_type == "correct_score"
                and _int_or_zero(row.get("quoted_scorelines", 0)) < CORRECT_SCORE_MIN_QUOTED_SCORELINES
            ):
                status = "excluded"
                reason = f"correct_score_fewer_than_{CORRECT_SCORE_MIN_QUOTED_SCORELINES}_scorelines"
            elif (
                market_type == "asian_handicap"
                and _int_or_zero(row.get("valid_bookmakers_on_line", 0))
                < ASIAN_HANDICAP_MIN_VALID_BOOKMAKERS_PER_LINE
            ):
                status = "excluded"
                reason = "asian_handicap_line_fewer_than_2_valid_bookmakers"
        statuses.append(status)
        reasons.append(reason)
    annotated["filter_status"] = statuses
    annotated["filter_reason"] = reasons
    return annotated


def summarise_overrounds(observations: pd.DataFrame, *, sample: str) -> pd.DataFrame:
    """Summarise overround distributions by year and market type."""

    if observations.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    usable = observations[pd.to_numeric(observations["overround"], errors="coerce").notna()].copy()
    for (year, market_type), group in usable.groupby(["year", "market_type"], sort=True):
        values = pd.to_numeric(group["overround"], errors="coerce").dropna().astype(float)
        if values.empty:
            continue
        q25 = float(values.quantile(0.25))
        q75 = float(values.quantile(0.75))
        mean = float(values.mean())
        rows.append(
            {
                "sample": sample,
                "year": int(year),
                "market_type": market_type,
                "n_observations": int(len(values)),
                "mean_overround": mean,
                "median_overround": float(values.median()),
                "std_overround": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "iqr_overround": q75 - q25,
                "p10_overround": float(values.quantile(0.10)),
                "p90_overround": float(values.quantile(0.90)),
                "min_overround": float(values.min()),
                "max_overround": float(values.max()),
                "mean_margin_percentage_points": (mean - 1.0) * 100.0,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def pairwise_year_differences(
    filtered_summary: pd.DataFrame,
    years: Iterable[int] = DEFAULT_YEARS,
) -> pd.DataFrame:
    """Compute descriptive mean-overround differences across years."""

    if filtered_summary.empty:
        return pd.DataFrame(columns=PAIRWISE_COLUMNS)
    expected_years = [int(year) for year in years]
    rows: list[dict[str, object]] = []
    for market_type, group in filtered_summary.groupby("market_type", sort=True):
        means = {
            int(row["year"]): float(row["mean_overround"])
            for _, row in group.iterrows()
            if pd.notna(row.get("mean_overround"))
        }
        available = sorted(year for year in expected_years if year in means)
        row: dict[str, object] = {
            "market_type": market_type,
            "available_years": ";".join(str(year) for year in available),
        }
        for year in expected_years:
            row[f"mean_overround_{year}"] = means.get(year, pd.NA)
        for earlier, later in combinations(expected_years, 2):
            key = f"diff_{later}_minus_{earlier}"
            row[key] = means[later] - means[earlier] if earlier in means and later in means else pd.NA
        if means:
            spread = max(means.values()) - min(means.values())
            row["max_minus_min_mean_overround"] = spread
            row["max_minus_min_margin_percentage_points"] = spread * 100.0
        else:
            row["max_minus_min_mean_overround"] = pd.NA
            row["max_minus_min_margin_percentage_points"] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)


def filter_audit(observations: pd.DataFrame) -> pd.DataFrame:
    """Count included and excluded observations by year, market and reason."""

    if observations.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    audit = (
        observations.groupby(["year", "market_type", "filter_reason"], dropna=False)
        .size()
        .reset_index(name="n_observations")
        .sort_values(["year", "market_type", "filter_reason"], kind="stable")
    )
    return audit[AUDIT_COLUMNS]


def build_overround_diagnostics(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    years: Iterable[int] = DEFAULT_YEARS,
) -> OverroundDiagnosticResult:
    """Build raw, filtered, pairwise and audit tables for overround stability."""

    year_tuple = tuple(int(year) for year in years)
    observations = filter_overround_observations(load_overround_observations(source_root, year_tuple))
    raw_summary = summarise_overrounds(observations, sample="raw")
    filtered = observations[observations["filter_status"].eq("included")]
    filtered_summary = summarise_overrounds(filtered, sample="filtered")
    return OverroundDiagnosticResult(
        observations=observations,
        raw_summary=raw_summary,
        filtered_summary=filtered_summary,
        pairwise_year_differences=pairwise_year_differences(filtered_summary, year_tuple),
        filter_audit=filter_audit(observations),
    )


def format_console_summary(result: OverroundDiagnosticResult) -> str:
    """Return a compact plain-English summary of the filtered diagnostics."""

    filtered = result.filtered_summary
    pairwise = result.pairwise_year_differences
    audit = result.filter_audit
    if filtered.empty:
        return "No filtered overround observations were available."

    lines = ["Bookmaker Margin Stability Diagnostic", ""]
    for market_type in sorted(filtered["market_type"].unique()):
        rows = filtered[filtered["market_type"].eq(market_type)].sort_values("year")
        year_bits = [
            f"{int(row.year)} mean={float(row.mean_overround):.4f} ({float(row.mean_margin_percentage_points):.2f} pp)"
            for row in rows.itertuples(index=False)
        ]
        spread = pairwise.loc[pairwise["market_type"].eq(market_type), "max_minus_min_margin_percentage_points"]
        spread_text = f"{float(spread.iloc[0]):.2f} pp" if not spread.empty and pd.notna(spread.iloc[0]) else "n/a"
        lines.append(f"- {market_type}: {'; '.join(year_bits)}; max cycle spread={spread_text}.")

    cs_rows = filtered[filtered["market_type"].eq("correct_score")]
    liquid_rows = filtered[filtered["market_type"].isin(["1x2", "btts", "total_goals", "asian_handicap"])]
    if not cs_rows.empty and not liquid_rows.empty:
        lines.append(
            "- Correct-score markets carried much higher margins than liquid two-way/three-way markets "
            f"(mean {cs_rows['mean_margin_percentage_points'].mean():.2f} pp vs "
            f"{liquid_rows['mean_margin_percentage_points'].mean():.2f} pp across year-market summaries)."
        )

    ah_audit = audit[audit["market_type"].eq("asian_handicap")]
    if not ah_audit.empty:
        excluded = int(ah_audit[~ah_audit["filter_reason"].eq("included")]["n_observations"].sum())
        total = int(ah_audit["n_observations"].sum())
        lines.append(f"- Asian handicap required the most careful filtering: {excluded} of {total} raw observations were excluded.")

    liquid_pairwise = pairwise[pairwise["market_type"].isin(["1x2", "btts", "total_goals", "asian_handicap"])]
    liquid_spread = pd.to_numeric(
        liquid_pairwise["max_minus_min_margin_percentage_points"],
        errors="coerce",
    ).dropna()
    if not liquid_spread.empty:
        lines.append(
            "- Descriptively, bookmaker margins look broadly stable for the liquid markets in this sample; "
            f"the largest filtered mean-overround spread across liquid market/year summaries is {float(liquid_spread.max()):.2f} percentage points."
        )
    lines.append("- This is descriptive and based on available historical processed market data only.")
    return "\n".join(lines)
