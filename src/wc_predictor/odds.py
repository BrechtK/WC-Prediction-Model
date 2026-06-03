"""Odds validation, fair-probability extraction, and bookmaker aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from wc_predictor.margin import MarginRemovalResult, remove_margin

MARKETS: dict[str, tuple[str, ...]] = {
    "1x2": ("odds_a_win", "odds_draw", "odds_b_win"),
    "over_under_2_5": ("odds_over_2_5", "odds_under_2_5"),
    "btts": ("odds_btts_yes", "odds_btts_no"),
    "qualification": ("odds_a_qualifies", "odds_b_qualifies"),
}

FAIR_COLUMNS: dict[str, tuple[str, ...]] = {
    "1x2": ("fair_a_win", "fair_draw", "fair_b_win"),
    "over_under_2_5": ("fair_over_2_5", "fair_under_2_5"),
    "btts": ("fair_btts_yes", "fair_btts_no"),
    "qualification": ("fair_a_qualifies", "fair_b_qualifies"),
}

RAW_COLUMNS: dict[str, tuple[str, ...]] = {
    "1x2": ("raw_a_win", "raw_draw", "raw_b_win"),
    "over_under_2_5": ("raw_over_2_5", "raw_under_2_5"),
    "btts": ("raw_btts_yes", "raw_btts_no"),
    "qualification": ("raw_a_qualifies", "raw_b_qualifies"),
}

COMMON_CORRECT_SCORELINES = {
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 2),
}
TOTAL_GOALS_OUTPUT_COLUMNS = [
    "match_id",
    "bookmaker",
    "line",
    "odds_over",
    "odds_under",
    "raw_over",
    "raw_under",
    "fair_over",
    "fair_under",
    "total_goals_overround",
    "shin_z",
    "line_kind",
    "used_for_calibration",
    "warnings",
]


def decimal_odds_to_implied_probabilities(decimal_odds: Sequence[float]) -> np.ndarray:
    """Convert valid decimal odds O_i into raw implied probabilities 1 / O_i."""

    odds = np.asarray(decimal_odds, dtype=float)
    if odds.ndim != 1 or len(odds) == 0:
        raise ValueError("Decimal odds must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(odds)) or np.any(odds <= 1.0):
        raise ValueError("Decimal odds must be finite and greater than 1")
    return 1.0 / odds


def fair_probabilities_from_decimal_odds(
    decimal_odds: Sequence[float],
    method: str = "normalised_inverse_odds",
    suspicious_low: float = 1.0,
    suspicious_high: float = 1.20,
) -> MarginRemovalResult:
    """Convert decimal odds to fair probabilities for one complete market."""

    raw = decimal_odds_to_implied_probabilities(decimal_odds)
    return remove_margin(raw, method, suspicious_low, suspicious_high)


def process_bookmaker_odds(
    odds: pd.DataFrame,
    margin_method: str = "normalised_inverse_odds",
    suspicious_low: float = 1.0,
    suspicious_high: float = 1.20,
) -> pd.DataFrame:
    """Calculate bookmaker-level fair probabilities for each available complete market."""

    required = {"match_id", "bookmaker"}
    missing = required - set(odds.columns)
    if missing:
        raise ValueError(f"Odds data is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, source_row in odds.iterrows():
        output: dict[str, object] = {
            "match_id": source_row["match_id"],
            "bookmaker": source_row["bookmaker"],
            "warnings": [],
        }
        for market_name, odds_columns in MARKETS.items():
            if not set(odds_columns).issubset(odds.columns):
                continue
            values = source_row[list(odds_columns)]
            if values.isna().all():
                continue
            if values.isna().any():
                output["warnings"].append(f"Ignored incomplete {market_name} market")
                continue
            result = fair_probabilities_from_decimal_odds(
                values.astype(float).tolist(), margin_method, suspicious_low, suspicious_high
            )
            output.update(zip(RAW_COLUMNS[market_name], result.raw_probabilities, strict=True))
            output.update(zip(FAIR_COLUMNS[market_name], result.fair_probabilities, strict=True))
            output[f"overround_{market_name}"] = result.overround
            if result.diagnostics and "shin_z" in result.diagnostics:
                output[f"{market_name}_shin_z"] = result.diagnostics["shin_z"]
            output["warnings"].extend(result.warnings)
        output["warnings"] = "; ".join(output["warnings"])
        rows.append(output)
    return pd.DataFrame(rows)


def aggregate_bookmaker_probabilities(
    bookmaker_probabilities: pd.DataFrame,
    method: str = "mean",
    bookmaker_weights: Mapping[str, float] | None = None,
    sharp_bookmaker: str | None = None,
) -> pd.DataFrame:
    """Aggregate bookmaker-level fair probabilities match by match."""

    if method not in {"mean", "median", "weighted", "sharp"}:
        raise ValueError(f"Unsupported bookmaker aggregation method: {method}")
    rows: list[dict[str, object]] = []
    for match_id, group in bookmaker_probabilities.groupby("match_id", sort=False):
        selected = group
        if method == "sharp":
            if not sharp_bookmaker:
                raise ValueError("sharp_bookmaker is required for sharp aggregation")
            selected = group[group["bookmaker"] == sharp_bookmaker]
            if selected.empty:
                raise ValueError(f"No odds found for sharp bookmaker {sharp_bookmaker!r}")
        output: dict[str, object] = {"match_id": match_id}
        for columns in FAIR_COLUMNS.values():
            if not set(columns).issubset(selected.columns):
                continue
            available = selected[["bookmaker", *columns]].dropna()
            if available.empty:
                continue
            if method == "median":
                values = available[list(columns)].median(axis=0).to_numpy(dtype=float)
            elif method == "weighted":
                if not bookmaker_weights:
                    raise ValueError("bookmaker_weights are required for weighted aggregation")
                weights = available["bookmaker"].map(bookmaker_weights).fillna(0.0).astype(float)
                if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
                    raise ValueError(f"Bookmaker weights must be finite, non-negative, and positive in total for match {match_id}")
                values = np.average(available[list(columns)].to_numpy(dtype=float), axis=0, weights=weights)
            else:
                values = available[list(columns)].mean(axis=0).to_numpy(dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values <= 0) or values.sum() <= 0:
                raise ValueError(f"Aggregated fair probabilities are invalid for match {match_id}")
            values = values / values.sum()
            output.update(zip(columns, (float(value) for value in values), strict=True))
        output["warnings"] = "; ".join(filter(None, selected["warnings"].astype(str).unique()))
        rows.append(output)

    return pd.DataFrame(rows)


def process_correct_score_odds(
    correct_score_odds: pd.DataFrame,
    margin_method: str = "normalised_inverse_odds",
) -> pd.DataFrame:
    """Remove margin from long-format correct-score odds for each bookmaker.

    The resulting probabilities are ready for optional direct-scoreline
    aggregation and configurable market/Poisson blending.
    """

    required = {"match_id", "bookmaker", "score_a", "score_b", "decimal_odds"}
    missing = required - set(correct_score_odds.columns)
    if missing:
        raise ValueError(f"Correct-score odds are missing required columns: {sorted(missing)}")
    frames: list[pd.DataFrame] = []
    for (match_id, bookmaker), group in correct_score_odds.groupby(["match_id", "bookmaker"], sort=False):
        fair = fair_probabilities_from_decimal_odds(group["decimal_odds"].tolist(), margin_method)
        processed = group.copy()
        scorelines = set(zip(group["score_a"].astype(int), group["score_b"].astype(int)))
        has_other_bucket = (
            group["has_other_bucket"].fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).any()
            if "has_other_bucket" in group
            else False
        )
        processed["raw_implied_probability"] = fair.raw_probabilities
        processed["fair_score_probability"] = fair.fair_probabilities
        processed["market_overround"] = fair.overround
        processed["correct_score_overround"] = fair.overround
        processed["shin_z"] = (fair.diagnostics or {}).get("shin_z", pd.NA)
        processed["number_of_scorelines"] = len(scorelines)
        processed["common_scoreline_coverage"] = len(scorelines & COMMON_CORRECT_SCORELINES) / len(COMMON_CORRECT_SCORELINES)
        processed["has_other_bucket"] = bool(has_other_bucket)
        processed["suspicious_overround_warning"] = "; ".join(fair.warnings)
        processed["warnings"] = "; ".join(fair.warnings)
        frames.append(processed)
    if not frames:
        return pd.DataFrame(
            columns=[
                *correct_score_odds.columns,
                "raw_implied_probability",
                "fair_score_probability",
                "market_overround",
                "correct_score_overround",
                "shin_z",
                "number_of_scorelines",
                "common_scoreline_coverage",
                "has_other_bucket",
                "suspicious_overround_warning",
                "warnings",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def total_goals_line_kind(line: float) -> str:
    """Classify totals lines without silently applying Asian settlement rules."""

    remainder = float(line) % 1
    if np.isclose(remainder, 0.5):
        return "half_goal"
    if np.isclose(remainder, 0.0):
        return "integer_asian"
    if np.isclose(remainder, 0.25) or np.isclose(remainder, 0.75):
        return "quarter_asian"
    return "unsupported"


def process_total_goals_odds(
    total_goals_odds: pd.DataFrame,
    margin_method: str = "normalised_inverse_odds",
    suspicious_low: float = 1.0,
    suspicious_high: float = 1.20,
) -> pd.DataFrame:
    """Remove margin within each bookmaker's two-way totals line."""

    required = {"match_id", "bookmaker", "line", "odds_over", "odds_under"}
    missing = required - set(total_goals_odds.columns)
    if missing:
        raise ValueError(f"Total-goals odds are missing required columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for _, source_row in total_goals_odds.iterrows():
        line = float(source_row["line"])
        fair = fair_probabilities_from_decimal_odds(
            [float(source_row["odds_over"]), float(source_row["odds_under"])],
            margin_method,
            suspicious_low,
            suspicious_high,
        )
        kind = total_goals_line_kind(line)
        rows.append(
            {
                **source_row.to_dict(),
                "line": line,
                "raw_over": float(fair.raw_probabilities[0]),
                "raw_under": float(fair.raw_probabilities[1]),
                "fair_over": float(fair.fair_probabilities[0]),
                "fair_under": float(fair.fair_probabilities[1]),
                "total_goals_overround": fair.overround,
                "shin_z": (fair.diagnostics or {}).get("shin_z", pd.NA),
                "line_kind": kind,
                "used_for_calibration": kind == "half_goal",
                "warnings": "; ".join(fair.warnings),
            }
        )
    if not rows:
        return pd.DataFrame(columns=TOTAL_GOALS_OUTPUT_COLUMNS)
    return pd.DataFrame(rows)


def aggregate_total_goals_probabilities(processed_total_goals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fair over probabilities across bookmakers for each totals line."""

    if processed_total_goals.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "line",
                "fair_over",
                "fair_under",
                "bookmakers_count",
                "average_total_goals_overround",
                "line_kind",
                "used_for_calibration",
                "warnings",
            ]
        )
    rows: list[dict[str, object]] = []
    for (match_id, line), group in processed_total_goals.groupby(["match_id", "line"], sort=True):
        fair_over = float(group["fair_over"].mean())
        rows.append(
            {
                "match_id": match_id,
                "line": float(line),
                "fair_over": fair_over,
                "fair_under": 1.0 - fair_over,
                "bookmakers_count": int(group["bookmaker"].nunique()),
                "average_total_goals_overround": float(group["total_goals_overround"].mean()),
                "line_kind": total_goals_line_kind(float(line)),
                "used_for_calibration": bool(group["used_for_calibration"].all()),
                "warnings": "; ".join(filter(None, group["warnings"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)
