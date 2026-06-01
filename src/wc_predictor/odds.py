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
    method: str = "proportional",
    suspicious_low: float = 1.0,
    suspicious_high: float = 1.20,
) -> MarginRemovalResult:
    """Convert decimal odds to fair probabilities for one complete market."""

    raw = decimal_odds_to_implied_probabilities(decimal_odds)
    return remove_margin(raw, method, suspicious_low, suspicious_high)


def process_bookmaker_odds(
    odds: pd.DataFrame,
    margin_method: str = "proportional",
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
            output.update(zip(FAIR_COLUMNS[market_name], result.fair_probabilities, strict=True))
            output[f"overround_{market_name}"] = result.overround
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
    margin_method: str = "proportional",
) -> pd.DataFrame:
    """Remove margin from long-format correct-score odds for each bookmaker.

    The resulting probabilities are ready for a future direct-scoreline model or
    configurable market/Poisson blend. Version 1 does not blend them automatically.
    """

    required = {"match_id", "bookmaker", "score_a", "score_b", "decimal_odds"}
    missing = required - set(correct_score_odds.columns)
    if missing:
        raise ValueError(f"Correct-score odds are missing required columns: {sorted(missing)}")
    frames: list[pd.DataFrame] = []
    for (match_id, bookmaker), group in correct_score_odds.groupby(["match_id", "bookmaker"], sort=False):
        fair = fair_probabilities_from_decimal_odds(group["decimal_odds"].tolist(), margin_method)
        processed = group.copy()
        processed["raw_implied_probability"] = fair.raw_probabilities
        processed["fair_score_probability"] = fair.fair_probabilities
        processed["market_overround"] = fair.overround
        processed["warnings"] = "; ".join(fair.warnings)
        frames.append(processed)
    if not frames:
        return pd.DataFrame(columns=[*correct_score_odds.columns, "raw_implied_probability", "fair_score_probability", "market_overround", "warnings"])
    return pd.concat(frames, ignore_index=True)
