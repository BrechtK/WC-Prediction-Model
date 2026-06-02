"""Optional correct-score market matrices and Poisson-market blending."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc_predictor.probabilities import ScoreProbabilityMatrix

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
VALID_CORRECT_SCORE_AGGREGATION_METHODS = {
    "auto",
    "mean",
    "median",
    "trimmed_mean",
    "winsorized_mean",
    "reliability_weighted_mean",
}
DEFAULT_OUTLIER_Z_THRESHOLD = 3.0
OUTLIER_RELIABILITY_MULTIPLIER = 0.25
TRIM_PROPORTION = 0.20


@dataclass(frozen=True)
class CorrectScoreMarketAggregation:
    """Aggregated direct-market matrix and transparent source diagnostics."""

    matrix: ScoreProbabilityMatrix
    aggregation_method: str
    diagnostics: dict[str, object]
    bookmaker_diagnostics: pd.DataFrame
    observations: pd.DataFrame


def _scoreline_label(scoreline: tuple[int, int]) -> str:
    return f"{scoreline[0]}-{scoreline[1]}"


def summarise_correct_score_coverage(correct_score_rows: pd.DataFrame) -> dict[str, object]:
    """Summarise direct-market coverage without changing blend decisions."""

    if correct_score_rows.empty:
        return {
            "correct_score_scorelines_count": 0,
            "correct_score_bookmakers_count": 0,
            "correct_score_sparse_warning": "",
        }
    required = {"bookmaker", "score_a", "score_b"}
    missing = required - set(correct_score_rows.columns)
    if missing:
        raise ValueError(f"Correct-score odds are missing coverage columns: {sorted(missing)}")
    rows = correct_score_rows.copy()
    rows["score_a"] = pd.to_numeric(rows["score_a"], errors="coerce")
    rows["score_b"] = pd.to_numeric(rows["score_b"], errors="coerce")
    valid = rows.dropna(subset=["bookmaker", "score_a", "score_b"])
    scorelines = {
        (int(row["score_a"]), int(row["score_b"]))
        for _, row in valid.iterrows()
    }
    bookmakers = {str(value) for value in valid["bookmaker"].dropna()}
    warnings: list[str] = []
    if len(scorelines) < 10:
        warnings.append(f"fewer_than_10_scorelines:{len(scorelines)}")
    missing_common = sorted(COMMON_SCORELINES - scorelines)
    if missing_common:
        warnings.append("missing_common_scorelines:" + ",".join(_scoreline_label(scoreline) for scoreline in missing_common))
    scoreline_bookmakers = valid.groupby(["score_a", "score_b"])["bookmaker"].nunique()
    sparse_scorelines = [
        _scoreline_label((int(score_a), int(score_b)))
        for (score_a, score_b), count in scoreline_bookmakers.items()
        if count < 2
    ]
    if sparse_scorelines:
        warnings.append("scorelines_with_fewer_than_2_bookmakers:" + ",".join(sparse_scorelines))
    if len(bookmakers) == 1:
        warnings.append("only_one_correct_score_bookmaker")
    return {
        "correct_score_scorelines_count": len(scorelines),
        "correct_score_bookmakers_count": len(bookmakers),
        "correct_score_sparse_warning": "; ".join(warnings),
    }


def _resolve_correct_score_aggregation_method(method: str, number_of_bookmakers: int) -> str:
    if method not in VALID_CORRECT_SCORE_AGGREGATION_METHODS:
        raise ValueError(f"Unsupported correct-score aggregation method: {method}")
    if method != "auto":
        return method
    if number_of_bookmakers >= 5:
        return "winsorized_mean"
    if number_of_bookmakers >= 3:
        return "median"
    return "mean"


def _normalise_correct_score_rows(
    processed_correct_score_odds: pd.DataFrame,
    match_id: str,
    max_goals: int,
) -> pd.DataFrame:
    """Validate and consolidate bookmaker-level fair score probabilities."""

    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")
    required = {"match_id", "bookmaker", "score_a", "score_b", "fair_score_probability"}
    missing = required - set(processed_correct_score_odds.columns)
    if missing:
        raise ValueError(f"Processed correct-score odds are missing required columns: {sorted(missing)}")
    rows = processed_correct_score_odds[
        processed_correct_score_odds["match_id"].astype(str) == str(match_id)
    ].copy()
    if rows.empty:
        raise ValueError(f"No correct-score odds found for match {match_id!r}")
    for column in ("score_a", "score_b"):
        numeric = pd.to_numeric(rows[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any() or not np.all(np.equal(numeric, np.floor(numeric))):
            raise ValueError("Correct-score goals must be non-negative integers")
        rows[column] = numeric.astype(int)
    probabilities = pd.to_numeric(rows["fair_score_probability"], errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Correct-score probabilities must be finite and non-negative")
    rows["fair_score_probability"] = probabilities

    per_bookmaker = (
        rows.groupby(["bookmaker", "score_a", "score_b"], as_index=False)["fair_score_probability"].sum()
    )
    bookmaker_totals = per_bookmaker.groupby("bookmaker")["fair_score_probability"].sum()
    if not np.allclose(bookmaker_totals.to_numpy(dtype=float), 1.0):
        raise ValueError("Each processed bookmaker correct-score market must sum to one")
    return per_bookmaker


def _source_bookmaker_diagnostics(
    processed_correct_score_odds: pd.DataFrame,
    match_id: str,
) -> pd.DataFrame:
    """Summarise coherent market quality once per bookmaker."""

    match_rows = processed_correct_score_odds[
        processed_correct_score_odds["match_id"].astype(str) == str(match_id)
    ].copy()
    rows: list[dict[str, object]] = []
    for bookmaker, group in match_rows.groupby("bookmaker", sort=False):
        scorelines = {
            (int(score_a), int(score_b))
            for score_a, score_b in zip(group["score_a"], group["score_b"], strict=True)
        }
        if "correct_score_overround" in group:
            overround = float(group["correct_score_overround"].iloc[0])
        elif "market_overround" in group:
            overround = float(group["market_overround"].iloc[0])
        elif "raw_implied_probability" in group:
            overround = float(group["raw_implied_probability"].sum())
        else:
            overround = float("nan")
        has_other_bucket = (
            group["has_other_bucket"].fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).any()
            if "has_other_bucket" in group
            else False
        )
        warning = (
            "; ".join(
                sorted(
                    {
                        str(value).strip()
                        for value in group["suspicious_overround_warning"].dropna()
                        if str(value).strip()
                    }
                )
            )
            if "suspicious_overround_warning" in group
            else ""
        )
        common_coverage = len(scorelines & COMMON_SCORELINES) / len(COMMON_SCORELINES)
        overround_penalty = 1.0 + abs(overround - 1.0) if np.isfinite(overround) else 2.0
        reliability_weight = max(common_coverage, 1.0 / len(COMMON_SCORELINES)) / overround_penalty
        rows.append(
            {
                "bookmaker": str(bookmaker),
                "correct_score_overround": overround,
                "number_of_scorelines": len(scorelines),
                "common_scoreline_coverage": common_coverage,
                "has_other_bucket": bool(has_other_bucket),
                "suspicious_overround_warning": warning,
                "reliability_weight": reliability_weight,
            }
        )
    return pd.DataFrame(rows)


def _annotate_scoreline_outliers(
    per_bookmaker: pd.DataFrame,
    bookmaker_diagnostics: pd.DataFrame,
    outlier_z_threshold: float,
) -> pd.DataFrame:
    """Flag robust log-probability outliers and prepare robust alternatives."""

    if outlier_z_threshold <= 0:
        raise ValueError("Correct-score outlier z threshold must be positive")
    reliability = bookmaker_diagnostics.set_index("bookmaker")["reliability_weight"].to_dict()
    frames: list[pd.DataFrame] = []
    for _, group in per_bookmaker.groupby(["score_a", "score_b"], sort=False):
        annotated = group.copy()
        probabilities = annotated["fair_score_probability"].to_numpy(dtype=float)
        log_probabilities = np.log(probabilities)
        median = float(np.median(log_probabilities))
        deviations = np.abs(log_probabilities - median)
        mad = float(np.median(deviations))
        if len(annotated) < 3:
            robust_z_scores = np.zeros(len(annotated), dtype=float)
            outliers = np.zeros(len(annotated), dtype=bool)
            winsorized_logs = log_probabilities
        elif mad <= np.finfo(float).eps:
            outliers = deviations > np.finfo(float).eps
            robust_z_scores = np.where(outliers, np.inf, 0.0)
            winsorized_logs = np.where(outliers, median, log_probabilities)
        else:
            robust_scale = 1.4826 * mad
            robust_z_scores = deviations / robust_scale
            outliers = robust_z_scores > outlier_z_threshold
            winsorized_logs = np.clip(
                log_probabilities,
                median - outlier_z_threshold * robust_scale,
                median + outlier_z_threshold * robust_scale,
            )
        annotated["log_fair_score_probability"] = log_probabilities
        annotated["robust_z_score"] = robust_z_scores
        annotated["is_scoreline_outlier"] = outliers
        annotated["winsorized_score_probability"] = np.exp(winsorized_logs)
        annotated["reliability_weight"] = annotated["bookmaker"].astype(str).map(reliability).astype(float)
        annotated.loc[annotated["is_scoreline_outlier"], "reliability_weight"] *= OUTLIER_RELIABILITY_MULTIPLIER
        frames.append(annotated)
    return pd.concat(frames, ignore_index=True)


def _trimmed_mean(values: np.ndarray) -> float:
    """Return a symmetric 20% trimmed mean where the sample supports trimming."""

    ordered = np.sort(values)
    trim_count = int(np.floor(len(ordered) * TRIM_PROPORTION))
    if trim_count == 0 or trim_count * 2 >= len(ordered):
        return float(np.mean(ordered))
    return float(np.mean(ordered[trim_count:-trim_count]))


def _aggregate_scoreline(group: pd.DataFrame, method: str) -> float:
    probabilities = group["fair_score_probability"].to_numpy(dtype=float)
    if method == "mean":
        return float(np.mean(probabilities))
    if method == "median":
        return float(np.median(probabilities))
    if method == "trimmed_mean":
        return _trimmed_mean(probabilities)
    if method == "winsorized_mean":
        return float(np.mean(group["winsorized_score_probability"].to_numpy(dtype=float)))
    if method == "reliability_weighted_mean":
        return float(
            np.average(
                probabilities,
                weights=group["reliability_weight"].to_numpy(dtype=float),
            )
        )
    raise ValueError(f"Unsupported correct-score aggregation method: {method}")


def _scoreline_coverage_warning(
    per_bookmaker: pd.DataFrame,
    bookmaker_diagnostics: pd.DataFrame,
    out_of_grid_scorelines: set[tuple[int, int]],
) -> str:
    warnings: list[str] = []
    scorelines = {
        (int(score_a), int(score_b))
        for score_a, score_b in zip(per_bookmaker["score_a"], per_bookmaker["score_b"], strict=True)
    }
    missing_common = sorted(COMMON_SCORELINES - scorelines)
    if missing_common:
        warnings.append("missing_common_scorelines:" + ",".join(_scoreline_label(scoreline) for scoreline in missing_common))
    incomplete_bookmakers = bookmaker_diagnostics.loc[
        bookmaker_diagnostics["common_scoreline_coverage"] < 1.0,
        "bookmaker",
    ].tolist()
    if incomplete_bookmakers:
        warnings.append("bookmakers_missing_common_scorelines:" + ",".join(incomplete_bookmakers))
    scoreline_counts = per_bookmaker.groupby(["score_a", "score_b"])["bookmaker"].nunique()
    sparse_scorelines = [
        _scoreline_label((int(score_a), int(score_b)))
        for (score_a, score_b), count in scoreline_counts.items()
        if count < 2
    ]
    if sparse_scorelines:
        warnings.append("scorelines_with_fewer_than_2_bookmakers:" + ",".join(sparse_scorelines))
    if out_of_grid_scorelines:
        warnings.append(
            "out_of_grid_scorelines:"
            + ",".join(_scoreline_label(scoreline) for scoreline in sorted(out_of_grid_scorelines))
        )
    return "; ".join(warnings)


def _format_top_outlier_examples(observations: pd.DataFrame, limit: int = 5) -> str:
    outliers = observations[observations["is_scoreline_outlier"]].copy()
    if outliers.empty:
        return ""
    outliers = outliers.sort_values(
        ["robust_z_score", "bookmaker", "score_a", "score_b"],
        ascending=[False, True, True, True],
    )
    return "; ".join(
        (
            f"{row['bookmaker']}:{int(row['score_a'])}-{int(row['score_b'])} "
            f"p={row['fair_score_probability']:.6f} z={row['robust_z_score']:.2f}"
        )
        for _, row in outliers.head(limit).iterrows()
    )


def format_correct_score_bookmaker_diagnostics(bookmaker_diagnostics: pd.DataFrame) -> str:
    """Format bookmaker-level direct-market quality diagnostics for exports."""

    return "; ".join(
        (
            f"{row['bookmaker']}: overround={row['correct_score_overround']:.4f}, "
            f"scorelines={int(row['number_of_scorelines'])}, "
            f"common_coverage={row['common_scoreline_coverage']:.1%}, "
            f"other_bucket={'yes' if row['has_other_bucket'] else 'no'}, "
            f"warning={row['suspicious_overround_warning'] or 'none'}"
        )
        for _, row in bookmaker_diagnostics.iterrows()
    )


def aggregate_correct_score_market(
    processed_correct_score_odds: pd.DataFrame,
    match_id: str,
    max_goals: int,
    method: str = "auto",
    outlier_z_threshold: float = DEFAULT_OUTLIER_Z_THRESHOLD,
) -> CorrectScoreMarketAggregation:
    """Aggregate fair bookmaker score probabilities without averaging odds."""

    per_bookmaker = _normalise_correct_score_rows(processed_correct_score_odds, match_id, max_goals)
    bookmaker_diagnostics = _source_bookmaker_diagnostics(processed_correct_score_odds, match_id)
    resolved_method = _resolve_correct_score_aggregation_method(method, len(bookmaker_diagnostics))
    observations = _annotate_scoreline_outliers(per_bookmaker, bookmaker_diagnostics, outlier_z_threshold)
    represented = observations[
        (observations["score_a"] <= max_goals) & (observations["score_b"] <= max_goals)
    ]
    if represented.empty:
        raise ValueError(f"No correct-score odds fall within configured max_goals={max_goals}")
    out_of_grid_scorelines = {
        (int(score_a), int(score_b))
        for score_a, score_b in zip(
            observations.loc[~observations.index.isin(represented.index), "score_a"],
            observations.loc[~observations.index.isin(represented.index), "score_b"],
            strict=True,
        )
    }
    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for (score_a, score_b), group in represented.groupby(["score_a", "score_b"], sort=False):
        matrix[int(score_a), int(score_b)] = _aggregate_scoreline(group, resolved_method)
    matrix = matrix / matrix.sum()
    overrounds = bookmaker_diagnostics["correct_score_overround"].dropna().to_numpy(dtype=float)
    diagnostics = {
        "correct_score_aggregation_method": resolved_method,
        "number_of_correct_score_bookmakers": len(bookmaker_diagnostics),
        "average_correct_score_overround": float(np.mean(overrounds)) if len(overrounds) else pd.NA,
        "max_correct_score_overround": float(np.max(overrounds)) if len(overrounds) else pd.NA,
        "outlier_count": int(observations["is_scoreline_outlier"].sum()),
        "top_outlier_examples": _format_top_outlier_examples(observations),
        "scoreline_coverage_warning": _scoreline_coverage_warning(
            per_bookmaker,
            bookmaker_diagnostics,
            out_of_grid_scorelines,
        ),
        "out_of_grid_scorelines_count": len(out_of_grid_scorelines),
    }
    return CorrectScoreMarketAggregation(
        ScoreProbabilityMatrix(matrix),
        resolved_method,
        diagnostics,
        bookmaker_diagnostics,
        observations,
    )


def correct_score_market_matrix(
    processed_correct_score_odds: pd.DataFrame,
    match_id: str,
    max_goals: int,
    method: str = "auto",
    outlier_z_threshold: float = DEFAULT_OUTLIER_Z_THRESHOLD,
) -> ScoreProbabilityMatrix:
    """Return the robustly aggregated direct correct-score market matrix."""

    return aggregate_correct_score_market(
        processed_correct_score_odds,
        match_id,
        max_goals,
        method,
        outlier_z_threshold,
    ).matrix


def blend_score_matrices(
    poisson_matrix: ScoreProbabilityMatrix,
    market_matrix: ScoreProbabilityMatrix,
    poisson_weight: float,
) -> ScoreProbabilityMatrix:
    """Blend Poisson and direct correct-score market distributions."""

    if not np.isfinite(poisson_weight) or not 0 <= poisson_weight <= 1:
        raise ValueError("Correct-score Poisson weight must lie between zero and one")
    if poisson_matrix.probabilities.shape != market_matrix.probabilities.shape:
        raise ValueError("Poisson and correct-score matrices must have matching shapes")
    probabilities = (
        poisson_weight * poisson_matrix.probabilities
        + (1.0 - poisson_weight) * market_matrix.probabilities
    )
    return ScoreProbabilityMatrix(
        probabilities,
        tail_probability=poisson_matrix.tail_probability,
        lambda_a=poisson_matrix.lambda_a,
        lambda_b=poisson_matrix.lambda_b,
    )


def market_to_poisson_kl_divergence(
    poisson_matrix: ScoreProbabilityMatrix,
    market_matrix: ScoreProbabilityMatrix,
) -> float:
    """Return D_KL(P_market || P_poisson) over the represented score grid."""

    if poisson_matrix.probabilities.shape != market_matrix.probabilities.shape:
        raise ValueError("Poisson and correct-score matrices must have matching shapes")
    poisson = poisson_matrix.probabilities
    market = market_matrix.probabilities
    positive_market = market > 0
    if np.any(poisson[positive_market] <= 0):
        return float("inf")
    return float(np.sum(market[positive_market] * np.log(market[positive_market] / poisson[positive_market])))


def format_top_scorelines(matrix: ScoreProbabilityMatrix, top_n: int = 10) -> str:
    """Format the highest-probability scorelines for transparent diagnostics."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    ranked = sorted(
        (
            ((int(score_a), int(score_b)), float(matrix.probabilities[score_a, score_b]))
            for score_a, score_b in np.ndindex(matrix.probabilities.shape)
            if matrix.probabilities[score_a, score_b] > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return "; ".join(f"{score_a}-{score_b} ({probability:.4%})" for (score_a, score_b), probability in ranked[:top_n])
