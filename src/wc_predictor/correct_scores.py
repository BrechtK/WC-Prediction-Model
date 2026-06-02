"""Optional correct-score market matrices and Poisson-market blending."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wc_predictor.probabilities import ScoreProbabilityMatrix


def correct_score_market_matrix(
    processed_correct_score_odds: pd.DataFrame,
    match_id: str,
    max_goals: int,
) -> ScoreProbabilityMatrix:
    """Aggregate bookmaker-level fair correct-score prices into one score matrix."""

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
        if (numeric > max_goals).any():
            raise ValueError(f"Correct-score goals exceed configured max_goals={max_goals}")
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

    matrices = []
    for _, bookmaker_rows in per_bookmaker.groupby("bookmaker", sort=False):
        matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        for _, row in bookmaker_rows.iterrows():
            matrix[int(row["score_a"]), int(row["score_b"])] = float(row["fair_score_probability"])
        matrices.append(matrix)
    aggregate = np.mean(matrices, axis=0)
    aggregate = aggregate / aggregate.sum()
    return ScoreProbabilityMatrix(aggregate)


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
