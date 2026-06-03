"""Diagnostic public-field scoreline strategy layer."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from wc_predictor.config import ProjectConfig, PublicStrategyConfig
from wc_predictor.optimiser import (
    GroupPredictionEvaluation,
    KnockoutPredictionEvaluation,
    KnockoutPredictionRecommendation,
    evaluate_group_prediction,
    evaluate_knockout_prediction,
)
from wc_predictor.probabilities import ScoreProbabilityMatrix
from wc_predictor.utils import result_sign

COMMON_PUBLIC_SCORELINES = ((1, 0), (2, 0), (2, 1), (1, 1), (0, 0), (0, 1), (1, 2), (0, 2))
SEVERE_WARNING_FLAGS = {
    "high_calibration_error",
    "high_tail_mass",
    "lambda_a_near_bound",
    "lambda_b_near_bound",
}
SCORE_PATTERN = re.compile(r"(?P<a>\d+)-(?P<b>\d+)")


@dataclass(frozen=True)
class PublicStrategyResult:
    """One match-level public-ranking diagnostic recommendation."""

    public_strategy_score: str
    public_strategy_reason: str
    public_strategy_ev_cost: float
    public_strategy_public_pick_share: float
    public_strategy_leverage_score: float
    public_strategy_mode: str
    estimated_most_crowded_public_score: str
    estimated_most_crowded_public_pick_share: float
    public_strategy_public_ranking_score: float
    public_strategy_exact_score_probability: float
    public_strategy_result_probability: float
    public_strategy_candidate_count: int
    friend_strategy_score: str
    friend_strategy_reason: str


def _score_label(score: tuple[int, int]) -> str:
    return f"{score[0]}-{score[1]}"


def _normalise_team(value: object) -> str:
    return " ".join(str(value).casefold().replace("&", " and ").split())


def _team_popularity(team: object, config: PublicStrategyConfig) -> float:
    normalised = _normalise_team(team)
    for name, weight in config.team_popularity_weights.items():
        if _normalise_team(name) == normalised:
            return float(weight)
    return 0.25


def _parse_score_labels(value: object, limit: int = 10) -> tuple[tuple[int, int], ...]:
    if pd.isna(value):
        return ()
    scores: list[tuple[int, int]] = []
    for match in SCORE_PATTERN.finditer(str(value)):
        score = int(match.group("a")), int(match.group("b"))
        if score not in scores:
            scores.append(score)
        if len(scores) == limit:
            break
    return tuple(scores)


def _top_probability_scorelines(matrix: ScoreProbabilityMatrix, limit: int = 10) -> tuple[tuple[int, int], ...]:
    flat_indexes = np.argsort(matrix.probabilities.ravel())[::-1]
    scores: list[tuple[int, int]] = []
    for flat_index in flat_indexes:
        score = tuple(int(value) for value in np.unravel_index(flat_index, matrix.probabilities.shape))
        scores.append(score)
        if len(scores) == limit:
            break
    return tuple(scores)


def _candidate_evaluation(
    score: tuple[int, int],
    matrix: ScoreProbabilityMatrix,
    knockout_recommendation: KnockoutPredictionRecommendation | None,
    qualifier_probabilities: dict[str, float] | None,
    config: ProjectConfig,
) -> GroupPredictionEvaluation | KnockoutPredictionEvaluation:
    if knockout_recommendation is None:
        return evaluate_group_prediction(matrix, score[0], score[1])
    qualifier = knockout_recommendation.best.predicted_qualifier
    return evaluate_knockout_prediction(
        matrix,
        score[0],
        score[1],
        qualifier,
        qualifier_probabilities or {},
        config.knockout_scoring,
    )


def _candidate_result_probability(evaluation: object, matrix: ScoreProbabilityMatrix) -> float:
    if hasattr(evaluation, "correct_result_probability"):
        return float(evaluation.correct_result_probability)
    return matrix.result_probability(result_sign(*evaluation.predicted_score))


def public_pick_distribution(
    *,
    matrix: ScoreProbabilityMatrix,
    team_a: object,
    team_b: object,
    favourite_probability: float,
    correct_score_market_top_10: object = "",
    friend_predictions: pd.DataFrame | None = None,
    config: PublicStrategyConfig | None = None,
) -> dict[tuple[int, int], float]:
    """Estimate transparent public pick shares over scorelines."""

    config = config or PublicStrategyConfig()
    weights: dict[tuple[int, int], float] = {}
    popularity_a = _team_popularity(team_a, config)
    popularity_b = _team_popularity(team_b, config)
    popularity_delta = popularity_a - popularity_b
    favourite_is_a = matrix.outcome_probabilities()["a_win"] >= matrix.outcome_probabilities()["b_win"]
    market_scores = _parse_score_labels(correct_score_market_top_10)
    market_boosts = {score: 1.0 + 0.35 * (len(market_scores) - index) / max(len(market_scores), 1) for index, score in enumerate(market_scores)}
    friend_counts: dict[tuple[int, int], int] = {}
    if friend_predictions is not None and not friend_predictions.empty:
        for _, row in friend_predictions.iterrows():
            score = int(row["predicted_team_a_goals"]), int(row["predicted_team_b_goals"])
            friend_counts[score] = friend_counts.get(score, 0) + 1
    friend_total = max(sum(friend_counts.values()), 1)

    for score in np.ndindex(matrix.probabilities.shape):
        probability = max(float(matrix.probabilities[score]), 1e-12)
        weight = probability ** 0.55
        if score in COMMON_PUBLIC_SCORELINES:
            weight *= 2.5
        if score in market_boosts:
            weight *= market_boosts[score]
        sign = result_sign(*score)
        if sign == 1:
            weight *= max(0.35, 1.0 + 0.45 * popularity_delta)
            if favourite_is_a:
                weight *= 1.0 + 0.30 * favourite_probability
        elif sign == -1:
            weight *= max(0.35, 1.0 - 0.45 * popularity_delta)
            if not favourite_is_a:
                weight *= 1.0 + 0.30 * favourite_probability
        else:
            weight *= 1.0 + 0.10 * max(popularity_a, popularity_b)
        normalised_a = _normalise_team(team_a)
        normalised_b = _normalise_team(team_b)
        if "belgium" in {normalised_a, normalised_b} and sign != 0:
            belgium_is_a = normalised_a == "belgium"
            if (belgium_is_a and sign == 1) or (not belgium_is_a and sign == -1):
                weight *= 1.35
        if score in friend_counts:
            weight *= 1.0 + config.friend_sample_weight * friend_counts[score] / friend_total
        weights[score] = weight

    total = sum(weights.values())
    return {score: weight / total for score, weight in weights.items()}


def build_public_strategy(
    *,
    match_row: pd.Series,
    matrix: ScoreProbabilityMatrix,
    recommendation: object,
    config: ProjectConfig,
    qualifier_probabilities: dict[str, float] | None = None,
    correct_score_market_matrix: ScoreProbabilityMatrix | None = None,
    friend_predictions: pd.DataFrame | None = None,
) -> PublicStrategyResult:
    """Build a diagnostic public-ranking recommendation for one match."""

    public_config = config.public_strategy
    pure_score = recommendation.best.predicted_score
    pure_label = _score_label(pure_score)
    public_distribution = public_pick_distribution(
        matrix=matrix,
        team_a=match_row["team_a"],
        team_b=match_row["team_b"],
        favourite_probability=float(match_row["favourite_probability"]),
        correct_score_market_top_10=match_row.get("correct_score_market_top_10", ""),
        friend_predictions=friend_predictions,
        config=public_config,
    )
    crowded_score, crowded_share = max(public_distribution.items(), key=lambda item: item[1])

    disabled = public_config.mode == "ev"
    flags = set(str(match_row.get("warning_flags", "")).split("; ")) - {""}
    severe_flags = sorted(flags & SEVERE_WARNING_FLAGS)
    if disabled:
        reason = "Public strategy disabled: strategy mode is ev; using pure EV recommendation."
    elif severe_flags:
        reason = "Public strategy kept pure EV recommendation because severe warning flags are present: " + ", ".join(severe_flags)
    else:
        reason = ""

    candidate_scores = set(COMMON_PUBLIC_SCORELINES)
    candidate_scores.update(evaluation.predicted_score for evaluation in (recommendation.best, *recommendation.alternatives))
    candidate_scores.update(_top_probability_scorelines(matrix, 10))
    candidate_scores.update(_parse_score_labels(match_row.get("correct_score_market_top_10", ""), 10))
    if correct_score_market_matrix is not None:
        candidate_scores.update(_top_probability_scorelines(correct_score_market_matrix, 10))
    candidate_scores = {
        score
        for score in candidate_scores
        if score[0] <= config.max_candidate_goals and score[1] <= config.max_candidate_goals
    }
    knockout_recommendation = recommendation if isinstance(recommendation, KnockoutPredictionRecommendation) else None
    pure_ev = float(recommendation.best.expected_points)
    rows: list[dict[str, object]] = []
    for score in sorted(candidate_scores):
        evaluation = _candidate_evaluation(score, matrix, knockout_recommendation, qualifier_probabilities, config)
        ev_gap = pure_ev - float(evaluation.expected_points)
        public_share = public_distribution.get(score, 0.0)
        contrarian_score = 1.0 - public_share
        exact_probability = float(evaluation.exact_score_probability)
        result_probability = _candidate_result_probability(evaluation, matrix)
        leverage_score = contrarian_score * (1.0 + 4.0 * exact_probability)
        upside_score = 10.0 * exact_probability + 2.0 * result_probability
        public_ranking_score = (
            float(evaluation.expected_points)
            + public_config.alpha * contrarian_score
            + public_config.beta * upside_score
            - public_config.gamma * max(ev_gap, 0.0)
        )
        eligible = (
            not disabled
            and not severe_flags
            and ev_gap <= public_config.effective_max_ev_loss + 1e-12
            and exact_probability >= public_config.min_exact_score_probability
            and result_probability >= public_config.min_result_probability
        )
        rows.append(
            {
                "score": score,
                "evaluation": evaluation,
                "ev_gap": ev_gap,
                "public_share": public_share,
                "contrarian_score": contrarian_score,
                "exact_probability": exact_probability,
                "result_probability": result_probability,
                "leverage_score": leverage_score,
                "public_ranking_score": public_ranking_score,
                "eligible": eligible,
            }
        )
    if reason:
        selected = next(row for row in rows if row["score"] == pure_score)
    else:
        eligible_rows = [row for row in rows if row["eligible"]]
        selected = max(eligible_rows or [row for row in rows if row["score"] == pure_score][0:1], key=lambda row: row["public_ranking_score"])
        if selected["score"] == pure_score:
            reason = "Public strategy kept pure EV recommendation: no eligible alternative beat it after EV-loss and probability safety checks."
        else:
            reason = (
                f"EV cost is {selected['ev_gap']:.2f} points versus {pure_label}, "
                "but estimated public crowding is lower and upside/leverage is attractive."
            )

    friend_strategy_score = ""
    friend_strategy_reason = "Friend predictions unavailable."
    if friend_predictions is not None and not friend_predictions.empty:
        friend_counts = friend_predictions.assign(
            score=friend_predictions["predicted_team_a_goals"].astype(int).astype(str)
            + "-"
            + friend_predictions["predicted_team_b_goals"].astype(int).astype(str)
        )["score"].value_counts()
        friend_rows = [row for row in rows if row["ev_gap"] <= public_config.effective_max_ev_loss + 1e-12]
        if friend_rows:
            friend_selected = min(friend_rows, key=lambda row: (friend_counts.get(_score_label(row["score"]), 0), row["ev_gap"]))
            friend_strategy_score = _score_label(friend_selected["score"])
            friend_strategy_reason = "Lowest friend crowding among close-EV candidates."

    return PublicStrategyResult(
        public_strategy_score=_score_label(selected["score"]),
        public_strategy_reason=reason,
        public_strategy_ev_cost=float(selected["ev_gap"]),
        public_strategy_public_pick_share=float(selected["public_share"]),
        public_strategy_leverage_score=float(selected["leverage_score"]),
        public_strategy_mode=public_config.mode,
        estimated_most_crowded_public_score=_score_label(crowded_score),
        estimated_most_crowded_public_pick_share=float(crowded_share),
        public_strategy_public_ranking_score=float(selected["public_ranking_score"]),
        public_strategy_exact_score_probability=float(selected["exact_probability"]),
        public_strategy_result_probability=float(selected["result_probability"]),
        public_strategy_candidate_count=len(rows),
        friend_strategy_score=friend_strategy_score,
        friend_strategy_reason=friend_strategy_reason,
    )
