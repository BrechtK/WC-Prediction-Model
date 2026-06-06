"""Application workflow joining market extraction, calibration, and optimisation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import time
from typing import Any

import numpy as np
import pandas as pd

from wc_predictor.calibration import CalibrationTargets, calibrate_poisson_model, poisson_over_total_probability
from wc_predictor.config import KnockoutScoringConfig, ProjectConfig
from wc_predictor.correct_scores import (
    aggregate_correct_score_market,
    blend_score_matrices,
    format_correct_score_bookmaker_diagnostics,
    format_top_scorelines,
    market_to_poisson_kl_divergence,
    summarise_correct_score_coverage,
)
from wc_predictor.friends import analyse_friend_predictions
from wc_predictor.margin import IMPLEMENTED_MARGIN_REMOVAL_METHODS
from wc_predictor.margin_diagnostics import draw_vs_decisive_diagnostics, margin_ev_diagnostics
from wc_predictor.margin_model import fit_skellam_margin_model
from wc_predictor.market_consistent import (
    MarketConsistentWeights,
    fit_market_consistent_matrix,
    select_asian_handicap_constraints,
)
from wc_predictor.odds import (
    aggregate_asian_handicap_probabilities,
    aggregate_bookmaker_probabilities,
    aggregate_total_goals_probabilities,
    process_bookmaker_odds,
    process_asian_handicap_odds,
    process_correct_score_odds,
    process_total_goals_odds,
)
from wc_predictor.optimiser import (
    GroupPredictionEvaluation,
    GroupPredictionRecommendation,
    KnockoutPredictionEvaluation,
    KnockoutPredictionRecommendation,
    evaluate_group_prediction,
    evaluate_knockout_prediction,
    optimise_group_prediction,
    optimise_knockout_prediction,
)
from wc_predictor.probabilities import ScoreProbabilityMatrix
from wc_predictor.public_strategy import build_public_strategy
from wc_predictor.scoring_rules import DEFAULT_GROUP_SCORING
from wc_predictor.score_models import (
    BivariatePoissonScoreModel,
    DixonColesScoreModel,
    Match,
    build_challenger_score_matrices,
    conservative_bivariate_covariance,
    estimate_dixon_coles_rho_from_market,
)
from wc_predictor.utils import favourite_strength_bucket, is_knockout_stage

HIGH_TAIL_MASS_THRESHOLD = 0.01
STALE_ODDS_THRESHOLD = pd.Timedelta(hours=24)
BTTS_FIT_WARNING_THRESHOLD = 0.05
STRONG_BTTS_SIGNAL_THRESHOLD = 0.55
EXTREME_FAVOURITE_PROBABILITY_THRESHOLD = 0.75
HIGH_SCORE_EV_CLUSTER_THRESHOLD = 0.10
LARGER_GRID_MAX_GOALS = 15
HIGH_CONFIDENCE_EV_GAP_THRESHOLD = 0.15
MEDIUM_CONFIDENCE_EV_GAP_THRESHOLD = 0.07


@dataclass(frozen=True)
class PredictionWorkflowResult:
    """All report-level outputs and reusable probability objects from one run."""

    bookmaker_probabilities: pd.DataFrame
    aggregated_probabilities: pd.DataFrame
    match_report: pd.DataFrame
    friend_report: pd.DataFrame
    score_matrices: dict[str, ScoreProbabilityMatrix]
    recommendations: dict[str, GroupPredictionRecommendation | KnockoutPredictionRecommendation]
    qualifier_probabilities: dict[str, dict[str, float]]
    processed_correct_score_probabilities: pd.DataFrame
    correct_score_market_matrices: dict[str, ScoreProbabilityMatrix]
    processed_total_goals_probabilities: pd.DataFrame
    aggregated_total_goals_probabilities: pd.DataFrame
    baseline_score_matrices: dict[str, ScoreProbabilityMatrix]
    challenger_score_matrices: dict[str, dict[str, ScoreProbabilityMatrix]]
    margin_method_comparison: pd.DataFrame
    final_decision_dashboard: pd.DataFrame
    processed_asian_handicap_probabilities: pd.DataFrame
    aggregated_asian_handicap_probabilities: pd.DataFrame
    margin_diagnostics: pd.DataFrame


def _optional_probability(row: pd.Series, column: str) -> float | None:
    return float(row[column]) if column in row and pd.notna(row[column]) else None


def _format_evaluations(evaluations: tuple[Any, ...]) -> str:
    formatted: list[str] = []
    for evaluation in evaluations:
        score = f"{evaluation.predicted_score[0]}-{evaluation.predicted_score[1]}"
        if hasattr(evaluation, "predicted_qualifier"):
            score += f"/{evaluation.predicted_qualifier}"
        formatted.append(f"{score} ({evaluation.expected_points:.3f})")
    return "; ".join(formatted)


def _format_alternatives(recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation) -> str:
    return _format_evaluations(recommendation.alternatives)


def _format_bookmaker_fair_probabilities(bookmaker_probabilities: pd.DataFrame, match_id: str) -> str:
    rows = bookmaker_probabilities[bookmaker_probabilities["match_id"] == match_id]
    return "; ".join(
        f"{row['bookmaker']}: {row['fair_a_win']:.4f}/{row['fair_draw']:.4f}/{row['fair_b_win']:.4f}"
        for _, row in rows.iterrows()
    )


def _format_bookmaker_raw_probabilities(bookmaker_probabilities: pd.DataFrame, match_id: str) -> str:
    rows = bookmaker_probabilities[bookmaker_probabilities["match_id"] == match_id]
    return "; ".join(
        f"{row['bookmaker']}: {row['raw_a_win']:.4f}/{row['raw_draw']:.4f}/{row['raw_b_win']:.4f} "
        f"(overround={row['overround_1x2']:.4f})"
        for _, row in rows.iterrows()
    )


def _format_top_ev_predictions(recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation) -> str:
    return _format_evaluations((recommendation.best, *recommendation.alternatives[:4]))


def _score_has_btts(score: tuple[int, int]) -> bool:
    return score[0] > 0 and score[1] > 0


def _score_label(score: tuple[int, int]) -> str:
    return f"{score[0]}-{score[1]}"


def _score_matrix_audit(matrix: ScoreProbabilityMatrix) -> dict[str, float]:
    scores_a, scores_b = np.indices(matrix.probabilities.shape)
    total_goals = scores_a + scores_b
    btts_yes = matrix.btts_yes_probability()
    audit = {
        "model_implied_btts_yes_probability": btts_yes,
        "model_implied_btts_no_probability": 1.0 - btts_yes,
        "model_probability_team_a_clean_sheet": float(matrix.probabilities[:, 0].sum()),
        "model_probability_team_b_clean_sheet": float(matrix.probabilities[0, :].sum()),
        "model_probability_no_btts": 1.0 - btts_yes,
        "model_probability_btts": btts_yes,
        "expected_total_goals": float((total_goals * matrix.probabilities).sum()),
        "expected_team_a_goals": float((scores_a * matrix.probabilities).sum()),
        "expected_team_b_goals": float((scores_b * matrix.probabilities).sum()),
        "probability_total_goals_0": float(matrix.probabilities[total_goals == 0].sum()),
        "probability_total_goals_1": float(matrix.probabilities[total_goals == 1].sum()),
        "probability_total_goals_2": float(matrix.probabilities[total_goals == 2].sum()),
        "probability_total_goals_3": float(matrix.probabilities[total_goals == 3].sum()),
        "probability_total_goals_4_plus": float(matrix.probabilities[total_goals >= 4].sum()),
    }
    for goals in range(6):
        audit[f"probability_team_a_scores_{goals}"] = (
            float(matrix.probabilities[goals, :].sum()) if goals < matrix.probabilities.shape[0] else 0.0
        )
        audit[f"probability_team_b_scores_{goals}"] = (
            float(matrix.probabilities[:, goals].sum()) if goals < matrix.probabilities.shape[1] else 0.0
        )
    audit["probability_team_a_scores_6_plus"] = float(matrix.probabilities[6:, :].sum())
    audit["probability_team_b_scores_6_plus"] = float(matrix.probabilities[:, 6:].sum())
    for goals in range(7):
        audit[f"probability_total_goals_{goals}"] = float(matrix.probabilities[total_goals == goals].sum())
    audit["probability_total_goals_7_plus"] = float(matrix.probabilities[total_goals >= 7].sum())
    for margin in range(1, 6):
        audit[f"probability_team_a_wins_by_{margin}"] = float(
            matrix.probabilities[scores_a - scores_b == margin].sum()
        )
        audit[f"probability_team_b_wins_by_{margin}"] = float(
            matrix.probabilities[scores_b - scores_a == margin].sum()
        )
    audit["probability_team_a_wins_by_6_plus"] = float(matrix.probabilities[scores_a - scores_b >= 6].sum())
    audit["probability_team_b_wins_by_6_plus"] = float(matrix.probabilities[scores_b - scores_a >= 6].sum())
    audit["probability_draw"] = float(np.trace(matrix.probabilities))
    return audit


def _group_ev_components(evaluation: GroupPredictionEvaluation) -> dict[str, float]:
    scoring = DEFAULT_GROUP_SCORING
    return {
        "participation_component": float(scoring.participation_points),
        "result_component": float(scoring.result_increment * evaluation.correct_result_probability),
        "goal_difference_component": float(
            scoring.goal_difference_increment * evaluation.correct_goal_difference_probability
        ),
        "exact_score_component": float(scoring.exact_increment * evaluation.exact_score_probability),
        "correct_result_probability": evaluation.correct_result_probability,
        "correct_goal_difference_probability": evaluation.correct_goal_difference_probability,
    }


def _knockout_ev_components(
    evaluation: KnockoutPredictionEvaluation,
    config: KnockoutScoringConfig,
) -> dict[str, float]:
    return {
        "participation_component": float(config.participation_points),
        "result_component": float(config.qualifier_points * evaluation.qualifier_probability),
        "goal_difference_component": float(config.goal_difference_points * evaluation.correct_goal_difference_probability),
        "exact_score_component": float(config.exact_score_points * evaluation.exact_score_probability),
        "correct_result_probability": evaluation.qualifier_probability,
        "correct_goal_difference_probability": evaluation.correct_goal_difference_probability,
    }


def _top_ev_evaluations(
    matrix: ScoreProbabilityMatrix,
    knockout: bool,
    qualifier_probabilities: dict[str, float] | None,
    config: ProjectConfig,
    top_n: int,
) -> tuple[GroupPredictionEvaluation | KnockoutPredictionEvaluation, ...]:
    if knockout:
        evaluations = [
            evaluate_knockout_prediction(
                matrix,
                pred_a,
                pred_b,
                qualifier,
                qualifier_probabilities or {},
                config.knockout_scoring,
            )
            for pred_a in range(config.max_candidate_goals + 1)
            for pred_b in range(config.max_candidate_goals + 1)
            for qualifier in (qualifier_probabilities or {})
        ]
        evaluations.sort(key=lambda item: (-item.expected_points, item.predicted_score, item.predicted_qualifier))
    else:
        evaluations = [
            evaluate_group_prediction(matrix, pred_a, pred_b)
            for pred_a in range(config.max_candidate_goals + 1)
            for pred_b in range(config.max_candidate_goals + 1)
        ]
        evaluations.sort(key=lambda item: (-item.expected_points, item.predicted_score))
    return tuple(evaluations[:top_n])


def _ev_decomposition_records_from_evaluations(
    match_id: str,
    evaluations: tuple[GroupPredictionEvaluation | KnockoutPredictionEvaluation, ...],
    config: ProjectConfig,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for rank, evaluation in enumerate(evaluations, start=1):
        if isinstance(evaluation, KnockoutPredictionEvaluation):
            components = _knockout_ev_components(evaluation, config.knockout_scoring)
        else:
            components = _group_ev_components(evaluation)
        total_from_components = (
            components["participation_component"]
            + components["result_component"]
            + components["goal_difference_component"]
            + components["exact_score_component"]
        )
        records.append(
            {
                "match_id": match_id,
                "rank": rank,
                "predicted_score": _score_label(evaluation.predicted_score),
                "total_expected_points": evaluation.expected_points,
                **components,
                "total_expected_points_from_components": total_from_components,
                "exact_score_probability": evaluation.exact_score_probability,
                "btts_status_of_prediction": "btts_yes" if _score_has_btts(evaluation.predicted_score) else "btts_no",
            }
        )
    return records


def _ev_decomposition_records(
    match_id: str,
    recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation,
    config: ProjectConfig,
) -> list[dict[str, object]]:
    return _ev_decomposition_records_from_evaluations(
        match_id,
        (recommendation.best, *recommendation.alternatives[:4]),
        config,
    )


def _format_ev_decomposition(records: list[dict[str, object]]) -> str:
    parts: list[str] = []
    for record in records:
        parts.append(
            f"{record['predicted_score']}: EV={float(record['total_expected_points']):.3f} "
            f"participation={float(record['participation_component']):.3f} "
            f"result={float(record['result_component']):.3f} "
            f"goal_difference={float(record['goal_difference_component']):.3f} "
            f"exact_score={float(record['exact_score_component']):.3f} "
            f"P_exact={float(record['exact_score_probability']):.4f} "
            f"P_result={float(record['correct_result_probability']):.4f} "
            f"P_goal_difference={float(record['correct_goal_difference_probability']):.4f} "
            f"{record['btts_status_of_prediction']}"
        )
    return "; ".join(parts)


def _ev_explanation(records: list[dict[str, object]]) -> str:
    if len(records) < 2:
        return "No second-best score available for comparison."
    best, second = records[0], records[1]
    component_names = [
        ("exact score", "exact_score_component"),
        ("result", "result_component"),
        ("goal difference", "goal_difference_component"),
        ("participation", "participation_component"),
    ]
    deltas = {label: float(best[column]) - float(second[column]) for label, column in component_names}
    positive_deltas = {label: delta for label, delta in deltas.items() if delta > 0}
    main_label, main_delta = max(
        (positive_deltas or deltas).items(),
        key=lambda item: abs(item[1]),
    )
    shared: list[str] = []
    if np.isclose(float(best["result_component"]), float(second["result_component"]), atol=1e-9):
        shared.append("result")
    if np.isclose(float(best["goal_difference_component"]), float(second["goal_difference_component"]), atol=1e-9):
        shared.append("goal-difference")
    shared_text = f", while both share the same {' and '.join(shared)} payoff" if shared else ""
    direction = "more" if main_delta >= 0 else "less"
    if main_label == "goal difference":
        best_margin = int(str(best["predicted_score"]).split("-")[0]) - int(str(best["predicted_score"]).split("-")[1])
        second_margin = int(str(second["predicted_score"]).split("-")[0]) - int(
            str(second["predicted_score"]).split("-")[1]
        )
        return (
            f"{best['predicted_score']} beats {second['predicted_score']} mainly because the model assigns "
            f"{direction} expected value to the {best_margin:+d} margin than the {second_margin:+d} margin"
            f"{shared_text}."
        )
    return (
        f"{best['predicted_score']} beats {second['predicted_score']} mainly because the model assigns "
        f"{direction} expected value to the {main_label} component{shared_text}."
    )


def _decision_aid(
    records: list[dict[str, object]],
    high_score_cluster: bool,
    high_score_cluster_alternatives: str,
    plausible_top_alternatives: str = "",
    suppressed_ev_candidates: str = "",
) -> dict[str, object]:
    best_ev = float(records[0]["total_expected_points"]) if records else np.nan
    second_ev = float(records[1]["total_expected_points"]) if len(records) > 1 else np.nan
    third_ev = float(records[2]["total_expected_points"]) if len(records) > 2 else np.nan
    ev_gap_to_second = best_ev - second_ev if np.isfinite(best_ev) and np.isfinite(second_ev) else pd.NA
    ev_gap_to_third = best_ev - third_ev if np.isfinite(best_ev) and np.isfinite(third_ev) else pd.NA
    if pd.isna(ev_gap_to_second):
        confidence = "unknown"
    elif float(ev_gap_to_second) >= HIGH_CONFIDENCE_EV_GAP_THRESHOLD:
        confidence = "high"
    elif float(ev_gap_to_second) >= MEDIUM_CONFIDENCE_EV_GAP_THRESHOLD:
        confidence = "medium"
    else:
        confidence = "low / clustered"

    close_scores = [
        str(record["predicted_score"])
        for record in records[1:]
        if np.isfinite(best_ev)
        and best_ev - float(record["total_expected_points"]) < MEDIUM_CONFIDENCE_EV_GAP_THRESHOLD
    ]
    if high_score_cluster_alternatives:
        close_scores.extend(score.strip() for score in high_score_cluster_alternatives.split(",") if score.strip())
    close_alternatives = ", ".join(dict.fromkeys(close_scores))
    notes: list[str] = []
    if confidence == "low / clustered":
        notes.append("EV cluster: top alternatives are very close; manual review recommended.")
    if high_score_cluster:
        notes.append("High-score cluster: consider manual choice among 3-0 / 4-0 / 5-0.")
    if suppressed_ev_candidates:
        notes.append("Some raw EV alternatives were suppressed because their exact-score probability is tiny or their scoreline is too high for practical review.")
    return {
        "recommendation_confidence": confidence,
        "manual_review_flag": "yes" if confidence == "low / clustered" or high_score_cluster else "no",
        "close_alternatives": close_alternatives,
        "plausible_top_alternatives": plausible_top_alternatives,
        "suppressed_ev_candidates": suppressed_ev_candidates,
        "ev_gap_to_second": ev_gap_to_second,
        "ev_gap_to_third": ev_gap_to_third,
        "decision_note": " ".join(notes) if notes else "No manual review signal.",
    }


def _parse_score_label(label: str) -> tuple[int, int]:
    left, right = str(label).split("-", maxsplit=1)
    return int(left), int(right)


def _plausible_extreme_favourite_clean_sheet(
    score: tuple[int, int],
    *,
    extreme_favourite: bool,
    favourite_is_team_a: bool,
) -> bool:
    if not extreme_favourite:
        return False
    if favourite_is_team_a:
        return score in {(3, 0), (4, 0), (5, 0), (6, 0)}
    return score in {(0, 3), (0, 4), (0, 5), (0, 6)}


def _result_goal_difference_bucket(score: tuple[int, int]) -> tuple[int, int]:
    return (1 if score[0] > score[1] else -1 if score[0] < score[1] else 0, score[0] - score[1])


def _plausible_alternative_decision_layer(
    records: list[dict[str, object]],
    *,
    config: ProjectConfig,
    extreme_favourite: bool,
    favourite_is_team_a: bool,
) -> dict[str, object]:
    """Filter raw EV alternatives into practical manual-review candidates."""

    exact_threshold = config.strategies.plausible_alternative_min_exact_probability
    total_goals_limit = config.strategies.plausible_alternative_max_total_goals
    best_exact_by_bucket: dict[tuple[int, int], float] = {}
    for record in records:
        score = _parse_score_label(str(record["predicted_score"]))
        bucket = _result_goal_difference_bucket(score)
        exact_probability = float(record["exact_score_probability"])
        best_exact_by_bucket[bucket] = max(best_exact_by_bucket.get(bucket, 0.0), exact_probability)

    plausible: list[str] = []
    suppressed: list[str] = []
    reasons: list[str] = []
    for record in records:
        score_label = str(record["predicted_score"])
        score = _parse_score_label(score_label)
        total_goals = score[0] + score[1]
        exact_probability = float(record["exact_score_probability"])
        bucket = _result_goal_difference_bucket(score)
        extreme_clean_sheet = _plausible_extreme_favourite_clean_sheet(
            score,
            extreme_favourite=extreme_favourite,
            favourite_is_team_a=favourite_is_team_a,
        )
        low_probability_duplicate = (
            exact_probability < exact_threshold
            and total_goals > total_goals_limit
            and best_exact_by_bucket.get(bucket, exact_probability) > exact_probability
        )
        include = (
            exact_probability >= exact_threshold
            and total_goals <= total_goals_limit
            and not low_probability_duplicate
        ) or extreme_clean_sheet
        if include:
            plausible.append(f"{score_label} ({float(record['total_expected_points']):.3f})")
            continue

        suppressed.append(score_label)
        score_reasons: list[str] = []
        if exact_probability < exact_threshold:
            score_reasons.append(f"P_exact<{exact_threshold:g}")
        if total_goals > total_goals_limit:
            score_reasons.append(f"total_goals>{total_goals_limit}")
        if low_probability_duplicate:
            score_reasons.append("lower-probability duplicate result/margin bucket")
        reasons.append(f"{score_label}: {', '.join(score_reasons) if score_reasons else 'filtered'}")

    return {
        "plausible_top_alternatives": "; ".join(plausible),
        "suppressed_ev_candidates": ", ".join(suppressed),
        "suppression_reason": "; ".join(reasons),
    }


def _btts_warning_flags(
    *,
    has_btts: bool,
    market_btts_yes: float | object,
    market_btts_no: float | object,
    model_btts_yes: float,
    recommended_score: tuple[int, int],
) -> list[str]:
    if not has_btts:
        return ["btts_market_missing"]
    flags: list[str] = []
    if pd.notna(market_btts_yes) and abs(model_btts_yes - float(market_btts_yes)) > BTTS_FIT_WARNING_THRESHOLD:
        flags.append("btts_model_market_mismatch_gt_5pp")
    if (
        not _score_has_btts(recommended_score)
        and pd.notna(market_btts_yes)
        and float(market_btts_yes) > STRONG_BTTS_SIGNAL_THRESHOLD
    ):
        flags.append("recommended_no_btts_against_strong_btts_yes_market")
    if (
        _score_has_btts(recommended_score)
        and pd.notna(market_btts_no)
        and float(market_btts_no) > STRONG_BTTS_SIGNAL_THRESHOLD
    ):
        flags.append("recommended_btts_against_strong_btts_no_market")
    return flags


def _modal_draw_challenger_diagnostic(
    *,
    recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation,
    favourite_probability: float,
    draw_decisive: Mapping[str, object],
    ev_minus_modal_expected_gap: float,
    config: ProjectConfig,
) -> dict[str, object]:
    """Flag balanced matches where modal draw is a plausible manual-review challenger."""

    ev_score = recommendation.best.predicted_score
    modal_score = recommendation.most_likely_scoreline
    draw_vs_decisive_gap = draw_decisive.get("draw_vs_decisive_gap", pd.NA)
    flag = (
        ev_score[0] != ev_score[1]
        and modal_score[0] == modal_score[1]
        and favourite_probability < config.strategies.modal_draw_favourite_probability_threshold
        and pd.notna(draw_vs_decisive_gap)
        and float(draw_vs_decisive_gap) > config.strategies.modal_draw_gap_threshold
        and ev_minus_modal_expected_gap <= config.strategies.modal_draw_ev_gap_threshold
    )
    reason = ""
    if flag:
        reason = (
            "Balanced draw-regime diagnostic: EV score is decisive while the modal score is a draw; "
            "WC 2022 backtest showed EV may over-select narrow decisive no-BTTS scores when the draw "
            "candidate is close in expected points."
        )
    return {
        "modal_draw_challenger_flag": "yes" if flag else "no",
        "modal_draw_challenger_score": _score_label(modal_score) if flag else "",
        "modal_draw_challenger_reason": reason,
        "ev_score": _score_label(ev_score),
        "modal_score": _score_label(modal_score),
        "ev_minus_modal_expected_gap": ev_minus_modal_expected_gap,
    }


def _btts_conflict_diagnostic(
    *,
    recommended_score: tuple[int, int],
    recommended_expected_points: float,
    market_btts_yes: float | object,
    model_btts_yes: float,
    favourite_probability: float,
    draw_prone_flag: bool,
    evaluations: tuple[GroupPredictionEvaluation | KnockoutPredictionEvaluation, ...],
    config: ProjectConfig,
) -> dict[str, object]:
    """Flag no-BTTS EV picks when BTTS signal and a close BTTS alternative coexist."""

    best_btts = next((item for item in evaluations if _score_has_btts(item.predicted_score)), None)
    market_signal = float(market_btts_yes) if pd.notna(market_btts_yes) else np.nan
    signal = np.nanmax([market_signal, float(model_btts_yes)])
    ev_gap_to_btts = (
        float(recommended_expected_points - best_btts.expected_points)
        if best_btts is not None
        else pd.NA
    )
    balanced_context = (
        draw_prone_flag
        or favourite_probability < config.strategies.btts_conflict_favourite_probability_threshold
    )
    flag = (
        not _score_has_btts(recommended_score)
        and np.isfinite(signal)
        and signal >= config.strategies.btts_conflict_probability_threshold
        and balanced_context
        and best_btts is not None
        and pd.notna(ev_gap_to_btts)
        and float(ev_gap_to_btts) <= config.strategies.btts_conflict_ev_gap_threshold
    )
    reason = ""
    if flag:
        reason = (
            f"No-BTTS EV pick conflicts with BTTS probability {signal:.3f}; "
            f"best close BTTS alternative is {_score_label(best_btts.predicted_score)} "
            f"({float(ev_gap_to_btts):.3f} EV behind)."
        )
    return {
        "btts_conflict_flag": "yes" if flag else "no",
        "btts_conflict_reason": reason,
        "btts_conflict_note": reason,
        "btts_probability": signal if np.isfinite(signal) else pd.NA,
        "best_btts_alternative_score": _score_label(best_btts.predicted_score) if best_btts is not None else "",
        "ev_gap_to_btts_alternative": ev_gap_to_btts,
        "ev_gap_to_best_btts_alternative": ev_gap_to_btts,
    }


def _median_total_goals_line(
    calibratable_total_goals: pd.DataFrame,
    match_total_goals: pd.DataFrame,
    fallback_over_2_5: object,
) -> object:
    rows = calibratable_total_goals if not calibratable_total_goals.empty else match_total_goals
    if not rows.empty and "line" in rows:
        lines = pd.to_numeric(rows["line"], errors="coerce").dropna()
        if not lines.empty:
            return float(lines.median())
    return 2.5 if pd.notna(fallback_over_2_5) else pd.NA


def _draw_prone_diagnostic(
    *,
    ou_median_total: object,
    favourite_probability: float,
    market_draw_probability: float,
    recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation,
    draw_decisive: Mapping[str, object],
    config: ProjectConfig,
) -> dict[str, object]:
    flag = (
        pd.notna(ou_median_total)
        and float(ou_median_total) < config.strategies.draw_prone_ou_median_total_threshold
        and favourite_probability < config.strategies.draw_prone_favourite_probability_threshold
    )
    soft_reasons: list[str] = []
    if market_draw_probability >= 0.28:
        soft_reasons.append(f"market draw {market_draw_probability:.3f}")
    if recommendation.best.predicted_score[0] != recommendation.best.predicted_score[1]:
        soft_reasons.append("EV score decisive")
    if recommendation.most_likely_scoreline[0] == recommendation.most_likely_scoreline[1]:
        soft_reasons.append("modal score draw")
    if pd.notna(draw_decisive.get("draw_vs_decisive_gap")):
        soft_reasons.append(f"draw gap {float(draw_decisive['draw_vs_decisive_gap']):+.3f}")
    reason = ""
    if flag:
        reason = (
            "Draw-prone market profile: low total and balanced favourite probability. "
            "Review best draw/modal alternative."
        )
        if soft_reasons:
            reason += " Signals: " + ", ".join(soft_reasons) + "."
    return {
        "draw_prone_flag": "yes" if flag else "no",
        "draw_prone_reason": reason,
        "market_draw_probability": market_draw_probability,
    }


def _format_high_margin_alternatives(
    evaluations: tuple[GroupPredictionEvaluation | KnockoutPredictionEvaluation, ...],
    *,
    favourite_is_team_a: bool,
    limit: int = 5,
) -> str:
    alternatives: list[str] = []
    for evaluation in evaluations:
        score = evaluation.predicted_score
        margin = score[0] - score[1] if favourite_is_team_a else score[1] - score[0]
        if margin >= 2:
            alternatives.append(f"{_score_label(score)} ({evaluation.expected_points:.3f})")
        if len(alternatives) >= limit:
            break
    return "; ".join(alternatives)


def _blowout_risk_diagnostic(
    *,
    favourite_probability: float,
    ou_median_total: object,
    high_margin_alternatives: str,
    high_score_tail_mass: float,
    larger_grid_recommendation: object,
    current_recommendation: str,
    config: ProjectConfig,
) -> dict[str, object]:
    flag = (
        pd.notna(ou_median_total)
        and favourite_probability > config.strategies.blowout_favourite_probability_threshold
        and float(ou_median_total) > config.strategies.blowout_ou_median_total_threshold
    )
    reason = ""
    if flag:
        reason = (
            "Blowout-risk profile: strong favourite and high total. "
            "Review higher-margin alternatives."
        )
    larger_grid_recommended = (
        "yes"
        if pd.notna(larger_grid_recommendation)
        and str(larger_grid_recommendation)
        and str(larger_grid_recommendation) != str(current_recommendation)
        else "no"
    )
    return {
        "blowout_risk_flag": "yes" if flag else "no",
        "blowout_risk_reason": reason,
        "best_high_margin_alternatives": high_margin_alternatives,
        "high_score_tail_mass": high_score_tail_mass,
        "larger_grid_recommended": larger_grid_recommended,
    }


def _asian_handicap_favourite_summary(
    match_asian_handicap: pd.DataFrame,
    *,
    favourite_is_team_a: bool,
) -> dict[str, object]:
    if match_asian_handicap.empty or "handicap" not in match_asian_handicap:
        return {"ah_implied_favourite_margin": pd.NA, "favourite_covered_ah": pd.NA}
    rows = match_asian_handicap.copy()
    if "selected_for_market_consistent" in rows:
        selected = rows[rows["selected_for_market_consistent"].astype(str).str.lower().eq("yes")]
        if not selected.empty:
            rows = selected
    direction_rows = rows[pd.to_numeric(rows["handicap"], errors="coerce").lt(0 if favourite_is_team_a else np.inf)]
    if not favourite_is_team_a:
        direction_rows = rows[pd.to_numeric(rows["handicap"], errors="coerce").gt(0)]
    if direction_rows.empty:
        direction_rows = rows
    if "bookmakers_count" in direction_rows:
        direction_rows = direction_rows.sort_values(["bookmakers_count", "handicap"], ascending=[False, True])
    row = direction_rows.iloc[0]
    probability_column = "fair_team_a" if favourite_is_team_a else "fair_team_b"
    return {
        "ah_implied_favourite_margin": abs(float(row["handicap"])) if pd.notna(row.get("handicap")) else pd.NA,
        "favourite_covered_ah": row.get(probability_column, pd.NA),
    }


def _format_probability_items(items: list[tuple[str, float]], limit: int = 5) -> str:
    return "; ".join(f"{label}: {probability:.4f}" for label, probability in items[:limit])


def _candidate_ev(
    matrix: ScoreProbabilityMatrix,
    score: tuple[int, int],
    recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation,
    qualifier_probabilities: dict[str, float] | None,
    config: ProjectConfig,
) -> float:
    if isinstance(recommendation, KnockoutPredictionRecommendation):
        evaluation = evaluate_knockout_prediction(
            matrix,
            score[0],
            score[1],
            recommendation.best.predicted_qualifier,
            qualifier_probabilities or {},
            config.knockout_scoring,
        )
    else:
        evaluation = evaluate_group_prediction(matrix, score[0], score[1])
    return evaluation.expected_points


def _larger_grid_sensitivity(
    *,
    targets: CalibrationTargets,
    config: ProjectConfig,
    knockout: bool,
    qualifier_probabilities: dict[str, float] | None,
    final_recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation,
    normal_poisson_matrix: ScoreProbabilityMatrix,
    normal_poisson_recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation,
    favourite_is_team_a: bool,
    larger_max_goals: int = LARGER_GRID_MAX_GOALS,
) -> dict[str, object]:
    final_score = _score_label(final_recommendation.best.predicted_score)
    normal_poisson_score = _score_label(normal_poisson_recommendation.best.predicted_score)
    larger_calibration = calibrate_poisson_model(
        targets,
        larger_max_goals,
        config.calibration_weights,
        config.renormalise_score_matrix,
        config.poor_calibration_loss_threshold,
    )
    larger_recommendation = _optimise_score_matrix(
        larger_calibration.score_matrix,
        knockout,
        qualifier_probabilities,
        config,
    )
    larger_score = _score_label(larger_recommendation.best.predicted_score)
    watched_scores = ((3, 0), (4, 0), (5, 0)) if favourite_is_team_a else ((0, 3), (0, 4), (0, 5))
    clean_sheet_evs: dict[str, object] = {}
    for favourite_goals, score in zip((3, 4, 5), watched_scores, strict=True):
        label = _score_label(score).replace("-", "_")
        normal_ev = _candidate_ev(
            normal_poisson_matrix,
            score,
            normal_poisson_recommendation,
            qualifier_probabilities,
            config,
        )
        larger_ev = _candidate_ev(
            larger_calibration.score_matrix,
            score,
            larger_recommendation,
            qualifier_probabilities,
            config,
        )
        clean_sheet_evs[f"ev_{label}_normal_grid"] = normal_ev
        clean_sheet_evs[f"ev_{label}_larger_grid"] = larger_ev
        clean_sheet_evs[f"ev_favourite_{favourite_goals}_0_normal_grid"] = normal_ev
        clean_sheet_evs[f"ev_favourite_{favourite_goals}_0_larger_grid"] = larger_ev
    return {
        "current_final_recommendation": final_score,
        "normal_grid_recommendation": normal_poisson_score,
        "normal_grid_poisson_recommendation": normal_poisson_score,
        "larger_grid_recommendation": larger_score,
        "larger_grid_poisson_recommendation": larger_score,
        "recommendation_changes_with_larger_grid": normal_poisson_score != larger_score,
        "larger_grid_poisson_changes_recommendation": normal_poisson_score != larger_score,
        "larger_grid_differs_from_live_recommendation": final_score != larger_score,
        "larger_grid_tail_mass": larger_calibration.score_matrix.tail_probability,
        "grid_max_goals_used": int(larger_max_goals),
        "tail_mass_before_grid_extension": float(normal_poisson_matrix.tail_probability),
        "tail_mass_after_grid_extension": float(larger_calibration.score_matrix.tail_probability),
        "recommendation_changed_due_to_larger_grid": "yes" if normal_poisson_score != larger_score else "no",
        **clean_sheet_evs,
    }


def _extreme_favourite_audit(
    *,
    matrix: ScoreProbabilityMatrix,
    favourite_is_team_a: bool,
    recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation,
    qualifier_probabilities: dict[str, float] | None,
    config: ProjectConfig,
) -> dict[str, object]:
    scores_a, scores_b = np.indices(matrix.probabilities.shape)
    favourite_scores = scores_a if favourite_is_team_a else scores_b
    underdog_scores = scores_b if favourite_is_team_a else scores_a
    favourite_margins = favourite_scores - underdog_scores
    clean_sheet_mask = underdog_scores == 0
    clean_sheet_items = [
        (
            _score_label((int(a), int(b))),
            float(matrix.probabilities[int(a), int(b)]),
        )
        for a, b in zip(scores_a[clean_sheet_mask], scores_b[clean_sheet_mask], strict=True)
    ]
    clean_sheet_items.sort(key=lambda item: (-item[1], item[0]))
    margin_items = [
        (f"+{margin}", float(matrix.probabilities[favourite_margins == margin].sum()))
        for margin in range(1, 6)
    ]
    margin_items.append(("+6+", float(matrix.probabilities[favourite_margins >= 6].sum())))
    score_count_items = [
        (str(goals), float(matrix.probabilities[favourite_scores == goals].sum()))
        for goals in range(6)
    ]
    score_count_items.append(("6+", float(matrix.probabilities[favourite_scores >= 6].sum())))
    score_count_items.sort(key=lambda item: (-item[1], item[0]))
    watched_scores = ((3, 0), (4, 0), (5, 0)) if favourite_is_team_a else ((0, 3), (0, 4), (0, 5))
    watched_evs = {
        _score_label(score): _candidate_ev(matrix, score, recommendation, qualifier_probabilities, config)
        for score in watched_scores
    }
    watched_values = list(watched_evs.values())
    cluster = max(watched_values) - min(watched_values) <= HIGH_SCORE_EV_CLUSTER_THRESHOLD
    cluster_ceiling = max(watched_values) if watched_values else np.nan
    cluster_alternatives = [
        score for score, ev in watched_evs.items() if np.isfinite(cluster_ceiling) and cluster_ceiling - ev <= HIGH_SCORE_EV_CLUSTER_THRESHOLD
    ]
    return {
        "top_clean_sheet_scores": _format_probability_items(clean_sheet_items),
        "top_favourite_margin_probabilities": _format_probability_items(margin_items, limit=len(margin_items)),
        "top_5_favourite_score_count_probabilities": _format_probability_items(score_count_items),
        "high_score_cluster_scores": "; ".join(f"{score}: {ev:.3f}" for score, ev in watched_evs.items()),
        "three_four_five_nil_within_0_10_ev": cluster,
        "high_score_cluster": "yes" if cluster else "no",
        "high_score_cluster_close_alternatives": ", ".join(cluster_alternatives) if cluster else "",
    }


def _market_consistent_high_score_comparison(
    *,
    poisson_matrix: ScoreProbabilityMatrix,
    market_consistent_matrix: ScoreProbabilityMatrix,
    poisson_recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation,
    market_consistent_recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation,
    qualifier_probabilities: dict[str, float] | None,
    config: ProjectConfig,
    favourite_is_team_a: bool,
) -> dict[str, object]:
    watched_scores = ((3, 0), (4, 0), (5, 0)) if favourite_is_team_a else ((0, 3), (0, 4), (0, 5))
    comparison: dict[str, object] = {}
    for favourite_goals, score in zip((3, 4, 5), watched_scores, strict=True):
        comparison[f"market_consistent_poisson_ev_favourite_{favourite_goals}_0"] = _candidate_ev(
            poisson_matrix,
            score,
            poisson_recommendation,
            qualifier_probabilities,
            config,
        )
        comparison[f"market_consistent_matrix_ev_favourite_{favourite_goals}_0"] = _candidate_ev(
            market_consistent_matrix,
            score,
            market_consistent_recommendation,
            qualifier_probabilities,
            config,
        )
    return comparison


def _market_consistent_weights(config: ProjectConfig) -> MarketConsistentWeights:
    """Scale the baseline market-consistent weights by configured group multipliers."""

    base = MarketConsistentWeights()
    groups = config.market_consistent_group_weights
    return MarketConsistentWeights(
        one_x_two=base.one_x_two * groups.one_x_two,
        liquid_total_goals=base.liquid_total_goals * groups.total_goals,
        asian_handicap=base.asian_handicap * groups.asian_handicap,
        btts=base.btts * groups.btts,
        correct_score=base.correct_score * groups.correct_score,
        max_correct_score_cells=base.max_correct_score_cells,
    )


_RESULT_MARGIN_OVERLAP_GROUPS = ("1x2", "asian_handicap", "correct_score")


def _market_consistent_correlation_note(diagnostics: object) -> tuple[str, str]:
    """Return active constraint groups and a double-counting scaffold note.

    Constraints are weighted but treated as independent. 1X2, Asian handicap and
    correct score all carry result/margin information, so activating several at
    once can over-count shared market signal. This is a transparent diagnostic,
    not a fitted covariance model.
    """

    get = diagnostics.get if hasattr(diagnostics, "get") else (lambda *_: 0)
    counts = {
        "1x2": int(get("market_consistent_1x2_constraint_count", 0) or 0),
        "btts": int(get("market_consistent_btts_constraint_count", 0) or 0),
        "total_goals": int(get("market_consistent_total_goals_constraint_count", 0) or 0),
        "asian_handicap": int(get("market_consistent_asian_handicap_constraint_count", 0) or 0),
        "correct_score": int(get("market_consistent_correct_score_constraint_count", 0) or 0),
    }
    active = [group for group, count in counts.items() if count > 0]
    overlap = [group for group in _RESULT_MARGIN_OVERLAP_GROUPS if counts[group] > 0]
    note = "constraints_treated_independently"
    if len(overlap) >= 2:
        note += "; potential_result_margin_double_counting:" + "+".join(overlap)
    return ", ".join(active), note


def _build_bivariate_prior_matrix(
    *,
    match: Match,
    lambda_a: float,
    lambda_b: float,
    config: ProjectConfig,
    max_goals: int,
) -> tuple[ScoreProbabilityMatrix | None, float, str]:
    """Return a conservative bivariate-Poisson matrix, or None when unstable."""

    covariance = conservative_bivariate_covariance(lambda_a, lambda_b, config.bivariate_poisson_covariance)
    try:
        matrix = BivariatePoissonScoreModel(
            lambda_a, lambda_b, covariance, config.renormalise_score_matrix
        ).predict_score_matrix(match, max_goals)
    except ValueError as exc:
        return None, covariance, f"bivariate_poisson_unstable:{exc}"
    return matrix, covariance, ""


def _empty_bivariate_diagnostic() -> dict[str, object]:
    return {
        "bivariate_lambda_1": pd.NA,
        "bivariate_lambda_2": pd.NA,
        "bivariate_lambda_3": pd.NA,
        "bivariate_prior_recommended_score": "",
        "bivariate_differs_from_default": "no",
        "bivariate_warnings": "",
    }


def _bivariate_poisson_diagnostic(
    *,
    match: Match,
    calibration: object,
    config: ProjectConfig,
    knockout: bool,
    qualifier_probabilities: dict[str, float] | None,
    default_score: tuple[int, int],
) -> dict[str, object]:
    """Optional research-only bivariate-Poisson challenger recommendation."""

    if not config.enable_bivariate_poisson_diagnostic:
        return _empty_bivariate_diagnostic()
    lambda_a = float(calibration.lambda_a)
    lambda_b = float(calibration.lambda_b)
    covariance = conservative_bivariate_covariance(lambda_a, lambda_b, config.bivariate_poisson_covariance)
    try:
        matrix = BivariatePoissonScoreModel(
            lambda_a, lambda_b, covariance, config.renormalise_score_matrix
        ).predict_score_matrix(match, config.max_goals_score_matrix)
    except ValueError as exc:
        diagnostic = _empty_bivariate_diagnostic()
        diagnostic["bivariate_lambda_3"] = float(covariance)
        diagnostic["bivariate_warnings"] = f"bivariate_poisson_unstable:{exc}"
        return diagnostic
    recommendation = _optimise_score_matrix(matrix, knockout, qualifier_probabilities, config)
    score = recommendation.best.predicted_score
    warning = "bivariate_reduces_to_independent_poisson" if covariance == 0.0 else ""
    return {
        "bivariate_lambda_1": float(lambda_a - covariance),
        "bivariate_lambda_2": float(lambda_b - covariance),
        "bivariate_lambda_3": float(covariance),
        "bivariate_prior_recommended_score": _score_label(score),
        "bivariate_differs_from_default": "yes" if score != default_score else "no",
        "bivariate_warnings": warning,
    }


def _empty_margin_model_diagnostic() -> dict[str, object]:
    return {
        "asian_handicap_margin_model_type": "",
        "asian_handicap_margin_parameters": "",
        "asian_handicap_margin_fit_error": pd.NA,
        "asian_handicap_margin_distribution_comparison": "",
        "asian_handicap_margin_recommendation_if_adjusted": "",
        "asian_handicap_margin_model_warnings": "",
    }


def _asian_handicap_margin_model_diagnostic(
    *,
    config: ProjectConfig,
    match_asian_handicap: pd.DataFrame,
    lambda_a: float,
    lambda_b: float,
) -> dict[str, object]:
    """Optional research-only Skellam margin model fitted to Asian-handicap lines."""

    if not config.enable_asian_handicap_margin_model:
        return _empty_margin_model_diagnostic()
    model = fit_skellam_margin_model(
        match_asian_handicap if match_asian_handicap is not None and not match_asian_handicap.empty else None,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
    )
    if model is None:
        return _empty_margin_model_diagnostic()
    parameters = "; ".join(f"{key}={value:.4f}" for key, value in model.fitted_margin_parameters.items())
    return {
        "asian_handicap_margin_model_type": model.margin_model_type,
        "asian_handicap_margin_parameters": parameters,
        "asian_handicap_margin_fit_error": model.margin_fit_error,
        "asian_handicap_margin_distribution_comparison": model.margin_distribution_comparison,
        "asian_handicap_margin_recommendation_if_adjusted": model.recommendation_if_margin_adjusted,
        "asian_handicap_margin_model_warnings": model.warnings,
    }


def _select_market_consistent_prior(
    *,
    prior_choice: str,
    poisson_matrix: ScoreProbabilityMatrix,
    dixon_coles_matrix: ScoreProbabilityMatrix,
    dixon_coles_rho_estimate: object,
    bivariate_matrix: ScoreProbabilityMatrix | None,
    bivariate_warning: str,
) -> tuple[ScoreProbabilityMatrix, str, float, str]:
    """Pick the prior matrix for the market-consistent projection.

    Returns ``(prior_matrix, prior_source, dixon_coles_rho_used, warning)``.
    Defaults to independent Poisson; a Dixon-Coles prior uses the estimated or
    configured rho; a bivariate prior falls back to independent Poisson with a
    warning when estimation is unstable.
    """

    if prior_choice == "dixon_coles":
        rho_used = float(getattr(dixon_coles_rho_estimate, "rho", 0.0))
        return dixon_coles_matrix, "dixon_coles", rho_used, str(getattr(dixon_coles_rho_estimate, "warning", ""))
    if prior_choice == "bivariate_poisson":
        if bivariate_matrix is not None:
            return bivariate_matrix, "bivariate_poisson", float("nan"), bivariate_warning
        return (
            poisson_matrix,
            "independent_poisson_fallback",
            float("nan"),
            bivariate_warning or "bivariate prior unavailable; used independent Poisson",
        )
    return poisson_matrix, "independent_poisson", float("nan"), ""


def _should_run_market_consistent_challenger(
    *,
    setting: bool | str,
    ev_gap_to_second: object,
    extreme_favourite: bool,
    warning_flags: str,
    config: ProjectConfig,
) -> bool:
    if setting is True:
        return True
    if setting is False:
        return False
    if setting != "only_if_close":
        raise ValueError("enable_market_consistent_challenger must be True, False, or 'only_if_close'")
    severe_warning = any(
        flag
        for flag in str(warning_flags or "").split("; ")
        if flag
        and (
            "poor_calibration" in flag
            or "high_tail_mass" in flag
            or "larger_grid_sensitivity" in flag
            or "challenger_model_recommendations_disagree" in flag
        )
    )
    return (
        pd.notna(ev_gap_to_second)
        and float(ev_gap_to_second) < config.strategies.low_confidence_ev_gap_threshold
    ) or extreme_favourite or severe_warning


def _empty_market_consistent_high_score_comparison() -> dict[str, object]:
    return {
        "market_consistent_poisson_ev_favourite_3_0": pd.NA,
        "market_consistent_poisson_ev_favourite_4_0": pd.NA,
        "market_consistent_poisson_ev_favourite_5_0": pd.NA,
        "market_consistent_matrix_ev_favourite_3_0": pd.NA,
        "market_consistent_matrix_ev_favourite_4_0": pd.NA,
        "market_consistent_matrix_ev_favourite_5_0": pd.NA,
    }


def _optimise_score_matrix(
    matrix: ScoreProbabilityMatrix,
    knockout: bool,
    qualifier_probabilities: dict[str, float] | None,
    config: ProjectConfig,
) -> GroupPredictionRecommendation | KnockoutPredictionRecommendation:
    """Run the existing EV optimiser against any baseline, live, or challenger matrix."""

    if knockout:
        return optimise_knockout_prediction(
            matrix,
            qualifier_probabilities or {},
            config.max_candidate_goals,
            config.strategies.top_alternatives,
            config.knockout_scoring,
        )
    return optimise_group_prediction(matrix, config.max_candidate_goals, config.strategies.top_alternatives)


def _canonical_margin_method(method: str) -> str:
    return "normalised_inverse_odds" if method == "proportional" else method


def _margin_methods_to_compare(active_method: str) -> tuple[str, ...]:
    ordered = [_canonical_margin_method(active_method), *IMPLEMENTED_MARGIN_REMOVAL_METHODS]
    return tuple(dict.fromkeys(ordered))


def _margin_method_warning_flags(
    *,
    method_row: pd.Series,
    default_row: pd.Series,
    differs_from_default: bool,
) -> str:
    flags: list[str] = []
    for column in ("market_a_win", "market_draw", "market_b_win"):
        if (
            pd.notna(method_row.get(column))
            and pd.notna(default_row.get(column))
            and abs(float(method_row[column]) - float(default_row[column])) > 0.02
        ):
            flags.append("margin_method_materially_changes_1x2_probabilities")
            break
    if differs_from_default:
        flags.append("margin_method_changes_recommendation")
    if bool(method_row.get("three_four_five_nil_within_0_10_ev", False)) != bool(
        default_row.get("three_four_five_nil_within_0_10_ev", False)
    ):
        flags.append("margin_method_changes_extreme_favourite_high_score_cluster")
    return "; ".join(dict.fromkeys(flags))


def _failed_margin_method_rows(
    odds: pd.DataFrame,
    method: str,
    error: Exception,
) -> list[dict[str, object]]:
    matches = odds[["match_id", "team_a", "team_b"]].drop_duplicates("match_id")
    return [
        {
            "match_id": row["match_id"],
            "team_a": row["team_a"],
            "team_b": row["team_b"],
            "method": method,
            "method_status": "failed",
            "method_failure_message": str(error),
            "differs_from_default": pd.NA,
            "warning_flags": "margin_method_failed",
        }
        for _, row in matches.iterrows()
    ]


def _first_nonempty_method(series: pd.Series) -> str:
    for value in series:
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "<na>"}:
            return text
    return ""


def _devig_method_summary(
    *,
    match_id: str,
    bookmaker_probabilities: pd.DataFrame,
    processed_total_goals: pd.DataFrame,
    processed_asian_handicap: pd.DataFrame,
    processed_correct_scores: pd.DataFrame,
) -> dict[str, object]:
    """Summarise requested vs actual devig method per market for one match."""

    parts: list[str] = []
    fallbacks: list[str] = []

    def record(label: str, requested: str, actual: str) -> None:
        if not requested:
            return
        parts.append(f"{label}:{requested}->{actual}")
        if actual and requested != actual:
            fallbacks.append(f"{label}:{requested}->{actual}")

    if bookmaker_probabilities is not None and not bookmaker_probabilities.empty:
        match_rows = bookmaker_probabilities[
            bookmaker_probabilities["match_id"].astype(str) == match_id
        ]
        for market in ("1x2", "btts", "over_under_2_5", "qualification"):
            requested_column = f"{market}_margin_removal_requested_method"
            actual_column = f"{market}_margin_removal_actual_method"
            if not match_rows.empty and requested_column in match_rows:
                record(
                    market,
                    _first_nonempty_method(match_rows[requested_column]),
                    _first_nonempty_method(match_rows[actual_column]),
                )

    for label, frame in (
        ("total_goals", processed_total_goals),
        ("asian_handicap", processed_asian_handicap),
        ("correct_score", processed_correct_scores),
    ):
        if frame is None or frame.empty or "margin_removal_requested_method" not in frame:
            continue
        match_rows = frame[frame["match_id"].astype(str) == match_id]
        if match_rows.empty:
            continue
        record(
            label,
            _first_nonempty_method(match_rows["margin_removal_requested_method"]),
            _first_nonempty_method(match_rows["margin_removal_actual_method"]),
        )

    return {
        "devig_methods_by_market": "; ".join(parts),
        "devig_fallback_warnings": "; ".join(fallbacks),
    }


def _build_margin_method_comparison(
    *,
    odds: pd.DataFrame,
    config: ProjectConfig,
    default_report: pd.DataFrame,
    correct_score_odds: pd.DataFrame | None,
    total_goals_odds: pd.DataFrame | None,
    asian_handicap_odds: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    default_by_match = default_report.set_index("match_id")
    for method in _margin_methods_to_compare(config.margin_removal_method):
        comparison_config = replace(
            config,
            margin_removal_method=method,
            enable_margin_method_comparison=False,
        )
        try:
            comparison = run_prediction_workflow(
                odds,
                config=comparison_config,
                correct_score_odds=correct_score_odds,
                total_goals_odds=total_goals_odds,
                asian_handicap_odds=asian_handicap_odds,
            )
        except (ValueError, NotImplementedError) as error:
            rows.extend(_failed_margin_method_rows(odds, method, error))
            continue
        for _, method_row in comparison.match_report.iterrows():
            match_id = method_row["match_id"]
            default_row = default_by_match.loc[match_id]
            differs = str(method_row["recommended_score"]) != str(default_row["recommended_score"])
            total_goals_targets = str(method_row.get("total_goals_lines_used_for_calibration", ""))
            warning_flags = _margin_method_warning_flags(
                method_row=method_row,
                default_row=default_row,
                differs_from_default=differs,
            )
            rows.append(
                {
                    "match_id": match_id,
                    "team_a": method_row["team_a"],
                    "team_b": method_row["team_b"],
                    "method": method,
                    "method_status": "ok",
                    "method_failure_message": "",
                    "fair_1x2_home": method_row["market_a_win"],
                    "fair_1x2_draw": method_row["market_draw"],
                    "fair_1x2_away": method_row["market_b_win"],
                    "fair_btts_yes": method_row.get("market_fair_btts_yes_probability", pd.NA),
                    "fair_over_2_5": method_row.get("fair_over_2_5", pd.NA),
                    "selected_total_goals_targets": total_goals_targets,
                    "lambda_a": method_row["lambda_a"],
                    "lambda_b": method_row["lambda_b"],
                    "recommended_score": method_row["recommended_score"],
                    "best_expected_points": method_row["best_expected_points"],
                    "top_5_ev_scorelines": method_row["top_5_ev_predictions"],
                    "differs_from_default": "yes" if differs else "no",
                    "calibration_error": method_row["calibration_loss"],
                    "btts_fit_error": method_row.get("btts_fit_error", pd.NA),
                    "total_goals_fit_error": method_row.get("total_goals_line_fit_error", pd.NA),
                    "warning_flags": warning_flags,
                }
            )
    return pd.DataFrame(rows)


def _extract_score_from_formatted_candidate(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split(" ", maxsplit=1)[0]


def _format_margin_method_scores(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""
    parts: list[str] = []
    for _, row in rows.iterrows():
        if str(row.get("method_status", "ok")) == "ok":
            parts.append(f"{row['method']}:{row['recommended_score']}")
        else:
            parts.append(f"{row['method']}:failed")
    return "; ".join(parts)


def _score_counts_for_consensus(default_score: str, challenger_scores: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for score in [default_score, *challenger_scores]:
        if score:
            counts[score] = counts.get(score, 0) + 1
    return counts


def _model_consensus(default_score: str, challenger_scores: list[str]) -> str:
    serious_scores = [score for score in challenger_scores if score]
    if not serious_scores or all(score == default_score for score in serious_scores):
        return "strong_consensus"
    differing = sum(score != default_score for score in serious_scores)
    unique_scores = set([default_score, *serious_scores])
    if differing == 1:
        return "moderate_consensus"
    if len(unique_scores) >= 4:
        return "no_consensus"
    return "weak_consensus"


def _dashboard_confidence(
    *,
    ev_gap_to_second: object,
    model_consensus: str,
    high_score_cluster: bool,
    strategic_alternative_score: str,
    config: ProjectConfig,
) -> str:
    if high_score_cluster:
        return "clustered"
    if strategic_alternative_score:
        return "strategic"
    if pd.notna(ev_gap_to_second):
        gap = float(ev_gap_to_second)
        if gap < config.strategies.low_confidence_ev_gap_threshold:
            return "low"
        if gap >= config.strategies.high_confidence_ev_gap_threshold and model_consensus == "strong_consensus":
            return "high"
    if model_consensus in {"weak_consensus", "no_consensus"}:
        return "low"
    return "medium"


def _dashboard_main_alternative(
    *,
    default_score: str,
    strategic_alternative_score: str,
    plausible_alternatives: str,
    challenger_scores: list[str],
) -> str:
    if strategic_alternative_score and strategic_alternative_score != default_score:
        return strategic_alternative_score
    for score in challenger_scores:
        if score and score != default_score:
            return score
    for candidate in str(plausible_alternatives or "").split("; "):
        score = _extract_score_from_formatted_candidate(candidate)
        if score and score != default_score:
            return score
    return ""


def _dashboard_risk_notes(
    *,
    ev_gap_small: bool,
    market_consistent_differs: bool,
    margin_method_sensitive: bool,
    dixon_coles_differs: bool,
    public_strategy_differs: bool,
    high_score_cluster: bool,
    modal_draw_challenger: bool,
    btts_conflict: bool,
    draw_prone: bool,
    blowout_risk: bool,
    warning_flags: str,
) -> str:
    notes: list[str] = []
    if ev_gap_small:
        notes.append("EV gap to second is small")
    if market_consistent_differs:
        notes.append("market-consistent challenger differs")
    if margin_method_sensitive:
        notes.append("margin-removal methods change recommendation")
    if dixon_coles_differs:
        notes.append("Dixon-Coles challenger differs")
    if public_strategy_differs:
        notes.append("public strategy suggests a different score")
    if high_score_cluster:
        notes.append("high-score cluster present")
    if modal_draw_challenger:
        notes.append("modal/draw challenger flagged")
    if btts_conflict:
        notes.append("BTTS conflict warning")
    if draw_prone:
        notes.append("draw-prone market profile")
    if blowout_risk:
        notes.append("blowout-risk market profile")
    severe_flags = [
        flag
        for flag in str(warning_flags or "").split("; ")
        if flag
        and (
            "poor_calibration" in flag
            or "high_tail_mass" in flag
            or "larger_grid_sensitivity" in flag
            or "stale_odds" in flag
            or "market_consistent_optimisation_failed" in flag
            or "market_consistent_constraint_fit_poor" in flag
            or "knockout_scoring_unverified" in flag
            or "asian_handicap_orientation_suspicious" in flag
        )
    ]
    if severe_flags:
        notes.append("severe warning flags: " + ", ".join(severe_flags))
    return "; ".join(notes)


def _dashboard_decision_note(
    *,
    default_score: str,
    model_consensus: str,
    confidence_level: str,
    risk_notes: str,
    main_alternative_score: str,
    high_score_cluster: bool,
    modal_draw_challenger: bool,
    modal_draw_challenger_score: str,
    draw_prone: bool,
    blowout_risk: bool,
) -> str:
    if draw_prone:
        return (
            "Draw-prone market profile: low total and balanced favourite probability. "
            "Review best draw/modal alternative."
        )
    if blowout_risk:
        return (
            "Blowout-risk profile: strong favourite and high total. "
            "Review higher-margin alternatives."
        )
    if modal_draw_challenger:
        challenger_text = (
            f" Consider {modal_draw_challenger_score} for manual review."
            if modal_draw_challenger_score
            else ""
        )
        return (
            f"Default remains {default_score}.{challenger_text} WC 2022 backtest showed EV may "
            "over-select narrow decisive no-BTTS scores in balanced draw regimes."
        )
    if not risk_notes and model_consensus == "strong_consensus" and confidence_level == "high":
        return "Default recommendation supported by all challenger diagnostics."
    if high_score_cluster and main_alternative_score:
        return (
            f"High-score cluster: {default_score} is model-favoured, but "
            f"{main_alternative_score} is a plausible strategic alternative."
        )
    if main_alternative_score:
        return (
            f"Default remains {default_score}; consider {main_alternative_score} for manual review. "
            f"{risk_notes or 'Challenger diagnostics provide an alternative signal.'}"
        )
    return risk_notes or "Default recommendation retained; no override signal."


def _build_final_decision_dashboard(
    match_report: pd.DataFrame,
    margin_method_comparison: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in match_report.iterrows():
        match_id = str(row["match_id"])
        default_score = str(row["recommended_score"])
        margin_rows = (
            margin_method_comparison[margin_method_comparison["match_id"].astype(str) == match_id]
            if not margin_method_comparison.empty
            else pd.DataFrame()
        )
        margin_method_scores = _format_margin_method_scores(margin_rows)
        margin_method_sensitive_value = (
            "not_run"
            if margin_rows.empty
            else "yes"
            if margin_rows["differs_from_default"].astype(str).str.lower().eq("yes").any()
            else "no"
        )
        margin_method_sensitive = margin_method_sensitive_value == "yes"
        margin_scores = [
            str(score)
            for score in margin_rows.loc[
                margin_rows.get("method_status", pd.Series(dtype=str)).astype(str).eq("ok"),
                "recommended_score",
            ].dropna()
            if str(score)
        ] if not margin_rows.empty and "recommended_score" in margin_rows else []
        market_consistent_score = str(row.get("market_consistent_recommended_score", ""))
        dixon_coles_score = str(row.get("dixon_coles_recommended_score", ""))
        public_strategy_score = str(row.get("public_strategy_score", ""))
        asian_handicap_shift = str(row.get("asian_handicap_shift_recommendation", "no")).lower() == "yes"
        correct_score_blended_score = str(row.get("correct_score_blended_recommended_score", ""))
        challenger_scores = [
            market_consistent_score,
            dixon_coles_score,
            correct_score_blended_score,
            *margin_scores,
        ]
        model_consensus = _model_consensus(default_score, challenger_scores)
        ev_gap_small = (
            pd.notna(row.get("ev_gap_to_second"))
            and float(row["ev_gap_to_second"]) < config.strategies.low_confidence_ev_gap_threshold
        )
        high_score_cluster = str(row.get("high_score_cluster", "no")).lower() == "yes"
        modal_draw_challenger = str(row.get("modal_draw_challenger_flag", "no")).lower() == "yes"
        btts_conflict = str(row.get("btts_conflict_flag", "no")).lower() == "yes"
        draw_prone = str(row.get("draw_prone_flag", "no")).lower() == "yes"
        blowout_risk = str(row.get("blowout_risk_flag", "no")).lower() == "yes"
        modal_draw_challenger_score = str(row.get("modal_draw_challenger_score", ""))
        market_consistent_differs = market_consistent_score not in {"", default_score}
        dixon_coles_differs = dixon_coles_score not in {"", default_score}
        public_strategy_differs = public_strategy_score not in {"", default_score}
        strategic_alternative_score = (
            public_strategy_score
            if public_strategy_differs
            and pd.notna(row.get("public_strategy_ev_cost"))
            and float(row.get("public_strategy_ev_cost", 0.0)) <= config.public_strategy.effective_max_ev_loss
            and str(row.get("warning_flags", "")).find("poor_calibration") == -1
            else ""
        )
        confidence_level = _dashboard_confidence(
            ev_gap_to_second=row.get("ev_gap_to_second"),
            model_consensus=model_consensus,
            high_score_cluster=high_score_cluster,
            strategic_alternative_score=strategic_alternative_score,
            config=config,
        )
        plausible_alternatives = str(row.get("plausible_top_alternatives", ""))
        main_alternative_score = _dashboard_main_alternative(
            default_score=default_score,
            strategic_alternative_score=strategic_alternative_score,
            plausible_alternatives=plausible_alternatives,
            challenger_scores=challenger_scores,
        )
        risk_notes = _dashboard_risk_notes(
            ev_gap_small=ev_gap_small,
            market_consistent_differs=market_consistent_differs,
            margin_method_sensitive=margin_method_sensitive,
            dixon_coles_differs=dixon_coles_differs,
            public_strategy_differs=public_strategy_differs,
            high_score_cluster=high_score_cluster,
            modal_draw_challenger=modal_draw_challenger,
            btts_conflict=btts_conflict,
            draw_prone=draw_prone,
            blowout_risk=blowout_risk,
            warning_flags=str(row.get("warning_flags", "")),
        )
        if asian_handicap_shift:
            risk_notes = "; ".join(filter(None, [risk_notes, "asian_handicap_shifts_market_consistent_score"]))
        manual_review = any(
            [
                ev_gap_small,
                market_consistent_differs,
                margin_method_sensitive,
                dixon_coles_differs,
                bool(strategic_alternative_score),
                high_score_cluster,
                modal_draw_challenger,
                btts_conflict,
                draw_prone,
                blowout_risk,
                bool(risk_notes),
            ]
        )
        override_candidate = bool(main_alternative_score and manual_review)
        decision_note = _dashboard_decision_note(
            default_score=default_score,
            model_consensus=model_consensus,
            confidence_level=confidence_level,
            risk_notes=risk_notes,
            main_alternative_score=main_alternative_score,
            high_score_cluster=high_score_cluster,
            modal_draw_challenger=modal_draw_challenger,
            modal_draw_challenger_score=modal_draw_challenger_score,
            draw_prone=draw_prone,
            blowout_risk=blowout_risk,
        )
        rows.append(
            {
                "match_id": row["match_id"],
                "date": row["date"],
                "time": row.get("time", ""),
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "default_score": default_score,
                "final_decision_score": default_score,
                "market_consistent_score": market_consistent_score,
                "dixon_coles_score": dixon_coles_score,
                "public_strategy_score": public_strategy_score,
                "margin_method_scores": margin_method_scores,
                "model_consensus": model_consensus,
                "confidence_level": confidence_level,
                "manual_review_flag": "yes" if manual_review else "no",
                "ev_gap_to_second": row.get("ev_gap_to_second", pd.NA),
                "ev_gap_to_third": row.get("ev_gap_to_third", pd.NA),
                "main_alternative_score": main_alternative_score,
                "plausible_alternatives": plausible_alternatives,
                "strategic_alternative_score": strategic_alternative_score,
                "override_candidate": "yes" if override_candidate else "no",
                "high_score_cluster": "yes" if high_score_cluster else "no",
                "modal_draw_challenger_flag": "yes" if modal_draw_challenger else "no",
                "modal_draw_challenger_score": modal_draw_challenger_score,
                "btts_conflict_flag": "yes" if btts_conflict else "no",
                "btts_conflict_note": row.get("btts_conflict_note", ""),
                "best_btts_alternative_score": row.get("best_btts_alternative_score", ""),
                "draw_prone_flag": "yes" if draw_prone else "no",
                "draw_prone_reason": row.get("draw_prone_reason", ""),
                "blowout_risk_flag": "yes" if blowout_risk else "no",
                "blowout_risk_reason": row.get("blowout_risk_reason", ""),
                "best_high_margin_alternatives": row.get("best_high_margin_alternatives", ""),
                "margin_method_sensitive": margin_method_sensitive_value,
                "market_consistent_differs": "yes" if market_consistent_differs else "no",
                "dixon_coles_differs": "yes" if dixon_coles_differs else "no",
                "public_strategy_differs": "yes" if public_strategy_differs else "no",
                "asian_handicap_shift_recommendation": "yes" if asian_handicap_shift else "no",
                "margin_ev_gap": row.get("margin_ev_gap", pd.NA),
                "draw_vs_decisive_gap": row.get("draw_vs_decisive_gap", pd.NA),
                "margin_diagnostic_note": row.get("margin_diagnostic_note", ""),
                "warning_flags": row.get("warning_flags", ""),
                "risk_notes": risk_notes,
                "decision_note": decision_note,
                "final_decision_basis": "default_ev_recommendation",
            }
        )
    return pd.DataFrame(rows)


def _ev_gap_best_vs_second(
    recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation,
) -> float | object:
    """Return the EV separation between a model's first and second choices."""

    return (
        recommendation.best.expected_points - recommendation.alternatives[0].expected_points
        if recommendation.alternatives
        else pd.NA
    )


def _summarise_text_values(rows: pd.DataFrame, column: str) -> object:
    """Join distinct non-empty optional source metadata values."""

    if column not in rows:
        return pd.NA
    values = sorted({str(value).strip() for value in rows[column].dropna() if str(value).strip()})
    return "; ".join(values) if values else pd.NA


def _format_total_goals_lines(rows: pd.DataFrame) -> str:
    """Format distinct totals lines compactly for recommendation diagnostics."""

    if rows.empty:
        return ""
    return "; ".join(f"{float(line):g}" for line in sorted(rows["line"].unique()))


def _format_asian_handicap_lines(rows: pd.DataFrame) -> str:
    """Format distinct handicap lines compactly for recommendation diagnostics."""

    if rows.empty:
        return ""
    return "; ".join(f"{float(line):+g}" for line in sorted(rows["handicap"].unique()))


def _asian_handicap_orientation_diagnostic(
    rows: pd.DataFrame,
    *,
    market_a_win: float,
    market_b_win: float,
) -> tuple[pd.DataFrame, bool, str]:
    """Flag AH ladders that appear reversed relative to the 1X2 favourite."""

    if rows.empty or "handicap" not in rows:
        return rows, False, ""
    annotated = rows.copy()
    annotated["asian_handicap_orientation_suspicious"] = "no"
    annotated["asian_handicap_orientation_reason"] = ""
    favourite_gap = float(market_a_win) - float(market_b_win)
    if abs(favourite_gap) < 0.10:
        return annotated, False, "1x2_favourite_not_strong_enough_for_orientation_check"

    candidates = annotated
    if "selected_for_market_consistent" in candidates:
        selected = candidates[candidates["selected_for_market_consistent"].astype(str).str.lower().eq("yes")]
        if not selected.empty:
            candidates = selected
    if "fair_team_a" in candidates:
        near_money = candidates[candidates["fair_team_a"].astype(float).between(0.25, 0.75)]
        if not near_money.empty:
            candidates = near_money
    candidates = candidates[candidates["handicap"].astype(float).abs() >= 0.25]
    if candidates.empty:
        return annotated, False, "no_near_money_directional_asian_handicap_lines"

    weights = (
        candidates["selection_weight"].astype(float)
        if "selection_weight" in candidates
        else candidates.get("bookmakers_count", pd.Series(1.0, index=candidates.index)).astype(float)
    )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = pd.Series(1.0, index=candidates.index)
    weighted_handicap = float(np.average(candidates["handicap"].astype(float), weights=weights))
    ah_favourite = "team_a" if weighted_handicap < -0.25 else "team_b" if weighted_handicap > 0.25 else ""
    one_x_two_favourite = "team_a" if favourite_gap > 0 else "team_b"
    if not ah_favourite:
        return annotated, False, f"ah_direction_neutral:weighted_handicap={weighted_handicap:+.2f}"
    suspicious = ah_favourite != one_x_two_favourite
    reason = (
        f"asian_handicap_orientation_suspicious:1x2={one_x_two_favourite};"
        f"ah={ah_favourite};weighted_handicap={weighted_handicap:+.2f}"
        if suspicious
        else f"orientation_ok:1x2={one_x_two_favourite};ah={ah_favourite};weighted_handicap={weighted_handicap:+.2f}"
    )
    if suspicious:
        annotated["asian_handicap_orientation_suspicious"] = "yes"
        annotated["asian_handicap_orientation_reason"] = reason
    else:
        annotated["asian_handicap_orientation_reason"] = reason
    return annotated, suspicious, reason


def _format_total_goals_fit_diagnostics(
    lambda_a: float,
    lambda_b: float,
    targets: tuple[tuple[float, float], ...],
) -> str:
    """Show the market target, fitted model probability, and error per totals line."""

    diagnostics: list[str] = []
    for line, market_probability in targets:
        model_probability = poisson_over_total_probability(lambda_a, lambda_b, line)
        diagnostics.append(
            f"{line:g}: market_over={market_probability:.4f} "
            f"model_over={model_probability:.4f} error={model_probability - market_probability:+.4f}"
        )
    return "; ".join(diagnostics)


def _summarise_odds_metadata(odds: pd.DataFrame, bookmaker_probabilities: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Summarise optional source metadata and valid 1X2 bookmaker coverage by match."""

    valid_columns = {"fair_a_win", "fair_draw", "fair_b_win"}
    valid_rows = (
        bookmaker_probabilities.dropna(subset=list(valid_columns))
        if valid_columns.issubset(bookmaker_probabilities.columns)
        else pd.DataFrame(columns=bookmaker_probabilities.columns)
    )
    summaries: dict[str, dict[str, object]] = {}
    for match_id, rows in odds.groupby("match_id", sort=False):
        timestamps = (
            pd.to_datetime(rows["odds_timestamp"], errors="coerce", utc=True).dropna()
            if "odds_timestamp" in rows
            else pd.Series(dtype="datetime64[ns, UTC]")
        )
        valid_match_rows = valid_rows[valid_rows["match_id"] == match_id]
        bookmakers = sorted({str(value) for value in valid_match_rows["bookmaker"].dropna()})
        summaries[str(match_id)] = {
            "odds_timestamp_min": timestamps.min().isoformat() if not timestamps.empty else pd.NA,
            "odds_timestamp_max": timestamps.max().isoformat() if not timestamps.empty else pd.NA,
            "latest_odds_timestamp": timestamps.max() if not timestamps.empty else None,
            "bookmakers_used": "; ".join(bookmakers),
            "number_of_bookmakers": len(bookmakers),
            "odds_source_urls": _summarise_text_values(rows, "odds_source_url"),
            "source_qualities": _summarise_text_values(rows, "source_quality"),
            "source_notes": _summarise_text_values(rows, "notes"),
        }
    return summaries


def _ev_gap_vs_modal(
    recommendation: GroupPredictionRecommendation | KnockoutPredictionRecommendation,
    matrix: ScoreProbabilityMatrix,
    qualifier_probabilities: dict[str, float] | None,
    config: ProjectConfig,
) -> float:
    """Compare the chosen prediction EV with the modal-score prediction EV."""

    modal_a, modal_b = recommendation.most_likely_scoreline
    if isinstance(recommendation, KnockoutPredictionRecommendation):
        modal = evaluate_knockout_prediction(
            matrix,
            modal_a,
            modal_b,
            recommendation.best.predicted_qualifier,
            qualifier_probabilities or {},
            config.knockout_scoring,
        )
    else:
        modal = evaluate_group_prediction(matrix, modal_a, modal_b)
    return recommendation.best.expected_points - modal.expected_points


def _warning_flags(
    *,
    number_of_bookmakers: int,
    has_over_under: bool,
    has_btts: bool,
    has_qualification_odds: bool,
    knockout: bool,
    favourite_bucket: str,
    calibration_loss: float,
    tail_mass: float,
    latest_odds_timestamp: pd.Timestamp | None,
    poor_calibration_loss_threshold: float,
    lambda_a_near_bound: bool,
    lambda_b_near_bound: bool,
    correct_score_blend_suppressed: bool,
    challenger_models_disagree: bool,
) -> str:
    """Build concise model-risk flags without changing any model decisions."""

    flags: list[str] = []
    if number_of_bookmakers == 1:
        flags.append("only_one_bookmaker")
    if not has_over_under:
        flags.append("no_over_under")
    if not has_btts:
        flags.append("no_btts")
    if calibration_loss > poor_calibration_loss_threshold:
        flags.append("high_calibration_error")
    if lambda_a_near_bound:
        flags.append("lambda_a_near_bound")
    if lambda_b_near_bound:
        flags.append("lambda_b_near_bound")
    if tail_mass > HIGH_TAIL_MASS_THRESHOLD:
        flags.append("high_tail_mass")
    if latest_odds_timestamp is not None and latest_odds_timestamp < pd.Timestamp.now(tz="UTC") - STALE_ODDS_THRESHOLD:
        flags.append("stale_odds_timestamp")
    if favourite_bucket == "extreme_favourite":
        flags.append("extreme_favourite")
    if knockout and not has_qualification_odds:
        flags.append("knockout_missing_qualification_odds")
    if correct_score_blend_suppressed:
        flags.append("correct_score_blend_suppressed_sparse_market")
    if challenger_models_disagree:
        flags.append("challenger_model_recommendations_disagree")
    return "; ".join(flags)


def run_prediction_workflow(
    odds: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    config: ProjectConfig | None = None,
    correct_score_odds: pd.DataFrame | None = None,
    total_goals_odds: pd.DataFrame | None = None,
    asian_handicap_odds: pd.DataFrame | None = None,
    runtime_timings: dict[str, float] | None = None,
    extra_warning_flags_by_match: Mapping[str, Sequence[str]] | None = None,
) -> PredictionWorkflowResult:
    """Produce market-implied score recommendations and optional friend EV analysis."""

    config = config or ProjectConfig()
    extra_warning_flags_by_match = extra_warning_flags_by_match or {}
    devig = config.resolved_devig()
    bookmaker_probabilities = process_bookmaker_odds(
        odds,
        devig.default_method,
        config.suspicious_overround_low,
        config.suspicious_overround_high,
        market_methods={
            "1x2": devig.method_for("1x2"),
            "over_under_2_5": devig.method_for("over_under_2_5"),
            "btts": devig.method_for("btts"),
            "qualification": devig.method_for("qualification"),
        },
    )
    odds_metadata = _summarise_odds_metadata(odds, bookmaker_probabilities)
    processed_correct_scores = (
        process_correct_score_odds(correct_score_odds, devig.method_for("correct_score"))
        if correct_score_odds is not None
        else pd.DataFrame()
    )
    processed_total_goals = (
        process_total_goals_odds(
            total_goals_odds,
            devig.method_for("total_goals"),
            config.suspicious_overround_low,
            config.suspicious_overround_high,
        )
        if total_goals_odds is not None
        else pd.DataFrame()
    )
    aggregated_total_goals = aggregate_total_goals_probabilities(processed_total_goals)
    processed_asian_handicap = (
        process_asian_handicap_odds(
            asian_handicap_odds,
            devig.method_for("asian_handicap"),
            config.suspicious_overround_low,
            config.suspicious_overround_high,
        )
        if asian_handicap_odds is not None
        else pd.DataFrame()
    )
    aggregated_asian_handicap = aggregate_asian_handicap_probabilities(processed_asian_handicap)
    aggregated = aggregate_bookmaker_probabilities(bookmaker_probabilities, config.bookmaker_aggregation_method)
    required_1x2 = {"fair_a_win", "fair_draw", "fair_b_win"}
    if not required_1x2.issubset(aggregated.columns):
        raise ValueError("No complete 1X2 market is available for calibration")
    missing_1x2 = aggregated[list(required_1x2)].isna().any(axis=1)
    if missing_1x2.any():
        match_ids = aggregated.loc[missing_1x2, "match_id"].astype(str).tolist()
        raise ValueError(f"No complete 1X2 market is available for matches: {match_ids}")
    odds_with_group = odds.assign(
        group=odds["group"] if "group" in odds else pd.NA,
        time=odds["time"] if "time" in odds else "",
    )
    matches = odds_with_group[["match_id", "date", "time", "stage", "group", "team_a", "team_b"]].drop_duplicates("match_id")
    market = matches.merge(aggregated, on="match_id", validate="one_to_one")

    report_rows: list[dict[str, object]] = []
    matrices: dict[str, ScoreProbabilityMatrix] = {}
    recommendations: dict[str, GroupPredictionRecommendation | KnockoutPredictionRecommendation] = {}
    qualification: dict[str, dict[str, float]] = {}
    correct_score_matrices: dict[str, ScoreProbabilityMatrix] = {}
    baseline_matrices: dict[str, ScoreProbabilityMatrix] = {}
    challenger_matrices: dict[str, dict[str, ScoreProbabilityMatrix]] = {}
    margin_diagnostic_frames: list[pd.DataFrame] = []
    asian_handicap_diagnostic_frames: list[pd.DataFrame] = []
    for _, row in market.iterrows():
        match_id = str(row["match_id"])
        metadata = odds_metadata[match_id]
        match_total_goals = (
            aggregated_total_goals[aggregated_total_goals["match_id"].astype(str) == match_id]
            if not aggregated_total_goals.empty
            else pd.DataFrame()
        )
        match_asian_handicap = (
            aggregated_asian_handicap[aggregated_asian_handicap["match_id"].astype(str) == match_id]
            if not aggregated_asian_handicap.empty
            else pd.DataFrame()
        )
        calibratable_total_goals = (
            match_total_goals[match_total_goals["used_for_calibration"]]
            if not match_total_goals.empty
            else pd.DataFrame()
        )
        skipped_total_goals = (
            match_total_goals[~match_total_goals["used_for_calibration"]]
            if not match_total_goals.empty
            else pd.DataFrame()
        )
        total_goals_targets = tuple(
            (float(total_row["line"]), float(total_row["fair_over"]))
            for _, total_row in calibratable_total_goals.iterrows()
        )
        legacy_over_2_5 = _optional_probability(row, "fair_over_2_5") if not total_goals_targets else None
        ou_median_total = _median_total_goals_line(
            calibratable_total_goals,
            match_total_goals,
            row.get("fair_over_2_5", pd.NA),
        )
        market_total_line_used = _format_total_goals_lines(calibratable_total_goals)
        targets = CalibrationTargets(
            float(row["fair_a_win"]),
            float(row["fair_draw"]),
            float(row["fair_b_win"]),
            legacy_over_2_5,
            _optional_probability(row, "fair_btts_yes"),
            total_goals_targets,
        )
        calibration = calibrate_poisson_model(
            targets,
            config.max_goals_score_matrix,
            config.calibration_weights,
            config.renormalise_score_matrix,
            config.poor_calibration_loss_threshold,
        )
        poisson_matrix = calibration.score_matrix
        match_asian_handicap = select_asian_handicap_constraints(
            match_asian_handicap,
            poisson_matrix.probabilities.shape,
            {"a_win": targets.a_win, "draw": targets.draw, "b_win": targets.b_win},
        )
        (
            match_asian_handicap,
            asian_handicap_orientation_suspicious,
            asian_handicap_orientation_reason,
        ) = _asian_handicap_orientation_diagnostic(
            match_asian_handicap,
            market_a_win=targets.a_win,
            market_b_win=targets.b_win,
        )
        if not match_asian_handicap.empty:
            asian_handicap_diagnostic_frames.append(match_asian_handicap)
        baseline_matrices[match_id] = poisson_matrix
        correct_score_blend_suppressed = False
        correct_score_blend_note = ""
        effective_correct_score_poisson_weight = 1.0
        correct_score_rows = (
            processed_correct_scores[processed_correct_scores["match_id"].astype(str) == match_id]
            if not processed_correct_scores.empty
            else pd.DataFrame()
        )
        if not correct_score_rows.empty:
            correct_score_coverage = summarise_correct_score_coverage(correct_score_rows)
            correct_score_aggregation = aggregate_correct_score_market(
                correct_score_rows,
                match_id,
                config.max_goals_score_matrix,
                config.correct_score_aggregation_method,
                config.correct_score_outlier_z_threshold,
            )
            correct_score_matrix = correct_score_aggregation.matrix
            correct_score_aggregation_diagnostics = correct_score_aggregation.diagnostics
            correct_score_bookmaker_diagnostics = format_correct_score_bookmaker_diagnostics(
                correct_score_aggregation.bookmaker_diagnostics
            )
            correct_score_matrices[match_id] = correct_score_matrix
            scoreline_count = int(correct_score_coverage["correct_score_scorelines_count"])
            correct_score_blend_suppressed = scoreline_count < config.min_scorelines_for_blend
            if correct_score_blend_suppressed:
                score_matrix = poisson_matrix
                correct_score_blend_note = (
                    "Correct-score blend suppressed: "
                    f"{scoreline_count} usable scorelines is below configured minimum "
                    f"{config.min_scorelines_for_blend}; used pure Poisson"
                )
            else:
                score_matrix = blend_score_matrices(
                    poisson_matrix,
                    correct_score_matrix,
                    config.correct_score_poisson_weight,
                )
                effective_correct_score_poisson_weight = config.correct_score_poisson_weight
            correct_score_market_top_10 = format_top_scorelines(correct_score_matrix)
            correct_score_blended_top_10 = format_top_scorelines(score_matrix)
            correct_score_kl_divergence = market_to_poisson_kl_divergence(poisson_matrix, correct_score_matrix)
            has_other_bucket = (
                "yes"
                if "has_other_bucket" in correct_score_rows
                and correct_score_rows["has_other_bucket"]
                .fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes"})
                .any()
                else "no"
            )
            correct_score_tail_mass = float(correct_score_matrix.tail_probability)
            has_correct_score_market = True
        else:
            correct_score_coverage = summarise_correct_score_coverage(correct_score_rows)
            correct_score_aggregation_diagnostics = {
                "correct_score_aggregation_method": "",
                "number_of_correct_score_bookmakers": 0,
                "average_correct_score_overround": pd.NA,
                "max_correct_score_overround": pd.NA,
                "outlier_count": 0,
                "top_outlier_examples": "",
                "scoreline_coverage_warning": "",
                "out_of_grid_scorelines_count": 0,
            }
            correct_score_bookmaker_diagnostics = ""
            score_matrix = poisson_matrix
            correct_score_market_top_10 = ""
            correct_score_blended_top_10 = ""
            correct_score_kl_divergence = pd.NA
            has_other_bucket = "no"
            correct_score_tail_mass = pd.NA
            has_correct_score_market = False
        matrices[match_id] = score_matrix
        if config.enable_dixon_coles_rho_estimation:
            dixon_coles_rho_estimate = estimate_dixon_coles_rho_from_market(
                lambda_a=calibration.lambda_a,
                lambda_b=calibration.lambda_b,
                market_matrix=correct_score_matrices.get(match_id),
                max_goals=config.max_goals_score_matrix,
                fallback_rho=config.dixon_coles_rho,
                rho_min=config.dixon_coles_rho_min,
                rho_max=config.dixon_coles_rho_max,
                grid_size=config.dixon_coles_rho_grid_size,
                renormalise=config.renormalise_score_matrix,
            )
        else:
            dixon_coles_rho_estimate = type(
                "ConfiguredDixonColesRho",
                (),
                {
                    "rho": config.dixon_coles_rho,
                    "source": "configured",
                    "fit_error": pd.NA,
                    "warning": "",
                },
            )()
        match_challenger_matrices = build_challenger_score_matrices(
            Match(match_id, str(row["stage"]), str(row["team_a"]), str(row["team_b"])),
            config.max_goals_score_matrix,
            (
                DixonColesScoreModel(
                    calibration.lambda_a,
                    calibration.lambda_b,
                    float(dixon_coles_rho_estimate.rho),
                    config.renormalise_score_matrix,
                ),
            ),
        )
        challenger_matrices[match_id] = match_challenger_matrices
        dixon_coles_matrix = match_challenger_matrices["dixon_coles"]
        notes = list(calibration.warnings)
        if str(getattr(dixon_coles_rho_estimate, "warning", "")):
            notes.append(str(dixon_coles_rho_estimate.warning))
        if not skipped_total_goals.empty:
            notes.append(
                "Stored but skipped from default Poisson calibration; used by market-consistent diagnostics: "
                + _format_total_goals_lines(skipped_total_goals)
            )
        if not match_asian_handicap.empty:
            notes.append(
                "Asian handicap market supplied; used only by market-consistent challenger diagnostics: "
                + _format_asian_handicap_lines(match_asian_handicap)
            )
        if asian_handicap_orientation_suspicious:
            notes.append("Asian handicap orientation is suspicious relative to 1X2 prices; inspect pasted team order")
        if correct_score_blend_note:
            notes.append(correct_score_blend_note)
        knockout = is_knockout_stage(str(row["stage"]))
        has_over_under = pd.notna(row.get("fair_over_2_5")) or bool(total_goals_targets)
        has_btts = pd.notna(row.get("fair_btts_yes"))
        has_qualification_odds = pd.notna(row.get("fair_a_qualifies")) and pd.notna(row.get("fair_b_qualifies"))
        if knockout:
            if config.knockout_scoring.knockout_scoring_mode == "unverified":
                notes.append("Knockout scoring mode is unverified; confirm Sporza rules before relying on knockout EV")
            if pd.notna(row.get("fair_a_qualifies")) and pd.notna(row.get("fair_b_qualifies")):
                qualifier_probabilities = {
                    str(row["team_a"]): float(row["fair_a_qualifies"]),
                    str(row["team_b"]): float(row["fair_b_qualifies"]),
                }
            else:
                qualifier_probabilities = {
                    str(row["team_a"]): targets.a_win + 0.5 * targets.draw,
                    str(row["team_b"]): targets.b_win + 0.5 * targets.draw,
                }
                notes.append("Qualification odds unavailable; used weak 90-minute draw-split approximation")
            qualification[match_id] = qualifier_probabilities
        else:
            qualifier_probabilities = None
        recommendation = _optimise_score_matrix(score_matrix, knockout, qualifier_probabilities, config)
        baseline_recommendation = _optimise_score_matrix(poisson_matrix, knockout, qualifier_probabilities, config)
        correct_score_blended_recommendation = recommendation
        dixon_coles_recommendation = _optimise_score_matrix(
            dixon_coles_matrix,
            knockout,
            qualifier_probabilities,
            config,
        )
        bivariate_diagnostic = _bivariate_poisson_diagnostic(
            match=Match(match_id, str(row["stage"]), str(row["team_a"]), str(row["team_b"])),
            calibration=calibration,
            config=config,
            knockout=knockout,
            qualifier_probabilities=qualifier_probabilities,
            default_score=recommendation.best.predicted_score,
        )
        margin_model_diagnostic = _asian_handicap_margin_model_diagnostic(
            config=config,
            match_asian_handicap=match_asian_handicap,
            lambda_a=calibration.lambda_a,
            lambda_b=calibration.lambda_b,
        )
        preliminary_ev_gap_to_second = _ev_gap_best_vs_second(recommendation)
        preliminary_extreme_favourite = (
            max(targets.a_win, targets.b_win) > EXTREME_FAVOURITE_PROBABILITY_THRESHOLD
            or poisson_matrix.tail_probability > HIGH_TAIL_MASS_THRESHOLD
        )
        run_market_consistent = _should_run_market_consistent_challenger(
            setting=config.enable_market_consistent_challenger,
            ev_gap_to_second=preliminary_ev_gap_to_second,
            extreme_favourite=preliminary_extreme_favourite,
            warning_flags="",
            config=config,
        )
        market_consistent_prior_source = "independent_poisson"
        market_consistent_prior_rho_used: object = pd.NA
        market_consistent_prior_covariance_used: object = pd.NA
        market_consistent_prior_kl_vs_independent: object = pd.NA
        market_consistent_prior_changed_recommendation = "no"
        market_consistent_prior_warning = ""
        if run_market_consistent:
            market_consistent_start = time.perf_counter()
            market_consistent_kwargs: dict[str, object] = {
                "total_goals": match_total_goals,
                "correct_score_matrix": correct_score_matrices.get(match_id),
            }
            if not match_asian_handicap.empty:
                market_consistent_kwargs["asian_handicap"] = match_asian_handicap
            market_consistent_kwargs["weights"] = _market_consistent_weights(config)
            bivariate_prior_matrix, bivariate_prior_covariance, bivariate_prior_warning = (
                _build_bivariate_prior_matrix(
                    match=Match(match_id, str(row["stage"]), str(row["team_a"]), str(row["team_b"])),
                    lambda_a=calibration.lambda_a,
                    lambda_b=calibration.lambda_b,
                    config=config,
                    max_goals=config.max_goals_score_matrix,
                )
                if config.market_consistent_prior == "bivariate_poisson"
                else (None, pd.NA, "")
            )
            prior_matrix, market_consistent_prior_source, prior_rho_used, market_consistent_prior_warning = (
                _select_market_consistent_prior(
                    prior_choice=config.market_consistent_prior,
                    poisson_matrix=poisson_matrix,
                    dixon_coles_matrix=dixon_coles_matrix,
                    dixon_coles_rho_estimate=dixon_coles_rho_estimate,
                    bivariate_matrix=bivariate_prior_matrix,
                    bivariate_warning=bivariate_prior_warning,
                )
            )
            if market_consistent_prior_source == "dixon_coles":
                market_consistent_prior_rho_used = prior_rho_used
            if market_consistent_prior_source == "bivariate_poisson":
                market_consistent_prior_covariance_used = float(bivariate_prior_covariance)
            market_consistent_prior_kl_vs_independent = (
                0.0
                if market_consistent_prior_source in {"independent_poisson", "independent_poisson_fallback"}
                else market_to_poisson_kl_divergence(poisson_matrix, prior_matrix)
            )
            market_consistent_result = fit_market_consistent_matrix(
                prior_matrix,
                {
                    "a_win": targets.a_win,
                    "draw": targets.draw,
                    "b_win": targets.b_win,
                    "btts_yes": _optional_probability(row, "fair_btts_yes"),
                },
                **market_consistent_kwargs,
            )
            market_consistent_matrix = market_consistent_result.matrix
            match_challenger_matrices["market_consistent_matrix"] = market_consistent_matrix
            market_consistent_recommendation = _optimise_score_matrix(
                market_consistent_matrix,
                knockout,
                qualifier_probabilities,
                config,
            )
            if market_consistent_prior_source not in {"independent_poisson", "independent_poisson_fallback"}:
                reference_result = fit_market_consistent_matrix(
                    poisson_matrix,
                    {
                        "a_win": targets.a_win,
                        "draw": targets.draw,
                        "b_win": targets.b_win,
                        "btts_yes": _optional_probability(row, "fair_btts_yes"),
                    },
                    **market_consistent_kwargs,
                )
                reference_recommendation = _optimise_score_matrix(
                    reference_result.matrix,
                    knockout,
                    qualifier_probabilities,
                    config,
                )
                market_consistent_prior_changed_recommendation = (
                    "yes"
                    if reference_recommendation.best.predicted_score
                    != market_consistent_recommendation.best.predicted_score
                    else "no"
                )
            if runtime_timings is not None:
                runtime_timings["market-consistent challenger"] = (
                    runtime_timings.get("market-consistent challenger", 0.0)
                    + time.perf_counter()
                    - market_consistent_start
                )
        else:
            market_consistent_matrix = None
            market_consistent_recommendation = None
            skip_note = (
                "market-consistent skipped: high-confidence default"
                if config.enable_market_consistent_challenger == "only_if_close"
                else "market-consistent skipped: disabled"
            )
            market_consistent_result = type(
                "SkippedMarketConsistentResult",
                (),
                {
                    "diagnostics": {
                        "market_consistent_status": "skipped",
                        "market_consistent_skip_reason": skip_note,
                        "market_consistent_optimisation_classification": "skipped",
                        "market_consistent_optimisation_success": pd.NA,
                        "market_consistent_optimisation_status_code": pd.NA,
                        "market_consistent_optimisation_message": skip_note,
                        "market_consistent_optimisation_iterations": pd.NA,
                        "market_consistent_final_objective_value": pd.NA,
                        "market_consistent_gradient_norm": pd.NA,
                        "market_consistent_max_constraint_error": pd.NA,
                        "market_consistent_constraint_count": 0,
                        "market_consistent_1x2_constraint_count": 0,
                        "market_consistent_btts_constraint_count": 0,
                        "market_consistent_total_goals_constraint_count": 0,
                        "market_consistent_asian_handicap_constraint_count": 0,
                        "market_consistent_correct_score_constraint_count": 0,
                        "market_consistent_kl_divergence_vs_prior": pd.NA,
                        "market_consistent_1x2_fit_error": pd.NA,
                        "market_consistent_btts_fit_error": pd.NA,
                        "market_consistent_total_goals_fit_error": pd.NA,
                        "market_consistent_asian_handicap_fit_error": pd.NA,
                        "market_consistent_correct_score_fit_error": pd.NA,
                        "market_consistent_asian_totals_used": "",
                        "market_consistent_asian_handicap_lines_available": len(match_asian_handicap),
                        "market_consistent_asian_handicap_lines_selected": int(
                            match_asian_handicap["selected_for_market_consistent"]
                            .astype(str)
                            .str.lower()
                            .eq("yes")
                            .sum()
                        )
                        if not match_asian_handicap.empty and "selected_for_market_consistent" in match_asian_handicap
                        else 0,
                        "market_consistent_asian_handicap_lines_skipped": int(
                            match_asian_handicap["selected_for_market_consistent"]
                            .astype(str)
                            .str.lower()
                            .ne("yes")
                            .sum()
                        )
                        if not match_asian_handicap.empty and "selected_for_market_consistent" in match_asian_handicap
                        else 0,
                        "market_consistent_asian_handicap_lines_used": "",
                        "market_consistent_asian_handicap_lines_skipped_detail": "",
                        "market_consistent_asian_handicap_fit_error_selected": pd.NA,
                        "market_consistent_asian_handicap_fit_error_all": pd.NA,
                        "market_consistent_margin_distribution_before": "",
                        "market_consistent_margin_distribution_after": "",
                        "market_consistent_largest_margin_shift": pd.NA,
                        "market_consistent_largest_margin_shift_value": pd.NA,
                    }
                },
            )()
        recommended_qualifier = (
            recommendation.best.predicted_qualifier
            if isinstance(recommendation, KnockoutPredictionRecommendation)
            else ""
        )
        recommendations[match_id] = recommendation
        model_outcomes = calibration.model_probabilities
        total_goals_line_fit_error = (
            float(
                np.mean(
                    [
                        (
                            poisson_over_total_probability(calibration.lambda_a, calibration.lambda_b, line)
                            - probability
                        )
                        ** 2
                        for line, probability in total_goals_targets
                    ]
                )
            )
            if total_goals_targets
            else pd.NA
        )
        total_goals_line_diagnostics = _format_total_goals_fit_diagnostics(
            calibration.lambda_a,
            calibration.lambda_b,
            total_goals_targets,
        )
        favourite_probability = max(targets.a_win, targets.b_win)
        favourite_is_team_a = targets.a_win >= targets.b_win
        ah_favourite_summary = _asian_handicap_favourite_summary(
            match_asian_handicap,
            favourite_is_team_a=favourite_is_team_a,
        )
        bucket = favourite_strength_bucket(favourite_probability)
        ev_gap_best_vs_second = _ev_gap_best_vs_second(recommendation)
        ev_gap_best_vs_modal = _ev_gap_vs_modal(
            recommendation,
            score_matrix,
            qualification.get(match_id),
            config,
        )
        most_likely_expected_points = recommendation.best.expected_points - ev_gap_best_vs_modal
        baseline_poisson_recommended_score = baseline_recommendation.best.predicted_score
        correct_score_blended_recommended_score = correct_score_blended_recommendation.best.predicted_score
        final_live_recommended_score = recommendation.best.predicted_score
        dixon_coles_recommended_score = dixon_coles_recommendation.best.predicted_score
        market_consistent_recommended_score = (
            market_consistent_recommendation.best.predicted_score
            if market_consistent_recommendation is not None
            else final_live_recommended_score
        )
        market_consistent_differs_from_default = (
            market_consistent_recommendation is not None
            and market_consistent_recommended_score != final_live_recommended_score
        )
        market_consistent_asian_totals_used = str(
            market_consistent_result.diagnostics.get("market_consistent_asian_totals_used", "")
        )
        market_consistent_asian_totals_shift_recommendation = (
            bool(market_consistent_asian_totals_used)
            and market_consistent_recommended_score != baseline_poisson_recommended_score
        )
        market_consistent_asian_handicap_lines_used = str(
            market_consistent_result.diagnostics.get("market_consistent_asian_handicap_lines_used", "")
        )
        asian_handicap_shift_recommendation = (
            bool(market_consistent_asian_handicap_lines_used)
            and market_consistent_recommended_score != baseline_poisson_recommended_score
        )
        score_audit = _score_matrix_audit(score_matrix)
        market_btts_yes = row.get("fair_btts_yes", pd.NA)
        market_btts_no = row.get("fair_btts_no", pd.NA)
        btts_fit_error = (
            score_audit["model_implied_btts_yes_probability"] - float(market_btts_yes)
            if pd.notna(market_btts_yes)
            else pd.NA
        )
        ev_decomposition_records = _ev_decomposition_records(match_id, recommendation, config)
        top_10_evaluations = _top_ev_evaluations(score_matrix, knockout, qualifier_probabilities, config, 10)
        all_candidate_evaluations = _top_ev_evaluations(
            score_matrix,
            knockout,
            qualifier_probabilities,
            config,
            (config.max_candidate_goals + 1) ** 2,
        )
        top_10_ev_decomposition_records = _ev_decomposition_records_from_evaluations(
            match_id,
            top_10_evaluations,
            config,
        )
        ev_decomposition_json = json.dumps(ev_decomposition_records)
        top_10_ev_decomposition_json = json.dumps(top_10_ev_decomposition_records)
        if market_consistent_matrix is not None:
            market_consistent_top_10_evaluations = _top_ev_evaluations(
                market_consistent_matrix,
                knockout,
                qualifier_probabilities,
                config,
                10,
            )
            market_consistent_top_10_ev_decomposition_records = _ev_decomposition_records_from_evaluations(
                match_id,
                market_consistent_top_10_evaluations,
                config,
            )
            market_consistent_top_10_ev_decomposition_json = json.dumps(
                market_consistent_top_10_ev_decomposition_records
            )
            market_consistent_top_10_probability_scorelines = format_top_scorelines(market_consistent_matrix)
        else:
            market_consistent_top_10_evaluations = ()
            market_consistent_top_10_ev_decomposition_records = []
            market_consistent_top_10_ev_decomposition_json = "[]"
            market_consistent_top_10_probability_scorelines = ""
        if not knockout:
            match_margin_diagnostics = margin_ev_diagnostics(
                score_matrix,
                match_id,
                config.max_candidate_goals,
            )
            margin_diagnostic_frames.append(match_margin_diagnostics)
            top_margin = match_margin_diagnostics.iloc[0] if not match_margin_diagnostics.empty else pd.Series(dtype=object)
            second_margin = (
                match_margin_diagnostics.iloc[1]
                if len(match_margin_diagnostics) > 1
                else pd.Series(dtype=object)
            )
            margin_ev_gap = (
                float(top_margin["representative_scoreline_ev"] - second_margin["representative_scoreline_ev"])
                if not top_margin.empty and not second_margin.empty
                else pd.NA
            )
            margin_diagnostic_note = (
                f"Best margin {int(top_margin['margin']):+d} via {top_margin['best_scoreline_on_margin']} "
                f"beats next margin by {float(margin_ev_gap):.3f} EV."
                if pd.notna(margin_ev_gap)
                else ""
            )
            largest_margin_shift_value = market_consistent_result.diagnostics.get(
                "market_consistent_largest_margin_shift_value",
                pd.NA,
            )
            if (
                bool(market_consistent_asian_handicap_lines_used)
                and pd.notna(largest_margin_shift_value)
                and abs(float(largest_margin_shift_value)) >= 0.02
            ):
                margin_diagnostic_note = " ".join(
                    filter(
                        None,
                        [
                            margin_diagnostic_note,
                            (
                                "AH constraints shift margin distribution: "
                                f"largest shift at margin "
                                f"{market_consistent_result.diagnostics.get('market_consistent_largest_margin_shift')} "
                                f"({float(largest_margin_shift_value):+.3f})."
                            ),
                        ],
                    )
                )
            draw_decisive = draw_vs_decisive_diagnostics(
                score_matrix,
                config.max_candidate_goals,
                config.strategies.low_confidence_ev_gap_threshold,
            )
        else:
            margin_ev_gap = pd.NA
            margin_diagnostic_note = "Margin decomposition diagnostics are group-stage only."
            draw_decisive = {
                "best_draw_score": "",
                "best_draw_ev": pd.NA,
                "best_decisive_score": "",
                "best_decisive_ev": pd.NA,
                "draw_vs_decisive_gap": pd.NA,
                "draw_boundary_flag": "",
            }
        market_consistent_status = str(
            market_consistent_result.diagnostics.get("market_consistent_status", "")
        )
        market_consistent_classification = str(
            market_consistent_result.diagnostics.get("market_consistent_optimisation_classification", "")
        )
        market_consistent_warning_flags = []
        if market_consistent_classification == "optimisation_failed_severe":
            market_consistent_warning_flags.extend(
                ["market_consistent_optimisation_failed", "market_consistent_optimisation_failed_severe"]
            )
        elif market_consistent_classification == "optimisation_not_fully_converged_but_fit_acceptable":
            market_consistent_warning_flags.append(
                "market_consistent_optimisation_not_fully_converged_but_fit_acceptable"
            )
        elif market_consistent_classification == "constraint_fit_poor" or market_consistent_status == "poor_fit":
            market_consistent_warning_flags.append("market_consistent_constraint_fit_poor")
        normal_grid_tail_mass = poisson_matrix.tail_probability
        extreme_favourite_audit_triggered = (
            favourite_probability > EXTREME_FAVOURITE_PROBABILITY_THRESHOLD
            or normal_grid_tail_mass > HIGH_TAIL_MASS_THRESHOLD
        )
        market_consistent_high_score_comparison = (
            _market_consistent_high_score_comparison(
                poisson_matrix=poisson_matrix,
                market_consistent_matrix=market_consistent_matrix,
                poisson_recommendation=baseline_recommendation,
                market_consistent_recommendation=market_consistent_recommendation,
                qualifier_probabilities=qualifier_probabilities,
                config=config,
                favourite_is_team_a=favourite_is_team_a,
            )
            if market_consistent_matrix is not None and market_consistent_recommendation is not None
            else _empty_market_consistent_high_score_comparison()
        )
        if extreme_favourite_audit_triggered and config.dynamic_grid_enabled:
            extreme_audit = _extreme_favourite_audit(
                matrix=score_matrix,
                favourite_is_team_a=favourite_is_team_a,
                recommendation=recommendation,
                qualifier_probabilities=qualifier_probabilities,
                config=config,
            )
            larger_grid_sensitivity = _larger_grid_sensitivity(
                targets=targets,
                config=config,
                knockout=knockout,
                qualifier_probabilities=qualifier_probabilities,
                final_recommendation=recommendation,
                normal_poisson_matrix=poisson_matrix,
                normal_poisson_recommendation=baseline_recommendation,
                favourite_is_team_a=favourite_is_team_a,
                larger_max_goals=config.extreme_favourite_max_goals,
            )
        else:
            extreme_audit = {
                "top_clean_sheet_scores": "",
                "top_favourite_margin_probabilities": "",
                "top_5_favourite_score_count_probabilities": "",
                "high_score_cluster_scores": "",
                "three_four_five_nil_within_0_10_ev": False,
                "high_score_cluster": "no",
                "high_score_cluster_close_alternatives": "",
            }
            larger_grid_sensitivity = {
                "current_final_recommendation": _score_label(final_live_recommended_score),
                "normal_grid_recommendation": _score_label(final_live_recommended_score),
                "normal_grid_poisson_recommendation": _score_label(baseline_poisson_recommended_score),
                "larger_grid_recommendation": "",
                "larger_grid_poisson_recommendation": "",
                "recommendation_changes_with_larger_grid": False,
                "larger_grid_poisson_changes_recommendation": False,
                "larger_grid_differs_from_live_recommendation": False,
                "larger_grid_tail_mass": pd.NA,
                "grid_max_goals_used": int(config.max_goals_score_matrix),
                "tail_mass_before_grid_extension": float(normal_grid_tail_mass),
                "tail_mass_after_grid_extension": float(normal_grid_tail_mass),
                "recommendation_changed_due_to_larger_grid": "no",
                "ev_3_0_normal_grid": pd.NA,
                "ev_4_0_normal_grid": pd.NA,
                "ev_5_0_normal_grid": pd.NA,
                "ev_3_0_larger_grid": pd.NA,
                "ev_4_0_larger_grid": pd.NA,
                "ev_5_0_larger_grid": pd.NA,
                "ev_favourite_3_0_normal_grid": pd.NA,
                "ev_favourite_4_0_normal_grid": pd.NA,
                "ev_favourite_5_0_normal_grid": pd.NA,
                "ev_favourite_3_0_larger_grid": pd.NA,
                "ev_favourite_4_0_larger_grid": pd.NA,
                "ev_favourite_5_0_larger_grid": pd.NA,
            }
        model_recommendations_agree = len(
            {
                baseline_poisson_recommended_score,
                correct_score_blended_recommended_score,
                final_live_recommended_score,
                dixon_coles_recommended_score,
            }
        ) == 1
        model_disagreement_warning = "" if model_recommendations_agree else "challenger_model_recommendations_disagree"
        warning_flags = _warning_flags(
            number_of_bookmakers=int(metadata["number_of_bookmakers"]),
            has_over_under=has_over_under,
            has_btts=has_btts,
            has_qualification_odds=has_qualification_odds,
            knockout=knockout,
            favourite_bucket=bucket,
            calibration_loss=calibration.loss,
            tail_mass=poisson_matrix.tail_probability,
            latest_odds_timestamp=metadata["latest_odds_timestamp"],
            poor_calibration_loss_threshold=config.poor_calibration_loss_threshold,
            lambda_a_near_bound="lambda_a_near_bound" in calibration.warnings,
            lambda_b_near_bound="lambda_b_near_bound" in calibration.warnings,
            correct_score_blend_suppressed=correct_score_blend_suppressed,
            challenger_models_disagree=not model_recommendations_agree,
        )
        btts_warning_flags = _btts_warning_flags(
            has_btts=has_btts,
            market_btts_yes=market_btts_yes,
            market_btts_no=market_btts_no,
            model_btts_yes=score_audit["model_implied_btts_yes_probability"],
            recommended_score=final_live_recommended_score,
        )
        high_score_warning_flags = []
        if bool(extreme_audit["three_four_five_nil_within_0_10_ev"]):
            high_score_warning_flags.append("high_score_ev_cluster")
        if larger_grid_sensitivity["recommendation_changes_with_larger_grid"] is True:
            high_score_warning_flags.append("larger_grid_changes_recommendation")
        if larger_grid_sensitivity["larger_grid_differs_from_live_recommendation"] is True:
            high_score_warning_flags.append("larger_grid_sensitivity")
        knockout_warning_flags = []
        if knockout and config.knockout_scoring.knockout_scoring_mode == "unverified":
            knockout_warning_flags.append("knockout_scoring_unverified")
        dixon_coles_warning_flags = []
        if str(getattr(dixon_coles_rho_estimate, "warning", "")):
            dixon_coles_warning_flags.append("dixon_coles_rho_not_estimated")
        asian_handicap_warning_flags = []
        if asian_handicap_orientation_suspicious:
            asian_handicap_warning_flags.append("asian_handicap_orientation_suspicious")
        extra_warning_flags = [
            str(flag).strip()
            for flag in extra_warning_flags_by_match.get(match_id, ())
            if str(flag).strip()
        ]
        warning_flags = "; ".join(
            filter(
                None,
                [
                    warning_flags,
                    *btts_warning_flags,
                    *high_score_warning_flags,
                    *knockout_warning_flags,
                    *market_consistent_warning_flags,
                    *dixon_coles_warning_flags,
                    *asian_handicap_warning_flags,
                    *extra_warning_flags,
                ],
            )
        )
        if market_consistent_warning_flags:
            if "market_consistent_optimisation_not_fully_converged_but_fit_acceptable" in market_consistent_warning_flags:
                notes.append(
                    "Market-consistent optimiser did not fully converge, but constraint fit is acceptable: "
                    + str(market_consistent_result.diagnostics.get("market_consistent_optimisation_message", ""))
                )
            elif "market_consistent_constraint_fit_poor" in market_consistent_warning_flags:
                notes.append(
                    "Market-consistent optimiser converged with poor constraint fit; inspect market consistency diagnostics"
                )
            else:
                notes.append(
                    "Market-consistent optimiser did not converge: "
                    + str(market_consistent_result.diagnostics.get("market_consistent_optimisation_message", ""))
                )
        plausible_decision_layer = _plausible_alternative_decision_layer(
            top_10_ev_decomposition_records,
            config=config,
            extreme_favourite=extreme_favourite_audit_triggered,
            favourite_is_team_a=favourite_is_team_a,
        )
        decision_aid = _decision_aid(
            top_10_ev_decomposition_records,
            bool(extreme_audit["three_four_five_nil_within_0_10_ev"]),
            str(extreme_audit["high_score_cluster_close_alternatives"]),
            plausible_top_alternatives=str(plausible_decision_layer["plausible_top_alternatives"]),
            suppressed_ev_candidates=str(plausible_decision_layer["suppressed_ev_candidates"]),
        )
        public_strategy = build_public_strategy(
            match_row=pd.Series(
                {
                    "team_a": row["team_a"],
                    "team_b": row["team_b"],
                    "favourite_probability": favourite_probability,
                    "warning_flags": warning_flags,
                    "correct_score_market_top_10": correct_score_market_top_10,
                }
            ),
            matrix=score_matrix,
            recommendation=recommendation,
            config=config,
            qualifier_probabilities=qualification.get(match_id),
            correct_score_market_matrix=correct_score_matrices.get(match_id),
            friend_predictions=(
                predictions[predictions["match_id"].astype(str) == match_id]
                if predictions is not None and not predictions.empty
                else None
            ),
        )
        modal_draw_challenger = _modal_draw_challenger_diagnostic(
            recommendation=recommendation,
            favourite_probability=favourite_probability,
            draw_decisive=draw_decisive,
            ev_minus_modal_expected_gap=ev_gap_best_vs_modal,
            config=config,
        )
        draw_prone = _draw_prone_diagnostic(
            ou_median_total=ou_median_total,
            favourite_probability=favourite_probability,
            market_draw_probability=targets.draw,
            recommendation=recommendation,
            draw_decisive=draw_decisive,
            config=config,
        )
        high_margin_alternatives = _format_high_margin_alternatives(
            all_candidate_evaluations,
            favourite_is_team_a=favourite_is_team_a,
        )
        blowout_risk = _blowout_risk_diagnostic(
            favourite_probability=favourite_probability,
            ou_median_total=ou_median_total,
            high_margin_alternatives=high_margin_alternatives,
            high_score_tail_mass=float(normal_grid_tail_mass),
            larger_grid_recommendation=larger_grid_sensitivity.get("larger_grid_poisson_recommendation", ""),
            current_recommendation=_score_label(final_live_recommended_score),
            config=config,
        )
        btts_conflict = _btts_conflict_diagnostic(
            recommended_score=final_live_recommended_score,
            recommended_expected_points=recommendation.best.expected_points,
            market_btts_yes=market_btts_yes,
            model_btts_yes=score_audit["model_implied_btts_yes_probability"],
            favourite_probability=favourite_probability,
            draw_prone_flag=draw_prone["draw_prone_flag"] == "yes",
            evaluations=all_candidate_evaluations,
            config=config,
        )
        if draw_prone["draw_prone_flag"] == "yes":
            warning_flags = "; ".join(filter(None, [warning_flags, "draw_prone_profile"]))
        if blowout_risk["blowout_risk_flag"] == "yes":
            warning_flags = "; ".join(filter(None, [warning_flags, "blowout_risk_profile"]))
        if btts_conflict["btts_conflict_flag"] == "yes":
            warning_flags = "; ".join(filter(None, [warning_flags, "btts_conflict"]))
        (
            market_consistent_active_constraint_groups,
            market_consistent_constraint_correlation_note,
        ) = _market_consistent_correlation_note(market_consistent_result.diagnostics)
        report_rows.append(
            {
                "match_id": match_id,
                "date": row["date"],
                "time": row.get("time", ""),
                "stage": row["stage"],
                "group": row["group"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "bookmaker_raw_1x2": _format_bookmaker_raw_probabilities(bookmaker_probabilities, match_id),
                "bookmaker_fair_1x2": _format_bookmaker_fair_probabilities(bookmaker_probabilities, match_id),
                "market_a_win": targets.a_win,
                "market_draw": targets.draw,
                "market_b_win": targets.b_win,
                "favourite_probability": favourite_probability,
                "favourite_team": row["team_a"] if targets.a_win >= targets.b_win else row["team_b"],
                "favourite_bucket": bucket,
                "odds_timestamp_min": metadata["odds_timestamp_min"],
                "odds_timestamp_max": metadata["odds_timestamp_max"],
                "bookmakers_used": metadata["bookmakers_used"],
                "number_of_bookmakers": metadata["number_of_bookmakers"],
                "margin_removal_method": _canonical_margin_method(config.margin_removal_method),
                **_devig_method_summary(
                    match_id=match_id,
                    bookmaker_probabilities=bookmaker_probabilities,
                    processed_total_goals=processed_total_goals,
                    processed_asian_handicap=processed_asian_handicap,
                    processed_correct_scores=processed_correct_scores,
                ),
                "market_consistent_prior_source": market_consistent_prior_source,
                "market_consistent_prior_dixon_coles_rho_used": market_consistent_prior_rho_used,
                "market_consistent_prior_covariance_used": market_consistent_prior_covariance_used,
                "market_consistent_prior_kl_vs_independent": market_consistent_prior_kl_vs_independent,
                "market_consistent_prior_changed_recommendation": market_consistent_prior_changed_recommendation,
                "market_consistent_prior_warning": market_consistent_prior_warning,
                **bivariate_diagnostic,
                **margin_model_diagnostic,
                "market_consistent_active_constraint_groups": market_consistent_active_constraint_groups,
                "market_consistent_constraint_correlation_note": market_consistent_constraint_correlation_note,
                "odds_source_urls": metadata["odds_source_urls"],
                "source_qualities": metadata["source_qualities"],
                "source_notes": metadata["source_notes"],
                "has_over_under": has_over_under,
                "total_goals_lines_available": _format_total_goals_lines(match_total_goals),
                "total_goals_lines_used_for_calibration": _format_total_goals_lines(calibratable_total_goals),
                "market_total_line_used": market_total_line_used,
                "ou_median_total": ou_median_total,
                "total_goals_lines_skipped_for_calibration": _format_total_goals_lines(skipped_total_goals),
                "total_goals_lines_skipped": _format_total_goals_lines(skipped_total_goals),
                "total_goals_line_fit_error": total_goals_line_fit_error,
                "total_goals_line_diagnostics": total_goals_line_diagnostics,
                "over_under_2_5_used": targets.over_2_5 is not None
                or any(np.isclose(line, 2.5) for line, _ in total_goals_targets),
                "multi_line_totals_used": len(total_goals_targets) > 1,
                "has_btts": has_btts,
                "btts_market_available": "yes" if has_btts else "no",
                "market_fair_btts_yes_probability": market_btts_yes,
                "market_fair_btts_no_probability": market_btts_no,
                "btts_fit_error": btts_fit_error,
                "has_qualification_odds": has_qualification_odds,
                "knockout_scoring_mode": config.knockout_scoring.knockout_scoring_mode,
                "has_correct_score_market": has_correct_score_market,
                "correct_score_poisson_weight": config.correct_score_poisson_weight,
                "selected_blend_weight": config.correct_score_poisson_weight,
                "effective_correct_score_poisson_weight": effective_correct_score_poisson_weight,
                "correct_score_blend_applied": has_correct_score_market and not correct_score_blend_suppressed,
                "correct_score_blend_suppressed": correct_score_blend_suppressed,
                "correct_score_blend_note": correct_score_blend_note,
                **correct_score_coverage,
                **correct_score_aggregation_diagnostics,
                "correct_score_bookmaker_diagnostics": correct_score_bookmaker_diagnostics,
                "correct_score_market_top_10": correct_score_market_top_10,
                "correct_score_top_scores": correct_score_market_top_10,
                "has_other_bucket": has_other_bucket,
                "correct_score_tail_mass": correct_score_tail_mass,
                "correct_score_blended_top_10": correct_score_blended_top_10,
                "correct_score_kl_divergence": correct_score_kl_divergence,
                "top_10_market_scorelines": correct_score_market_top_10,
                "top_10_blended_scorelines": correct_score_blended_top_10,
                "kl_divergence": correct_score_kl_divergence,
                "lambda_a": calibration.lambda_a,
                "lambda_b": calibration.lambda_b,
                "lambda_a_near_bound": "lambda_a_near_bound" in calibration.warnings,
                "lambda_b_near_bound": "lambda_b_near_bound" in calibration.warnings,
                "model_a_win": model_outcomes["a_win"],
                "model_draw": model_outcomes["draw"],
                "model_b_win": model_outcomes["b_win"],
                **score_audit,
                "calibration_loss": calibration.loss,
                "recommended_score": f"{recommendation.best.predicted_score[0]}-{recommendation.best.predicted_score[1]}",
                "ev_default_score": _score_label(final_live_recommended_score),
                "most_likely_score": _score_label(recommendation.most_likely_scoreline),
                "recommended_qualifier": recommended_qualifier,
                "best_expected_points": recommendation.best.expected_points,
                "ev_expected_points": recommendation.best.expected_points,
                "modal_expected_points": most_likely_expected_points,
                "ev_gap_best_vs_second": ev_gap_best_vs_second,
                "ev_gap_best_vs_modal": ev_gap_best_vs_modal,
                "baseline_poisson_recommended_score": f"{baseline_poisson_recommended_score[0]}-{baseline_poisson_recommended_score[1]}",
                "baseline_poisson_best_expected_points": baseline_recommendation.best.expected_points,
                "baseline_poisson_ev_gap_best_vs_second": _ev_gap_best_vs_second(baseline_recommendation),
                "correct_score_blended_recommended_score": f"{correct_score_blended_recommended_score[0]}-{correct_score_blended_recommended_score[1]}",
                "correct_score_blended_best_expected_points": correct_score_blended_recommendation.best.expected_points,
                "correct_score_blended_ev_gap_best_vs_second": _ev_gap_best_vs_second(
                    correct_score_blended_recommendation
                ),
                "final_live_recommended_score": f"{final_live_recommended_score[0]}-{final_live_recommended_score[1]}",
                "final_live_ev_gap_best_vs_second": ev_gap_best_vs_second,
                "model_recommendations_agree": model_recommendations_agree,
                "model_disagreement_warning": model_disagreement_warning,
                "dixon_coles_rho": float(dixon_coles_rho_estimate.rho),
                "dixon_coles_rho_used": float(dixon_coles_rho_estimate.rho),
                "dixon_coles_rho_source": str(dixon_coles_rho_estimate.source),
                "dixon_coles_rho_fit_error": dixon_coles_rho_estimate.fit_error,
                "dixon_coles_recommended_score": f"{dixon_coles_recommended_score[0]}-{dixon_coles_recommended_score[1]}",
                "dixon_coles_best_expected_points": dixon_coles_recommendation.best.expected_points,
                "dixon_coles_ev_gap_best_vs_second": _ev_gap_best_vs_second(dixon_coles_recommendation),
                "dixon_coles_top_5_ev_predictions": _format_top_ev_predictions(dixon_coles_recommendation),
                "dixon_coles_changes_recommendation": dixon_coles_recommended_score != final_live_recommended_score,
                "market_consistent_recommended_score": (
                    _score_label(market_consistent_recommended_score)
                    if market_consistent_recommendation is not None
                    else ""
                ),
                "market_consistent_best_expected_points": (
                    market_consistent_recommendation.best.expected_points
                    if market_consistent_recommendation is not None
                    else pd.NA
                ),
                "market_consistent_ev_gap_best_vs_second": (
                    _ev_gap_best_vs_second(market_consistent_recommendation)
                    if market_consistent_recommendation is not None
                    else pd.NA
                ),
                "market_consistent_top_10_ev_scorelines": _format_evaluations(
                    market_consistent_top_10_evaluations
                ),
                "market_consistent_top_10_ev_decomposition": _format_ev_decomposition(
                    market_consistent_top_10_ev_decomposition_records
                ),
                "market_consistent_top_10_ev_decomposition_json": market_consistent_top_10_ev_decomposition_json,
                "market_consistent_top_10_probability_scorelines": market_consistent_top_10_probability_scorelines,
                **market_consistent_result.diagnostics,
                "market_consistent_warning_flags": "; ".join(market_consistent_warning_flags),
                "market_consistent_differs_from_default": (
                    "yes"
                    if market_consistent_differs_from_default
                    else "no"
                    if market_consistent_recommendation is not None
                    else "not_run"
                ),
                "market_consistent_asian_totals_shift_recommendation": (
                    "yes" if market_consistent_asian_totals_shift_recommendation else "no"
                ),
                "asian_handicap_lines_available": _format_asian_handicap_lines(match_asian_handicap),
                "asian_handicap_lines_used": market_consistent_asian_handicap_lines_used,
                "asian_handicap_orientation_suspicious": (
                    "yes" if asian_handicap_orientation_suspicious else "no"
                ),
                "asian_handicap_orientation_reason": asian_handicap_orientation_reason,
                "asian_handicap_shift_recommendation": "yes" if asian_handicap_shift_recommendation else "no",
                **ah_favourite_summary,
                "margin_ev_gap": margin_ev_gap,
                "margin_diagnostic_note": margin_diagnostic_note,
                **draw_decisive,
                **market_consistent_high_score_comparison,
                "exact_score_probability": recommendation.best.exact_score_probability,
                "correct_goal_difference_probability": recommendation.best.correct_goal_difference_probability,
                "correct_result_probability": getattr(recommendation.best, "correct_result_probability", pd.NA),
                "most_likely_scoreline": f"{recommendation.most_likely_scoreline[0]}-{recommendation.most_likely_scoreline[1]}",
                "most_likely_expected_points": most_likely_expected_points,
                "ev_optimal_differs_from_most_likely": recommendation.differs_from_most_likely,
                "ev_vs_modal_differs": "yes" if final_live_recommended_score != recommendation.most_likely_scoreline else "no",
                **modal_draw_challenger,
                **draw_prone,
                **blowout_risk,
                **btts_conflict,
                "top_5_ev_predictions": _format_top_ev_predictions(recommendation),
                "top_5_ev_decomposition": _format_ev_decomposition(ev_decomposition_records),
                "top_5_ev_decomposition_json": ev_decomposition_json,
                "top_10_ev_decomposition": _format_ev_decomposition(top_10_ev_decomposition_records),
                "top_10_ev_decomposition_json": top_10_ev_decomposition_json,
                "ev_explanation": _ev_explanation(ev_decomposition_records),
                "top_alternatives": _format_alternatives(recommendation),
                "tail_probability_before_renormalisation": normal_grid_tail_mass,
                "extreme_favourite_audit_triggered": extreme_favourite_audit_triggered,
                **extreme_audit,
                **larger_grid_sensitivity,
                "normal_grid_tail_mass": normal_grid_tail_mass,
                **plausible_decision_layer,
                **decision_aid,
                "warnings": "; ".join(filter(None, [str(row.get("warnings", "")), *notes])),
                "warning_flags": warning_flags,
                "estimated_most_crowded_public_score": public_strategy.estimated_most_crowded_public_score,
                "estimated_most_crowded_public_pick_share": public_strategy.estimated_most_crowded_public_pick_share,
                "public_strategy_score": public_strategy.public_strategy_score,
                "public_strategy_reason": public_strategy.public_strategy_reason,
                "public_strategy_ev_cost": public_strategy.public_strategy_ev_cost,
                "public_strategy_public_pick_share": public_strategy.public_strategy_public_pick_share,
                "public_strategy_leverage_score": public_strategy.public_strategy_leverage_score,
                "public_strategy_mode": public_strategy.public_strategy_mode,
                "public_strategy_public_ranking_score": public_strategy.public_strategy_public_ranking_score,
                "public_strategy_exact_score_probability": public_strategy.public_strategy_exact_score_probability,
                "public_strategy_result_probability": public_strategy.public_strategy_result_probability,
                "public_strategy_candidate_count": public_strategy.public_strategy_candidate_count,
                "public_strategy_target": public_strategy.public_strategy_target,
                "public_field_size": public_strategy.public_field_size,
                "friend_strategy_score": public_strategy.friend_strategy_score,
                "friend_strategy_reason": public_strategy.friend_strategy_reason,
            }
        )
    match_report = pd.DataFrame(report_rows)
    margin_diagnostics_frame = (
        pd.concat(margin_diagnostic_frames, ignore_index=True)
        if margin_diagnostic_frames
        else pd.DataFrame()
    )
    asian_handicap_diagnostics_frame = (
        pd.concat(asian_handicap_diagnostic_frames, ignore_index=True)
        if asian_handicap_diagnostic_frames
        else aggregated_asian_handicap
    )
    friend_report = (
        analyse_friend_predictions(predictions, matches, matrices, recommendations, qualification, config)
        if predictions is not None
        else pd.DataFrame()
    )
    if config.enable_margin_method_comparison:
        margin_start = time.perf_counter()
        margin_method_comparison = _build_margin_method_comparison(
            odds=odds,
            config=config,
            default_report=match_report,
            correct_score_odds=correct_score_odds,
            total_goals_odds=total_goals_odds,
            asian_handicap_odds=asian_handicap_odds,
        )
        if runtime_timings is not None:
            runtime_timings["margin-method comparison"] = (
                runtime_timings.get("margin-method comparison", 0.0) + time.perf_counter() - margin_start
            )
    else:
        margin_method_comparison = pd.DataFrame()
        if runtime_timings is not None:
            runtime_timings["margin-method comparison"] = runtime_timings.get("margin-method comparison", 0.0)
    final_decision_dashboard = (
        _build_final_decision_dashboard(match_report, margin_method_comparison, config)
        if config.enable_final_decision_dashboard
        else pd.DataFrame()
    )
    if not final_decision_dashboard.empty:
        decision_columns = [
            "match_id",
            "final_decision_score",
            "final_decision_basis",
            "model_consensus",
            "confidence_level",
            "manual_review_flag",
            "main_alternative_score",
            "plausible_alternatives",
            "strategic_alternative_score",
            "override_candidate",
            "modal_draw_challenger_flag",
            "modal_draw_challenger_score",
            "btts_conflict_flag",
            "btts_conflict_note",
            "best_btts_alternative_score",
            "draw_prone_flag",
            "draw_prone_reason",
            "blowout_risk_flag",
            "blowout_risk_reason",
            "best_high_margin_alternatives",
            "risk_notes",
            "decision_note",
            "asian_handicap_shift_recommendation",
            "margin_ev_gap",
            "draw_vs_decisive_gap",
            "margin_diagnostic_note",
        ]
        match_report = match_report.drop(
            columns=[column for column in decision_columns if column in match_report and column != "match_id"],
            errors="ignore",
        ).merge(
            final_decision_dashboard[decision_columns],
            on="match_id",
            how="left",
            validate="one_to_one",
        )
    return PredictionWorkflowResult(
        bookmaker_probabilities,
        aggregated,
        match_report,
        friend_report,
        matrices,
        recommendations,
        qualification,
        processed_correct_scores,
        correct_score_matrices,
        processed_total_goals,
        aggregated_total_goals,
        baseline_matrices,
        challenger_matrices,
        margin_method_comparison,
        final_decision_dashboard,
        processed_asian_handicap,
        asian_handicap_diagnostics_frame,
        margin_diagnostics_frame,
    )
