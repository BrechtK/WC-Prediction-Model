"""Application workflow joining market extraction, calibration, and optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from wc_predictor.calibration import CalibrationTargets, calibrate_poisson_model, poisson_over_total_probability
from wc_predictor.config import ProjectConfig
from wc_predictor.correct_scores import (
    aggregate_correct_score_market,
    blend_score_matrices,
    format_correct_score_bookmaker_diagnostics,
    format_top_scorelines,
    market_to_poisson_kl_divergence,
    summarise_correct_score_coverage,
)
from wc_predictor.friends import analyse_friend_predictions
from wc_predictor.odds import (
    aggregate_bookmaker_probabilities,
    aggregate_total_goals_probabilities,
    process_bookmaker_odds,
    process_correct_score_odds,
    process_total_goals_odds,
)
from wc_predictor.optimiser import (
    GroupPredictionRecommendation,
    KnockoutPredictionRecommendation,
    evaluate_group_prediction,
    evaluate_knockout_prediction,
    optimise_group_prediction,
    optimise_knockout_prediction,
)
from wc_predictor.probabilities import ScoreProbabilityMatrix
from wc_predictor.utils import favourite_strength_bucket, is_knockout_stage

HIGH_TAIL_MASS_THRESHOLD = 0.01
STALE_ODDS_THRESHOLD = pd.Timedelta(hours=24)


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
    return "; ".join(flags)


def run_prediction_workflow(
    odds: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    config: ProjectConfig | None = None,
    correct_score_odds: pd.DataFrame | None = None,
    total_goals_odds: pd.DataFrame | None = None,
) -> PredictionWorkflowResult:
    """Produce market-implied score recommendations and optional friend EV analysis."""

    config = config or ProjectConfig()
    bookmaker_probabilities = process_bookmaker_odds(
        odds,
        config.margin_removal_method,
        config.suspicious_overround_low,
        config.suspicious_overround_high,
    )
    odds_metadata = _summarise_odds_metadata(odds, bookmaker_probabilities)
    processed_correct_scores = (
        process_correct_score_odds(correct_score_odds, config.margin_removal_method)
        if correct_score_odds is not None
        else pd.DataFrame()
    )
    processed_total_goals = (
        process_total_goals_odds(
            total_goals_odds,
            config.margin_removal_method,
            config.suspicious_overround_low,
            config.suspicious_overround_high,
        )
        if total_goals_odds is not None
        else pd.DataFrame()
    )
    aggregated_total_goals = aggregate_total_goals_probabilities(processed_total_goals)
    aggregated = aggregate_bookmaker_probabilities(bookmaker_probabilities, config.bookmaker_aggregation_method)
    required_1x2 = {"fair_a_win", "fair_draw", "fair_b_win"}
    if not required_1x2.issubset(aggregated.columns):
        raise ValueError("No complete 1X2 market is available for calibration")
    missing_1x2 = aggregated[list(required_1x2)].isna().any(axis=1)
    if missing_1x2.any():
        match_ids = aggregated.loc[missing_1x2, "match_id"].astype(str).tolist()
        raise ValueError(f"No complete 1X2 market is available for matches: {match_ids}")
    odds_with_group = odds.assign(group=odds["group"] if "group" in odds else pd.NA)
    matches = odds_with_group[["match_id", "date", "stage", "group", "team_a", "team_b"]].drop_duplicates("match_id")
    market = matches.merge(aggregated, on="match_id", validate="one_to_one")

    report_rows: list[dict[str, object]] = []
    matrices: dict[str, ScoreProbabilityMatrix] = {}
    recommendations: dict[str, GroupPredictionRecommendation | KnockoutPredictionRecommendation] = {}
    qualification: dict[str, dict[str, float]] = {}
    correct_score_matrices: dict[str, ScoreProbabilityMatrix] = {}
    for _, row in market.iterrows():
        match_id = str(row["match_id"])
        metadata = odds_metadata[match_id]
        match_total_goals = (
            aggregated_total_goals[aggregated_total_goals["match_id"].astype(str) == match_id]
            if not aggregated_total_goals.empty
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
            has_correct_score_market = False
        matrices[match_id] = score_matrix
        notes = list(calibration.warnings)
        if not skipped_total_goals.empty:
            notes.append(
                "Stored but skipped Asian total-goals calibration lines pending push/half-stake settlement support: "
                + _format_total_goals_lines(skipped_total_goals)
            )
        if correct_score_blend_note:
            notes.append(correct_score_blend_note)
        knockout = is_knockout_stage(str(row["stage"]))
        has_over_under = pd.notna(row.get("fair_over_2_5")) or bool(total_goals_targets)
        has_btts = pd.notna(row.get("fair_btts_yes"))
        has_qualification_odds = pd.notna(row.get("fair_a_qualifies")) and pd.notna(row.get("fair_b_qualifies"))
        if knockout:
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
            recommendation = optimise_knockout_prediction(
                score_matrix,
                qualifier_probabilities,
                config.max_candidate_goals,
                config.strategies.top_alternatives,
                config.knockout_scoring,
            )
            recommended_qualifier = recommendation.best.predicted_qualifier
        else:
            recommendation = optimise_group_prediction(
                score_matrix, config.max_candidate_goals, config.strategies.top_alternatives
            )
            recommended_qualifier = ""
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
        bucket = favourite_strength_bucket(favourite_probability)
        ev_gap_best_vs_second = (
            recommendation.best.expected_points - recommendation.alternatives[0].expected_points
            if recommendation.alternatives
            else pd.NA
        )
        ev_gap_best_vs_modal = _ev_gap_vs_modal(
            recommendation,
            score_matrix,
            qualification.get(match_id),
            config,
        )
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
        )
        report_rows.append(
            {
                "match_id": match_id,
                "date": row["date"],
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
                "favourite_bucket": bucket,
                "odds_timestamp_min": metadata["odds_timestamp_min"],
                "odds_timestamp_max": metadata["odds_timestamp_max"],
                "bookmakers_used": metadata["bookmakers_used"],
                "number_of_bookmakers": metadata["number_of_bookmakers"],
                "odds_source_urls": metadata["odds_source_urls"],
                "source_qualities": metadata["source_qualities"],
                "source_notes": metadata["source_notes"],
                "has_over_under": has_over_under,
                "total_goals_lines_available": _format_total_goals_lines(match_total_goals),
                "total_goals_lines_used_for_calibration": _format_total_goals_lines(calibratable_total_goals),
                "total_goals_lines_skipped_for_calibration": _format_total_goals_lines(skipped_total_goals),
                "total_goals_lines_skipped": _format_total_goals_lines(skipped_total_goals),
                "total_goals_line_fit_error": total_goals_line_fit_error,
                "total_goals_line_diagnostics": total_goals_line_diagnostics,
                "over_under_2_5_used": targets.over_2_5 is not None
                or any(np.isclose(line, 2.5) for line, _ in total_goals_targets),
                "multi_line_totals_used": len(total_goals_targets) > 1,
                "has_btts": has_btts,
                "has_qualification_odds": has_qualification_odds,
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
                "calibration_loss": calibration.loss,
                "recommended_score": f"{recommendation.best.predicted_score[0]}-{recommendation.best.predicted_score[1]}",
                "recommended_qualifier": recommended_qualifier,
                "best_expected_points": recommendation.best.expected_points,
                "ev_gap_best_vs_second": ev_gap_best_vs_second,
                "ev_gap_best_vs_modal": ev_gap_best_vs_modal,
                "exact_score_probability": recommendation.best.exact_score_probability,
                "correct_goal_difference_probability": recommendation.best.correct_goal_difference_probability,
                "correct_result_probability": getattr(recommendation.best, "correct_result_probability", pd.NA),
                "most_likely_scoreline": f"{recommendation.most_likely_scoreline[0]}-{recommendation.most_likely_scoreline[1]}",
                "ev_optimal_differs_from_most_likely": recommendation.differs_from_most_likely,
                "top_5_ev_predictions": _format_top_ev_predictions(recommendation),
                "top_alternatives": _format_alternatives(recommendation),
                "tail_probability_before_renormalisation": poisson_matrix.tail_probability,
                "warnings": "; ".join(filter(None, [str(row.get("warnings", "")), *notes])),
                "warning_flags": warning_flags,
            }
        )
    match_report = pd.DataFrame(report_rows)
    friend_report = (
        analyse_friend_predictions(predictions, matches, matrices, recommendations, qualification, config)
        if predictions is not None
        else pd.DataFrame()
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
    )
