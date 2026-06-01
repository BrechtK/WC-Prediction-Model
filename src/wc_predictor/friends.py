"""Pre-match expected-value analysis of friends' predictions."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from wc_predictor.config import ProjectConfig
from wc_predictor.optimiser import (
    GroupPredictionRecommendation,
    KnockoutPredictionRecommendation,
    evaluate_group_prediction,
    evaluate_knockout_prediction,
)
from wc_predictor.probabilities import ScoreProbabilityMatrix
from wc_predictor.utils import is_knockout_stage


def _prediction_key(row: pd.Series, knockout: bool) -> str:
    score = f"{int(row['predicted_team_a_goals'])}-{int(row['predicted_team_b_goals'])}"
    if knockout:
        return f"{score}; qualifier={row['predicted_qualifier']}"
    return score


def analyse_friend_predictions(
    predictions: pd.DataFrame,
    matches: pd.DataFrame,
    score_matrices: Mapping[str, ScoreProbabilityMatrix],
    recommendations: Mapping[str, GroupPredictionRecommendation | KnockoutPredictionRecommendation],
    qualifier_probabilities: Mapping[str, dict[str, float]] | None = None,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Calculate model-implied EV, optimal-EV gap, rank, and consensus labels."""

    config = config or ProjectConfig()
    qualifier_probabilities = qualifier_probabilities or {}
    match_metadata = matches.drop_duplicates("match_id").set_index("match_id")
    rows: list[dict[str, object]] = []
    counts: dict[tuple[str, str], int] = {}
    for _, prediction in predictions.iterrows():
        match_id = str(prediction["match_id"])
        knockout = is_knockout_stage(str(match_metadata.loc[match_id, "stage"]))
        key = _prediction_key(prediction, knockout)
        counts[(match_id, key)] = counts.get((match_id, key), 0) + 1

    for _, prediction in predictions.iterrows():
        match_id = str(prediction["match_id"])
        metadata = match_metadata.loc[match_id]
        knockout = is_knockout_stage(str(metadata["stage"]))
        matrix = score_matrices[match_id]
        pred_a = int(prediction["predicted_team_a_goals"])
        pred_b = int(prediction["predicted_team_b_goals"])
        recommendation = recommendations[match_id]
        if knockout:
            qualifier = str(prediction["predicted_qualifier"])
            if qualifier in {"", "<NA>", "nan", "None"}:
                raise ValueError(f"Knockout prediction for {match_id} requires predicted_qualifier")
            evaluation = evaluate_knockout_prediction(
                matrix, pred_a, pred_b, qualifier, qualifier_probabilities[match_id], config.knockout_scoring
            )
            candidate_evs = [
                evaluate_knockout_prediction(matrix, a, b, q, qualifier_probabilities[match_id], config.knockout_scoring).expected_points
                for a in range(config.max_candidate_goals + 1)
                for b in range(config.max_candidate_goals + 1)
                for q in qualifier_probabilities[match_id]
            ]
            recommended_prediction = (
                f"{recommendation.best.predicted_score[0]}-{recommendation.best.predicted_score[1]}; "
                f"qualifier={recommendation.best.predicted_qualifier}"
            )
        else:
            evaluation = evaluate_group_prediction(matrix, pred_a, pred_b)
            candidate_evs = [
                evaluate_group_prediction(matrix, a, b).expected_points
                for a in range(config.max_candidate_goals + 1)
                for b in range(config.max_candidate_goals + 1)
            ]
            recommended_prediction = f"{recommendation.best.predicted_score[0]}-{recommendation.best.predicted_score[1]}"
        key = _prediction_key(prediction, knockout)
        same_prediction_count = counts[(match_id, key)]
        largest_count = max(count for (candidate_match, _), count in counts.items() if candidate_match == match_id)
        rows.append(
            {
                "player": prediction["player"],
                "match_id": match_id,
                "prediction": key,
                "model_expected_points": evaluation.expected_points,
                "difference_vs_optimal_ev": recommendation.best.expected_points - evaluation.expected_points,
                "candidate_rank": 1 + sum(ev > evaluation.expected_points + 1e-12 for ev in candidate_evs),
                "model_recommendation": recommended_prediction,
                "is_consensus": same_prediction_count == largest_count and largest_count > 1,
                "is_contrarian": same_prediction_count == 1,
            }
        )
    return pd.DataFrame(rows)

