"""Realised scoring and standings calculation."""

from __future__ import annotations

import pandas as pd

from wc_predictor.config import KnockoutScoringConfig
from wc_predictor.scoring_rules import score_group_prediction, score_knockout_prediction
from wc_predictor.utils import goal_difference, is_knockout_stage, result_sign


def select_knockout_score(result: pd.Series, config: KnockoutScoringConfig) -> tuple[int, int]:
    """Choose the score basis used by the configured knockout interpretation."""

    score_90 = (int(result["team_a_goals_90"]), int(result["team_b_goals_90"]))
    if config.score_basis == "90min":
        return score_90
    went_to_extra_time = bool(result.get("went_to_extra_time", False))
    if went_to_extra_time and pd.notna(result.get("team_a_goals_120")) and pd.notna(result.get("team_b_goals_120")):
        return int(result["team_a_goals_120"]), int(result["team_b_goals_120"])
    return score_90


def score_submitted_predictions(
    predictions: pd.DataFrame,
    matches: pd.DataFrame,
    results: pd.DataFrame,
    knockout_config: KnockoutScoringConfig | None = None,
) -> pd.DataFrame:
    """Score submitted group and knockout predictions against realised results."""

    knockout_config = knockout_config or KnockoutScoringConfig()
    metadata = matches[["match_id", "stage"]].drop_duplicates("match_id")
    merged = predictions.merge(metadata, on="match_id", how="left", validate="many_to_one")
    merged = merged.merge(results, on="match_id", how="left", validate="many_to_one")
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        if pd.isna(row["team_a_goals_90"]) or pd.isna(row["team_b_goals_90"]):
            continue
        pred_a = int(row["predicted_team_a_goals"])
        pred_b = int(row["predicted_team_b_goals"])
        knockout = is_knockout_stage(str(row["stage"]))
        if knockout:
            actual_a, actual_b = select_knockout_score(row, knockout_config)
            qualifier_correct = str(row["predicted_qualifier"]) == str(row["qualifier"])
            points = score_knockout_prediction(
                pred_a,
                pred_b,
                str(row["predicted_qualifier"]),
                actual_a,
                actual_b,
                str(row["qualifier"]),
                knockout_config,
            )
        else:
            actual_a, actual_b = int(row["team_a_goals_90"]), int(row["team_b_goals_90"])
            qualifier_correct = False
            points = score_group_prediction(pred_a, pred_b, actual_a, actual_b)
        rows.append(
            {
                "player": row["player"],
                "match_id": row["match_id"],
                "stage": row["stage"],
                "realised_points": points,
                "is_exact_score": (pred_a, pred_b) == (actual_a, actual_b),
                "is_correct_goal_difference": goal_difference(pred_a, pred_b) == goal_difference(actual_a, actual_b),
                "is_correct_result": result_sign(pred_a, pred_b) == result_sign(actual_a, actual_b),
                "is_correct_qualifier": qualifier_correct,
            }
        )
    return pd.DataFrame(rows)


def calculate_standings(scored_predictions: pd.DataFrame, friend_ev_report: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate realised and pre-match expected points into a player leaderboard."""

    if scored_predictions.empty:
        return pd.DataFrame()
    standings = (
        scored_predictions.groupby("player", as_index=False)
        .agg(
            total_realised_points=("realised_points", "sum"),
            predictions_scored=("match_id", "count"),
            exact_scores=("is_exact_score", "sum"),
            correct_goal_differences=("is_correct_goal_difference", "sum"),
            correct_results=("is_correct_result", "sum"),
            correct_qualifiers=("is_correct_qualifier", "sum"),
        )
    )
    if friend_ev_report is not None and not friend_ev_report.empty:
        expected = (
            friend_ev_report.groupby("player", as_index=False)
            .agg(
                total_expected_points=("model_expected_points", "sum"),
                average_ev_per_prediction=("model_expected_points", "mean"),
            )
        )
        standings = standings.merge(expected, on="player", how="left")
    standings = standings.sort_values(["total_realised_points", "player"], ascending=[False, True]).reset_index(drop=True)
    standings.insert(0, "ranking", standings.index + 1)
    return standings

