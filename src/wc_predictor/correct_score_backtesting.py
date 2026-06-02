"""Group-stage realised-points backtesting for correct-score blend weights."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from wc_predictor.config import ProjectConfig
from wc_predictor.scoring_rules import score_group_prediction
from wc_predictor.utils import ensure_parent_directory, goal_difference, is_knockout_stage, result_sign
from wc_predictor.workflow import run_prediction_workflow

DEFAULT_CORRECT_SCORE_BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class CorrectScoreBlendBacktestReport:
    """Ranked weight summary and match-level realised points."""

    summary: pd.DataFrame
    predictions: pd.DataFrame

    def export_csv(self, path: str | Path) -> None:
        """Export the ranked blend-weight comparison."""

        path = Path(path)
        ensure_parent_directory(path)
        self.summary.to_csv(path, index=False)


def backtest_correct_score_blend_weights(
    odds: pd.DataFrame,
    correct_score_odds: pd.DataFrame,
    results: pd.DataFrame,
    weights: tuple[float, ...] = DEFAULT_CORRECT_SCORE_BLEND_WEIGHTS,
    config: ProjectConfig | None = None,
) -> CorrectScoreBlendBacktestReport:
    """Rank Poisson-market blend weights by realised group-stage pool points."""

    config = config or ProjectConfig()
    required_results = {"match_id", "team_a_goals_90", "team_b_goals_90"}
    missing = required_results - set(results.columns)
    if missing:
        raise ValueError(f"Result data is missing required columns: {sorted(missing)}")
    result_rows = results.set_index(results["match_id"].astype(str))
    stages = odds[["match_id", "stage"]].drop_duplicates("match_id")
    group_match_ids = {
        str(row["match_id"])
        for _, row in stages.iterrows()
        if not is_knockout_stage(str(row["stage"]))
    }

    prediction_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for weight in weights:
        weighted_config = replace(config, correct_score_poisson_weight=float(weight))
        workflow = run_prediction_workflow(odds, config=weighted_config, correct_score_odds=correct_score_odds)
        scores: list[dict[str, object]] = []
        for match_id in sorted(group_match_ids & set(workflow.correct_score_market_matrices)):
            if match_id not in result_rows.index:
                continue
            result = result_rows.loc[match_id]
            actual_a = int(result["team_a_goals_90"])
            actual_b = int(result["team_b_goals_90"])
            predicted_a, predicted_b = workflow.recommendations[match_id].best.predicted_score
            scored = {
                "correct_score_poisson_weight": float(weight),
                "match_id": match_id,
                "predicted_team_a_goals": predicted_a,
                "predicted_team_b_goals": predicted_b,
                "actual_team_a_goals": actual_a,
                "actual_team_b_goals": actual_b,
                "realised_points": score_group_prediction(predicted_a, predicted_b, actual_a, actual_b),
                "is_exact_score": (predicted_a, predicted_b) == (actual_a, actual_b),
                "is_correct_goal_difference": goal_difference(predicted_a, predicted_b)
                == goal_difference(actual_a, actual_b),
                "is_correct_result": result_sign(predicted_a, predicted_b) == result_sign(actual_a, actual_b),
            }
            scores.append(scored)
            prediction_rows.append(scored)
        scored_frame = pd.DataFrame(scores)
        summary_rows.append(
            {
                "correct_score_poisson_weight": float(weight),
                "average_realised_points": float(scored_frame["realised_points"].mean()) if not scored_frame.empty else np.nan,
                "total_points": int(scored_frame["realised_points"].sum()) if not scored_frame.empty else 0,
                "exact_score_hit_rate": float(scored_frame["is_exact_score"].mean()) if not scored_frame.empty else np.nan,
                "correct_goal_difference_hit_rate": (
                    float(scored_frame["is_correct_goal_difference"].mean()) if not scored_frame.empty else np.nan
                ),
                "correct_result_hit_rate": float(scored_frame["is_correct_result"].mean()) if not scored_frame.empty else np.nan,
                "points_variance": float(scored_frame["realised_points"].var(ddof=0)) if not scored_frame.empty else np.nan,
                "matches_used": len(scored_frame),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["rank"] = summary["average_realised_points"].rank(method="min", ascending=False).astype("Int64")
    summary = summary.sort_values(
        ["average_realised_points", "total_points", "correct_score_poisson_weight"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return CorrectScoreBlendBacktestReport(summary, pd.DataFrame(prediction_rows))

