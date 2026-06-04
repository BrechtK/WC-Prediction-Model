"""Margin-level group-stage EV diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wc_predictor.optimiser import evaluate_group_prediction
from wc_predictor.probabilities import ScoreProbabilityMatrix
from wc_predictor.utils import result_sign


def _score_label(score: tuple[int, int]) -> str:
    return f"{score[0]}-{score[1]}"


def margin_ev_diagnostics(
    matrix: ScoreProbabilityMatrix,
    match_id: str,
    max_candidate_goals: int = 5,
) -> pd.DataFrame:
    """Return one group-stage diagnostic row per candidate goal-difference margin."""

    probabilities = matrix.probabilities
    scores_a, scores_b = np.indices(probabilities.shape)
    margins = scores_a - scores_b
    rows: list[dict[str, object]] = []
    for margin in range(-max_candidate_goals, max_candidate_goals + 1):
        candidate_scores = [
            (score_a, score_b)
            for score_a in range(max_candidate_goals + 1)
            for score_b in range(max_candidate_goals + 1)
            if score_a - score_b == margin
            and score_a < probabilities.shape[0]
            and score_b < probabilities.shape[1]
        ]
        if not candidate_scores:
            continue
        best_score = max(candidate_scores, key=lambda score: (probabilities[score], -score[0], -score[1]))
        margin_probability = float(probabilities[margins == margin].sum())
        if margin == 0:
            result_bucket = "draw"
            correct_result_probability = float(np.trace(probabilities))
            expected_value = 1.0 + 6.0 * correct_result_probability + 3.0 * float(probabilities[best_score])
        else:
            result_bucket = "home win" if margin > 0 else "away win"
            correct_result_probability = matrix.result_probability(result_sign(margin, 0))
            expected_value = (
                1.0
                + 4.0 * correct_result_probability
                + 2.0 * margin_probability
                + 3.0 * float(probabilities[best_score])
            )
        rows.append(
            {
                "match_id": match_id,
                "margin": margin,
                "result_bucket": result_bucket,
                "margin_probability": margin_probability,
                "correct_result_probability": correct_result_probability,
                "best_scoreline_on_margin": _score_label(best_score),
                "best_scoreline_probability_on_margin": float(probabilities[best_score]),
                "representative_scoreline_ev": expected_value,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["rank_among_margins"] = (
        frame["representative_scoreline_ev"].rank(method="min", ascending=False).astype(int)
    )
    return frame.sort_values(["rank_among_margins", "margin"], kind="stable").reset_index(drop=True)


def draw_vs_decisive_diagnostics(
    matrix: ScoreProbabilityMatrix,
    max_candidate_goals: int = 5,
    boundary_threshold: float = 0.07,
) -> dict[str, object]:
    """Compare the best draw score with the best decisive score."""

    draw_evaluations = []
    decisive_evaluations = []
    for score_a in range(max_candidate_goals + 1):
        for score_b in range(max_candidate_goals + 1):
            evaluation = evaluate_group_prediction(matrix, score_a, score_b)
            if score_a == score_b:
                draw_evaluations.append(evaluation)
            else:
                decisive_evaluations.append(evaluation)
    best_draw = min(draw_evaluations, key=lambda item: (-item.expected_points, item.predicted_score))
    best_decisive = min(decisive_evaluations, key=lambda item: (-item.expected_points, item.predicted_score))
    gap = best_draw.expected_points - best_decisive.expected_points
    return {
        "best_draw_score": _score_label(best_draw.predicted_score),
        "best_draw_ev": best_draw.expected_points,
        "best_decisive_score": _score_label(best_decisive.predicted_score),
        "best_decisive_ev": best_decisive.expected_points,
        "draw_vs_decisive_gap": gap,
        "draw_boundary_flag": "yes" if abs(gap) <= boundary_threshold or gap > 0 else "no",
    }
