from __future__ import annotations

import pytest

from wc_predictor.margin_diagnostics import draw_vs_decisive_diagnostics, margin_ev_diagnostics
from wc_predictor.optimiser import evaluate_group_prediction, optimise_group_prediction
from wc_predictor.probabilities import poisson_score_matrix


def test_margin_ev_matches_best_raw_scoreline_ev() -> None:
    matrix = poisson_score_matrix(1.5, 1.0, max_goals=7)
    diagnostics = margin_ev_diagnostics(matrix, "M001", max_candidate_goals=5)
    best_margin = diagnostics.iloc[0]
    score_a, score_b = (int(value) for value in str(best_margin["best_scoreline_on_margin"]).split("-"))
    raw = evaluate_group_prediction(matrix, score_a, score_b)
    recommendation = optimise_group_prediction(matrix, max_candidate_goals=5)

    assert best_margin["representative_scoreline_ev"] == pytest.approx(raw.expected_points)
    assert best_margin["representative_scoreline_ev"] == pytest.approx(recommendation.best.expected_points)
    assert best_margin["best_scoreline_on_margin"] == (
        f"{recommendation.best.predicted_score[0]}-{recommendation.best.predicted_score[1]}"
    )


def test_best_margin_representative_is_selected_correctly() -> None:
    matrix = poisson_score_matrix(2.2, 0.8, max_goals=7)
    diagnostics = margin_ev_diagnostics(matrix, "M001", max_candidate_goals=5)
    plus_two = diagnostics[diagnostics["margin"] == 2].iloc[0]

    assert plus_two["best_scoreline_on_margin"] in {"2-0", "3-1", "4-2", "5-3"}
    score_a, score_b = (int(value) for value in plus_two["best_scoreline_on_margin"].split("-"))
    selected_probability = matrix.probabilities[score_a, score_b]
    for candidate_a, candidate_b in ((2, 0), (3, 1), (4, 2), (5, 3)):
        assert selected_probability >= matrix.probabilities[candidate_a, candidate_b] - 1e-12


def test_draw_vs_decisive_diagnostics_work() -> None:
    matrix = poisson_score_matrix(1.0, 1.0, max_goals=7)

    diagnostics = draw_vs_decisive_diagnostics(matrix, max_candidate_goals=5, boundary_threshold=0.20)

    assert diagnostics["best_draw_score"] == "1-1"
    assert diagnostics["best_decisive_score"] in {"1-0", "0-1"}
    assert diagnostics["draw_boundary_flag"] in {"yes", "no"}
    assert diagnostics["draw_vs_decisive_gap"] == pytest.approx(
        diagnostics["best_draw_ev"] - diagnostics["best_decisive_ev"]
    )
