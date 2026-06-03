import numpy as np
import pandas as pd

from wc_predictor.config import ProjectConfig, PublicStrategyConfig
from wc_predictor.optimiser import optimise_group_prediction
from wc_predictor.probabilities import ScoreProbabilityMatrix
from wc_predictor.public_strategy import build_public_strategy, public_pick_distribution
from wc_predictor.workflow import run_prediction_workflow


def _matrix() -> ScoreProbabilityMatrix:
    probabilities = np.asarray(
        [
            [0.08, 0.03, 0.01, 0.00],
            [0.18, 0.10, 0.04, 0.01],
            [0.16, 0.13, 0.05, 0.01],
            [0.08, 0.08, 0.03, 0.01],
        ],
        dtype=float,
    )
    return ScoreProbabilityMatrix(probabilities / probabilities.sum())


def _match_row(warnings: str = "") -> pd.Series:
    return pd.Series(
        {
            "team_a": "Belgium",
            "team_b": "Canada",
            "favourite_probability": 0.62,
            "warning_flags": warnings,
            "correct_score_market_top_10": "1-0 (0.12); 2-0 (0.11); 2-1 (0.10)",
        }
    )


def test_public_strategy_ev_mode_keeps_pure_ev_recommendation() -> None:
    matrix = _matrix()
    recommendation = optimise_group_prediction(matrix, max_candidate_goals=3, top_n=5)

    result = build_public_strategy(
        match_row=_match_row(),
        matrix=matrix,
        recommendation=recommendation,
        config=ProjectConfig(public_strategy=PublicStrategyConfig(mode="ev")),
    )

    assert result.public_strategy_score == f"{recommendation.best.predicted_score[0]}-{recommendation.best.predicted_score[1]}"
    assert result.public_strategy_mode == "ev"
    assert "strategy mode is ev" in result.public_strategy_reason


def test_public_ranking_can_pick_close_lower_crowding_alternative() -> None:
    matrix = _matrix()
    recommendation = optimise_group_prediction(matrix, max_candidate_goals=3, top_n=5)
    config = ProjectConfig(
        public_strategy=PublicStrategyConfig(
            mode="aggressive-public-ranking",
            alpha=25.0,
            beta=0.0,
            gamma=0.0,
            max_public_strategy_ev_loss=3.0,
            min_exact_score_probability=0.0,
            min_result_probability=0.0,
        )
    )

    result = build_public_strategy(
        match_row=_match_row(),
        matrix=matrix,
        recommendation=recommendation,
        config=config,
    )

    pure_score = f"{recommendation.best.predicted_score[0]}-{recommendation.best.predicted_score[1]}"
    assert result.public_strategy_score != pure_score
    assert result.public_strategy_ev_cost >= 0
    assert "EV cost" in result.public_strategy_reason


def test_public_strategy_keeps_pure_ev_when_safety_flag_is_severe() -> None:
    matrix = _matrix()
    recommendation = optimise_group_prediction(matrix, max_candidate_goals=3, top_n=5)

    result = build_public_strategy(
        match_row=_match_row("high_calibration_error"),
        matrix=matrix,
        recommendation=recommendation,
        config=ProjectConfig(public_strategy=PublicStrategyConfig(mode="public-ranking")),
    )

    assert result.public_strategy_score == f"{recommendation.best.predicted_score[0]}-{recommendation.best.predicted_score[1]}"
    assert "severe warning flags" in result.public_strategy_reason


def test_belgium_popularity_bias_increases_belgium_win_public_share() -> None:
    matrix = _matrix()

    belgium_home = public_pick_distribution(
        matrix=matrix,
        team_a="Belgium",
        team_b="Canada",
        favourite_probability=0.62,
    )
    belgium_away = public_pick_distribution(
        matrix=matrix,
        team_a="Canada",
        team_b="Belgium",
        favourite_probability=0.62,
    )

    assert sum(share for (a, b), share in belgium_home.items() if a > b) > sum(
        share for (a, b), share in belgium_away.items() if a > b
    )


def test_workflow_output_includes_public_strategy_diagnostics() -> None:
    odds = pd.DataFrame(
        [
            {
                "match_id": "M001",
                "date": "2026-06-01",
                "stage": "group stage",
                "group": "Group A",
                "team_a": "Belgium",
                "team_b": "Canada",
                "bookmaker": "Book",
                "odds_a_win": 1.80,
                "odds_draw": 3.50,
                "odds_b_win": 5.00,
            }
        ]
    )

    report = run_prediction_workflow(
        odds,
        config=ProjectConfig(public_strategy=PublicStrategyConfig(mode="public-ranking")),
    ).match_report

    assert {
        "estimated_most_crowded_public_score",
        "estimated_most_crowded_public_pick_share",
        "public_strategy_score",
        "public_strategy_ev_cost",
        "public_strategy_public_pick_share",
        "public_strategy_leverage_score",
        "public_strategy_mode",
        "public_strategy_reason",
        "friend_strategy_score",
    }.issubset(report.columns)
    assert report.loc[0, "recommended_score"] == report.loc[0, "final_live_recommended_score"]
