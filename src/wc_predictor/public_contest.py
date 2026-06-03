"""Public-field contest simulation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc_predictor.config import ProjectConfig
from wc_predictor.optimiser import KnockoutPredictionRecommendation
from wc_predictor.public_strategy import public_pick_distribution
from wc_predictor.scoring_rules import score_group_prediction, score_knockout_prediction
from wc_predictor.workflow import PredictionWorkflowResult


@dataclass(frozen=True)
class PublicContestSimulation:
    """Summary outputs for a public-field strategy stress test."""

    strategy_summary: pd.DataFrame
    match_setup: pd.DataFrame
    simulation_totals: pd.DataFrame


def _parse_score_label(value: object) -> tuple[int, int]:
    left, right = str(value).split("-", maxsplit=1)
    return int(left), int(right)


def _sample_score(matrix_probabilities: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat_index = int(rng.choice(matrix_probabilities.size, p=matrix_probabilities.ravel()))
    return tuple(int(value) for value in np.unravel_index(flat_index, matrix_probabilities.shape))


def _field_score_choices(distribution: dict[tuple[int, int], float]) -> tuple[list[tuple[int, int]], np.ndarray]:
    scores = list(distribution)
    probabilities = np.asarray([distribution[score] for score in scores], dtype=float)
    probabilities = probabilities / probabilities.sum()
    return scores, probabilities


def _score_prediction(
    *,
    predicted_score: tuple[int, int],
    actual_score: tuple[int, int],
    recommendation: object,
    actual_qualifier: str | None,
    config: ProjectConfig,
) -> int:
    if isinstance(recommendation, KnockoutPredictionRecommendation):
        return score_knockout_prediction(
            predicted_score[0],
            predicted_score[1],
            recommendation.best.predicted_qualifier,
            actual_score[0],
            actual_score[1],
            actual_qualifier or recommendation.best.predicted_qualifier,
            config.knockout_scoring,
        )
    return score_group_prediction(predicted_score[0], predicted_score[1], actual_score[0], actual_score[1])


def simulate_public_contest(
    workflow: PredictionWorkflowResult,
    *,
    config: ProjectConfig | None = None,
    field_size: int = 1000,
    simulations: int = 1000,
    seed: int | None = 42,
) -> PublicContestSimulation:
    """Stress-test pure EV and public-ranking entries against a heuristic public field."""

    if field_size <= 0:
        raise ValueError("field_size must be positive")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    config = config or ProjectConfig()
    rng = np.random.default_rng(seed)
    match_report = workflow.match_report.reset_index(drop=True)
    match_setup_rows: list[dict[str, object]] = []
    setup: list[dict[str, object]] = []

    for _, row in match_report.iterrows():
        match_id = str(row["match_id"])
        matrix = workflow.score_matrices[match_id]
        recommendation = workflow.recommendations[match_id]
        public_distribution = public_pick_distribution(
            matrix=matrix,
            team_a=row["team_a"],
            team_b=row["team_b"],
            favourite_probability=float(row["favourite_probability"]),
            correct_score_market_top_10=row.get("correct_score_market_top_10", ""),
            config=config.public_strategy,
        )
        field_scores, field_probabilities = _field_score_choices(public_distribution)
        qualifier_probabilities = workflow.qualifier_probabilities.get(match_id, {})
        setup.append(
            {
                "match_id": match_id,
                "matrix": matrix.probabilities,
                "recommendation": recommendation,
                "pure_ev_score": _parse_score_label(row["recommended_score"]),
                "public_strategy_score": _parse_score_label(row["public_strategy_score"]),
                "field_scores": field_scores,
                "field_probabilities": field_probabilities,
                "qualifier_names": list(qualifier_probabilities),
                "qualifier_probabilities": np.asarray(list(qualifier_probabilities.values()), dtype=float),
            }
        )
        match_setup_rows.append(
            {
                "match_id": match_id,
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "pure_ev_score": row["recommended_score"],
                "public_strategy_score": row["public_strategy_score"],
                "estimated_most_crowded_public_score": row["estimated_most_crowded_public_score"],
                "estimated_most_crowded_public_pick_share": row["estimated_most_crowded_public_pick_share"],
                "public_strategy_public_pick_share": row["public_strategy_public_pick_share"],
                "public_strategy_ev_cost": row["public_strategy_ev_cost"],
            }
        )

    total_rows: list[dict[str, object]] = []
    pure_totals = np.zeros(simulations, dtype=float)
    strategy_totals = np.zeros(simulations, dtype=float)
    rank_pure = np.zeros(simulations, dtype=int)
    rank_strategy = np.zeros(simulations, dtype=int)

    for simulation_index in range(simulations):
        field_totals = np.zeros(field_size, dtype=int)
        pure_total = 0
        strategy_total = 0
        for item in setup:
            actual_score = _sample_score(item["matrix"], rng)
            actual_qualifier = None
            if len(item["qualifier_names"]) > 0:
                actual_qualifier = str(
                    rng.choice(item["qualifier_names"], p=item["qualifier_probabilities"])
                )
            field_choice_indexes = rng.choice(
                len(item["field_scores"]),
                size=field_size,
                p=item["field_probabilities"],
            )
            recommendation = item["recommendation"]
            field_points = [
                _score_prediction(
                    predicted_score=item["field_scores"][int(index)],
                    actual_score=actual_score,
                    recommendation=recommendation,
                    actual_qualifier=actual_qualifier,
                    config=config,
                )
                for index in field_choice_indexes
            ]
            field_totals += np.asarray(field_points, dtype=int)
            pure_total += _score_prediction(
                predicted_score=item["pure_ev_score"],
                actual_score=actual_score,
                recommendation=recommendation,
                actual_qualifier=actual_qualifier,
                config=config,
            )
            strategy_total += _score_prediction(
                predicted_score=item["public_strategy_score"],
                actual_score=actual_score,
                recommendation=recommendation,
                actual_qualifier=actual_qualifier,
                config=config,
            )
        pure_rank = int(1 + np.sum(field_totals > pure_total))
        strategy_rank = int(1 + np.sum(field_totals > strategy_total))
        pure_totals[simulation_index] = pure_total
        strategy_totals[simulation_index] = strategy_total
        rank_pure[simulation_index] = pure_rank
        rank_strategy[simulation_index] = strategy_rank
        total_rows.extend(
            [
                {
                    "simulation": simulation_index + 1,
                    "strategy": "pure_ev",
                    "total_points": pure_total,
                    "rank_vs_public_field": pure_rank,
                },
                {
                    "simulation": simulation_index + 1,
                    "strategy": "public_strategy",
                    "total_points": strategy_total,
                    "rank_vs_public_field": strategy_rank,
                },
            ]
        )

    def summary_row(name: str, totals: np.ndarray, ranks: np.ndarray) -> dict[str, object]:
        return {
            "strategy": name,
            "simulations": simulations,
            "field_size": field_size,
            "average_total_points": float(np.mean(totals)),
            "median_total_points": float(np.median(totals)),
            "average_rank_vs_public_field": float(np.mean(ranks)),
            "probability_top_50": float(np.mean(ranks <= 50)),
            "probability_top_10": float(np.mean(ranks <= 10)),
            "probability_rank_1": float(np.mean(ranks <= 1)),
        }

    strategy_summary = pd.DataFrame(
        [
            summary_row("pure_ev", pure_totals, rank_pure),
            summary_row("public_strategy", strategy_totals, rank_strategy),
        ]
    )
    strategy_summary["public_strategy_beats_pure_ev_rate"] = float(np.mean(strategy_totals > pure_totals))
    strategy_summary["public_strategy_ties_pure_ev_rate"] = float(np.mean(strategy_totals == pure_totals))
    return PublicContestSimulation(
        strategy_summary=strategy_summary,
        match_setup=pd.DataFrame(match_setup_rows),
        simulation_totals=pd.DataFrame(total_rows),
    )
