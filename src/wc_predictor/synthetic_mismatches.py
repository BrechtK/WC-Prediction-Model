"""Synthetic 1X2 mismatch scenarios for manual EV inspection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from wc_predictor.calibration import CalibrationTargets, calibrate_poisson_model
from wc_predictor.config import ProjectConfig
from wc_predictor.optimiser import optimise_group_prediction
from wc_predictor.utils import favourite_strength_bucket

DEFAULT_SYNTHETIC_MISMATCHES = (
    (0.55, 0.25, 0.20),
    (0.65, 0.22, 0.13),
    (0.75, 0.16, 0.09),
    (0.85, 0.10, 0.05),
    (0.90, 0.07, 0.03),
)


@dataclass(frozen=True)
class SyntheticMismatchResult:
    """Calibrated and optimised output for one synthetic fair 1X2 scenario."""

    fair_1x2: tuple[float, float, float]
    favourite_probability: float
    favourite_bucket: str
    lambda_a: float
    lambda_b: float
    model_1x2: tuple[float, float, float]
    most_likely_scoreline: tuple[int, int]
    ev_optimal_scoreline: tuple[int, int]
    top_5_ev_predictions: tuple[tuple[tuple[int, int], float], ...]


def inspect_synthetic_mismatches(
    scenarios: Iterable[tuple[float, float, float]] = DEFAULT_SYNTHETIC_MISMATCHES,
    config: ProjectConfig | None = None,
) -> tuple[SyntheticMismatchResult, ...]:
    """Calibrate and optimise transparent synthetic fair 1X2 mismatch scenarios."""

    config = config or ProjectConfig()
    results: list[SyntheticMismatchResult] = []
    for a_win, draw, b_win in scenarios:
        calibration = calibrate_poisson_model(
            CalibrationTargets(a_win, draw, b_win),
            config.max_goals_score_matrix,
            config.calibration_weights,
            config.renormalise_score_matrix,
            config.poor_calibration_loss_threshold,
        )
        recommendation = optimise_group_prediction(
            calibration.score_matrix,
            config.max_candidate_goals,
            top_n=4,
        )
        evaluations = (recommendation.best, *recommendation.alternatives)
        favourite_probability = max(a_win, b_win)
        model = calibration.model_probabilities
        results.append(
            SyntheticMismatchResult(
                (a_win, draw, b_win),
                favourite_probability,
                favourite_strength_bucket(favourite_probability),
                calibration.lambda_a,
                calibration.lambda_b,
                (model["a_win"], model["draw"], model["b_win"]),
                recommendation.most_likely_scoreline,
                recommendation.best.predicted_score,
                tuple((evaluation.predicted_score, evaluation.expected_points) for evaluation in evaluations),
            )
        )
    return tuple(results)


def format_synthetic_mismatch_report(results: Iterable[SyntheticMismatchResult]) -> str:
    """Render synthetic mismatch calibration and EV output for manual inspection."""

    sections: list[str] = []
    for result in results:
        fair_a, fair_draw, fair_b = result.fair_1x2
        model_a, model_draw, model_b = result.model_1x2
        top_five = "; ".join(
            f"{score_a}-{score_b} ({expected_points:.3f} EV)"
            for (score_a, score_b), expected_points in result.top_5_ev_predictions
        )
        sections.append(
            "\n".join(
                [
                    f"Fair 1X2: A={fair_a:.2%} D={fair_draw:.2%} B={fair_b:.2%}",
                    f"  Favourite strength: p_fav={result.favourite_probability:.2%} ({result.favourite_bucket})",
                    f"  Calibrated lambdas: A={result.lambda_a:.4f} B={result.lambda_b:.4f}",
                    f"  Model-implied 1X2: A={model_a:.2%} D={model_draw:.2%} B={model_b:.2%}",
                    f"  Most likely scoreline: {result.most_likely_scoreline[0]}-{result.most_likely_scoreline[1]}",
                    f"  EV-optimal scoreline: {result.ev_optimal_scoreline[0]}-{result.ev_optimal_scoreline[1]}",
                    f"  Top 5 EV predictions: {top_five}",
                ]
            )
        )
    return "\n\n".join(sections)

