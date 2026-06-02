"""Run the complete live OddsPortal paste-to-submission workflow."""

from __future__ import annotations

import argparse

from wc_predictor.config import ProjectConfig
from wc_predictor.live_prediction import (
    LivePredictionError,
    LivePredictionSettings,
    format_live_prediction_summary,
    run_live_prediction,
)


def main() -> None:
    """Parse live pastes, export recommendations, and print the final submission."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id")
    parser.add_argument("--correct-score-poisson-weight", type=float, default=0.85)
    parser.add_argument("--correct-score-aggregation-method", default="auto")
    parser.add_argument("--skip-weight-sensitivity", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    try:
        result = run_live_prediction(
            LivePredictionSettings(
                match_id=args.match_id,
                strict=args.strict,
                skip_weight_sensitivity=args.skip_weight_sensitivity,
            ),
            ProjectConfig(
                correct_score_poisson_weight=args.correct_score_poisson_weight,
                correct_score_aggregation_method=args.correct_score_aggregation_method,
            ),
        )
    except (LivePredictionError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(format_live_prediction_summary(result))


if __name__ == "__main__":
    main()
