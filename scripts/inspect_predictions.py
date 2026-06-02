"""Print a compact manual-inspection report for the market-implied baseline."""

from __future__ import annotations

import argparse

from wc_predictor.config import ProjectConfig
from wc_predictor.market_data import load_correct_score_odds, load_odds
from wc_predictor.reporting import format_model_inspection_report
from wc_predictor.workflow import run_prediction_workflow


def main() -> None:
    """Print probabilities, calibration diagnostics, and EV rankings by match."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--odds", default="data/examples/example_odds.csv")
    parser.add_argument("--correct-score-odds")
    parser.add_argument("--correct-score-poisson-weight", type=float, default=1.0)
    parser.add_argument("--correct-score-aggregation-method", default="auto")
    args = parser.parse_args()

    correct_score_odds = load_correct_score_odds(args.correct_score_odds) if args.correct_score_odds else None
    workflow = run_prediction_workflow(
        load_odds(args.odds),
        config=ProjectConfig(
            correct_score_poisson_weight=args.correct_score_poisson_weight,
            correct_score_aggregation_method=args.correct_score_aggregation_method,
        ),
        correct_score_odds=correct_score_odds,
    )
    print(format_model_inspection_report(workflow.match_report))


if __name__ == "__main__":
    main()
