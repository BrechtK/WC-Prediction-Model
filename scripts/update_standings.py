"""Score submitted predictions against results and export standings."""

from __future__ import annotations

import argparse

from wc_predictor.config import ProjectConfig
from wc_predictor.market_data import load_correct_score_odds, load_odds, load_predictions, load_results
from wc_predictor.reporting import export_standings_bundle
from wc_predictor.results import calculate_standings, score_submitted_predictions
from wc_predictor.workflow import run_prediction_workflow


def main() -> None:
    """Calculate realised pool points with pre-match expected points alongside them."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--odds", default="data/examples/example_odds.csv")
    parser.add_argument("--predictions", default="data/examples/example_predictions.csv")
    parser.add_argument("--results", default="data/examples/example_results.csv")
    parser.add_argument("--correct-score-odds")
    parser.add_argument("--correct-score-poisson-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    config = ProjectConfig(
        output_dir=args.output_dir,
        correct_score_poisson_weight=args.correct_score_poisson_weight,
    )
    odds = load_odds(args.odds)
    predictions = load_predictions(args.predictions)
    results = load_results(args.results)
    correct_score_odds = load_correct_score_odds(args.correct_score_odds) if args.correct_score_odds else None
    workflow = run_prediction_workflow(odds, predictions, config, correct_score_odds)
    scored = score_submitted_predictions(predictions, odds, results, config.knockout_scoring)
    standings = calculate_standings(scored, workflow.friend_report)
    export_standings_bundle(scored, standings, config.output_dir)
    print(standings.to_string(index=False))
    print(f"\nStandings reports written to {config.output_dir}")


if __name__ == "__main__":
    main()
