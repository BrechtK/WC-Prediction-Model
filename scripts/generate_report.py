"""Generate both recommendation and realised-standings report bundles."""

from __future__ import annotations

import argparse

from wc_predictor.config import ProjectConfig
from wc_predictor.market_data import load_odds, load_predictions, load_results
from wc_predictor.reporting import export_report_bundle, export_standings_bundle
from wc_predictor.results import calculate_standings, score_submitted_predictions
from wc_predictor.workflow import run_prediction_workflow


def main() -> None:
    """Write all Version 1 reports in one command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--odds", default="data/examples/example_odds.csv")
    parser.add_argument("--predictions", default="data/examples/example_predictions.csv")
    parser.add_argument("--results", default="data/examples/example_results.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    config = ProjectConfig(output_dir=args.output_dir)
    odds = load_odds(args.odds)
    predictions = load_predictions(args.predictions)
    workflow = run_prediction_workflow(odds, predictions, config)
    scored = score_submitted_predictions(predictions, odds, load_results(args.results), config.knockout_scoring)
    standings = calculate_standings(scored, workflow.friend_report)
    export_report_bundle(workflow.match_report, workflow.friend_report, config.output_dir)
    export_standings_bundle(scored, standings, config.output_dir)
    print(f"All reports written to {config.output_dir}")


if __name__ == "__main__":
    main()

