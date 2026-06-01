"""Generate match recommendations and friend-EV reports from bookmaker odds."""

from __future__ import annotations

import argparse

from wc_predictor.config import ProjectConfig
from wc_predictor.market_data import load_odds, load_predictions
from wc_predictor.reporting import export_report_bundle
from wc_predictor.workflow import run_prediction_workflow


def main() -> None:
    """Run the Version 1 market-implied recommendation workflow."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--odds", default="data/examples/example_odds.csv")
    parser.add_argument("--predictions", default="data/examples/example_predictions.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    config = ProjectConfig(output_dir=args.output_dir)
    workflow = run_prediction_workflow(load_odds(args.odds), load_predictions(args.predictions), config)
    export_report_bundle(workflow.match_report, workflow.friend_report, config.output_dir)
    print(workflow.match_report[["match_id", "team_a", "team_b", "recommended_score", "recommended_qualifier", "best_expected_points"]].to_string(index=False))
    print(f"\nReports written to {config.output_dir}")


if __name__ == "__main__":
    main()

