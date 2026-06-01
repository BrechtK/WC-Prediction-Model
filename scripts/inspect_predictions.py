"""Print a compact manual-inspection report for the market-implied baseline."""

from __future__ import annotations

import argparse

from wc_predictor.config import ProjectConfig
from wc_predictor.market_data import load_odds
from wc_predictor.reporting import format_model_inspection_report
from wc_predictor.workflow import run_prediction_workflow


def main() -> None:
    """Print probabilities, calibration diagnostics, and EV rankings by match."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--odds", default="data/examples/example_odds.csv")
    args = parser.parse_args()

    workflow = run_prediction_workflow(load_odds(args.odds), config=ProjectConfig())
    print(format_model_inspection_report(workflow.match_report))


if __name__ == "__main__":
    main()

