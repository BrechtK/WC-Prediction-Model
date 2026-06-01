"""Run the first historical group-stage-style strategy backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.backtesting import BacktestRunner, BacktestSettings, FootballDataCSVLoader


def main() -> None:
    """Load Football-Data-like history, export summary metrics, and print them."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/examples/example_historical_matches.csv")
    parser.add_argument("--output", default="data/processed/backtest_results.csv")
    args = parser.parse_args()

    settings = BacktestSettings(output_path=Path(args.output))
    report = BacktestRunner(FootballDataCSVLoader(args.input), settings=settings).run()
    print(report.summary.to_string(index=False))
    print(f"\nBacktest summary written to {settings.output_path}")


if __name__ == "__main__":
    main()

