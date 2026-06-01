"""Run the first historical group-stage-style strategy backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.backtesting import BatchBacktestRunner, BatchBacktestSettings
from wc_predictor.reporting import format_backtest_console_summary


def main() -> None:
    """Load one CSV or a CSV folder, export summaries, and print them."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/examples/example_historical_matches.csv")
    parser.add_argument(
        "--detailed-output",
        "--output",
        dest="detailed_output",
        default="data/processed/backtest_results_by_file.csv",
    )
    parser.add_argument(
        "--aggregate-output",
        default="data/processed/backtest_results_aggregate.csv",
    )
    parser.add_argument(
        "--skipped-output",
        default="data/processed/backtest_skipped_by_file.csv",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full per-file, aggregate, and skipped-match tables after the summary.",
    )
    args = parser.parse_args()

    settings = BatchBacktestSettings(
        detailed_output_path=Path(args.detailed_output),
        aggregate_output_path=Path(args.aggregate_output),
        skipped_output_path=Path(args.skipped_output),
    )
    report = BatchBacktestRunner(args.input, settings=settings).run()
    print(format_backtest_console_summary(report, args.input, settings, args.verbose))


if __name__ == "__main__":
    main()
