"""Run the first historical group-stage-style strategy backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.backtesting import BatchBacktestRunner, BatchBacktestSettings


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
    args = parser.parse_args()

    settings = BatchBacktestSettings(
        detailed_output_path=Path(args.detailed_output),
        aggregate_output_path=Path(args.aggregate_output),
        skipped_output_path=Path(args.skipped_output),
    )
    report = BatchBacktestRunner(args.input, settings=settings).run()
    print("Per-file strategy results")
    print(report.detailed_summary.to_string(index=False))
    print("\nAggregate strategy results")
    print(report.aggregate_summary.to_string(index=False))
    print("\nSkipped matches by file and reason")
    if report.skipped_by_file_reason.empty:
        print("None")
    else:
        print(report.skipped_by_file_reason.to_string(index=False))
    print(f"\nDetailed results written to {settings.detailed_output_path}")
    print(f"Aggregate results written to {settings.aggregate_output_path}")
    print(f"Skipped-match report written to {settings.skipped_output_path}")


if __name__ == "__main__":
    main()
