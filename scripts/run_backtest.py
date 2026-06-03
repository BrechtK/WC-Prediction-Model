"""Run the first historical group-stage-style strategy backtest."""

from __future__ import annotations

import argparse

from wc_predictor.backtest_cli import add_backtest_runtime_arguments, run_and_print_backtest


def main() -> None:
    """Load one CSV or a CSV folder, export summaries, and print them."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/examples/example_historical_matches.csv")
    parser.add_argument(
        "--detailed-output",
        "--output",
        dest="detailed_output",
        default="output/research/backtest_results_by_file.csv",
    )
    parser.add_argument(
        "--aggregate-output",
        default="output/research/backtest_results_aggregate.csv",
    )
    parser.add_argument(
        "--skipped-output",
        default="output/research/backtest_skipped_by_file.csv",
    )
    parser.add_argument(
        "--favourite-strength-output",
        default="output/research/backtest_favourite_strength.csv",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full per-file, aggregate, and skipped-match tables after the summary.",
    )
    add_backtest_runtime_arguments(parser)
    args = parser.parse_args()

    run_and_print_backtest(
        input_path=args.input,
        detailed_output_path=args.detailed_output,
        aggregate_output_path=args.aggregate_output,
        skipped_output_path=args.skipped_output,
        favourite_strength_output_path=args.favourite_strength_output,
        verbose=args.verbose,
        fast=args.fast,
        max_files=args.max_files,
        max_matches=args.max_matches,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
