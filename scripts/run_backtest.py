"""Run the first historical group-stage-style strategy backtest."""

from __future__ import annotations

import argparse

from wc_predictor.backtest_cli import run_and_print_backtest


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
        "--favourite-strength-output",
        default="data/processed/backtest_favourite_strength.csv",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full per-file, aggregate, and skipped-match tables after the summary.",
    )
    args = parser.parse_args()

    run_and_print_backtest(
        input_path=args.input,
        detailed_output_path=args.detailed_output,
        aggregate_output_path=args.aggregate_output,
        skipped_output_path=args.skipped_output,
        favourite_strength_output_path=args.favourite_strength_output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
