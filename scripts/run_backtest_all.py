"""Run the historical backtest for every CSV below data/raw."""

from __future__ import annotations

import argparse

from wc_predictor.backtest_cli import add_backtest_runtime_arguments, run_and_print_backtest


def main() -> None:
    """Run all nested historical CSVs with stable output paths for VS Code."""

    parser = argparse.ArgumentParser()
    add_backtest_runtime_arguments(parser)
    args = parser.parse_args()

    run_and_print_backtest(
        input_path="data/raw",
        detailed_output_path="data/processed/backtest_all_by_file.csv",
        aggregate_output_path="data/processed/backtest_all_aggregate.csv",
        skipped_output_path="data/processed/backtest_all_skipped.csv",
        favourite_strength_output_path="data/processed/backtest_all_favourite_strength.csv",
        fast=args.fast,
        max_files=args.max_files,
        max_matches=args.max_matches,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
