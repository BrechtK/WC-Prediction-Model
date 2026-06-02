"""Run the historical backtest for the Belgium CSV folder."""

from __future__ import annotations

import argparse

from wc_predictor.backtest_cli import add_backtest_runtime_arguments, run_and_print_backtest


def main() -> None:
    """Run Belgium history with stable output paths for VS Code."""

    parser = argparse.ArgumentParser()
    add_backtest_runtime_arguments(parser)
    args = parser.parse_args()

    run_and_print_backtest(
        input_path="data/raw/Belgium",
        detailed_output_path="data/processed/backtest_belgium_by_file.csv",
        aggregate_output_path="data/processed/backtest_belgium_aggregate.csv",
        skipped_output_path="data/processed/backtest_belgium_skipped.csv",
        favourite_strength_output_path="data/processed/backtest_belgium_favourite_strength.csv",
        fast=args.fast,
        max_files=args.max_files,
        max_matches=args.max_matches,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
