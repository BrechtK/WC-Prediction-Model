"""Run the historical backtest for every CSV below data/raw."""

from __future__ import annotations

from wc_predictor.backtest_cli import run_and_print_backtest


def main() -> None:
    """Run all nested historical CSVs with stable output paths for VS Code."""

    run_and_print_backtest(
        input_path="data/raw",
        detailed_output_path="data/processed/backtest_all_by_file.csv",
        aggregate_output_path="data/processed/backtest_all_aggregate.csv",
        skipped_output_path="data/processed/backtest_all_skipped.csv",
        favourite_strength_output_path="data/processed/backtest_all_favourite_strength.csv",
    )


if __name__ == "__main__":
    main()
