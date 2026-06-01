"""Run the historical backtest for the England CSV folder."""

from __future__ import annotations

from wc_predictor.backtest_cli import run_and_print_backtest


def main() -> None:
    """Run England history with stable output paths for VS Code."""

    run_and_print_backtest(
        input_path="data/raw/England",
        detailed_output_path="data/processed/backtest_england_by_file.csv",
        aggregate_output_path="data/processed/backtest_england_aggregate.csv",
        skipped_output_path="data/processed/backtest_england_skipped.csv",
    )


if __name__ == "__main__":
    main()

