"""Shared command-line orchestration for historical backtest runner scripts."""

from __future__ import annotations

from pathlib import Path

from wc_predictor.backtesting import BatchBacktestReport, BatchBacktestRunner, BatchBacktestSettings
from wc_predictor.reporting import format_backtest_console_summary


def run_and_print_backtest(
    input_path: str | Path,
    detailed_output_path: str | Path,
    aggregate_output_path: str | Path,
    skipped_output_path: str | Path,
    verbose: bool = False,
) -> BatchBacktestReport:
    """Run an existing batch backtest configuration and print its console summary."""

    settings = BatchBacktestSettings(
        detailed_output_path=Path(detailed_output_path),
        aggregate_output_path=Path(aggregate_output_path),
        skipped_output_path=Path(skipped_output_path),
    )
    report = BatchBacktestRunner(input_path, settings=settings).run()
    print(format_backtest_console_summary(report, input_path, settings, verbose))
    return report

