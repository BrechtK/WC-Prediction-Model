"""Shared command-line orchestration for historical backtest runner scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.backtesting import BatchBacktestReport, BatchBacktestRunner, BatchBacktestSettings
from wc_predictor.reporting import format_backtest_console_summary


def add_backtest_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared historical runtime controls to one command-line parser."""

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use single-start Poisson calibration for exploratory historical runs.",
    )
    parser.add_argument("--max-files", type=int, help="Process at most N CSV files.")
    parser.add_argument("--max-matches", type=int, help="Process at most N historical matches in total.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print match progress every N matches within a historical CSV file.",
    )


def run_and_print_backtest(
    input_path: str | Path,
    detailed_output_path: str | Path,
    aggregate_output_path: str | Path,
    skipped_output_path: str | Path,
    favourite_strength_output_path: str | Path = "output/research/backtest_favourite_strength.csv",
    verbose: bool = False,
    fast: bool = False,
    max_files: int | None = None,
    max_matches: int | None = None,
    progress_interval: int = 100,
) -> BatchBacktestReport:
    """Run an existing batch backtest configuration and print its console summary."""

    settings = BatchBacktestSettings(
        detailed_output_path=Path(detailed_output_path),
        aggregate_output_path=Path(aggregate_output_path),
        skipped_output_path=Path(skipped_output_path),
        favourite_strength_output_path=Path(favourite_strength_output_path),
    )
    report = BatchBacktestRunner(
        input_path,
        settings=settings,
        fast=fast,
        max_files=max_files,
        max_matches=max_matches,
        progress=True,
        progress_interval=progress_interval,
    ).run()
    print(format_backtest_console_summary(report, input_path, settings, verbose))
    return report
