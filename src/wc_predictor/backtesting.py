"""Scaffold for historical out-of-sample strategy evaluation.

TODO:
    Load historical odds and results through provider-specific adapters.
    Reuse margin removal, score calibration, strategy prediction, and realised
    competition scoring without introducing World Cup-specific assumptions.
    Compare strategies using realised points, hit rates, variance, time-series
    cumulative points, and relevant favourite/draw splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class HistoricalOddsLoader(Protocol):
    """Interface for future providers such as Football-Data.co.uk."""

    def load(self) -> pd.DataFrame:
        """Return normalised historical odds and results."""


@dataclass(frozen=True)
class BacktestSettings:
    """Initial settings placeholder for future historical evaluation."""

    train_start: str | None = None
    test_start: str | None = None
    output_path: str = "data/processed/backtest_results.csv"


class BacktestRunner:
    """Placeholder orchestrator; full historical evaluation is reserved for Version 1.5."""

    def run(self) -> pd.DataFrame:
        """Run historical evaluation once provider adapters are implemented."""

        raise NotImplementedError("Backtesting is intentionally a Version 1.5 placeholder")

