"""Central configuration for the Version 1 market-implied baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CalibrationWeights:
    """Relative weights for market constraints during Poisson calibration."""

    one_x_two: float = 1.0
    over_under_2_5: float = 0.75
    btts: float = 0.50


@dataclass(frozen=True)
class KnockoutScoringConfig:
    """Configurable interpretation of knockout-stage scoring."""

    score_basis: str = "final_before_penalties"
    exact_score_points: int = 6
    goal_difference_points: int = 4
    qualifier_points: int = 10
    participation_points: int = 1
    additive: bool = True

    def __post_init__(self) -> None:
        valid_bases = {"90min", "120min_if_extra_time", "final_before_penalties"}
        if self.score_basis not in valid_bases:
            raise ValueError(f"score_basis must be one of {sorted(valid_bases)}")


@dataclass(frozen=True)
class BacktestingConfig:
    """Reserved knobs for the Version 1.5 historical evaluation runner."""

    enabled: bool = False
    output_path: Path = Path("data/processed/backtest_results.csv")


@dataclass(frozen=True)
class StrategyConfig:
    """Shared strategy settings for prediction generation and later backtests."""

    top_alternatives: int = 5


@dataclass(frozen=True)
class ProjectConfig:
    """Defaults used by scripts; library functions remain individually configurable."""

    max_goals_score_matrix: int = 8
    max_candidate_goals: int = 5
    margin_removal_method: str = "proportional"
    bookmaker_aggregation_method: str = "mean"
    renormalise_score_matrix: bool = True
    suspicious_overround_low: float = 1.0
    suspicious_overround_high: float = 1.20
    poor_calibration_loss_threshold: float = 0.01
    output_dir: Path = Path("data/processed")
    calibration_weights: CalibrationWeights = field(default_factory=CalibrationWeights)
    knockout_scoring: KnockoutScoringConfig = field(default_factory=KnockoutScoringConfig)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
