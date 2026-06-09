"""Central configuration for the Version 1 market-implied baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wc_predictor.margin import IMPLEMENTED_MARGIN_REMOVAL_METHODS


@dataclass(frozen=True)
class CalibrationWeights:
    """Relative weights for market constraints during Poisson calibration."""

    one_x_two: float = 1.0
    over_under_2_5: float = 0.75
    total_goals_lines: float = 0.75
    btts: float = 0.50


@dataclass(frozen=True)
class DevigConfig:
    """Market-type-specific bookmaker-margin removal.

    Different markets behave differently. 1X2, BTTS, O/U totals and Asian
    handicap are 2-way or 3-way markets where Shin or power can be reasonable.
    Correct-score markets are many-outcome, high-overround, sparse, and often
    contain an "Other" bucket, so they default to the conservative normalised
    inverse-odds method. A ``None`` per-market method falls back to
    ``default_method``; ``correct_score_method`` keeps its own conservative
    default rather than following ``default_method``. If a requested method
    fails on a given market, the lower-level devig falls back to
    ``fallback_method`` and records a warning.
    """

    default_method: str = "normalised_inverse_odds"
    one_x_two_method: str | None = None
    btts_method: str | None = None
    total_goals_method: str | None = None
    asian_handicap_method: str | None = None
    correct_score_method: str = "normalised_inverse_odds"
    qualification_method: str | None = None
    fallback_method: str = "normalised_inverse_odds"

    def __post_init__(self) -> None:
        allowed = {*IMPLEMENTED_MARGIN_REMOVAL_METHODS, "proportional"}
        for name in (
            "default_method",
            "one_x_two_method",
            "btts_method",
            "total_goals_method",
            "asian_handicap_method",
            "correct_score_method",
            "qualification_method",
            "fallback_method",
        ):
            value = getattr(self, name)
            if value is not None and value not in allowed:
                raise ValueError(
                    f"DevigConfig.{name} must be one of {sorted(allowed)} or None"
                )

    def method_for(self, market: str) -> str:
        """Return the devig method for one market key, resolving None to default."""

        if market == "correct_score":
            return self.correct_score_method or self.default_method
        per_market = {
            "1x2": self.one_x_two_method,
            "one_x_two": self.one_x_two_method,
            "over_under_2_5": self.total_goals_method,
            "total_goals": self.total_goals_method,
            "btts": self.btts_method,
            "asian_handicap": self.asian_handicap_method,
            "qualification": self.qualification_method or self.one_x_two_method,
        }
        if market not in per_market:
            raise ValueError(f"Unknown devig market key: {market!r}")
        return per_market[market] or self.default_method


@dataclass(frozen=True)
class GroupScoringConfig:
    """Single source of truth for group-stage scoring."""

    exact_score_points: int = 10
    goal_difference_points: int = 7
    result_points: int = 5
    participation_points: int = 1

    @property
    def exact_increment(self) -> int:
        return self.exact_score_points - self.goal_difference_points

    @property
    def goal_difference_increment(self) -> int:
        return self.goal_difference_points - self.result_points

    @property
    def result_increment(self) -> int:
        return self.result_points - self.participation_points

    @property
    def draw_increment(self) -> int:
        return self.goal_difference_points - self.participation_points


@dataclass(frozen=True)
class KnockoutScoringConfig:
    """Configurable interpretation of knockout-stage scoring."""

    score_basis: str = "final_before_penalties"
    knockout_scoring_mode: str = "unverified"
    exact_score_points: int = 6
    goal_difference_points: int = 4
    qualifier_points: int = 10
    participation_points: int = 1
    additive: bool = True

    def __post_init__(self) -> None:
        valid_modes = {"additive", "hierarchical", "unverified"}
        if self.knockout_scoring_mode not in valid_modes:
            raise ValueError(f"knockout_scoring_mode must be one of {sorted(valid_modes)}")
        if not self.additive and self.knockout_scoring_mode == "additive":
            raise ValueError("additive=False conflicts with knockout_scoring_mode='additive'")
        valid_bases = {"90min", "120min_if_extra_time", "final_before_penalties"}
        if self.score_basis not in valid_bases:
            raise ValueError(f"score_basis must be one of {sorted(valid_bases)}")


@dataclass(frozen=True)
class MarketConsistentGroupWeights:
    """Group-level multipliers for market-consistent soft constraints.

    Research scaffold for reliability-aware weighting. The market-consistent
    projection currently treats every constraint as independent even though
    1X2, Asian handicap, total goals, BTTS and correct score share information
    (result, margin and total-goal structure overlap). A true correlated-error
    covariance model needs data that is not yet available, so these multipliers
    let a researcher down-weight an over-represented group instead. All default
    to ``1.0``, which exactly reproduces the unweighted baseline.
    """

    one_x_two: float = 1.0
    total_goals: float = 1.0
    asian_handicap: float = 1.0
    btts: float = 1.0
    correct_score: float = 1.0

    def __post_init__(self) -> None:
        for name in ("one_x_two", "total_goals", "asian_handicap", "btts", "correct_score"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"market-consistent group weight {name} must be finite and non-negative")


@dataclass(frozen=True)
class BacktestingConfig:
    """Project-level defaults for historical evaluation outputs."""

    enabled: bool = False
    output_path: Path = Path("output/research/backtest_results.csv")


@dataclass(frozen=True)
class StrategyConfig:
    """Shared strategy settings for prediction generation and later backtests."""

    top_alternatives: int = 5
    plausible_exact_probability_threshold: float = 0.005
    plausible_total_goals_limit: int = 5
    plausible_alternative_min_exact_probability: float = 0.005
    plausible_alternative_max_total_goals: int = 5
    low_confidence_ev_gap_threshold: float = 0.07
    high_confidence_ev_gap_threshold: float = 0.15
    high_score_cluster_ev_gap_threshold: float = 0.10
    modal_draw_favourite_probability_threshold: float = 0.45
    modal_draw_gap_threshold: float = -0.70
    modal_draw_ev_gap_threshold: float = 0.70
    btts_conflict_probability_threshold: float = 0.45
    btts_conflict_favourite_probability_threshold: float = 0.50
    btts_conflict_ev_gap_threshold: float = 0.70
    draw_prone_ou_median_total_threshold: float = 2.25
    draw_prone_expected_total_goals_threshold: float = 2.35
    draw_prone_favourite_probability_threshold: float = 0.50
    blowout_favourite_probability_threshold: float = 0.70
    blowout_ou_median_total_threshold: float = 2.60

    def __post_init__(self) -> None:
        if self.top_alternatives <= 0:
            raise ValueError("top_alternatives must be positive")
        for name in (
            "plausible_exact_probability_threshold",
            "plausible_alternative_min_exact_probability",
            "low_confidence_ev_gap_threshold",
            "high_confidence_ev_gap_threshold",
            "high_score_cluster_ev_gap_threshold",
            "modal_draw_ev_gap_threshold",
            "btts_conflict_probability_threshold",
            "btts_conflict_favourite_probability_threshold",
            "btts_conflict_ev_gap_threshold",
            "draw_prone_ou_median_total_threshold",
            "draw_prone_expected_total_goals_threshold",
            "draw_prone_favourite_probability_threshold",
            "blowout_favourite_probability_threshold",
            "blowout_ou_median_total_threshold",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        modal_draw_gap = float(self.modal_draw_gap_threshold)
        if not np.isfinite(modal_draw_gap):
            raise ValueError("modal_draw_gap_threshold must be finite")
        for name in ("plausible_total_goals_limit", "plausible_alternative_max_total_goals"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PublicStrategyConfig:
    """Diagnostic settings for public-field ranking strategy.

    These parameters are heuristic and unvalidated; keep public strategy
    diagnostic-only until backtests justify changing live submissions.
    """

    mode: str = "ev"
    alpha: float = 0.20
    beta: float = 0.15
    gamma: float = 1.0
    max_public_strategy_ev_loss: float | None = None
    min_exact_score_probability: float = 0.025
    min_result_probability: float = 0.20
    friend_sample_weight: float = 0.15
    public_strategy_target: str = "balanced"
    public_field_size: int = 100
    team_popularity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "Belgium": 1.00,
            "France": 0.85,
            "Netherlands": 0.85,
            "England": 0.85,
            "Brazil": 0.85,
            "Argentina": 0.85,
            "Germany": 0.85,
            "Spain": 0.85,
            "Portugal": 0.85,
            "Mexico": 0.55,
            "USA": 0.55,
            "Canada": 0.45,
            "Switzerland": 0.45,
        }
    )

    def __post_init__(self) -> None:
        valid_modes = {"ev", "balanced", "public-ranking", "aggressive-public-ranking"}
        if self.mode not in valid_modes:
            raise ValueError(f"strategy mode must be one of {sorted(valid_modes)}")
        valid_targets = {"friends", "balanced", "national"}
        if self.public_strategy_target not in valid_targets:
            raise ValueError(f"public_strategy_target must be one of {sorted(valid_targets)}")
        if self.public_field_size <= 0:
            raise ValueError("public_field_size must be positive")
        for name in ("alpha", "beta", "gamma", "min_exact_score_probability", "min_result_probability"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_public_strategy_ev_loss is not None:
            value = float(self.max_public_strategy_ev_loss)
            if not np.isfinite(value) or value < 0:
                raise ValueError("max_public_strategy_ev_loss must be finite and non-negative")

    @property
    def effective_max_ev_loss(self) -> float:
        """Return the mode-specific EV loss cap."""

        if self.max_public_strategy_ev_loss is not None:
            return self.max_public_strategy_ev_loss
        if self.mode == "balanced":
            return 0.15
        if self.mode == "public-ranking":
            return 0.15
        if self.mode == "aggressive-public-ranking":
            return 0.30
        return 0.0

    @property
    def field_influence_scale(self) -> float:
        """Return diagnostic public-field influence by target size."""

        target_scale = {
            "friends": 0.45,
            "balanced": 1.0,
            "national": 1.75,
        }[self.public_strategy_target]
        size_scale = min(2.0, max(0.35, np.log10(max(self.public_field_size, 2)) / 2.0))
        return target_scale * size_scale


@dataclass(frozen=True)
class ProjectConfig:
    """Defaults used by scripts; library functions remain individually configurable."""

    max_goals_score_matrix: int = 8
    max_candidate_goals: int = 5
    margin_removal_method: str = "normalised_inverse_odds"
    devig: "DevigConfig | None" = None
    enable_margin_method_comparison: bool = True
    enable_final_decision_dashboard: bool = True
    enable_market_consistent_challenger: bool | str = True
    market_consistent_prior: str = "independent_poisson"
    bivariate_poisson_covariance: float = 0.0
    enable_bivariate_poisson_diagnostic: bool = False
    enable_asian_handicap_margin_model: bool = False
    market_consistent_group_weights: MarketConsistentGroupWeights = field(
        default_factory=MarketConsistentGroupWeights
    )
    bookmaker_aggregation_method: str = "mean"
    renormalise_score_matrix: bool = True
    suspicious_overround_low: float = 1.0
    suspicious_overround_high: float = 1.20
    poor_calibration_loss_threshold: float = 0.01
    correct_score_poisson_weight: float = 1.0
    correct_score_aggregation_method: str = "auto"
    correct_score_outlier_z_threshold: float = 3.0
    min_scorelines_for_blend: int = 10
    dynamic_grid_enabled: bool = True
    extreme_favourite_max_goals: int = 12
    research_max_goals: int = 15
    dixon_coles_rho: float = 0.0
    enable_dixon_coles_rho_estimation: bool = True
    dixon_coles_rho_min: float = -0.20
    dixon_coles_rho_max: float = 0.20
    dixon_coles_rho_grid_size: int = 81
    output_dir: Path = Path("output/research")
    calibration_weights: CalibrationWeights = field(default_factory=CalibrationWeights)
    knockout_scoring: KnockoutScoringConfig = field(default_factory=KnockoutScoringConfig)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    public_strategy: PublicStrategyConfig = field(default_factory=PublicStrategyConfig)

    def __post_init__(self) -> None:
        allowed_margin_methods = {*IMPLEMENTED_MARGIN_REMOVAL_METHODS, "proportional"}
        if self.margin_removal_method not in allowed_margin_methods:
            raise ValueError(
                "margin_removal_method must be one of "
                f"{sorted(IMPLEMENTED_MARGIN_REMOVAL_METHODS)}"
            )
        if self.enable_market_consistent_challenger not in {True, False, "only_if_close"}:
            raise ValueError("enable_market_consistent_challenger must be True, False, or 'only_if_close'")
        valid_priors = {"independent_poisson", "dixon_coles", "bivariate_poisson"}
        if self.market_consistent_prior not in valid_priors:
            raise ValueError(f"market_consistent_prior must be one of {sorted(valid_priors)}")
        if not np.isfinite(self.bivariate_poisson_covariance) or self.bivariate_poisson_covariance < 0:
            raise ValueError("bivariate_poisson_covariance must be finite and non-negative")
        if not 0 <= self.correct_score_poisson_weight <= 1:
            raise ValueError("correct_score_poisson_weight must lie between zero and one")
        valid_correct_score_aggregation_methods = {
            "auto",
            "mean",
            "median",
            "trimmed_mean",
            "winsorized_mean",
            "reliability_weighted_mean",
        }
        if self.correct_score_aggregation_method not in valid_correct_score_aggregation_methods:
            raise ValueError(
                "correct_score_aggregation_method must be one of "
                f"{sorted(valid_correct_score_aggregation_methods)}"
            )
        if self.correct_score_outlier_z_threshold <= 0:
            raise ValueError("correct_score_outlier_z_threshold must be positive")
        if not isinstance(self.min_scorelines_for_blend, int) or isinstance(self.min_scorelines_for_blend, bool):
            raise ValueError("min_scorelines_for_blend must be a positive integer")
        if self.min_scorelines_for_blend <= 0:
            raise ValueError("min_scorelines_for_blend must be a positive integer")
        if not np.isfinite(self.dixon_coles_rho):
            raise ValueError("dixon_coles_rho must be finite")
        if not np.isfinite(self.dixon_coles_rho_min) or not np.isfinite(self.dixon_coles_rho_max):
            raise ValueError("Dixon-Coles rho bounds must be finite")
        if self.dixon_coles_rho_min >= self.dixon_coles_rho_max:
            raise ValueError("dixon_coles_rho_min must be below dixon_coles_rho_max")
        if self.dixon_coles_rho_grid_size < 2:
            raise ValueError("dixon_coles_rho_grid_size must be at least two")
        for name in ("extreme_favourite_max_goals", "research_max_goals"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < self.max_goals_score_matrix:
                raise ValueError(f"{name} must be an integer at least max_goals_score_matrix")

    def resolved_devig(self) -> "DevigConfig":
        """Return the effective per-market devig configuration.

        When ``devig`` is unset the legacy behaviour is preserved exactly: every
        market uses ``margin_removal_method`` with a normalised-inverse-odds
        fallback. When ``devig`` is supplied it is used as-is, enabling
        market-type-specific devig (for example power for 1X2 while correct
        score stays conservative).
        """

        if self.devig is not None:
            return self.devig
        return DevigConfig(
            default_method=self.margin_removal_method,
            correct_score_method=self.margin_removal_method,
            qualification_method=self.margin_removal_method,
            fallback_method="normalised_inverse_odds",
        )
