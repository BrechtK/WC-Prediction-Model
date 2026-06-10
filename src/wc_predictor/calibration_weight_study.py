"""Calibration-weight robustness and exploratory search utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from wc_predictor.config import CalibrationWeights, ProjectConfig
from wc_predictor.live_backtest import BacktestConfigEntry, LiveBacktestReport, run_combined_live_backtest
from wc_predictor.utils import ensure_parent_directory


DEFAULT_TOURNAMENT_FOLDERS: tuple[Path, ...] = (
    Path("input/historical/wc2014"),
    Path("input/historical/wc2018"),
    Path("input/historical/wc2022"),
)

SENSITIVITY_OUTPUT_PATH = Path("output/research/calibration_weight_sensitivity.csv")
SENSITIVITY_BY_TOURNAMENT_OUTPUT_PATH = Path("output/research/calibration_weight_sensitivity_by_tournament.csv")
SEARCH_OUTPUT_PATH = Path("output/research/calibration_weight_search.csv")
SEARCH_TOP_PROFILES_OUTPUT_PATH = Path("output/research/calibration_weight_search_top_profiles.csv")


@dataclass(frozen=True)
class CalibrationWeightProfile:
    """One named profile for the basic Poisson calibration weights."""

    name: str
    one_x_two: float
    totals: float
    btts: float
    study_type: str = "robustness"
    note: str = ""

    def weights(self) -> CalibrationWeights:
        return CalibrationWeights(
            one_x_two=self.one_x_two,
            over_under_2_5=self.totals,
            total_goals_lines=self.totals,
            btts=self.btts,
        )


ROBUSTNESS_PROFILES: tuple[CalibrationWeightProfile, ...] = (
    CalibrationWeightProfile("current", 1.00, 0.75, 0.50, note="live default; not changed by this study"),
    CalibrationWeightProfile("conservative", 1.00, 0.50, 0.25),
    CalibrationWeightProfile("equal", 1.00, 1.00, 1.00),
    CalibrationWeightProfile("totals_heavy", 1.00, 1.25, 0.75),
    CalibrationWeightProfile("btts_heavy", 1.00, 0.75, 1.00),
    CalibrationWeightProfile("result_anchor_strong", 1.50, 0.75, 0.50),
)

SEARCH_ONE_X_TWO_GRID: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0)
SEARCH_TOTALS_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
SEARCH_BTTS_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)

SEARCH_LABEL = "in-sample exploratory search, likely overfit, not used for live policy"


def make_search_profiles(
    *,
    one_x_two_grid: Sequence[float] = SEARCH_ONE_X_TWO_GRID,
    totals_grid: Sequence[float] = SEARCH_TOTALS_GRID,
    btts_grid: Sequence[float] = SEARCH_BTTS_GRID,
) -> tuple[CalibrationWeightProfile, ...]:
    """Return the exploratory in-sample grid of calibration-weight profiles."""

    profiles: list[CalibrationWeightProfile] = []
    for one_x_two in one_x_two_grid:
        for totals in totals_grid:
            for btts in btts_grid:
                profiles.append(
                    CalibrationWeightProfile(
                        f"w_1x2_{one_x_two:g}_totals_{totals:g}_btts_{btts:g}".replace(".", "p"),
                        float(one_x_two),
                        float(totals),
                        float(btts),
                        study_type="in_sample_exploratory",
                        note=SEARCH_LABEL,
                    )
                )
    return tuple(profiles)


def _profile_config_name(profile: CalibrationWeightProfile, base_name: str) -> str:
    return f"{profile.name}__{base_name}"


def build_profile_configs(
    profiles: Iterable[CalibrationWeightProfile],
    *,
    include_market_consistent: bool,
) -> tuple[BacktestConfigEntry, ...]:
    """Create live-backtest configs for each calibration-weight profile."""

    configs: list[BacktestConfigEntry] = []
    for profile in profiles:
        base_config = ProjectConfig(
            calibration_weights=profile.weights(),
            enable_market_consistent_challenger=False,
            enable_margin_method_comparison=False,
            enable_dixon_coles_rho_estimation=False,
        )
        configs.append(
            BacktestConfigEntry(
                name=_profile_config_name(profile, "baseline_ev"),
                config=base_config,
                description=f"Calibration-weight profile {profile.name}: EV/default matrix",
                use_asian_handicap=False,
            )
        )
        if include_market_consistent:
            configs.append(
                BacktestConfigEntry(
                    name=_profile_config_name(profile, "mc_with_ah"),
                    config=replace(base_config, enable_market_consistent_challenger=True),
                    description=f"Calibration-weight profile {profile.name}: MC+AH challenger",
                    use_asian_handicap=True,
                )
            )
    return tuple(configs)


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _numeric_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.sum()) if values.notna().any() else np.nan


def _strategy_points(predictions: pd.DataFrame, config: str, strategy: str) -> float:
    rows = predictions[
        predictions["config"].astype(str).eq(config)
        & predictions["strategy"].astype(str).eq(strategy)
    ]
    return _numeric_sum(rows, "realised_points")


def _gated_mc_ah_points(predictions: pd.DataFrame, config: str) -> float:
    ev_rows = predictions[
        predictions["config"].astype(str).eq(config)
        & predictions["strategy"].astype(str).eq("ev_default")
    ].copy()
    mc_rows = predictions[
        predictions["config"].astype(str).eq(config)
        & predictions["strategy"].astype(str).eq("market_consistent")
    ].copy()
    if ev_rows.empty:
        return np.nan
    key_columns = ["tournament", "match_id"] if "tournament" in ev_rows else ["match_id"]
    mc_by_key = mc_rows.set_index(key_columns) if not mc_rows.empty else pd.DataFrame()
    total = 0.0
    for _, row in ev_rows.iterrows():
        acceptable = str(row.get("mc_ah_optimisation_acceptable", "")).lower() == "yes"
        key = tuple(row[column] for column in key_columns) if len(key_columns) > 1 else row[key_columns[0]]
        if acceptable and not mc_by_key.empty and key in mc_by_key.index:
            mc_row = mc_by_key.loc[key]
            if isinstance(mc_row, pd.DataFrame):
                mc_row = mc_row.iloc[0]
            total += float(pd.to_numeric(mc_row.get("realised_points", np.nan), errors="coerce"))
        else:
            total += float(pd.to_numeric(row.get("realised_points", np.nan), errors="coerce"))
    return total


def _mean_abs_model_market_1x2_error(rows: pd.DataFrame) -> float:
    pairs = (
        ("predicted_probability_team_a_win", "market_a_win"),
        ("predicted_probability_draw", "market_draw"),
        ("predicted_probability_team_b_win", "market_b_win"),
    )
    values: list[float] = []
    for predicted_column, market_column in pairs:
        if predicted_column not in rows or market_column not in rows:
            continue
        predicted = pd.to_numeric(rows[predicted_column], errors="coerce")
        market = pd.to_numeric(rows[market_column], errors="coerce")
        valid = predicted.notna() & market.notna()
        values.extend((predicted[valid] - market[valid]).abs().tolist())
    return float(np.mean(values)) if values else np.nan


def _mean_abs_model_market_btts_error(rows: pd.DataFrame) -> float:
    if not {"predicted_btts_yes_probability", "market_fair_btts_yes_probability"}.issubset(rows.columns):
        return np.nan
    predicted = pd.to_numeric(rows["predicted_btts_yes_probability"], errors="coerce")
    market = pd.to_numeric(rows["market_fair_btts_yes_probability"], errors="coerce")
    valid = predicted.notna() & market.notna()
    return float((predicted[valid] - market[valid]).abs().mean()) if valid.any() else np.nan


def _mean_abs_model_market_total_error(total_rows: pd.DataFrame) -> float:
    required = {"predicted_over_probability", "market_fair_over_probability"}
    if total_rows.empty or not required.issubset(total_rows.columns):
        return np.nan
    predicted = pd.to_numeric(total_rows["predicted_over_probability"], errors="coerce")
    market = pd.to_numeric(total_rows["market_fair_over_probability"], errors="coerce")
    valid = predicted.notna() & market.notna()
    return float((predicted[valid] - market[valid]).abs().mean()) if valid.any() else np.nan


def _modal_ev_divergence(rows: pd.DataFrame, config: str) -> tuple[int, float]:
    ev = rows[
        rows["config"].astype(str).eq(config)
        & rows["strategy"].astype(str).eq("ev_default")
    ].copy()
    modal = rows[
        rows["config"].astype(str).eq(config)
        & rows["strategy"].astype(str).eq("most_likely")
    ].copy()
    if ev.empty or modal.empty:
        return 0, np.nan
    key_columns = ["tournament", "match_id"] if "tournament" in ev else ["match_id"]
    merged = ev[key_columns + ["predicted_score"]].merge(
        modal[key_columns + ["predicted_score"]],
        on=key_columns,
        suffixes=("_ev", "_modal"),
    )
    if merged.empty:
        return 0, np.nan
    changed = int((merged["predicted_score_ev"].astype(str) != merged["predicted_score_modal"].astype(str)).sum())
    return changed, changed / len(merged)


def _changed_vs_current(
    rows: pd.DataFrame,
    *,
    config: str,
    current_config: str,
) -> tuple[int, float]:
    ev = rows[
        rows["config"].astype(str).eq(config)
        & rows["strategy"].astype(str).eq("ev_default")
    ]
    current = rows[
        rows["config"].astype(str).eq(current_config)
        & rows["strategy"].astype(str).eq("ev_default")
    ]
    if ev.empty or current.empty:
        return 0, np.nan
    key_columns = ["tournament", "match_id"] if "tournament" in ev else ["match_id"]
    merged = ev[key_columns + ["predicted_score"]].merge(
        current[key_columns + ["predicted_score"]],
        on=key_columns,
        suffixes=("", "_current"),
    )
    if merged.empty:
        return 0, np.nan
    changed = int((merged["predicted_score"].astype(str) != merged["predicted_score_current"].astype(str)).sum())
    return changed, changed / len(merged)


def _profile_row(
    *,
    profile: CalibrationWeightProfile,
    predictions: pd.DataFrame,
    total_goals_rows: pd.DataFrame,
    scope: str,
    tournament: str,
    current_config: str,
    include_market_consistent: bool,
) -> dict[str, object]:
    baseline_config = _profile_config_name(profile, "baseline_ev")
    mc_config = _profile_config_name(profile, "mc_with_ah")
    baseline_ev = predictions[
        predictions["config"].astype(str).eq(baseline_config)
        & predictions["strategy"].astype(str).eq("ev_default")
    ]
    baseline_totals = total_goals_rows[
        total_goals_rows["config"].astype(str).eq(baseline_config)
        & total_goals_rows["strategy"].astype(str).eq("ev_default")
    ] if not total_goals_rows.empty and "config" in total_goals_rows else pd.DataFrame()
    modal_changed, modal_changed_fraction = _modal_ev_divergence(predictions, baseline_config)
    changed, changed_fraction = _changed_vs_current(
        predictions,
        config=baseline_config,
        current_config=current_config,
    )
    return {
        "profile": profile.name,
        "study_type": profile.study_type,
        "research_warning": profile.note,
        "scope": scope,
        "tournament": tournament,
        "weight_1x2": profile.one_x_two,
        "weight_totals": profile.totals,
        "weight_over_under_2_5": profile.totals,
        "weight_total_goals_lines": profile.totals,
        "weight_btts": profile.btts,
        "matches": int(baseline_ev[["tournament", "match_id"]].drop_duplicates().shape[0])
        if {"tournament", "match_id"}.issubset(baseline_ev.columns)
        else int(baseline_ev["match_id"].nunique()) if "match_id" in baseline_ev else 0,
        "ev_default_points": _strategy_points(predictions, baseline_config, "ev_default"),
        "modal_points": _strategy_points(predictions, baseline_config, "most_likely"),
        "raw_mc_ah_points": (
            _strategy_points(predictions, mc_config, "market_consistent")
            if include_market_consistent
            else np.nan
        ),
        "gated_mc_ah_points": (
            _gated_mc_ah_points(predictions, mc_config)
            if include_market_consistent
            else np.nan
        ),
        "mean_expected_total_goals": _numeric_mean(baseline_ev, "matrix_expected_total_goals"),
        "mean_actual_total_goals": _numeric_mean(baseline_ev, "actual_total_goals"),
        "expected_minus_actual_total_goals": (
            _numeric_mean(baseline_ev, "matrix_expected_total_goals")
            - _numeric_mean(baseline_ev, "actual_total_goals")
        ),
        "mean_abs_model_market_1x2_error": _mean_abs_model_market_1x2_error(baseline_ev),
        "mean_abs_model_market_total_goals_error": _mean_abs_model_market_total_error(baseline_totals),
        "mean_abs_model_market_btts_error": _mean_abs_model_market_btts_error(baseline_ev),
        "mean_brier_1x2": _numeric_mean(baseline_ev, "brier_score_1x2"),
        "mean_log_loss_1x2": _numeric_mean(baseline_ev, "log_loss_1x2"),
        "mean_rps_1x2": _numeric_mean(baseline_ev, "rps_1x2"),
        "mean_brier_btts": _numeric_mean(baseline_ev, "brier_score_btts"),
        "mean_log_loss_btts": _numeric_mean(baseline_ev, "log_loss_btts"),
        "modal_ev_divergence_count": modal_changed,
        "modal_ev_divergence_pct": modal_changed_fraction,
        "ev_picks_changed_vs_current_count": changed,
        "ev_picks_changed_vs_current_pct": changed_fraction,
    }


def summarise_calibration_weight_report(
    report: LiveBacktestReport,
    profiles: Sequence[CalibrationWeightProfile],
    *,
    include_market_consistent: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise a calibration-weight study report into combined and tournament rows."""

    predictions = report.predictions.copy()
    total_goals = report.total_goals_probability_diagnostics.copy()
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    current_config = _profile_config_name(profiles[0], "baseline_ev")
    combined_rows = [
        _profile_row(
            profile=profile,
            predictions=predictions,
            total_goals_rows=total_goals,
            scope="combined",
            tournament="combined",
            current_config=current_config,
            include_market_consistent=include_market_consistent,
        )
        for profile in profiles
    ]
    by_tournament_rows: list[dict[str, object]] = []
    tournaments = sorted(predictions["tournament"].dropna().astype(str).unique()) if "tournament" in predictions else []
    for tournament in tournaments:
        tournament_predictions = predictions[predictions["tournament"].astype(str).eq(tournament)]
        tournament_totals = (
            total_goals[total_goals["tournament"].astype(str).eq(tournament)]
            if not total_goals.empty and "tournament" in total_goals
            else pd.DataFrame()
        )
        for profile in profiles:
            by_tournament_rows.append(
                _profile_row(
                    profile=profile,
                    predictions=tournament_predictions,
                    total_goals_rows=tournament_totals,
                    scope="tournament",
                    tournament=tournament,
                    current_config=current_config,
                    include_market_consistent=include_market_consistent,
                )
            )
    return pd.DataFrame(combined_rows), pd.DataFrame(by_tournament_rows)


def add_search_rankings(summary: pd.DataFrame, by_tournament: pd.DataFrame) -> pd.DataFrame:
    """Add in-sample ranking and stability diagnostics to search output."""

    if summary.empty:
        return summary
    ranked = summary.copy()
    current_points = ranked.loc[ranked["profile"].eq("current"), "ev_default_points"]
    baseline_points = float(current_points.iloc[0]) if not current_points.empty else np.nan
    current_scores = ranked.loc[ranked["profile"].eq("current"), "mean_log_loss_1x2"]
    baseline_log_loss = float(current_scores.iloc[0]) if not current_scores.empty else np.nan
    tournament_points = by_tournament.pivot_table(
        index="profile",
        columns="tournament",
        values="ev_default_points",
        aggfunc="first",
    ) if not by_tournament.empty else pd.DataFrame()
    for column in tournament_points.columns:
        ranked[f"{column}_ev_default_points"] = ranked["profile"].map(tournament_points[column])
    tournament_point_columns = [
        column
        for column in ranked.columns
        if column.endswith("_ev_default_points") and column != "ev_default_points"
    ]
    if tournament_point_columns:
        ranked["minimum_tournament_points"] = ranked[tournament_point_columns].min(axis=1)
        ranked["tournament_points_range"] = (
            ranked[tournament_point_columns].max(axis=1)
            - ranked[tournament_point_columns].min(axis=1)
        )
        if "current" in tournament_points.index:
            current_by_tournament = tournament_points.loc["current"]
            improvements = []
            for _, row in ranked.iterrows():
                profile = str(row["profile"])
                if profile not in tournament_points.index:
                    improvements.append(False)
                    continue
                improvements.append(bool((tournament_points.loc[profile] >= current_by_tournament).all()))
            ranked["improves_or_matches_current_all_tournaments"] = improvements
        else:
            ranked["improves_or_matches_current_all_tournaments"] = False
    else:
        ranked["minimum_tournament_points"] = np.nan
        ranked["tournament_points_range"] = np.nan
        ranked["improves_or_matches_current_all_tournaments"] = False
    ranked["ev_points_delta_vs_current"] = ranked["ev_default_points"] - baseline_points
    ranked["log_loss_1x2_delta_vs_current"] = ranked["mean_log_loss_1x2"] - baseline_log_loss
    ranked["search_rank_by_points"] = (
        ranked["ev_default_points"].rank(method="min", ascending=False).astype("Int64")
    )
    ranked["search_rank_by_log_loss"] = (
        ranked["mean_log_loss_1x2"].rank(method="min", ascending=True).astype("Int64")
    )
    return ranked.sort_values(
        ["ev_default_points", "mean_log_loss_1x2", "tournament_points_range"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def run_calibration_weight_study(
    profiles: Sequence[CalibrationWeightProfile],
    *,
    tournament_folders: Sequence[str | Path] = DEFAULT_TOURNAMENT_FOLDERS,
    cache_folder: str | Path = Path("cache/calibration_weight_study"),
    include_market_consistent: bool = True,
    export_report: bool = False,
    progress: bool = True,
) -> tuple[LiveBacktestReport, pd.DataFrame, pd.DataFrame]:
    """Run profiles through the historical live pipeline and return summaries."""

    configs = build_profile_configs(profiles, include_market_consistent=include_market_consistent)
    report = run_combined_live_backtest(
        tournament_folders,
        cache_folder=cache_folder,
        configs=configs,
        export=export_report,
        export_csv_only=True,
        progress=progress,
    )
    summary, by_tournament = summarise_calibration_weight_report(
        report,
        profiles,
        include_market_consistent=include_market_consistent,
    )
    return report, summary, by_tournament


def write_calibration_weight_outputs(
    summary: pd.DataFrame,
    by_tournament: pd.DataFrame,
    *,
    summary_path: str | Path,
    by_tournament_path: str | Path | None = None,
) -> None:
    ensure_parent_directory(Path(summary_path))
    summary.to_csv(summary_path, index=False)
    if by_tournament_path is not None:
        ensure_parent_directory(Path(by_tournament_path))
        by_tournament.to_csv(by_tournament_path, index=False)
