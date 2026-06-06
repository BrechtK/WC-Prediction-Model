"""Full-pipeline historical backtest using the live OddsPortal paste workflow.

Input structure::

    input/historical/wc2022/
      results.csv               <- one row per match; see template
      odds/
        M001.txt                <- same combined-paste format as the live tool
        M002.txt                    (### MATCH, ### 1X2, ### OVER_UNDER, etc.)
        ...

The runner splits each paste with the live splitter, parses all markets
(1X2, O/U, BTTS, correct score, Asian handicap) exactly as in matchday mode,
and runs ``run_prediction_workflow`` with every configured ``ProjectConfig``.
It then scores each extracted strategy recommendation against the known result.

Because the same live pipeline is used end-to-end, all advanced features work
automatically: market-type-specific devig, Asian-handicap-informed
market-consistent matrices, Dixon-Coles or bivariate priors, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np
import pandas as pd

from wc_predictor.config import DevigConfig, ProjectConfig, StrategyConfig
from wc_predictor.oddsportal import parse_oddsportal_correct_score_folder
from wc_predictor.oddsportal_asian_handicap import parse_oddsportal_asian_handicap_folder
from wc_predictor.oddsportal_combined import split_combined_oddsportal_pastes
from wc_predictor.oddsportal_core import parse_oddsportal_core_odds_folder
from wc_predictor.scoring_rules import score_group_prediction
from wc_predictor.utils import ensure_parent_directory, goal_difference, result_sign

_DEFAULT_CACHE = Path("cache/live_backtest/split_pastes")
_DEFAULT_SUMMARY = Path("output/research/live_backtest_summary.csv")
_DEFAULT_PREDICTIONS = Path("output/research/live_backtest_predictions.csv")
_DEFAULT_EXCEL = Path("output/research/live_backtest.xlsx")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoricalResult:
    """One historical match: who played, when, and what the final score was."""

    match_id: str
    date: str
    stage: str
    group: str
    team_a: str
    team_b: str
    actual_score_a: int
    actual_score_b: int


@dataclass(frozen=True)
class BacktestConfigEntry:
    """One named configuration to evaluate in the backtest."""

    name: str
    config: ProjectConfig
    description: str = ""
    use_asian_handicap: bool = True
    use_correct_score: bool = True
    use_total_goals: bool = True


# Default configs cover the key questions:
#  1. Does market-consistent help vs pure EV?
#  2. Does AH information improve market-consistent?
#  3. Does a richer devig method matter?
#  4. Does a Dixon-Coles prior help?
DEFAULT_BACKTEST_CONFIGS: tuple[BacktestConfigEntry, ...] = (
    BacktestConfigEntry(
        name="baseline_ev",
        config=ProjectConfig(
            enable_market_consistent_challenger=False,
            enable_margin_method_comparison=False,
            enable_dixon_coles_rho_estimation=False,
        ),
        description="Pure EV, normalised inverse odds, no market-consistent challenger",
        use_asian_handicap=False,
    ),
    BacktestConfigEntry(
        name="power_devig_ev",
        config=ProjectConfig(
            devig=DevigConfig(
                one_x_two_method="power",
                btts_method="power",
                total_goals_method="power",
                asian_handicap_method="power",
                correct_score_method="normalised_inverse_odds",
            ),
            enable_market_consistent_challenger=False,
            enable_margin_method_comparison=False,
            enable_dixon_coles_rho_estimation=False,
        ),
        description="EV with power devig on 2-way markets; correct score stays conservative",
        use_asian_handicap=False,
    ),
    BacktestConfigEntry(
        name="mc_no_ah",
        config=ProjectConfig(
            enable_market_consistent_challenger=True,
            enable_margin_method_comparison=False,
        ),
        description="Market-consistent challenger (1X2 + O/U + BTTS + correct score, no AH)",
        use_asian_handicap=False,
    ),
    BacktestConfigEntry(
        name="mc_with_ah",
        config=ProjectConfig(
            enable_market_consistent_challenger=True,
            enable_margin_method_comparison=False,
        ),
        description="Market-consistent challenger with Asian-handicap constraints",
        use_asian_handicap=True,
    ),
    BacktestConfigEntry(
        name="mc_dc_prior_with_ah",
        config=ProjectConfig(
            market_consistent_prior="dixon_coles",
            enable_market_consistent_challenger=True,
            enable_margin_method_comparison=False,
        ),
        description="Market-consistent with Dixon-Coles prior and AH constraints",
        use_asian_handicap=True,
    ),
)

QUICK_BACKTEST_CONFIGS: tuple[BacktestConfigEntry, ...] = (
    DEFAULT_BACKTEST_CONFIGS[0],  # baseline_ev
    DEFAULT_BACKTEST_CONFIGS[3],  # mc_with_ah
)

CORRECT_SCORE_BLEND_SWEEP_CONFIGS: tuple[BacktestConfigEntry, ...] = tuple(
    BacktestConfigEntry(
        name=f"cs_weight_{str(weight).replace('.', '_')}",
        config=ProjectConfig(
            correct_score_poisson_weight=weight,
            enable_market_consistent_challenger=False,
            enable_margin_method_comparison=False,
            enable_dixon_coles_rho_estimation=False,
        ),
        description=f"Correct-score blend sweep: poisson weight {weight:g}",
        use_asian_handicap=False,
        use_correct_score=True,
    )
    for weight in (1.0, 0.85, 0.75, 0.50)
)

MODAL_DRAW_THRESHOLD_SWEEP_CONFIGS: tuple[BacktestConfigEntry, ...] = tuple(
    BacktestConfigEntry(
        name=(
            f"modal_draw_fav_{str(favourite_threshold).replace('.', '_')}"
            f"_gap_{str(draw_gap_threshold).replace('-', 'm').replace('.', '_')}"
        ),
        config=ProjectConfig(
            strategies=StrategyConfig(
                modal_draw_favourite_probability_threshold=favourite_threshold,
                modal_draw_gap_threshold=draw_gap_threshold,
            ),
            enable_market_consistent_challenger=False,
            enable_margin_method_comparison=False,
            enable_dixon_coles_rho_estimation=False,
        ),
        description=(
            "Modal/draw challenger threshold sweep: "
            f"fav<{favourite_threshold:g}, draw gap>{draw_gap_threshold:g}"
        ),
        use_asian_handicap=False,
    )
    for favourite_threshold in (0.40, 0.45, 0.50)
    for draw_gap_threshold in (-0.30, -0.50, -0.70, -1.00)
)

LARGER_GRID_SWEEP_CONFIGS: tuple[BacktestConfigEntry, ...] = (
    BacktestConfigEntry(
        name="larger_grid_15",
        config=ProjectConfig(
            max_goals_score_matrix=15,
            extreme_favourite_max_goals=15,
            research_max_goals=15,
            enable_market_consistent_challenger=False,
            enable_margin_method_comparison=False,
            enable_dixon_coles_rho_estimation=False,
        ),
        description="Larger score-matrix grid diagnostic for tail sensitivity",
        use_asian_handicap=False,
    ),
)

RESEARCH_BACKTEST_CONFIGS: tuple[BacktestConfigEntry, ...] = (
    *DEFAULT_BACKTEST_CONFIGS,
    *CORRECT_SCORE_BLEND_SWEEP_CONFIGS,
    *MODAL_DRAW_THRESHOLD_SWEEP_CONFIGS,
    *LARGER_GRID_SWEEP_CONFIGS,
)


@dataclass(frozen=True)
class LiveBacktestSettings:
    """Paths and configuration for one historical backtest run."""

    historical_odds_folder: Path
    results_path: Path
    cache_folder: Path = field(default_factory=lambda: _DEFAULT_CACHE)
    summary_output_path: Path = field(default_factory=lambda: _DEFAULT_SUMMARY)
    predictions_output_path: Path = field(default_factory=lambda: _DEFAULT_PREDICTIONS)
    excel_output_path: Path = field(default_factory=lambda: _DEFAULT_EXCEL)
    configs: tuple[BacktestConfigEntry, ...] = field(
        default_factory=lambda: DEFAULT_BACKTEST_CONFIGS
    )


@dataclass(frozen=True)
class LiveBacktestReport:
    """Summary, per-match predictions, and skip diagnostics."""

    summary: pd.DataFrame
    predictions: pd.DataFrame
    skipped: pd.DataFrame
    round_summary: pd.DataFrame

    def export(self, settings: LiveBacktestSettings) -> None:
        for path in (
            settings.summary_output_path,
            settings.predictions_output_path,
            settings.excel_output_path,
        ):
            ensure_parent_directory(path)
        self.summary.to_csv(settings.summary_output_path, index=False)
        self.predictions.to_csv(settings.predictions_output_path, index=False)
        with pd.ExcelWriter(settings.excel_output_path, engine="openpyxl") as writer:
            self.summary.to_excel(writer, index=False, sheet_name="summary")
            self.predictions.to_excel(writer, index=False, sheet_name="predictions")
            _sheet_or_status(
                self.round_summary,
                reason="No group-stage round rows were produced.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="group_stage_playing_round != ''",
            ).to_excel(writer, index=False, sheet_name="matchday_performance")
            _sheet_or_status(
                _build_ev_vs_modal_attribution(self.predictions),
                reason="EV and modal strategy rows were unavailable.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="strategy in {ev_default, most_likely}",
            ).to_excel(
                writer, index=False, sheet_name="ev_vs_modal_attribution"
            )
            _sheet_or_status(
                _filter_flagged_ev_rows(self.predictions, "modal_draw_challenger_flag"),
                reason="No EV-default rows matched modal_draw_challenger_flag = yes.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="strategy=ev_default; modal_draw_challenger_flag=yes",
            ).to_excel(
                writer, index=False, sheet_name="modal_draw_challenger_matches"
            )
            _sheet_or_status(
                _filter_flagged_ev_rows(self.predictions, "btts_conflict_flag"),
                reason="No EV-default rows matched btts_conflict_flag = yes.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="strategy=ev_default; btts_conflict_flag=yes",
            ).to_excel(
                writer, index=False, sheet_name="btts_conflict_matches"
            )
            _sheet_or_status(
                _filter_flagged_ev_rows(self.predictions, "draw_prone_flag"),
                reason="No EV-default rows matched draw_prone_flag = yes.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="strategy=ev_default; draw_prone_flag=yes",
            ).to_excel(
                writer, index=False, sheet_name="draw_prone_matches"
            )
            _sheet_or_status(
                _build_draw_prone_candidates(self.predictions),
                reason="No EV-default rows were available for draw-prone nearest-miss analysis.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="strategy=ev_default; draw_prone_flag!=yes",
            ).to_excel(writer, index=False, sheet_name="draw_prone_candidates")
            _sheet_or_status(
                _filter_flagged_ev_rows(self.predictions, "blowout_risk_flag"),
                reason="No EV-default rows matched blowout_risk_flag = yes.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="strategy=ev_default; blowout_risk_flag=yes",
            ).to_excel(
                writer, index=False, sheet_name="blowout_risk_matches"
            )
            cs_summary, cs_matches = _build_correct_score_blend_sweep(self.predictions)
            _sheet_or_status(
                cs_summary,
                reason="No cs_weight_* configs were run. Use BACKTEST_PROFILE = 'research'.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="config starts with cs_weight_; strategy=ev_default",
            ).to_excel(writer, index=False, sheet_name="blend_weight_sweep")
            _sheet_or_status(
                cs_summary,
                reason="No cs_weight_* configs were run. Use BACKTEST_PROFILE = 'research'.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="config starts with cs_weight_; strategy=ev_default",
            ).to_excel(writer, index=False, sheet_name="correct_score_blend_sweep")
            _sheet_or_status(
                cs_matches,
                reason="No cs_weight_* match rows were produced. Use BACKTEST_PROFILE = 'research'.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="config starts with cs_weight_; strategy=ev_default",
            ).to_excel(writer, index=False, sheet_name="cs_blend_matches")
            grid_summary, grid_matches = _build_larger_grid_sweep(self.predictions)
            _sheet_or_status(
                grid_summary,
                reason="No larger_grid_* configs were run. Use BACKTEST_PROFILE = 'research'.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="baseline_ev and larger_grid_*; strategy=ev_default",
            ).to_excel(writer, index=False, sheet_name="larger_grid_sweep")
            _sheet_or_status(
                grid_matches,
                reason="No larger_grid_* match rows were produced. Use BACKTEST_PROFILE = 'research'.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="baseline_ev and larger_grid_*; strategy=ev_default",
            ).to_excel(writer, index=False, sheet_name="larger_grid_sweep_matches")
            draw_summary, draw_matches = _build_draw_threshold_sweep(self.predictions)
            _sheet_or_status(
                draw_summary,
                reason="No baseline EV rows were available for draw-threshold sweep.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="config=baseline_ev; strategy=ev_default",
            ).to_excel(writer, index=False, sheet_name="draw_threshold_sweep")
            _sheet_or_status(
                draw_matches,
                reason="No draw-threshold configurations flagged a match.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="baseline_ev rows; draw/modal challenger conditions",
            ).to_excel(writer, index=False, sheet_name="draw_threshold_sweep_matches")
            _sheet_or_status(
                _build_pattern_flags_summary(self.predictions),
                reason="No pattern flag summary rows could be produced.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="strategy=ev_default; known pattern flag columns",
            ).to_excel(writer, index=False, sheet_name="pattern_flags_summary")
            _sheet_or_status(
                _build_grouped_performance(self.predictions, ["config", "strategy", "favourite_bucket"]),
                reason="Favourite-bucket columns were unavailable.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="group by config,strategy,favourite_bucket",
            ).to_excel(writer, index=False, sheet_name="favourite_bucket_performance")
            _sheet_or_status(
                _build_grouped_performance(self.predictions, ["config", "strategy", "manual_review_flag"]),
                reason="Manual-review columns were unavailable.",
                matches_evaluated=_matches_evaluated(self.predictions),
                filters_applied="group by config,strategy,manual_review_flag",
            ).to_excel(writer, index=False, sheet_name="manual_review_performance")
            if not self.skipped.empty:
                self.skipped.to_excel(writer, index=False, sheet_name="skipped")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_historical_results(path: str | Path) -> list[HistoricalResult]:
    """Load match results from the project template CSV format.

    Required columns: ``match_id``, ``team_a``, ``team_b``,
    ``actual_score_a``, ``actual_score_b``.

    Optional columns (defaults applied when absent): ``date``, ``stage``,
    ``group``.
    """
    path = Path(path)
    source = pd.read_csv(path, dtype=str)
    required = {"match_id", "team_a", "team_b", "actual_score_a", "actual_score_b"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(
            f"Historical results CSV is missing required columns: {sorted(missing)}. "
            f"See templates/historical_results_template.csv."
        )
    results: list[HistoricalResult] = []
    for _, row in source.iterrows():
        try:
            score_a = int(row["actual_score_a"])
            score_b = int(row["actual_score_b"])
        except (ValueError, TypeError):
            continue
        results.append(
            HistoricalResult(
                match_id=str(row["match_id"]).strip(),
                date=str(row.get("date", "")).strip(),
                stage=str(row.get("stage", "group stage")).strip(),
                group=str(row.get("group", "")).strip(),
                team_a=str(row["team_a"]).strip(),
                team_b=str(row["team_b"]).strip(),
                actual_score_a=score_a,
                actual_score_b=score_b,
            )
        )
    return results


def _inject_match_metadata(
    odds: pd.DataFrame,
    results: list[HistoricalResult],
) -> pd.DataFrame:
    """Overwrite metadata columns from results so run_prediction_workflow gets
    date, stage, team names even when no live schedule is present."""
    if odds.empty:
        return odds
    meta = {
        r.match_id: {
            "date": r.date,
            "stage": r.stage,
            "group": r.group,
            "team_a": r.team_a,
            "team_b": r.team_b,
        }
        for r in results
    }
    odds = odds.copy()
    for col in ("date", "stage", "group", "team_a", "team_b"):
        if col not in odds.columns:
            odds[col] = ""
    for match_id, row_meta in meta.items():
        mask = odds["match_id"].astype(str) == match_id
        if not mask.any():
            continue
        for col, value in row_meta.items():
            odds.loc[mask, col] = value
    return odds


def _classify_missing_odds(odds_folder: Path, match_id: str) -> str:
    """Explain why a match produced no parseable core odds.

    Distinguishes a genuinely absent paste file, an unedited template (still
    containing the ``<paste ...>`` placeholder), and a filled-in file whose
    1X2 section could not be parsed.
    """
    paste_path = odds_folder / f"{match_id}.txt"
    if not paste_path.exists():
        return "no_paste_file"
    try:
        text = paste_path.read_text(encoding="utf-8-sig")
    except OSError:
        return "paste_file_unreadable"
    if "<paste" in text:
        return "paste_file_not_filled_in_yet"
    return "no_valid_1x2_odds_parsed"


def _parse_score(value: object) -> tuple[int, int] | None:
    """Parse a formatted score label like '2-1' into a (goals_a, goals_b) tuple."""
    if not value or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if "-" not in text:
        return None
    parts = text.split("-", 1)
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def _match_number(match_id: object) -> int | None:
    """Extract the numeric tournament match number from labels like M001."""

    match = re.search(r"(\d+)", str(match_id))
    if match is None:
        return None
    return int(match.group(1))


def _group_stage_playing_round(match_id: object, stage: object) -> str:
    """Map World Cup group-stage match numbers to playing rounds."""

    if "group" not in str(stage).lower():
        return ""
    number = _match_number(match_id)
    if number is None:
        return ""
    if 1 <= number <= 16:
        return "round_1"
    if 17 <= number <= 32:
        return "round_2"
    if 33 <= number <= 48:
        return "round_3"
    return ""


def _round_match_range(playing_round: str) -> str:
    ranges = {
        "round_1": "M001-M016",
        "round_2": "M017-M032",
        "round_3": "M033-M048",
    }
    return ranges.get(playing_round, "")


def _score_strategies(
    report_row: pd.Series,
    actual_a: int,
    actual_b: int,
) -> list[dict[str, object]]:
    """Extract every candidate strategy score from one workflow report row and
    score it against the actual result."""

    mc_ran = str(report_row.get("market_consistent_status", "skipped")).lower() not in {
        "skipped", ""
    }
    ah_lines_used = str(report_row.get("market_consistent_asian_handicap_lines_used", "") or "")

    strategy_columns = {
        "ev_default": "recommended_score",
        "baseline_poisson": "baseline_poisson_recommended_score",
        "most_likely": "most_likely_scoreline",
        "dixon_coles": "dixon_coles_recommended_score",
        "market_consistent": "market_consistent_recommended_score",
    }
    expected_point_columns = {
        "ev_default": "best_expected_points",
        "baseline_poisson": "baseline_poisson_best_expected_points",
        "most_likely": "most_likely_expected_points",
        "dixon_coles": "dixon_coles_best_expected_points",
        "market_consistent": "market_consistent_best_expected_points",
    }
    diagnostic_columns = (
        "market_a_win",
        "market_draw",
        "market_b_win",
        "favourite_team",
        "lambda_a",
        "lambda_b",
        "expected_total_goals",
        "ou_median_total",
        "market_total_line_used",
        "market_fair_btts_yes_probability",
        "model_implied_btts_yes_probability",
        "btts_probability",
        "ev_default_score",
        "most_likely_score",
        "best_draw_score",
        "best_draw_ev",
        "best_decisive_score",
        "best_decisive_ev",
        "draw_vs_decisive_gap",
        "ev_vs_modal_differs",
        "ev_expected_points",
        "modal_expected_points",
        "ev_minus_modal_expected_gap",
        "manual_review_flag",
        "confidence_level",
        "modal_draw_challenger_flag",
        "modal_draw_challenger_score",
        "modal_draw_challenger_reason",
        "ev_score",
        "modal_score",
        "btts_conflict_flag",
        "btts_conflict_reason",
        "btts_conflict_note",
        "best_btts_alternative_score",
        "ev_gap_to_btts_alternative",
        "ev_gap_to_best_btts_alternative",
        "draw_prone_flag",
        "draw_prone_reason",
        "market_draw_probability",
        "blowout_risk_flag",
        "blowout_risk_reason",
        "best_high_margin_alternatives",
        "high_score_tail_mass",
        "larger_grid_recommended",
        "normal_grid_tail_mass",
        "larger_grid_tail_mass",
        "tail_probability_before_renormalisation",
        "tail_mass_before_grid_extension",
        "tail_mass_after_grid_extension",
        "recommendation_changed_due_to_larger_grid",
        "ah_implied_favourite_margin",
        "favourite_covered_ah",
        "correct_score_top_scores",
        "has_other_bucket",
        "correct_score_tail_mass",
        "market_consistent_status",
        "market_consistent_optimisation_classification",
        "market_consistent_1x2_fit_error",
        "market_consistent_btts_fit_error",
        "market_consistent_total_goals_fit_error",
        "market_consistent_correct_score_fit_error",
        "market_consistent_warning_flags",
        "correct_score_poisson_weight",
        "margin_removal_method",
        "grid_max_goals_used",
        "decision_note",
        "risk_notes",
    )
    shared_diagnostics = {
        column: report_row.get(column, np.nan)
        for column in diagnostic_columns
        if column in report_row
    }

    rows: list[dict[str, object]] = []
    for strategy_name, column in strategy_columns.items():
        if strategy_name == "market_consistent" and not mc_ran:
            continue
        score = _parse_score(report_row.get(column))
        if score is None:
            continue
        pred_a, pred_b = score
        points = score_group_prediction(pred_a, pred_b, actual_a, actual_b)
        rows.append(
            {
                "strategy": strategy_name,
                "predicted_score": f"{pred_a}-{pred_b}",
                "actual_score": f"{actual_a}-{actual_b}",
                "realised_points": points,
                "model_expected_points": report_row.get(expected_point_columns[strategy_name], np.nan),
                "is_exact_score": bool((pred_a, pred_b) == (actual_a, actual_b)),
                "is_correct_goal_difference": bool(
                    goal_difference(pred_a, pred_b) == goal_difference(actual_a, actual_b)
                ),
                "is_correct_result": bool(
                    result_sign(pred_a, pred_b) == result_sign(actual_a, actual_b)
                ),
                "mc_ran": mc_ran,
                "mc_status": str(report_row.get("market_consistent_status", "")),
                "ah_lines_used": ah_lines_used,
                "favourite_probability": report_row.get("favourite_probability", np.nan),
                "favourite_bucket": str(report_row.get("favourite_bucket", "")),
                "ev_gap_to_second": report_row.get("ev_gap_best_vs_second", np.nan),
                "calibration_loss": report_row.get("calibration_loss", np.nan),
                "warning_flags": str(report_row.get("warning_flags", "")),
                **shared_diagnostics,
            }
        )
    return rows


def _build_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-match predictions into config x strategy summary rows."""
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (config_name, strategy), group in predictions.groupby(["config", "strategy"]):
        expected_points = (
            pd.to_numeric(group["model_expected_points"], errors="coerce")
            if "model_expected_points" in group
            else pd.Series(dtype=float)
        )
        total_expected_points = (
            float(expected_points.sum())
            if expected_points.notna().any()
            else np.nan
        )
        average_expected_points = (
            float(expected_points.mean())
            if expected_points.notna().any()
            else np.nan
        )
        actual_points_sum = int(group["realised_points"].sum())
        rows.append(
            {
                "config": config_name,
                "strategy": strategy,
                "matches_used": len(group),
                "total_expected_points": total_expected_points,
                "average_expected_points": average_expected_points,
                "actual_points_sum": actual_points_sum,
                "total_points": actual_points_sum,
                "average_realised_points": float(group["realised_points"].mean()),
                "exact_score_rate": float(group["is_exact_score"].mean()),
                "correct_goal_difference_rate": float(
                    group["is_correct_goal_difference"].mean()
                ),
                "correct_result_rate": float(group["is_correct_result"].mean()),
                "points_variance": float(group["realised_points"].var()),
            }
        )
    summary = (
        pd.DataFrame(rows)
        .sort_values("average_realised_points", ascending=False)
        .reset_index(drop=True)
    )
    summary["rank"] = (
        summary["average_realised_points"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return summary


def _build_round_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate group-stage realised and expected points by playing round."""

    if predictions.empty or "group_stage_playing_round" not in predictions:
        return pd.DataFrame()
    round_predictions = predictions[predictions["group_stage_playing_round"].astype(str).ne("")].copy()
    if round_predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (config_name, strategy, playing_round), group in round_predictions.groupby(
        ["config", "strategy", "group_stage_playing_round"],
        sort=True,
    ):
        expected_points = (
            pd.to_numeric(group["model_expected_points"], errors="coerce")
            if "model_expected_points" in group
            else pd.Series(dtype=float)
        )
        total_expected_points = float(expected_points.sum()) if expected_points.notna().any() else np.nan
        average_expected_points = float(expected_points.mean()) if expected_points.notna().any() else np.nan
        actual_points_sum = int(group["realised_points"].sum())
        rows.append(
            {
                "config": config_name,
                "strategy": strategy,
                "group_stage_playing_round": playing_round,
                "round_match_range": _round_match_range(str(playing_round)),
                "matches_used": len(group),
                "total_expected_points": total_expected_points,
                "average_expected_points": average_expected_points,
                "actual_points_sum": actual_points_sum,
                "average_realised_points": float(group["realised_points"].mean()),
                "exact_score_rate": float(group["is_exact_score"].mean()),
                "correct_goal_difference_rate": float(group["is_correct_goal_difference"].mean()),
                "correct_result_rate": float(group["is_correct_result"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["config", "strategy", "group_stage_playing_round"],
        kind="stable",
    ).reset_index(drop=True)


def _matches_evaluated(predictions: pd.DataFrame) -> int:
    return int(predictions["match_id"].nunique()) if not predictions.empty and "match_id" in predictions else 0


def _status_frame(reason: str, *, matches_evaluated: int, filters_applied: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "status": "no_rows",
                "reason": reason,
                "matches_evaluated": matches_evaluated,
                "filters_applied": filters_applied,
            }
        ]
    )


def _sheet_or_status(
    frame: pd.DataFrame,
    *,
    reason: str,
    matches_evaluated: int,
    filters_applied: str,
) -> pd.DataFrame:
    if frame is not None and not frame.empty:
        return frame
    return _status_frame(reason, matches_evaluated=matches_evaluated, filters_applied=filters_applied)


def _ev_default_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "strategy" not in predictions:
        return pd.DataFrame()
    return predictions[predictions["strategy"].astype(str).eq("ev_default")].copy()


def _count_hits(group: pd.DataFrame) -> dict[str, int]:
    exact = int(group["is_exact_score"].fillna(False).astype(bool).sum()) if "is_exact_score" in group else 0
    gd = (
        int((group["is_correct_goal_difference"].fillna(False).astype(bool) & ~group["is_exact_score"].fillna(False).astype(bool)).sum())
        if {"is_correct_goal_difference", "is_exact_score"}.issubset(group.columns)
        else 0
    )
    result = (
        int(
            (
                group["is_correct_result"].fillna(False).astype(bool)
                & ~group["is_correct_goal_difference"].fillna(False).astype(bool)
                & ~group["is_exact_score"].fillna(False).astype(bool)
            ).sum()
        )
        if {"is_correct_result", "is_correct_goal_difference", "is_exact_score"}.issubset(group.columns)
        else 0
    )
    misses = len(group) - exact - gd - result
    return {
        "exact_hits": exact,
        "goal_difference_hits": gd,
        "GD_hits": gd,
        "result_hits": result,
        "misses": misses,
    }


def _performance_row(group: pd.DataFrame, extra: dict[str, object]) -> dict[str, object]:
    expected_points = pd.to_numeric(group.get("model_expected_points", pd.Series(dtype=float)), errors="coerce")
    total_expected_points = float(expected_points.sum()) if expected_points.notna().any() else np.nan
    return {
        **extra,
        "matches_used": len(group),
        "total_expected_points": total_expected_points,
        "average_expected_points": float(expected_points.mean()) if expected_points.notna().any() else np.nan,
        "actual_points_sum": int(group["realised_points"].sum()) if "realised_points" in group else 0,
        "average_realised_points": float(group["realised_points"].mean()) if "realised_points" in group else np.nan,
        "exact_score_rate": float(group["is_exact_score"].mean()) if "is_exact_score" in group else np.nan,
        "correct_goal_difference_rate": (
            float(group["is_correct_goal_difference"].mean()) if "is_correct_goal_difference" in group else np.nan
        ),
        "correct_result_rate": float(group["is_correct_result"].mean()) if "is_correct_result" in group else np.nan,
    }


def _build_grouped_performance(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if predictions.empty or any(column not in predictions for column in group_columns):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(_performance_row(group, dict(zip(group_columns, keys, strict=True))))
    return pd.DataFrame(rows).sort_values(group_columns, kind="stable").reset_index(drop=True)


def _build_named_config_summary(predictions: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    if predictions.empty or "config" not in predictions:
        return pd.DataFrame()
    subset = predictions[predictions["config"].astype(str).str.startswith(prefix)]
    return _build_grouped_performance(subset, ["config", "strategy"]) if not subset.empty else pd.DataFrame()


def _weight_from_config(config_name: object) -> float | object:
    text = str(config_name)
    match = re.search(r"cs_weight_(\d+)_(\d+)", text)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return pd.NA


def _build_correct_score_blend_sweep(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = _ev_default_rows(predictions)
    if ev.empty or "config" not in ev:
        return pd.DataFrame(), pd.DataFrame()
    sweep = ev[ev["config"].astype(str).str.startswith("cs_weight_")].copy()
    if sweep.empty:
        return pd.DataFrame(), pd.DataFrame()
    baseline = sweep[sweep["config"].astype(str).eq("cs_weight_1_0")].set_index("match_id")
    rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    for config_name, group in sweep.groupby("config", sort=True):
        group = group.copy()
        weight = _weight_from_config(config_name)
        changed = 0
        ev_costs: list[float] = []
        missing_cs = 0
        for _, row in group.iterrows():
            base_row = baseline.loc[row["match_id"]] if row["match_id"] in baseline.index else pd.Series(dtype=object)
            default_score = str(base_row.get("predicted_score", row.get("predicted_score", "")))
            changed_vs_w1 = str(row.get("predicted_score", "")) != default_score
            changed += int(changed_vs_w1)
            base_expected = pd.to_numeric(base_row.get("model_expected_points", np.nan), errors="coerce")
            current_expected = pd.to_numeric(row.get("model_expected_points", np.nan), errors="coerce")
            if pd.notna(base_expected) and pd.notna(current_expected):
                ev_costs.append(float(base_expected - current_expected))
            top_scores = str(row.get("correct_score_top_scores", "") or "")
            has_cs = bool(top_scores.strip())
            missing_cs += 0 if has_cs else 1
            match_rows.append(
                {
                    "match_id": row.get("match_id", ""),
                    "team_a": row.get("team_a", ""),
                    "team_b": row.get("team_b", ""),
                    "actual_score": row.get("actual_score", ""),
                    "weight": weight,
                    "correct_score_poisson_weight": weight,
                    "predicted_score": row.get("predicted_score", ""),
                    "points": row.get("realised_points", np.nan),
                    "expected_points": row.get("model_expected_points", np.nan),
                    "default_w_1_score": default_score,
                    "changed_vs_w_1": "yes" if changed_vs_w1 else "no",
                    "market_correct_score_top_scores": top_scores,
                    "has_other_bucket": row.get("has_other_bucket", ""),
                    "correct_score_tail_mass": row.get("correct_score_tail_mass", np.nan),
                    "notes_warnings": (
                        "correct_score_data_missing_or_unparsed" if not has_cs else str(row.get("warning_flags", ""))
                    ),
                }
            )
        expected_points = pd.to_numeric(group["model_expected_points"], errors="coerce")
        hits = _count_hits(group)
        rows.append(
            {
                "correct_score_poisson_weight": weight,
                "strategy_config_name": config_name,
                "config": config_name,
                "strategy": "ev_default",
                "n_matches": len(group),
                "total_points": int(group["realised_points"].sum()),
                "average_points": float(group["realised_points"].mean()),
                "expected_total_points": float(expected_points.sum()) if expected_points.notna().any() else np.nan,
                "expected_average_points": float(expected_points.mean()) if expected_points.notna().any() else np.nan,
                **hits,
                "changed_predictions_vs_w_1_0": changed,
                "average_ev_cost_vs_w_1_0": float(np.mean(ev_costs)) if ev_costs else np.nan,
                "notes_warnings": (
                    f"{missing_cs} matches missing correct-score data"
                    if missing_cs
                    else "correct-score data available for all rows"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("correct_score_poisson_weight", ascending=False), pd.DataFrame(match_rows)


def _build_larger_grid_sweep(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = _ev_default_rows(predictions)
    if ev.empty or "config" not in ev:
        return pd.DataFrame(), pd.DataFrame()
    baseline = ev[ev["config"].astype(str).eq("baseline_ev")].copy()
    larger = ev[ev["config"].astype(str).str.startswith("larger_grid_")].copy()
    if baseline.empty or larger.empty:
        return pd.DataFrame(), pd.DataFrame()
    tail_columns_available = any(
        column in ev.columns
        for column in (
            "normal_grid_tail_mass",
            "tail_probability_before_renormalisation",
            "tail_mass_before_grid_extension",
        )
    )
    baseline_by_match = baseline.set_index("match_id")
    summary_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []

    def add_summary(grid_config: str, group: pd.DataFrame, *, changed: int, high_tail: int, recommendation_changed: int) -> None:
        expected = pd.to_numeric(group["model_expected_points"], errors="coerce")
        summary_rows.append(
            {
                "grid_config": grid_config,
                "n_matches": len(group),
                "total_points": int(group["realised_points"].sum()),
                "average_points": float(group["realised_points"].mean()),
                "expected_total_points": float(expected.sum()) if expected.notna().any() else np.nan,
                **_count_hits(group),
                "changed_predictions_vs_default_grid": changed,
                "high_tail_mass_matches": high_tail,
                "recommendation_changed_due_to_larger_grid": recommendation_changed,
                "larger_grid_diagnostics_status": (
                    "ok" if tail_columns_available else "tail_diagnostics_missing"
                ),
                "notes": (
                    ""
                    if tail_columns_available
                    else "larger-grid score comparison available; tail diagnostics missing from predictions"
                ),
            }
        )

    default_tail_series = _first_existing_numeric_series(
        baseline,
        ("normal_grid_tail_mass", "tail_probability_before_renormalisation", "tail_mass_before_grid_extension"),
        0.0,
    )
    high_tail_default = int(default_tail_series.gt(0.01).sum())
    add_summary("default_grid", baseline, changed=0, high_tail=high_tail_default, recommendation_changed=0)
    for config_name, group in larger.groupby("config", sort=True):
        changed = 0
        high_tail = 0
        for _, row in group.iterrows():
            if row["match_id"] not in baseline_by_match.index:
                continue
            base = baseline_by_match.loc[row["match_id"]]
            default_score = str(base.get("predicted_score", ""))
            larger_score = str(row.get("predicted_score", ""))
            changed_flag = default_score != larger_score
            changed += int(changed_flag)
            tail_default = _safe_float(
                base.get(
                    "normal_grid_tail_mass",
                    base.get("tail_probability_before_renormalisation", base.get("tail_mass_before_grid_extension", np.nan)),
                )
            )
            tail_larger = _safe_float(
                row.get(
                    "normal_grid_tail_mass",
                    row.get("tail_probability_before_renormalisation", row.get("tail_mass_after_grid_extension", np.nan)),
                )
            )
            high_tail += int(pd.notna(tail_default) and float(tail_default) > 0.01)
            match_rows.append(
                {
                    "match_id": row.get("match_id", ""),
                    "teams": f"{row.get('team_a', '')} vs {row.get('team_b', '')}",
                    "team_a": row.get("team_a", ""),
                    "team_b": row.get("team_b", ""),
                    "actual_score": row.get("actual_score", ""),
                    "favourite_probability": row.get("favourite_probability", np.nan),
                    "expected_total_goals": row.get("expected_total_goals", np.nan),
                    "high_tail_mass_flag": "yes" if pd.notna(tail_default) and float(tail_default) > 0.01 else "no",
                    "default_grid_score": default_score,
                    "larger_grid_score": larger_score,
                    "points_default": base.get("realised_points", np.nan),
                    "points_larger_grid": row.get("realised_points", np.nan),
                    "changed": "yes" if changed_flag else "no",
                    "tail_mass_default": tail_default,
                    "tail_mass_larger_grid": tail_larger,
                    "grid_config": config_name,
                    "larger_grid_diagnostics_status": (
                        "ok" if tail_columns_available else "tail_diagnostics_missing"
                    ),
                }
            )
        add_summary(
            config_name,
            group,
            changed=changed,
            high_tail=high_tail,
            recommendation_changed=changed,
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(match_rows)


def _safe_float(value: object) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    return float(parsed) if pd.notna(parsed) else None


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _first_existing_numeric_series(frame: pd.DataFrame, columns: tuple[str, ...], default: float = 0.0) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return _numeric_series(frame, column, default)
    return pd.Series(default, index=frame.index, dtype=float)


def _draw_threshold_trigger(row: pd.Series, fav_threshold: float, gap_threshold: float, ou_threshold: float) -> bool:
    fav = _safe_float(row.get("favourite_probability"))
    gap = _safe_float(row.get("draw_vs_decisive_gap"))
    ou = _safe_float(row.get("ou_median_total"))
    default_score = _parse_score(row.get("predicted_score"))
    challenger_score = _parse_score(row.get("best_draw_score")) or _parse_score(row.get("modal_score"))
    return (
        fav is not None
        and gap is not None
        and ou is not None
        and default_score is not None
        and challenger_score is not None
        and default_score[0] != default_score[1]
        and challenger_score[0] == challenger_score[1]
        and fav < fav_threshold
        and gap > gap_threshold
        and ou <= ou_threshold
    )


def _build_draw_threshold_sweep(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = _ev_default_rows(predictions)
    baseline = ev[ev["config"].astype(str).eq("baseline_ev")].copy() if not ev.empty and "config" in ev else pd.DataFrame()
    if baseline.empty:
        return pd.DataFrame(), pd.DataFrame()
    fav_thresholds = (0.40, 0.45, 0.50)
    gap_thresholds = (-0.30, -0.50, -0.70, -1.00)
    ou_thresholds = (2.0, 2.25, 2.5)
    summary_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    for fav_threshold in fav_thresholds:
        for gap_threshold in gap_thresholds:
            for ou_threshold in ou_thresholds:
                threshold_config = (
                    f"fav<{fav_threshold:g}; draw_gap>{gap_threshold:g}; ou<={ou_threshold:g}"
                )
                flagged_rows: list[pd.Series] = [
                    row
                    for _, row in baseline.iterrows()
                    if _draw_threshold_trigger(row, fav_threshold, gap_threshold, ou_threshold)
                ]
                default_points_total = 0
                challenger_points_total = 0
                ev_costs: list[float] = []
                pseudo_rows: list[dict[str, object]] = []
                for row in flagged_rows:
                    challenger_score = str(row.get("best_draw_score") or row.get("modal_score") or "")
                    points_default = int(row.get("realised_points", 0))
                    points_challenger = _points_for_score_label(challenger_score, row.get("actual_score"))
                    ev_default = _safe_float(row.get("ev_expected_points")) or _safe_float(row.get("model_expected_points"))
                    ev_challenger = _safe_float(row.get("best_draw_ev")) or _safe_float(row.get("modal_expected_points"))
                    ev_cost = (
                        float(ev_default - ev_challenger)
                        if ev_default is not None and ev_challenger is not None
                        else np.nan
                    )
                    if pd.notna(ev_cost):
                        ev_costs.append(float(ev_cost))
                    default_points_total += points_default
                    challenger_points_total += int(points_challenger) if pd.notna(points_challenger) else 0
                    parsed_challenger = _parse_score(challenger_score)
                    actual = _parse_score(row.get("actual_score"))
                    pseudo = {
                        "is_exact_score": bool(parsed_challenger == actual) if parsed_challenger and actual else False,
                        "is_correct_goal_difference": (
                            bool(goal_difference(*parsed_challenger) == goal_difference(*actual))
                            if parsed_challenger and actual
                            else False
                        ),
                        "is_correct_result": (
                            bool(result_sign(*parsed_challenger) == result_sign(*actual))
                            if parsed_challenger and actual
                            else False
                        ),
                    }
                    pseudo_rows.append(pseudo)
                    match_rows.append(
                        {
                            "match_id": row.get("match_id", ""),
                            "teams": f"{row.get('team_a', '')} vs {row.get('team_b', '')}",
                            "team_a": row.get("team_a", ""),
                            "team_b": row.get("team_b", ""),
                            "actual_score": row.get("actual_score", ""),
                            "default_score": row.get("predicted_score", ""),
                            "challenger_score": challenger_score,
                            "threshold_config": threshold_config,
                            "points_default": points_default,
                            "points_challenger": points_challenger,
                            "ev_default": ev_default,
                            "ev_challenger": ev_challenger,
                            "ev_cost": ev_cost,
                            "draw_vs_decisive_gap": row.get("draw_vs_decisive_gap", np.nan),
                            "ou_median_total": row.get("ou_median_total", np.nan),
                            "favourite_probability": row.get("favourite_probability", np.nan),
                        }
                    )
                pseudo_frame = pd.DataFrame(pseudo_rows)
                hits = _count_hits(pseudo_frame) if not pseudo_frame.empty else {
                    "exact_hits": 0,
                    "goal_difference_hits": 0,
                    "GD_hits": 0,
                    "result_hits": 0,
                    "misses": 0,
                }
                summary_rows.append(
                    {
                        "favourite_threshold": fav_threshold,
                        "draw_gap_threshold": gap_threshold,
                        "ou_median_threshold": ou_threshold,
                        "n_flagged": len(flagged_rows),
                        "total_points_if_override": challenger_points_total,
                        "total_points_default_on_same_matches": default_points_total,
                        "points_gain": challenger_points_total - default_points_total,
                        "average_ev_cost": float(np.mean(ev_costs)) if ev_costs else np.nan,
                        **hits,
                        "overfitting_warning": "research_only_small_sample_do_not_promote_live_default",
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(match_rows)


def _filter_flagged_ev_rows(predictions: pd.DataFrame, flag_column: str) -> pd.DataFrame:
    if predictions.empty or flag_column not in predictions:
        return pd.DataFrame()
    rows = predictions[
        predictions["strategy"].astype(str).eq("ev_default")
        & predictions[flag_column].astype(str).str.lower().eq("yes")
    ].copy()
    if rows.empty:
        return rows.reset_index(drop=True)
    rows["default_score"] = rows.get("ev_score", rows.get("predicted_score", ""))
    rows["default_points"] = rows.get("realised_points", np.nan)
    rows["relevant_alternative"] = rows.apply(
        lambda row: _relevant_alternative_for_flag(row, flag_column),
        axis=1,
    )
    rows["alternative_points"] = rows.apply(
        lambda row: _points_for_score_label(row.get("relevant_alternative"), row.get("actual_score")),
        axis=1,
    )
    rows["caution_label"] = _caution_label(flag_column)
    return rows.reset_index(drop=True)


def _first_score_from_alternative_list(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split(";", maxsplit=1)[0].strip().split(" ", maxsplit=1)[0]


def _relevant_alternative_for_flag(row: pd.Series, flag_column: str) -> str:
    if flag_column in {"draw_prone_flag", "modal_draw_challenger_flag"}:
        return str(row.get("best_draw_score") or row.get("modal_score") or "")
    if flag_column == "btts_conflict_flag":
        return str(row.get("best_btts_alternative_score") or "")
    if flag_column == "blowout_risk_flag":
        return _first_score_from_alternative_list(row.get("best_high_margin_alternatives"))
    return ""


def _points_for_score_label(score_label: object, actual_score_label: object) -> object:
    score = _parse_score(score_label)
    actual = _parse_score(actual_score_label)
    if score is None or actual is None:
        return np.nan
    return score_group_prediction(score[0], score[1], actual[0], actual[1])


def _caution_label(flag_column: str) -> str:
    labels = {
        "draw_prone_flag": "low_total_balanced_draw_review",
        "modal_draw_challenger_flag": "modal_draw_manual_review",
        "btts_conflict_flag": "narrow_btts_conflict_review",
        "blowout_risk_flag": "strong_favourite_high_total_review",
    }
    return labels.get(flag_column, flag_column)


def _draw_prone_miss_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    ou = _safe_float(row.get("ou_median_total"))
    fav = _safe_float(row.get("favourite_probability"))
    if ou is None:
        reasons.append("ou_median_total_missing")
    elif ou > 2.25:
        reasons.append("ou_median_total_above_2_25")
    if fav is None:
        reasons.append("favourite_probability_missing")
    elif fav >= 0.50:
        reasons.append("favourite_probability_at_or_above_0_50")
    if str(row.get("draw_prone_flag", "no")).lower() == "yes":
        reasons.append("already_flagged")
    return "; ".join(reasons) or "nearest_miss_not_flagged_by_current_rule"


def _build_draw_prone_candidates(predictions: pd.DataFrame, *, limit: int = 20) -> pd.DataFrame:
    ev = _ev_default_rows(predictions)
    if ev.empty:
        return pd.DataFrame()
    rows = ev[ev.get("draw_prone_flag", pd.Series("", index=ev.index)).astype(str).str.lower().ne("yes")].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["reason_not_flagged"] = rows.apply(_draw_prone_miss_reason, axis=1)
    ou_numeric = pd.to_numeric(rows.get("ou_median_total", np.nan), errors="coerce")
    fav_numeric = pd.to_numeric(rows.get("favourite_probability", np.nan), errors="coerce")
    rows["_draw_prone_distance"] = (
        (ou_numeric - 2.25).clip(lower=0).fillna(9.0)
        + (fav_numeric - 0.50).clip(lower=0).fillna(9.0)
    )
    columns = [
        "match_id",
        "team_a",
        "team_b",
        "ou_median_total",
        "favourite_probability",
        "market_draw_probability",
        "ev_score",
        "modal_score",
        "actual_score",
        "reason_not_flagged",
    ]
    existing_columns = [column for column in columns if column in rows.columns]
    return rows.sort_values("_draw_prone_distance", kind="stable")[existing_columns].head(limit).reset_index(drop=True)


def _build_pattern_flags_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    ev = predictions[predictions["strategy"].astype(str).eq("ev_default")]
    if ev.empty:
        return pd.DataFrame()
    flags = ("draw_prone_flag", "modal_draw_challenger_flag", "btts_conflict_flag", "blowout_risk_flag")
    rows: list[dict[str, object]] = []
    for flag in flags:
        if flag not in ev:
            continue
        flagged = ev[ev[flag].astype(str).str.lower().eq("yes")]
        for config_name, group in flagged.groupby("config", sort=True):
            rows.append(
                _performance_row(
                    group,
                    {
                        "config": config_name,
                        "pattern_flag": flag,
                        "caution_label": _caution_label(flag),
                    },
                )
            )
        if flagged.empty:
            rows.append(
                {
                    "config": "all",
                    "pattern_flag": flag,
                    "caution_label": _caution_label(flag),
                    "matches_used": 0,
                    "total_expected_points": np.nan,
                    "average_expected_points": np.nan,
                    "actual_points_sum": 0,
                    "average_realised_points": np.nan,
                    "exact_score_rate": np.nan,
                    "correct_goal_difference_rate": np.nan,
                    "correct_result_rate": np.nan,
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True)


def _build_ev_vs_modal_attribution(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    required = {"config", "match_id", "strategy", "predicted_score", "realised_points", "model_expected_points"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()
    ev = predictions[predictions["strategy"].astype(str).eq("ev_default")].copy()
    modal = predictions[predictions["strategy"].astype(str).eq("most_likely")].copy()
    if ev.empty or modal.empty:
        return pd.DataFrame()
    columns = [
        "config",
        "match_id",
        "date",
        "team_a",
        "team_b",
        "actual_score",
        "predicted_score",
        "realised_points",
        "model_expected_points",
        "favourite_probability",
        "market_draw",
        "draw_vs_decisive_gap",
        "modal_draw_challenger_flag",
        "manual_review_flag",
    ]
    ev_columns = [column for column in columns if column in ev.columns]
    modal_columns = [
        column
        for column in ["config", "match_id", "predicted_score", "realised_points", "model_expected_points"]
        if column in modal.columns
    ]
    merged = ev[ev_columns].merge(
        modal[modal_columns],
        on=["config", "match_id"],
        how="inner",
        suffixes=("_ev", "_modal"),
    )
    if merged.empty:
        return merged
    merged["ev_vs_modal_differs"] = merged["predicted_score_ev"] != merged["predicted_score_modal"]
    merged["ev_minus_modal_realised_points"] = (
        merged["realised_points_ev"].astype(float) - merged["realised_points_modal"].astype(float)
    )
    merged["ev_minus_modal_expected_points"] = (
        pd.to_numeric(merged["model_expected_points_ev"], errors="coerce")
        - pd.to_numeric(merged["model_expected_points_modal"], errors="coerce")
    )
    return merged.sort_values(["config", "match_id"], kind="stable").reset_index(drop=True)


def _format_expected_vs_actual_summary(summary: pd.DataFrame, *, limit: int = 8) -> str:
    """Render a compact terminal table comparing expected and realised totals."""

    if summary.empty:
        return ""
    display = summary.head(limit)
    lines = ["Expected vs actual totals:"]
    for _, row in display.iterrows():
        expected = row.get("total_expected_points", np.nan)
        expected_text = f"{float(expected):.2f}" if pd.notna(expected) else "n/a"
        average_expected = row.get("average_expected_points", np.nan)
        average_expected_text = f"{float(average_expected):.2f}" if pd.notna(average_expected) else "n/a"
        lines.append(
            f"  {row['config']} / {row['strategy']} | "
            f"n={int(row['matches_used'])} | "
            f"expected total {expected_text} (avg {average_expected_text}) | "
            f"actual total {int(row['actual_points_sum'])} "
            f"(avg {float(row['average_realised_points']):.2f})"
        )
    return "\n".join(lines)


def _format_round_summary(round_summary: pd.DataFrame, *, limit: int = 12) -> str:
    """Render round-level group-stage totals for the terminal."""

    if round_summary.empty:
        return ""
    lines = ["Group-stage round totals:"]
    for _, row in round_summary.head(limit).iterrows():
        expected = row.get("total_expected_points", np.nan)
        expected_text = f"{float(expected):.2f}" if pd.notna(expected) else "n/a"
        lines.append(
            f"  {row['config']} / {row['strategy']} / {row['group_stage_playing_round']} "
            f"({row['round_match_range']}) | n={int(row['matches_used'])} | "
            f"expected {expected_text} | actual {int(row['actual_points_sum'])}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_live_backtest(
    settings: LiveBacktestSettings,
    *,
    export: bool = True,
    progress: bool = True,
) -> LiveBacktestReport:
    """Run the full live pipeline on historical paste files and score each strategy.

    Steps
    -----
    1. Load ``results.csv`` to discover which matches to score.
    2. Split the combined paste files in ``historical_odds_folder`` exactly as
       the live tool does.
    3. Parse 1X2, O/U, BTTS, correct-score, and Asian-handicap markets.
    4. Inject date / stage / team metadata from ``results.csv`` into the parsed
       odds (there is no live schedule file for historical runs).
    5. For every match x config, run ``run_prediction_workflow`` and extract
       all candidate strategy recommendations.
    6. Score each recommendation against the actual result and export.
    """
    from wc_predictor.workflow import run_prediction_workflow

    # 1. Load results
    results = load_historical_results(settings.results_path)
    if not results:
        raise ValueError(f"No valid historical results found in {settings.results_path}")
    result_by_id = {r.match_id: r for r in results}
    match_ids = tuple(result_by_id)

    if progress:
        print(f"Loaded {len(results)} historical results")
        print(f"Odds folder : {settings.historical_odds_folder}")
        print(f"Cache folder: {settings.cache_folder}")
        print()

    # 2. Split combined paste files
    settings.cache_folder.mkdir(parents=True, exist_ok=True)
    split_result = split_combined_oddsportal_pastes(
        settings.historical_odds_folder,
        settings.cache_folder,
        overwrite=True,
    )
    if progress:
        print(
            f"Paste split: {split_result.files_processed} files -> "
            f"{split_result.files_written} section files"
        )
        for warning in split_result.warnings:
            print(f"  Warning: {warning}")
        print()

    # 3. Parse all markets - write to per-backtest cache paths to avoid
    #    colliding with live output files
    cache = settings.cache_folder
    core_parse = parse_oddsportal_core_odds_folder(
        cache,
        output_path=cache.parent / "core_odds.csv",
        report_path=cache.parent / "core_parse_report.csv",
        total_goals_output_path=cache.parent / "total_goals_odds.csv",
        metadata_odds_path=None,   # no live schedule; metadata injected below
        match_ids=match_ids,
    )
    cs_parse = parse_oddsportal_correct_score_folder(
        cache,
        output_path=cache.parent / "correct_score_odds.csv",
        report_path=cache.parent / "correct_score_parse_report.csv",
        match_ids=match_ids,
    )
    ah_parse = parse_oddsportal_asian_handicap_folder(
        cache,
        output_path=cache.parent / "asian_handicap_odds.csv",
        report_path=cache.parent / "asian_handicap_parse_report.csv",
        match_ids=match_ids,
    )

    # 4. Inject metadata (date, stage, team names) from results into core odds
    core_odds = _inject_match_metadata(core_parse.odds, results)
    total_goals_odds = core_parse.total_goals_odds

    if progress:
        n_core = core_odds["match_id"].nunique() if not core_odds.empty else 0
        n_cs = cs_parse.odds["match_id"].nunique() if not cs_parse.odds.empty else 0
        n_ah = ah_parse.odds["match_id"].nunique() if not ah_parse.odds.empty else 0
        print(f"Parsed odds : {n_core} matches with 1X2/O/U, "
              f"{n_cs} with correct score, {n_ah} with AH")
        print()

    # 5. Run all configs for every match that has core odds and a result
    prediction_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    template_count = 0
    missing_count = 0

    for match_id, result in result_by_id.items():
        match_core = core_odds[core_odds["match_id"].astype(str) == match_id]
        if match_core.empty:
            reason = _classify_missing_odds(settings.historical_odds_folder, match_id)
            skipped_rows.append({"match_id": match_id, "config": "all", "reason": reason})
            if reason == "paste_file_not_filled_in_yet":
                template_count += 1
            elif reason == "no_paste_file":
                missing_count += 1
            elif progress:
                # Only print the genuinely surprising skips (file filled but unparseable)
                print(f"  SKIP {match_id}: {reason}")
            continue

        match_tg = (
            total_goals_odds[total_goals_odds["match_id"].astype(str) == match_id]
            if not total_goals_odds.empty
            else pd.DataFrame()
        )
        match_cs = (
            cs_parse.odds[cs_parse.odds["match_id"].astype(str) == match_id]
            if not cs_parse.odds.empty
            else pd.DataFrame()
        )
        match_ah = (
            ah_parse.odds[ah_parse.odds["match_id"].astype(str) == match_id]
            if not ah_parse.odds.empty
            else pd.DataFrame()
        )

        for config_entry in settings.configs:
            tg_input = match_tg if config_entry.use_total_goals and not match_tg.empty else None
            cs_input = match_cs if config_entry.use_correct_score and not match_cs.empty else None
            ah_input = match_ah if config_entry.use_asian_handicap and not match_ah.empty else None
            playing_round = _group_stage_playing_round(match_id, result.stage)

            try:
                workflow_result = run_prediction_workflow(
                    match_core,
                    config=config_entry.config,
                    correct_score_odds=cs_input,
                    total_goals_odds=tg_input,
                    asian_handicap_odds=ah_input,
                )
            except (ValueError, RuntimeError) as exc:
                skipped_rows.append(
                    {
                        "match_id": match_id,
                        "config": config_entry.name,
                        "reason": str(exc),
                    }
                )
                continue

            if workflow_result.match_report.empty:
                skipped_rows.append(
                    {
                        "match_id": match_id,
                        "config": config_entry.name,
                        "reason": "empty_workflow_report",
                    }
                )
                continue

            report_row = workflow_result.match_report.iloc[0]
            strategies = _score_strategies(
                report_row, result.actual_score_a, result.actual_score_b
            )
            for strat_row in strategies:
                strat_row.update(
                    {
                        "config": config_entry.name,
                        "match_id": match_id,
                        "date": result.date,
                        "stage": result.stage,
                        "team_a": result.team_a,
                        "team_b": result.team_b,
                        "group_stage_playing_round": playing_round,
                        "round_match_range": _round_match_range(playing_round),
                    }
                )
                prediction_rows.append(strat_row)

        if progress:
            ev_score_row = next(
                (r for r in prediction_rows[-len(settings.configs) * 5:]
                 if r.get("match_id") == match_id and r.get("strategy") == "ev_default"
                 and r.get("config") == settings.configs[0].name),
                None,
            )
            pred_str = ev_score_row["predicted_score"] if ev_score_row else "?"
            points_str = f"{ev_score_row['realised_points']}pts" if ev_score_row else ""
            print(
                f"  {match_id}: {result.team_a} vs {result.team_b} | "
                f"actual {result.actual_score_a}-{result.actual_score_b} | "
                f"baseline predicted {pred_str} -> {points_str}"
            )

    predictions = pd.DataFrame(prediction_rows)
    skipped = pd.DataFrame(skipped_rows)
    summary = _build_summary(predictions)
    round_summary = _build_round_summary(predictions)

    matches_scored = predictions["match_id"].nunique() if not predictions.empty else 0
    if progress and (template_count or missing_count):
        print()
        if template_count:
            print(f"  ({template_count} matches skipped: paste file not filled in yet)")
        if missing_count:
            print(f"  ({missing_count} matches skipped: no paste file)")

    report = LiveBacktestReport(summary, predictions, skipped, round_summary)
    if export:
        report.export(settings)
        if progress:
            print()
            print("Backtest complete")
            print(f"  {matches_scored} matches scored | {len(settings.configs)} configs")
            if matches_scored == 0:
                print("  WARNING: no matches were scored. Fill in odds paste files first.")
            expected_vs_actual = _format_expected_vs_actual_summary(summary)
            if expected_vs_actual:
                print()
                print(expected_vs_actual)
            round_totals = _format_round_summary(round_summary)
            if round_totals:
                print()
                print(round_totals)
            print(f"  Summary   : {settings.summary_output_path}")
            print(f"  Workbook  : {settings.excel_output_path}")
    return report
