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
            if not self.round_summary.empty:
                self.round_summary.to_excel(writer, index=False, sheet_name="group_stage_rounds")
            self.round_summary.to_excel(writer, index=False, sheet_name="matchday_performance")
            _build_ev_vs_modal_attribution(self.predictions).to_excel(
                writer, index=False, sheet_name="ev_vs_modal_attribution"
            )
            _filter_flagged_ev_rows(self.predictions, "modal_draw_challenger_flag").to_excel(
                writer, index=False, sheet_name="modal_draw_challenger_matches"
            )
            _filter_flagged_ev_rows(self.predictions, "btts_conflict_flag").to_excel(
                writer, index=False, sheet_name="btts_conflict_matches"
            )
            _filter_flagged_ev_rows(self.predictions, "draw_prone_flag").to_excel(
                writer, index=False, sheet_name="draw_prone_matches"
            )
            _filter_flagged_ev_rows(self.predictions, "blowout_risk_flag").to_excel(
                writer, index=False, sheet_name="blowout_risk_matches"
            )
            _build_named_config_summary(self.predictions, prefix="cs_weight_").to_excel(
                writer, index=False, sheet_name="blend_weight_sweep"
            )
            _build_named_config_summary(self.predictions, prefix="cs_weight_").to_excel(
                writer, index=False, sheet_name="correct_score_blend_sweep"
            )
            _build_named_config_summary(self.predictions, prefix="larger_grid_").to_excel(
                writer, index=False, sheet_name="larger_grid_sweep"
            )
            _build_named_config_summary(self.predictions, prefix="modal_draw_").to_excel(
                writer, index=False, sheet_name="draw_threshold_sweep"
            )
            _build_pattern_flags_summary(self.predictions).to_excel(
                writer, index=False, sheet_name="pattern_flags_summary"
            )
            _build_grouped_performance(self.predictions, ["config", "strategy", "favourite_bucket"]).to_excel(
                writer, index=False, sheet_name="favourite_bucket_performance"
            )
            _build_grouped_performance(self.predictions, ["config", "strategy", "manual_review_flag"]).to_excel(
                writer, index=False, sheet_name="manual_review_performance"
            )
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
