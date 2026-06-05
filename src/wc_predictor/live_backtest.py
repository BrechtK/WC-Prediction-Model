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

import numpy as np
import pandas as pd

from wc_predictor.config import DevigConfig, ProjectConfig
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
            }
        )
    return rows


def _build_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-match predictions into config × strategy summary rows."""
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (config_name, strategy), group in predictions.groupby(["config", "strategy"]):
        rows.append(
            {
                "config": config_name,
                "strategy": strategy,
                "matches_used": len(group),
                "total_points": int(group["realised_points"].sum()),
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
    5. For every match × config, run ``run_prediction_workflow`` and extract
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
            f"Paste split: {split_result.files_processed} files → "
            f"{split_result.files_written} section files"
        )
        for warning in split_result.warnings:
            print(f"  Warning: {warning}")
        print()

    # 3. Parse all markets — write to per-backtest cache paths to avoid
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

    for match_id, result in result_by_id.items():
        match_core = core_odds[core_odds["match_id"].astype(str) == match_id]
        if match_core.empty:
            skipped_rows.append(
                {
                    "match_id": match_id,
                    "config": "all",
                    "reason": "no_core_odds_file_found",
                }
            )
            if progress:
                print(f"  SKIP {match_id}: no odds file")
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
                f"baseline predicted {pred_str} → {points_str}"
            )

    predictions = pd.DataFrame(prediction_rows)
    skipped = pd.DataFrame(skipped_rows)
    summary = _build_summary(predictions)

    report = LiveBacktestReport(summary, predictions, skipped)
    if export:
        report.export(settings)
        if progress:
            print()
            print(f"Backtest complete")
            print(f"  {len(result_by_id)} matches | {len(settings.configs)} configs")
            print(f"  Summary   : {settings.summary_output_path}")
            print(f"  Workbook  : {settings.excel_output_path}")
    return report
