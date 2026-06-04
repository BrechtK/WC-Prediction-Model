"""One-click orchestration for live predictions from manually pasted odds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import re
import time

import pandas as pd

from wc_predictor.config import ProjectConfig
from wc_predictor.correct_score_weight_comparison import (
    DEFAULT_CORRECT_SCORE_WEIGHTS,
    DEFAULT_OUTPUT_PATH as DEFAULT_WEIGHT_COMPARISON_OUTPUT_PATH,
    CorrectScoreWeightComparison,
    compare_correct_score_weights,
    export_correct_score_weight_comparison,
)
from wc_predictor.oddsportal import (
    DEFAULT_OUTPUT_PATH as DEFAULT_CORRECT_SCORE_OUTPUT_PATH,
    DEFAULT_REPORT_PATH as DEFAULT_CORRECT_SCORE_REPORT_PATH,
    OddsPortalParseResult,
    parse_oddsportal_correct_score_folder,
)
from wc_predictor.oddsportal_asian_handicap import (
    DEFAULT_OUTPUT_PATH as DEFAULT_ASIAN_HANDICAP_OUTPUT_PATH,
    DEFAULT_REPORT_PATH as DEFAULT_ASIAN_HANDICAP_REPORT_PATH,
    OddsPortalAsianHandicapParseResult,
    parse_oddsportal_asian_handicap_folder,
)
from wc_predictor.oddsportal_core import (
    DEFAULT_INPUT_FOLDER,
    DEFAULT_METADATA_ODDS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_CORE_OUTPUT_PATH,
    DEFAULT_REPORT_PATH as DEFAULT_CORE_REPORT_PATH,
    DEFAULT_TOTAL_GOALS_OUTPUT_PATH,
    OddsPortalCoreParseResult,
    parse_oddsportal_core_odds_folder,
)
from wc_predictor.oddsportal_schedule import (
    DEFAULT_SCHEDULE_INPUT_PATH,
    DEFAULT_SCHEDULE_OUTPUT_PATH,
    DEFAULT_SCHEDULE_REPORT_PATH,
    OddsPortalScheduleParseResult,
    prepare_schedule_metadata,
)
from wc_predictor.paths import (
    OUTPUT_PREDICTIONS_CSV_PATH,
    OUTPUT_PREDICTIONS_XLSX_PATH,
    OUTPUT_SUBMISSION_XLSX_PATH,
)
from wc_predictor.world_cup import WorldCupPredictionSettings, run_world_cup_predictions
from wc_predictor.workflow import PredictionWorkflowResult

_PASTE_FILENAME_PATTERN = re.compile(
    r"^(?P<match_id>.+)_(?P<market>1x2|over_under|btts|correct_score|asian_handicap)\.txt$",
    re.IGNORECASE,
)
_OPTIONAL_MARKETS = ("over_under", "btts", "correct_score")
_STRICT_OPTIONAL_MARKETS = _OPTIONAL_MARKETS


class LivePredictionError(RuntimeError):
    """Raised when live paste inputs cannot produce a trustworthy baseline."""


@dataclass(frozen=True)
class LivePredictionSettings:
    """Paths and runtime switches for one live-prediction run."""

    input_folder: Path = DEFAULT_INPUT_FOLDER
    match_id: str | None = None
    match_ids: Sequence[str] | None = None
    strict: bool = False
    skip_weight_sensitivity: bool = True
    metadata_odds_path: Path | None = DEFAULT_METADATA_ODDS_PATH
    schedule_input_path: Path | None = DEFAULT_SCHEDULE_INPUT_PATH
    schedule_output_path: Path = DEFAULT_SCHEDULE_OUTPUT_PATH
    schedule_parse_report_path: Path = DEFAULT_SCHEDULE_REPORT_PATH
    skip_schedule_parse: bool = False
    core_odds_output_path: Path = DEFAULT_CORE_OUTPUT_PATH
    total_goals_output_path: Path = DEFAULT_TOTAL_GOALS_OUTPUT_PATH
    core_parse_report_path: Path = DEFAULT_CORE_REPORT_PATH
    correct_score_output_path: Path = DEFAULT_CORRECT_SCORE_OUTPUT_PATH
    correct_score_parse_report_path: Path = DEFAULT_CORRECT_SCORE_REPORT_PATH
    asian_handicap_output_path: Path = DEFAULT_ASIAN_HANDICAP_OUTPUT_PATH
    asian_handicap_parse_report_path: Path = DEFAULT_ASIAN_HANDICAP_REPORT_PATH
    recommendations_csv_output_path: Path = OUTPUT_PREDICTIONS_CSV_PATH
    recommendations_xlsx_output_path: Path = OUTPUT_PREDICTIONS_XLSX_PATH
    submission_xlsx_output_path: Path = OUTPUT_SUBMISSION_XLSX_PATH
    write_detailed_excel: bool = True
    weight_comparison_output_path: Path = DEFAULT_WEIGHT_COMPARISON_OUTPUT_PATH
    weight_sensitivity_weights: Sequence[float] = DEFAULT_CORRECT_SCORE_WEIGHTS
    initial_runtime_timings: dict[str, float] = field(default_factory=dict)
    extra_warning_flags_by_match: Mapping[str, Sequence[str]] | None = None


@dataclass(frozen=True)
class LivePredictionResult:
    """Parser, workflow, and optional sensitivity outputs from a live run."""

    settings: LivePredictionSettings
    core_parse: OddsPortalCoreParseResult
    correct_score_parse: OddsPortalParseResult
    asian_handicap_parse: OddsPortalAsianHandicapParseResult
    workflow: PredictionWorkflowResult
    weight_comparison: CorrectScoreWeightComparison | None
    paste_warnings: dict[str, tuple[str, ...]]
    schedule_metadata: pd.DataFrame
    runtime_timings: dict[str, float]


@dataclass(frozen=True)
class ParsedLivePredictionInputs:
    """Parsed OddsPortal inputs shared by live recommendations and diagnostics."""

    settings: LivePredictionSettings
    selected_match_ids: tuple[str, ...]
    core_parse: OddsPortalCoreParseResult
    correct_score_parse: OddsPortalParseResult
    asian_handicap_parse: OddsPortalAsianHandicapParseResult
    paste_warnings: dict[str, tuple[str, ...]]
    schedule_metadata: pd.DataFrame
    schedule_parse: OddsPortalScheduleParseResult | None

    @property
    def has_total_goals(self) -> bool:
        return not self.core_parse.total_goals_odds.empty

    @property
    def has_correct_scores(self) -> bool:
        return not self.correct_score_parse.odds.empty

    @property
    def has_asian_handicap(self) -> bool:
        return not self.asian_handicap_parse.odds.empty


def discover_live_paste_markets(input_folder: str | Path) -> dict[str, set[str]]:
    """Return recognised paste-market names grouped by filename match ID."""

    input_folder = Path(input_folder)
    discovered: dict[str, set[str]] = {}
    for path in sorted(input_folder.glob("*.txt")):
        match = _PASTE_FILENAME_PATTERN.fullmatch(path.name)
        if match:
            discovered.setdefault(match.group("match_id"), set()).add(match.group("market").lower())
    return discovered


def _append_warning(warnings: dict[str, list[str]], match_id: str, warning: str) -> None:
    match_warnings = warnings.setdefault(match_id, [])
    if warning not in match_warnings:
        match_warnings.append(warning)


def _select_and_validate_pastes(
    input_folder: Path,
    match_id: str | None,
    match_ids: Sequence[str] | None,
    strict: bool,
) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    """Choose matches and fail early when required paste files are absent."""

    discovered = discover_live_paste_markets(input_folder)
    if match_ids is not None:
        selected = tuple(str(selected_match_id) for selected_match_id in match_ids)
        missing = [selected_match_id for selected_match_id in selected if selected_match_id not in discovered]
        if missing:
            raise LivePredictionError(f"No OddsPortal paste files found for match IDs: {', '.join(missing)}")
    elif match_id is not None:
        if match_id not in discovered:
            raise LivePredictionError(f"No OddsPortal paste files found for match_id {match_id!r}")
        selected = (match_id,)
    else:
        selected = tuple(sorted(discovered))
    if not selected:
        raise LivePredictionError(f"No recognised OddsPortal paste files found below {input_folder}")

    missing_required = [match for match in selected if "1x2" not in discovered[match]]
    if missing_required:
        raise LivePredictionError(
            "Critical error: missing required *_1x2.txt paste for match IDs: "
            + ", ".join(missing_required)
        )

    warnings: dict[str, list[str]] = {}
    missing_optional: list[str] = []
    for selected_match_id in selected:
        for market in _OPTIONAL_MARKETS:
            if market not in discovered[selected_match_id]:
                warning = f"missing_optional_paste:{market}"
                _append_warning(warnings, selected_match_id, warning)
                if market in _STRICT_OPTIONAL_MARKETS:
                    missing_optional.append(f"{selected_match_id}:{market}")
    if strict and missing_optional:
        raise LivePredictionError(
            "Strict mode requires 1X2, O/U, BTTS, and correct-score pastes. Missing: "
            + ", ".join(missing_optional)
        )
    return selected, warnings


def _validate_parsed_markets(
    selected: Sequence[str],
    core_parse: OddsPortalCoreParseResult,
    correct_score_parse: OddsPortalParseResult,
    strict: bool,
    warnings: dict[str, list[str]],
) -> None:
    """Surface empty parser results before entering the model workflow."""

    for match_id in selected:
        core_rows = core_parse.odds[core_parse.odds["match_id"].astype(str) == match_id]
        complete_1x2 = core_rows.dropna(subset=["odds_a_win", "odds_draw", "odds_b_win"])
        if complete_1x2.empty:
            raise LivePredictionError(f"Critical error: no valid bookmaker 1X2 rows parsed for {match_id}")

        total_rows = core_parse.total_goals_odds[
            core_parse.total_goals_odds["match_id"].astype(str) == match_id
        ]
        if total_rows.empty:
            _append_warning(warnings, match_id, "no_total_goals_rows_parsed")

        btts_rows = core_rows.dropna(subset=["odds_btts_yes", "odds_btts_no"])
        if btts_rows.empty:
            _append_warning(warnings, match_id, "no_btts_rows_parsed")

        correct_score_rows = correct_score_parse.odds[
            correct_score_parse.odds["match_id"].astype(str) == match_id
        ]
        if correct_score_rows.empty:
            _append_warning(warnings, match_id, "no_correct_score_rows_parsed")

        if strict:
            required_optional_rows = {
                "over_under": total_rows,
                "btts": btts_rows,
                "correct_score": correct_score_rows,
            }
            empty_markets = [market for market, rows in required_optional_rows.items() if rows.empty]
            if empty_markets:
                raise LivePredictionError(
                    f"Strict mode found no usable parsed rows for {match_id}: {', '.join(empty_markets)}"
                )


def run_live_prediction(
    settings: LivePredictionSettings | None = None,
    config: ProjectConfig | None = None,
) -> LivePredictionResult:
    """Parse live pastes, export recommendations, and optionally compare blend weights."""

    settings = settings or LivePredictionSettings()
    total_start = time.perf_counter()
    config = config or ProjectConfig(
        correct_score_poisson_weight=1.0,
        enable_margin_method_comparison=False,
        enable_market_consistent_challenger="only_if_close",
    )
    runtime_timings = dict(settings.initial_runtime_timings)
    parsed = parse_live_prediction_inputs(settings, runtime_timings)
    has_total_goals = parsed.has_total_goals
    has_correct_scores = parsed.has_correct_scores
    has_asian_handicap = parsed.has_asian_handicap
    mutable_warnings = {match_id: list(values) for match_id, values in parsed.paste_warnings.items()}
    workflow = run_world_cup_predictions(
        WorldCupPredictionSettings(
            input_path=settings.core_odds_output_path,
            total_goals_input_path=settings.total_goals_output_path if has_total_goals else None,
            correct_score_input_path=settings.correct_score_output_path if has_correct_scores else None,
            asian_handicap_input_path=settings.asian_handicap_output_path if has_asian_handicap else None,
            csv_output_path=settings.recommendations_csv_output_path,
            xlsx_output_path=settings.recommendations_xlsx_output_path,
            submission_xlsx_output_path=settings.submission_xlsx_output_path,
            write_detailed_excel=settings.write_detailed_excel,
            extra_warning_flags_by_match=settings.extra_warning_flags_by_match,
        ),
        config,
        runtime_timings,
    )

    weight_comparison = None
    if has_correct_scores and not settings.skip_weight_sensitivity:
        weight_start = time.perf_counter()
        weight_comparison = compare_correct_score_weights(
            settings.core_odds_output_path,
            settings.correct_score_output_path,
            weights=settings.weight_sensitivity_weights,
            config=config,
            total_goals_odds_path=settings.total_goals_output_path if has_total_goals else None,
        )
        export_correct_score_weight_comparison(weight_comparison, settings.weight_comparison_output_path)
        runtime_timings["correct-score weight sensitivity"] = (
            runtime_timings.get("correct-score weight sensitivity", 0.0)
            + time.perf_counter()
            - weight_start
        )
    elif not has_correct_scores and not settings.skip_weight_sensitivity:
        for selected_match_id in parsed.selected_match_ids:
            _append_warning(mutable_warnings, selected_match_id, "weight_sensitivity_skipped:no_correct_score_odds")
        runtime_timings["correct-score weight sensitivity"] = runtime_timings.get(
            "correct-score weight sensitivity", 0.0
        )
    else:
        runtime_timings["correct-score weight sensitivity"] = runtime_timings.get(
            "correct-score weight sensitivity", 0.0
        )
    runtime_timings["total"] = time.perf_counter() - total_start

    return LivePredictionResult(
        settings,
        parsed.core_parse,
        parsed.correct_score_parse,
        parsed.asian_handicap_parse,
        workflow,
        weight_comparison,
        {match_id: tuple(values) for match_id, values in mutable_warnings.items()},
        parsed.schedule_metadata,
        runtime_timings,
    )


def parse_live_prediction_inputs(
    settings: LivePredictionSettings | None = None,
    runtime_timings: dict[str, float] | None = None,
) -> ParsedLivePredictionInputs:
    """Parse and validate the live OddsPortal pastes without running recommendations."""

    settings = settings or LivePredictionSettings()
    schedule_start = time.perf_counter()
    prepared_schedule = prepare_schedule_metadata(
        settings.schedule_input_path,
        settings.schedule_output_path,
        settings.schedule_parse_report_path,
        existing_metadata_path=settings.metadata_odds_path,
        skip_parse=settings.skip_schedule_parse,
    )
    if runtime_timings is not None and "schedule parse" not in runtime_timings:
        runtime_timings["schedule parse"] = time.perf_counter() - schedule_start
    selected, mutable_warnings = _select_and_validate_pastes(
        settings.input_folder,
        settings.match_id,
        settings.match_ids,
        settings.strict,
    )
    core_parse_start = time.perf_counter()
    core_parse = parse_oddsportal_core_odds_folder(
        settings.input_folder,
        settings.core_odds_output_path,
        settings.core_parse_report_path,
        total_goals_output_path=settings.total_goals_output_path,
        metadata_odds_path=prepared_schedule.metadata_path,
        match_ids=selected,
    )
    if runtime_timings is not None:
        runtime_timings["core odds parse"] = (
            runtime_timings.get("core odds parse", 0.0) + time.perf_counter() - core_parse_start
        )
    correct_score_parse_start = time.perf_counter()
    correct_score_parse = parse_oddsportal_correct_score_folder(
        settings.input_folder,
        settings.correct_score_output_path,
        settings.correct_score_parse_report_path,
        match_ids=selected,
    )
    if runtime_timings is not None:
        runtime_timings["correct-score odds parse"] = (
            runtime_timings.get("correct-score odds parse", 0.0)
            + time.perf_counter()
            - correct_score_parse_start
        )
    asian_handicap_parse_start = time.perf_counter()
    asian_handicap_parse = parse_oddsportal_asian_handicap_folder(
        settings.input_folder,
        settings.asian_handicap_output_path,
        settings.asian_handicap_parse_report_path,
        match_ids=selected,
    )
    if runtime_timings is not None:
        runtime_timings["asian-handicap odds parse"] = (
            runtime_timings.get("asian-handicap odds parse", 0.0)
            + time.perf_counter()
            - asian_handicap_parse_start
        )
    _validate_parsed_markets(selected, core_parse, correct_score_parse, settings.strict, mutable_warnings)
    return ParsedLivePredictionInputs(
        settings,
        selected,
        core_parse,
        correct_score_parse,
        asian_handicap_parse,
        {match_id: tuple(values) for match_id, values in mutable_warnings.items()},
        prepared_schedule.schedule,
        prepared_schedule.parse_result,
    )


def format_live_prediction_summary(
    result: LivePredictionResult,
    terminal_verbosity: str = "compact",
    run_profile: str = "live",
    show_runtime_summary: bool = False,
) -> str:
    """Render a concise submission-focused live terminal report."""

    if terminal_verbosity not in {"compact", "normal", "debug"}:
        raise ValueError("terminal_verbosity must be compact, normal, or debug")

    def format_probability(value: object) -> str:
        return f"{float(value):.1%}" if pd.notna(value) else "n/a"

    def runtime_lines() -> list[str]:
        if not show_runtime_summary:
            return []
        timings = result.runtime_timings
        measured_keys = (
            "schedule parse",
            "combined paste split",
            "core odds parse",
            "correct-score odds parse",
            "asian-handicap odds parse",
            "main model run",
            "market-consistent challenger",
            "margin-method comparison",
            "correct-score weight sensitivity",
            "Excel write",
        )
        total = timings.get("total", sum(float(timings.get(key, 0.0)) for key in measured_keys))
        if terminal_verbosity == "compact":
            return [
                "",
                "Runtime summary:",
                f"- total: {float(total):.2f}s",
            ]
        measured_total = sum(float(timings.get(key, 0.0)) for key in measured_keys)
        other = max(0.0, float(total) - measured_total)
        return [
            "",
            "Runtime summary:",
            f"- schedule parse: {timings.get('schedule parse', 0.0):.2f}s",
            f"- combined paste split: {timings.get('combined paste split', 0.0):.2f}s",
            f"- core odds parse: {timings.get('core odds parse', 0.0):.2f}s",
            f"- correct-score odds parse: {timings.get('correct-score odds parse', 0.0):.2f}s",
            f"- asian-handicap odds parse: {timings.get('asian-handicap odds parse', 0.0):.2f}s",
            f"- main model run: {timings.get('main model run', 0.0):.2f}s",
            f"- market-consistent challenger: {timings.get('market-consistent challenger', 0.0):.2f}s",
            f"- margin-method comparison: {timings.get('margin-method comparison', 0.0):.2f}s",
            f"- correct-score weight sensitivity: {timings.get('correct-score weight sensitivity', 0.0):.2f}s",
            f"- Excel write: {timings.get('Excel write', 0.0):.2f}s",
            f"- other/unmeasured: {other:.2f}s",
            f"- total: {float(total):.2f}s",
        ]

    def format_float(value: object, precision: int = 6) -> str:
        return f"{float(value):.{precision}f}" if pd.notna(value) else "n/a"

    def market_consistent_diagnostic_lines(row: pd.Series) -> list[str]:
        status = str(row.get("market_consistent_status", ""))
        classification = str(row.get("market_consistent_optimisation_classification", ""))
        if status in {"", "ok", "skipped"} and classification in {"", "success", "skipped"}:
            return []
        return [
            "Market-consistent optimiser:",
            f"- status: {status or 'n/a'}",
            f"- classification: {classification or 'n/a'}",
            f"- scipy success: {row.get('market_consistent_optimisation_success')}",
            f"- scipy status: {row.get('market_consistent_optimisation_status_code')}",
            f"- message: {row.get('market_consistent_optimisation_message')}",
            f"- iterations: {row.get('market_consistent_optimisation_iterations')}",
            f"- objective: {format_float(row.get('market_consistent_final_objective_value'))}",
            f"- gradient inf-norm: {format_float(row.get('market_consistent_gradient_norm'))}",
            f"- max constraint error: {format_float(row.get('market_consistent_max_constraint_error'))}",
            (
                "- fit rmse: "
                f"1x2={format_float(row.get('market_consistent_1x2_fit_error'))}, "
                f"btts={format_float(row.get('market_consistent_btts_fit_error'))}, "
                f"totals={format_float(row.get('market_consistent_total_goals_fit_error'))}, "
                f"correct-score={format_float(row.get('market_consistent_correct_score_fit_error'))}"
                f", asian-handicap={format_float(row.get('market_consistent_asian_handicap_fit_error'))}"
            ),
            f"- KL vs prior: {format_float(row.get('market_consistent_kl_divergence_vs_prior'))}",
            (
                "- constraints: "
                f"total={row.get('market_consistent_constraint_count', 'n/a')}, "
                f"1x2={row.get('market_consistent_1x2_constraint_count', 'n/a')}, "
                f"btts={row.get('market_consistent_btts_constraint_count', 'n/a')}, "
                f"totals={row.get('market_consistent_total_goals_constraint_count', 'n/a')}, "
                f"asian-handicap={row.get('market_consistent_asian_handicap_constraint_count', 'n/a')}, "
                f"correct-score={row.get('market_consistent_correct_score_constraint_count', 'n/a')}"
            ),
            (
                "- Asian handicap: "
                f"available={row.get('market_consistent_asian_handicap_lines_available', 'n/a')}, "
                f"selected={row.get('market_consistent_asian_handicap_lines_selected', 'n/a')}, "
                f"skipped={row.get('market_consistent_asian_handicap_lines_skipped', 'n/a')}"
            ),
            f"- Asian handicap lines selected: {row.get('market_consistent_asian_handicap_lines_used') or 'none'}",
            f"- Asian handicap fit all lines: {format_float(row.get('market_consistent_asian_handicap_fit_error_all'))}",
            f"- margin distribution before: {row.get('market_consistent_margin_distribution_before') or 'n/a'}",
            f"- margin distribution after: {row.get('market_consistent_margin_distribution_after') or 'n/a'}",
            (
                "- largest margin shift: "
                f"{row.get('market_consistent_largest_margin_shift', 'n/a')} "
                f"({format_float(row.get('market_consistent_largest_margin_shift_value'), 4)})"
            ),
        ]

    diagnostic_sections: list[str] = []
    final_submissions: list[str] = []
    margin_comparison = result.workflow.margin_method_comparison
    dashboard = result.workflow.final_decision_dashboard
    dashboard_by_match = dashboard.set_index("match_id") if not dashboard.empty else pd.DataFrame()
    manual_review_lines: list[str] = []
    for _, row in result.workflow.match_report.iterrows():
        match_id = str(row["match_id"])
        dashboard_row = dashboard_by_match.loc[match_id] if not dashboard_by_match.empty and match_id in dashboard_by_match.index else pd.Series(dtype=object)
        match_margin_comparison = (
            margin_comparison[margin_comparison["match_id"].astype(str) == match_id]
            if not margin_comparison.empty
            else pd.DataFrame()
        )
        margin_lines: list[str] = []
        if not match_margin_comparison.empty:
            methods = ", ".join(match_margin_comparison["method"].astype(str).tolist())
            changes = match_margin_comparison["differs_from_default"].astype(str).str.lower().eq("yes").any()
            margin_lines = [
                "Margin-removal sensitivity:",
                f"- default method: {row.get('margin_removal_method')}",
                f"- methods compared: {methods}",
                "- recommendations:",
                *[
                    (
                        f"  {comparison_row['method']}: {comparison_row['recommended_score']}"
                        if str(comparison_row.get("method_status", "ok")) == "ok"
                        else f"  {comparison_row['method']}: failed / invalid probabilities"
                    )
                    for _, comparison_row in match_margin_comparison.iterrows()
                ],
                f"- changes recommendation: {'yes' if changes else 'no'}",
            ]
        btts_difference = (
            row["model_implied_btts_yes_probability"] - row["market_fair_btts_yes_probability"]
            if pd.notna(row.get("market_fair_btts_yes_probability"))
            else pd.NA
        )
        if terminal_verbosity == "debug":
            diagnostic_sections.append(
                "\n".join(
                    [
                        f"{match_id} {row['team_a']} vs {row['team_b']}",
                        "BTTS diagnostics:",
                        f"- market BTTS Yes: {format_probability(row.get('market_fair_btts_yes_probability'))}",
                        f"- model BTTS Yes: {format_probability(row.get('model_implied_btts_yes_probability'))}",
                        f"- difference: {format_probability(btts_difference)}",
                        f"- P(no BTTS): {format_probability(row.get('model_probability_no_btts'))}",
                        "EV explanation:",
                        f"- {row.get('ev_explanation')}",
                        "Decision aid:",
                        f"- confidence: {row.get('recommendation_confidence')}",
                        f"- plausible alternatives: {row.get('plausible_top_alternatives') or 'none'}",
                        f"- manual review flag: {row.get('manual_review_flag')}",
                        f"- decision note: {row.get('decision_note')}",
                        "Margin diagnostics:",
                        f"- {row.get('margin_diagnostic_note') or 'n/a'}",
                        f"- draw vs decisive gap: {format_float(row.get('draw_vs_decisive_gap'), 3)}",
                        *market_consistent_diagnostic_lines(row),
                        *margin_lines,
                        *(
                            [
                                "Extreme-favourite audit:",
                                f"- top clean-sheet scores: {row.get('top_clean_sheet_scores')}",
                                f"- favourite margins: {row.get('top_favourite_margin_probabilities')}",
                                f"- favourite score counts: {row.get('top_5_favourite_score_count_probabilities')}",
                                f"- 3/4/5-nil EV cluster: {row.get('high_score_cluster_scores')}",
                                f"- current final: {row.get('current_final_recommendation')}",
                                f"- normal-grid Poisson: {row.get('normal_grid_poisson_recommendation')} "
                                f"(tail {format_probability(row.get('normal_grid_tail_mass'))})",
                                f"- larger-grid Poisson: {row.get('larger_grid_poisson_recommendation')} "
                                f"(tail {format_probability(row.get('larger_grid_tail_mass'))})",
                                f"- favourite 3-0 EV: normal {row.get('ev_favourite_3_0_normal_grid'):.3f} "
                                f"larger {row.get('ev_favourite_3_0_larger_grid'):.3f}",
                                f"- favourite 4-0 EV: normal {row.get('ev_favourite_4_0_normal_grid'):.3f} "
                                f"larger {row.get('ev_favourite_4_0_larger_grid'):.3f}",
                                f"- favourite 5-0 EV: normal {row.get('ev_favourite_5_0_normal_grid'):.3f} "
                                f"larger {row.get('ev_favourite_5_0_larger_grid'):.3f}",
                            ]
                            if bool(row.get("extreme_favourite_audit_triggered"))
                            else []
                        ),
                    ]
                )
            )
        confidence = dashboard_row.get("confidence_level", row.get("confidence_level", "unknown"))
        review = dashboard_row.get("manual_review_flag", row.get("manual_review_flag", "no"))
        alternative = dashboard_row.get("main_alternative_score", "")
        final_line = f"{match_id} {row['team_a']} vs {row['team_b']}: {row['recommended_score']} | {confidence} | review {review}"
        if alternative:
            final_line += f" | alt {alternative}"
        final_submissions.append(final_line)

        if (
            terminal_verbosity == "compact"
            and str(row.get("asian_handicap_shift_recommendation", "no")).lower() == "yes"
        ):
            manual_review_lines.extend(
                [
                    "",
                    "Asian handicap review:",
                    f"{match_id}: handicap constraints shift market-consistent score to "
                    f"{row.get('market_consistent_recommended_score')}; default remains {row.get('recommended_score')}.",
                ]
            )

    dashboard_summary: list[str] = []
    if not dashboard.empty:
        high_count = int(dashboard["confidence_level"].astype(str).eq("high").sum())
        medium_count = int(dashboard["confidence_level"].astype(str).eq("medium").sum())
        manual_rows = dashboard[dashboard["manual_review_flag"].astype(str).str.lower().eq("yes")]
        if terminal_verbosity != "compact":
            dashboard_summary = [
                "",
                "Decision dashboard summary:",
                f"- high confidence: {high_count} matches",
                f"- medium confidence: {medium_count} matches",
                f"- manual review: {len(manual_rows)} matches",
            ]
        if not manual_rows.empty:
            manual_review_lines.extend(["", "Manual review:"])
            for _, review in manual_rows.iterrows():
                if terminal_verbosity == "compact":
                    manual_review_lines.append(
                        f"{review['match_id']} {review['team_a']} vs {review['team_b']}: "
                        f"default {review['default_score']}, alt {review.get('main_alternative_score') or 'none'}, "
                        f"reason {review.get('risk_notes') or review['decision_note']}"
                    )
                    continue
                manual_review_lines.extend(
                    [
                        f"{review['match_id']} {review['team_a']} vs {review['team_b']}",
                        f"- default: {review['default_score']}",
                        f"- market-consistent: {review.get('market_consistent_score') or 'n/a'}",
                        f"- main alternative: {review.get('main_alternative_score') or 'none'}",
                        f"- confidence: {review['confidence_level']}",
                        f"- note: {review['decision_note']}",
                    ]
                )
        elif terminal_verbosity == "compact":
            manual_review_lines.extend(["", "Manual review:", "none"])

    warning_lines: list[str] = []
    paste_warnings = {
        match_id: tuple(warning for warning in warnings if str(warning).strip())
        for match_id, warnings in result.paste_warnings.items()
    }
    paste_warnings = {match_id: warnings for match_id, warnings in paste_warnings.items() if warnings}
    if paste_warnings:
        warning_lines = ["", "Input warnings:"]
        for match_id, warnings in sorted(paste_warnings.items()):
            warning_lines.append(f"{match_id}: {', '.join(warnings)}")

    normal_extra: list[str] = []
    market_consistent_warning_extra: list[str] = []
    if terminal_verbosity == "normal" or (run_profile == "research" and terminal_verbosity != "debug"):
        for _, row in result.workflow.match_report.iterrows():
            lines = market_consistent_diagnostic_lines(row)
            if lines:
                market_consistent_warning_extra.extend(["", f"{row['match_id']} {row['team_a']} vs {row['team_b']}:", *lines])
    if terminal_verbosity == "normal" and not dashboard.empty:
        disagreements = dashboard[
            dashboard[["market_consistent_differs", "dixon_coles_differs", "public_strategy_differs"]]
            .astype(str)
            .eq("yes")
            .any(axis=1)
        ]
        clusters = dashboard[dashboard["high_score_cluster"].astype(str).eq("yes")]
        normal_extra = [
            "",
            "Model disagreement summary:",
            f"- matches with challenger disagreement: {len(disagreements)}",
            "High-score cluster summary:",
            f"- matches with high-score cluster: {len(clusters)}",
        ]

    sections = [
        *(diagnostic_sections if terminal_verbosity == "debug" else []),
        "Live Prediction Summary",
        f"- run profile: {run_profile}",
        f"- matches processed: {len(result.workflow.match_report)}",
        "",
        "Final recommendations:",
        *final_submissions,
        *normal_extra,
        *market_consistent_warning_extra,
        *warning_lines,
        *manual_review_lines,
        *dashboard_summary,
        *runtime_lines(),
        "",
        "Outputs:",
        f"- {result.settings.submission_xlsx_output_path}",
        *([f"- {result.settings.recommendations_xlsx_output_path}"] if result.settings.write_detailed_excel else []),
        *([f"- Weight sensitivity: {result.settings.weight_comparison_output_path}"] if result.weight_comparison is not None else []),
    ]
    return "\n".join(sections)
