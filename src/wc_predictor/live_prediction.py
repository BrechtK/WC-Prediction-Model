"""One-click orchestration for live predictions from manually pasted odds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

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
    r"^(?P<match_id>.+)_(?P<market>1x2|over_under|btts|correct_score)\.txt$",
    re.IGNORECASE,
)
_OPTIONAL_MARKETS = ("over_under", "btts", "correct_score")


class LivePredictionError(RuntimeError):
    """Raised when live paste inputs cannot produce a trustworthy baseline."""


@dataclass(frozen=True)
class LivePredictionSettings:
    """Paths and runtime switches for one live-prediction run."""

    input_folder: Path = DEFAULT_INPUT_FOLDER
    match_id: str | None = None
    match_ids: Sequence[str] | None = None
    strict: bool = False
    skip_weight_sensitivity: bool = False
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
    recommendations_csv_output_path: Path = OUTPUT_PREDICTIONS_CSV_PATH
    recommendations_xlsx_output_path: Path = OUTPUT_PREDICTIONS_XLSX_PATH
    submission_xlsx_output_path: Path = OUTPUT_SUBMISSION_XLSX_PATH
    weight_comparison_output_path: Path = DEFAULT_WEIGHT_COMPARISON_OUTPUT_PATH
    weight_sensitivity_weights: Sequence[float] = DEFAULT_CORRECT_SCORE_WEIGHTS


@dataclass(frozen=True)
class LivePredictionResult:
    """Parser, workflow, and optional sensitivity outputs from a live run."""

    settings: LivePredictionSettings
    core_parse: OddsPortalCoreParseResult
    correct_score_parse: OddsPortalParseResult
    workflow: PredictionWorkflowResult
    weight_comparison: CorrectScoreWeightComparison | None
    paste_warnings: dict[str, tuple[str, ...]]
    schedule_metadata: pd.DataFrame


@dataclass(frozen=True)
class ParsedLivePredictionInputs:
    """Parsed OddsPortal inputs shared by live recommendations and diagnostics."""

    settings: LivePredictionSettings
    selected_match_ids: tuple[str, ...]
    core_parse: OddsPortalCoreParseResult
    correct_score_parse: OddsPortalParseResult
    paste_warnings: dict[str, tuple[str, ...]]
    schedule_metadata: pd.DataFrame
    schedule_parse: OddsPortalScheduleParseResult | None

    @property
    def has_total_goals(self) -> bool:
        return not self.core_parse.total_goals_odds.empty

    @property
    def has_correct_scores(self) -> bool:
        return not self.correct_score_parse.odds.empty


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
    config = config or ProjectConfig(correct_score_poisson_weight=0.85)
    parsed = parse_live_prediction_inputs(settings)
    has_total_goals = parsed.has_total_goals
    has_correct_scores = parsed.has_correct_scores
    mutable_warnings = {match_id: list(values) for match_id, values in parsed.paste_warnings.items()}
    workflow = run_world_cup_predictions(
        WorldCupPredictionSettings(
            input_path=settings.core_odds_output_path,
            total_goals_input_path=settings.total_goals_output_path if has_total_goals else None,
            correct_score_input_path=settings.correct_score_output_path if has_correct_scores else None,
            csv_output_path=settings.recommendations_csv_output_path,
            xlsx_output_path=settings.recommendations_xlsx_output_path,
            submission_xlsx_output_path=settings.submission_xlsx_output_path,
        ),
        config,
    )

    weight_comparison = None
    if has_correct_scores and not settings.skip_weight_sensitivity:
        weight_comparison = compare_correct_score_weights(
            settings.core_odds_output_path,
            settings.correct_score_output_path,
            weights=settings.weight_sensitivity_weights,
            config=config,
            total_goals_odds_path=settings.total_goals_output_path if has_total_goals else None,
        )
        export_correct_score_weight_comparison(weight_comparison, settings.weight_comparison_output_path)
    elif not has_correct_scores and not settings.skip_weight_sensitivity:
        for selected_match_id in parsed.selected_match_ids:
            _append_warning(mutable_warnings, selected_match_id, "weight_sensitivity_skipped:no_correct_score_odds")

    return LivePredictionResult(
        settings,
        parsed.core_parse,
        parsed.correct_score_parse,
        workflow,
        weight_comparison,
        {match_id: tuple(values) for match_id, values in mutable_warnings.items()},
        parsed.schedule_metadata,
    )


def parse_live_prediction_inputs(
    settings: LivePredictionSettings | None = None,
) -> ParsedLivePredictionInputs:
    """Parse and validate the live OddsPortal pastes without running recommendations."""

    settings = settings or LivePredictionSettings()
    prepared_schedule = prepare_schedule_metadata(
        settings.schedule_input_path,
        settings.schedule_output_path,
        settings.schedule_parse_report_path,
        existing_metadata_path=settings.metadata_odds_path,
        skip_parse=settings.skip_schedule_parse,
    )
    selected, mutable_warnings = _select_and_validate_pastes(
        settings.input_folder,
        settings.match_id,
        settings.match_ids,
        settings.strict,
    )
    core_parse = parse_oddsportal_core_odds_folder(
        settings.input_folder,
        settings.core_odds_output_path,
        settings.core_parse_report_path,
        total_goals_output_path=settings.total_goals_output_path,
        metadata_odds_path=prepared_schedule.metadata_path,
        match_ids=selected,
    )
    correct_score_parse = parse_oddsportal_correct_score_folder(
        settings.input_folder,
        settings.correct_score_output_path,
        settings.correct_score_parse_report_path,
        match_ids=selected,
    )
    _validate_parsed_markets(selected, core_parse, correct_score_parse, settings.strict, mutable_warnings)
    return ParsedLivePredictionInputs(
        settings,
        selected,
        core_parse,
        correct_score_parse,
        {match_id: tuple(values) for match_id, values in mutable_warnings.items()},
        prepared_schedule.schedule,
        prepared_schedule.parse_result,
    )


def format_live_prediction_summary(result: LivePredictionResult) -> str:
    """Render a concise submission-focused live terminal report."""

    def format_probability(value: object) -> str:
        return f"{float(value):.1%}" if pd.notna(value) else "n/a"

    diagnostic_sections: list[str] = []
    final_submissions: list[str] = []
    for _, row in result.workflow.match_report.iterrows():
        match_id = str(row["match_id"])
        btts_difference = (
            row["model_implied_btts_yes_probability"] - row["market_fair_btts_yes_probability"]
            if pd.notna(row.get("market_fair_btts_yes_probability"))
            else pd.NA
        )
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
                    f"- close alternatives: {row.get('close_alternatives') or 'none'}",
                    f"- note: {row.get('decision_note')}",
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
        final_submissions.append(f"{match_id} {row['team_a']} vs {row['team_b']}: {row['recommended_score']}")

    sections = [
        *diagnostic_sections,
        "",
        "Final recommended submission:",
        *final_submissions,
        "",
        "Full Excel outputs:",
        f"- Recommendations: {result.settings.recommendations_xlsx_output_path}",
        f"- Submission sheet: {result.settings.submission_xlsx_output_path}",
        (
            f"- Weight sensitivity: {result.settings.weight_comparison_output_path}"
            if result.weight_comparison is not None
            else "- Weight sensitivity: not written"
        ),
    ]
    return "\n".join(sections)
