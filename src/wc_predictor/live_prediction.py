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
    _weight_column,
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
    strict: bool = False
    skip_weight_sensitivity: bool = False
    metadata_odds_path: Path | None = DEFAULT_METADATA_ODDS_PATH
    core_odds_output_path: Path = DEFAULT_CORE_OUTPUT_PATH
    total_goals_output_path: Path = DEFAULT_TOTAL_GOALS_OUTPUT_PATH
    core_parse_report_path: Path = DEFAULT_CORE_REPORT_PATH
    correct_score_output_path: Path = DEFAULT_CORRECT_SCORE_OUTPUT_PATH
    correct_score_parse_report_path: Path = DEFAULT_CORRECT_SCORE_REPORT_PATH
    recommendations_csv_output_path: Path = Path("data/processed/world_cup_recommendations.csv")
    recommendations_xlsx_output_path: Path = Path("data/processed/world_cup_recommendations.xlsx")
    submission_xlsx_output_path: Path = Path("data/processed/world_cup_submission_sheet.xlsx")
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
    strict: bool,
) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    """Choose matches and fail early when required paste files are absent."""

    discovered = discover_live_paste_markets(input_folder)
    if match_id is not None:
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
    selected, mutable_warnings = _select_and_validate_pastes(settings.input_folder, settings.match_id, settings.strict)
    core_parse = parse_oddsportal_core_odds_folder(
        settings.input_folder,
        settings.core_odds_output_path,
        settings.core_parse_report_path,
        total_goals_output_path=settings.total_goals_output_path,
        metadata_odds_path=settings.metadata_odds_path,
        match_ids=selected,
    )
    correct_score_parse = parse_oddsportal_correct_score_folder(
        settings.input_folder,
        settings.correct_score_output_path,
        settings.correct_score_parse_report_path,
        match_ids=selected,
    )
    _validate_parsed_markets(selected, core_parse, correct_score_parse, settings.strict, mutable_warnings)

    has_total_goals = not core_parse.total_goals_odds.empty
    has_correct_scores = not correct_score_parse.odds.empty
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
        for selected_match_id in selected:
            _append_warning(mutable_warnings, selected_match_id, "weight_sensitivity_skipped:no_correct_score_odds")

    return LivePredictionResult(
        settings,
        core_parse,
        correct_score_parse,
        workflow,
        weight_comparison,
        {match_id: tuple(values) for match_id, values in mutable_warnings.items()},
    )


def _format_value(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if pd.notna(value) else "n/a"


def _top_scorelines(value: object, limit: int = 5) -> str:
    if pd.isna(value) or not str(value).strip():
        return "none"
    return "; ".join(str(value).split("; ")[:limit])


def _btts_bookmakers(workflow: PredictionWorkflowResult, match_id: str) -> str:
    rows = workflow.bookmaker_probabilities
    if "fair_btts_yes" not in rows:
        return "none"
    rows = rows[(rows["match_id"].astype(str) == match_id) & rows["fair_btts_yes"].notna()]
    bookmakers = sorted(set(rows["bookmaker"].astype(str)))
    return "; ".join(bookmakers) if bookmakers else "none"


def _sensitivity_lines(comparison: CorrectScoreWeightComparison | None, match_id: str) -> list[str]:
    if comparison is None:
        return ["  Weight sensitivity: not run"]
    rows = comparison.sensitivity[comparison.sensitivity["match_id"].astype(str) == match_id]
    if rows.empty:
        return ["  Weight sensitivity: no correct-score market rows"]
    row = rows.iloc[0]
    return [
        "  Weight sensitivity:",
        f"    w=1.00: {row[_weight_column(1.00)]}",
        f"    w=0.85: {row[_weight_column(0.85)]}",
        f"    w=0.75: {row[_weight_column(0.75)]}",
        f"    w=0.50: {row[_weight_column(0.50)]}",
        f"    w=0.00: {row[_weight_column(0.00)]}",
        f"    stable range around w=0.85: {row['stable_weight_range_around_0_85']}",
        f"    changes across weights: {row['recommendation_changes']}",
    ]


def format_live_prediction_summary(result: LivePredictionResult) -> str:
    """Render a concise submission-focused live terminal report."""

    sections = ["# Live Prediction Summary"]
    final_submissions: list[str] = []
    for _, row in result.workflow.match_report.iterrows():
        match_id = str(row["match_id"])
        group = str(row["group"]) if pd.notna(row["group"]) and str(row["group"]).strip() else "n/a"
        qualifier = (
            str(row["recommended_qualifier"])
            if pd.notna(row["recommended_qualifier"]) and str(row["recommended_qualifier"]).strip()
            else "n/a"
        )
        warnings = list(result.paste_warnings.get(match_id, ()))
        warnings.extend(str(row["warning_flags"]).split("; ") if str(row["warning_flags"]).strip() else [])
        if str(row["correct_score_sparse_warning"]).strip():
            warnings.append(str(row["correct_score_sparse_warning"]))
        warnings = list(dict.fromkeys(filter(None, warnings)))
        sections.append(
            "\n".join(
                [
                    "",
                    f"Match: {match_id} | {row['team_a']} vs {row['team_b']}",
                    f"  Date / stage / group: {row['date']} / {row['stage']} / {group}",
                    "Market data:",
                    f"  1X2 bookmakers: {row['bookmakers_used'] or 'none'}",
                    f"  BTTS bookmakers: {_btts_bookmakers(result.workflow, match_id)}",
                    f"  Total-goals lines available: {row['total_goals_lines_available'] or 'none'}",
                    f"  Total-goals lines used for calibration: {row['total_goals_lines_used_for_calibration'] or 'none'}",
                    f"  Correct-score bookmakers used: {row['correct_score_bookmakers_count']}",
                    f"  Correct-score scorelines: {row['correct_score_scorelines_count']}",
                    f"  Correct-score aggregation: {row['correct_score_aggregation_method'] or 'none'}",
                    f"  Average correct-score overround: {_format_value(row['average_correct_score_overround'])}",
                    f"  Correct-score outliers: {row['outlier_count']}",
                    "Fair probabilities:",
                    f"  {row['team_a']} win: {row['market_a_win']:.2%}",
                    f"  Draw: {row['market_draw']:.2%}",
                    f"  {row['team_b']} win: {row['market_b_win']:.2%}",
                    f"  Favourite: {row['favourite_probability']:.2%} ({row['favourite_bucket']})",
                    "Model fit:",
                    f"  Lambdas: {row['lambda_a']:.4f} / {row['lambda_b']:.4f}",
                    f"  Calibration error: {row['calibration_loss']:.6f}",
                    f"  Total-goals line fit error: {_format_value(row['total_goals_line_fit_error'], 6)}",
                    f"  Score-matrix tail mass: {row['tail_probability_before_renormalisation']:.4%}",
                    "Recommendation:",
                    f"  Recommended score: {row['recommended_score']}",
                    f"  Recommended qualifier: {qualifier}",
                    f"  Best expected points: {row['best_expected_points']:.3f}",
                    f"  Most likely scoreline: {row['most_likely_scoreline']}",
                    f"  EV-optimal differs from modal: {'yes' if row['ev_optimal_differs_from_most_likely'] else 'no'}",
                    f"  Top 5 EV scorelines: {row['top_5_ev_predictions']}",
                    f"  EV gap best vs second-best: {_format_value(row['ev_gap_best_vs_second'], 3)}",
                    "Correct-score market diagnostics:",
                    f"  Top 5 market-implied scores: {_top_scorelines(row['correct_score_market_top_10'])}",
                    f"  Top 5 blended scores: {_top_scorelines(row['correct_score_blended_top_10'])}",
                    f"  KL divergence market vs Poisson: {_format_value(row['correct_score_kl_divergence'], 6)}",
                    *_sensitivity_lines(result.weight_comparison, match_id),
                    "Warnings:",
                    *(f"  - {warning}" for warning in warnings),
                    *(["  - none"] if not warnings else []),
                ]
            )
        )
        final_submissions.append(f"{match_id} {row['team_a']} vs {row['team_b']}: {row['recommended_score']}")

    sections.extend(
        [
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
    )
    return "\n".join(sections)
