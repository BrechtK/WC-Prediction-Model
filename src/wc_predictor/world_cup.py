"""Real-tournament recommendation workflow built on the Version 1 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile

from wc_predictor.config import ProjectConfig
from wc_predictor.market_data import load_correct_score_odds, load_odds, load_total_goals_odds
from wc_predictor.paths import (
    INPUT_PREPARED_WORLD_CUP_ODDS_CSV_PATH,
    INPUT_PREPARED_WORLD_CUP_ODDS_XLSX_PATH,
    OUTPUT_PREDICTIONS_CSV_PATH,
    OUTPUT_PREDICTIONS_XLSX_PATH,
    OUTPUT_SUBMISSION_XLSX_PATH,
)
from wc_predictor.reporting import (
    export_dataframe,
    export_world_cup_recommendations_excel,
    export_world_cup_submission_sheet_excel,
    format_world_cup_console_summary,
)
from wc_predictor.workflow import PredictionWorkflowResult, run_prediction_workflow

DEFAULT_WORLD_CUP_XLSX_INPUT_PATH = INPUT_PREPARED_WORLD_CUP_ODDS_XLSX_PATH
DEFAULT_WORLD_CUP_CSV_INPUT_PATH = INPUT_PREPARED_WORLD_CUP_ODDS_CSV_PATH
DEFAULT_WORLD_CUP_TEMPLATE_PATH = Path("data/templates/world_cup_odds_template.xlsx")
WORLD_CUP_ODDS_MISSING_MESSAGE = (
    "No World Cup odds file found. Run scripts/create_world_cup_odds_file.py first, "
    "fill in input/prepared/world_cup_odds.xlsx, then rerun predictions."
)


@dataclass(frozen=True)
class WorldCupPredictionSettings:
    """Input and output paths for a real World Cup recommendation run."""

    input_path: Path | None = None
    correct_score_input_path: Path | None = None
    total_goals_input_path: Path | None = None
    csv_output_path: Path = OUTPUT_PREDICTIONS_CSV_PATH
    xlsx_output_path: Path = OUTPUT_PREDICTIONS_XLSX_PATH
    submission_xlsx_output_path: Path = OUTPUT_SUBMISSION_XLSX_PATH


def resolve_world_cup_odds_input(input_path: str | Path | None = None) -> Path:
    """Resolve an explicit odds path or prefer the user-friendly Excel default."""

    if input_path is not None:
        path = Path(input_path)
        if path.exists():
            return path
        raise FileNotFoundError(WORLD_CUP_ODDS_MISSING_MESSAGE)
    for path in (DEFAULT_WORLD_CUP_XLSX_INPUT_PATH, DEFAULT_WORLD_CUP_CSV_INPUT_PATH):
        if path.exists():
            return path
    raise FileNotFoundError(WORLD_CUP_ODDS_MISSING_MESSAGE)


def create_world_cup_odds_file(
    output_path: str | Path = DEFAULT_WORLD_CUP_XLSX_INPUT_PATH,
    template_path: str | Path = DEFAULT_WORLD_CUP_TEMPLATE_PATH,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    """Copy the Excel odds template into the raw-data folder when requested."""

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path, False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    copyfile(template_path, output_path)
    return output_path, True


def run_world_cup_predictions(
    settings: WorldCupPredictionSettings | None = None,
    config: ProjectConfig | None = None,
) -> PredictionWorkflowResult:
    """Generate and export recommendations for upcoming tournament matches."""

    settings = settings or WorldCupPredictionSettings()
    input_path = resolve_world_cup_odds_input(settings.input_path)
    correct_score_odds = (
        load_correct_score_odds(settings.correct_score_input_path)
        if settings.correct_score_input_path is not None
        else None
    )
    total_goals_odds = (
        load_total_goals_odds(settings.total_goals_input_path)
        if settings.total_goals_input_path is not None
        else None
    )
    workflow = run_prediction_workflow(
        load_odds(input_path),
        config=config,
        correct_score_odds=correct_score_odds,
        total_goals_odds=total_goals_odds,
    )
    export_dataframe(workflow.match_report, settings.csv_output_path)
    export_world_cup_recommendations_excel(workflow.match_report, settings.xlsx_output_path)
    export_world_cup_submission_sheet_excel(workflow.match_report, settings.submission_xlsx_output_path)
    return workflow


def run_and_print_world_cup_predictions(
    settings: WorldCupPredictionSettings | None = None,
    config: ProjectConfig | None = None,
) -> PredictionWorkflowResult:
    """Generate, export, and print upcoming tournament recommendations."""

    settings = settings or WorldCupPredictionSettings()
    input_path = resolve_world_cup_odds_input(settings.input_path)
    resolved_settings = WorldCupPredictionSettings(
        input_path=input_path,
        correct_score_input_path=settings.correct_score_input_path,
        total_goals_input_path=settings.total_goals_input_path,
        csv_output_path=settings.csv_output_path,
        xlsx_output_path=settings.xlsx_output_path,
        submission_xlsx_output_path=settings.submission_xlsx_output_path,
    )
    workflow = run_world_cup_predictions(resolved_settings, config)
    print(
        format_world_cup_console_summary(
            workflow.match_report,
            input_path,
            settings.csv_output_path,
            settings.xlsx_output_path,
            settings.submission_xlsx_output_path,
        )
    )
    return workflow
