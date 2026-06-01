"""Real-tournament recommendation workflow built on the Version 1 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wc_predictor.config import ProjectConfig
from wc_predictor.market_data import load_odds
from wc_predictor.reporting import export_dataframe, format_world_cup_console_summary
from wc_predictor.workflow import PredictionWorkflowResult, run_prediction_workflow


@dataclass(frozen=True)
class WorldCupPredictionSettings:
    """Input and output paths for a real World Cup recommendation run."""

    input_path: Path = Path("data/raw/world_cup_odds.csv")
    csv_output_path: Path = Path("data/processed/world_cup_recommendations.csv")
    xlsx_output_path: Path = Path("data/processed/world_cup_recommendations.xlsx")


def run_world_cup_predictions(
    settings: WorldCupPredictionSettings | None = None,
    config: ProjectConfig | None = None,
) -> PredictionWorkflowResult:
    """Generate and export recommendations for upcoming tournament matches."""

    settings = settings or WorldCupPredictionSettings()
    workflow = run_prediction_workflow(load_odds(settings.input_path), config=config)
    export_dataframe(workflow.match_report, settings.csv_output_path)
    export_dataframe(workflow.match_report, settings.xlsx_output_path)
    return workflow


def run_and_print_world_cup_predictions(
    settings: WorldCupPredictionSettings | None = None,
    config: ProjectConfig | None = None,
) -> PredictionWorkflowResult:
    """Generate, export, and print upcoming tournament recommendations."""

    settings = settings or WorldCupPredictionSettings()
    workflow = run_world_cup_predictions(settings, config)
    print(
        format_world_cup_console_summary(
            workflow.match_report,
            settings.input_path,
            settings.csv_output_path,
            settings.xlsx_output_path,
        )
    )
    return workflow
