from pathlib import Path
import runpy
import sys

from wc_predictor.config import ProjectConfig, PublicStrategyConfig
from wc_predictor.paths import CACHE_CORE_ODDS_PATH
from wc_predictor.public_contest import simulate_public_contest
from wc_predictor.world_cup import WorldCupPredictionSettings, run_world_cup_predictions


EXAMPLES = Path("data/examples")
RUN_SCRIPT = Path("scripts/simulate_public_contest.py").resolve()


def _workflow(tmp_path: Path):
    return run_world_cup_predictions(
        WorldCupPredictionSettings(
            input_path=EXAMPLES / "example_world_cup_odds.csv",
            csv_output_path=tmp_path / "predictions.csv",
            xlsx_output_path=tmp_path / "predictions.xlsx",
            submission_xlsx_output_path=tmp_path / "submission.xlsx",
        ),
        ProjectConfig(public_strategy=PublicStrategyConfig(mode="public-ranking")),
    )


def test_public_contest_simulation_reports_leaderboard_probabilities(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)

    simulation = simulate_public_contest(
        workflow,
        config=ProjectConfig(public_strategy=PublicStrategyConfig(mode="public-ranking")),
        field_size=25,
        simulations=20,
        seed=7,
    )

    assert simulation.strategy_summary["strategy"].tolist() == ["pure_ev", "public_strategy"]
    assert {
        "average_total_points",
        "probability_top_50",
        "probability_top_10",
        "probability_rank_1",
    }.issubset(simulation.strategy_summary.columns)
    assert len(simulation.match_setup) == len(workflow.match_report)
    assert len(simulation.simulation_totals) == 40


def test_public_contest_script_writes_workbook_from_cached_live_odds(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cache_odds = tmp_path / CACHE_CORE_ODDS_PATH
    cache_odds.parent.mkdir(parents=True)
    cache_odds.write_bytes((EXAMPLES / "example_world_cup_odds.csv").read_bytes())
    output_path = tmp_path / "public_contest.xlsx"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulate_public_contest.py",
            "--field-size",
            "10",
            "--simulations",
            "5",
            "--output",
            str(output_path),
        ],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    assert output_path.exists()
    assert "Public Contest Simulation" in capsys.readouterr().out
