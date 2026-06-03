"""Simulate diagnostic public-field contest outcomes from live cached odds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wc_predictor.config import ProjectConfig, PublicStrategyConfig
from wc_predictor.market_data import load_correct_score_odds, load_odds, load_total_goals_odds
from wc_predictor.paths import (
    CACHE_CORE_ODDS_PATH,
    CACHE_CORRECT_SCORE_ODDS_PATH,
    CACHE_TOTAL_GOALS_ODDS_PATH,
    OUTPUT_PUBLIC_CONTEST_SIMULATION_XLSX_PATH,
)
from wc_predictor.public_contest import simulate_public_contest
from wc_predictor.workflow import run_prediction_workflow


def main() -> None:
    """Run a public-field leaderboard stress test without changing live picks."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--odds", default=str(CACHE_CORE_ODDS_PATH))
    parser.add_argument("--correct-score-odds", default=str(CACHE_CORRECT_SCORE_ODDS_PATH))
    parser.add_argument("--total-goals-odds", default=str(CACHE_TOTAL_GOALS_ODDS_PATH))
    parser.add_argument("--field-size", type=int, default=1000)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy-mode",
        choices=("balanced", "public-ranking", "aggressive-public-ranking"),
        default="public-ranking",
    )
    parser.add_argument("--output", default=str(OUTPUT_PUBLIC_CONTEST_SIMULATION_XLSX_PATH))
    args = parser.parse_args()

    odds_path = Path(args.odds)
    if not odds_path.exists():
        raise SystemExit(
            f"No cached odds found at {odds_path}. Run python scripts/run_live_prediction.py first."
        )
    correct_score_path = Path(args.correct_score_odds)
    total_goals_path = Path(args.total_goals_odds)
    config = ProjectConfig(public_strategy=PublicStrategyConfig(mode=args.strategy_mode))
    workflow = run_prediction_workflow(
        load_odds(odds_path),
        config=config,
        correct_score_odds=load_correct_score_odds(correct_score_path) if correct_score_path.exists() else None,
        total_goals_odds=load_total_goals_odds(total_goals_path) if total_goals_path.exists() else None,
    )
    simulation = simulate_public_contest(
        workflow,
        config=config,
        field_size=args.field_size,
        simulations=args.simulations,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        simulation.strategy_summary.to_excel(writer, sheet_name="strategy_summary", index=False)
        simulation.match_setup.to_excel(writer, sheet_name="match_setup", index=False)
        simulation.simulation_totals.to_excel(writer, sheet_name="simulation_totals", index=False)

    changes = int((workflow.match_report["public_strategy_score"] != workflow.match_report["recommended_score"]).sum())
    summary = simulation.strategy_summary.set_index("strategy")
    print("Public Contest Simulation")
    print(f"- matches: {len(workflow.match_report)}")
    print(f"- field size: {args.field_size}")
    print(f"- simulations: {args.simulations}")
    print(f"- matches where public strategy differs from pure EV: {changes}")
    print(f"- pure EV top-10 probability: {summary.loc['pure_ev', 'probability_top_10']:.2%}")
    print(
        "- public strategy top-10 probability: "
        f"{summary.loc['public_strategy', 'probability_top_10']:.2%}"
    )
    print(f"- output: {output_path}")


if __name__ == "__main__":
    main()
