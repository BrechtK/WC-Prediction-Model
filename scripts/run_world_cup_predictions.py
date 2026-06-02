"""Generate recommendations for real upcoming World Cup matches."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.config import ProjectConfig
from wc_predictor.world_cup import (
    WORLD_CUP_ODDS_MISSING_MESSAGE,
    WorldCupPredictionSettings,
    run_and_print_world_cup_predictions,
)


def main() -> None:
    """Run the real-tournament prediction workflow and print a compact summary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--odds")
    parser.add_argument("--correct-score-odds")
    parser.add_argument("--total-goals-odds")
    parser.add_argument("--correct-score-poisson-weight", type=float, default=1.0)
    parser.add_argument("--correct-score-aggregation-method", default="auto")
    parser.add_argument("--csv-output", default="data/processed/world_cup_recommendations.csv")
    parser.add_argument("--xlsx-output", default="data/processed/world_cup_recommendations.xlsx")
    parser.add_argument("--submission-xlsx-output", default="data/processed/world_cup_submission_sheet.xlsx")
    args = parser.parse_args()

    try:
        run_and_print_world_cup_predictions(
            WorldCupPredictionSettings(
                input_path=Path(args.odds) if args.odds else None,
                correct_score_input_path=Path(args.correct_score_odds) if args.correct_score_odds else None,
                total_goals_input_path=Path(args.total_goals_odds) if args.total_goals_odds else None,
                csv_output_path=Path(args.csv_output),
                xlsx_output_path=Path(args.xlsx_output),
                submission_xlsx_output_path=Path(args.submission_xlsx_output),
            ),
            ProjectConfig(
                correct_score_poisson_weight=args.correct_score_poisson_weight,
                correct_score_aggregation_method=args.correct_score_aggregation_method,
            ),
        )
    except FileNotFoundError:
        if args.correct_score_odds and not Path(args.correct_score_odds).exists():
            print(f"Correct-score odds file not found: {args.correct_score_odds}")
        elif args.total_goals_odds and not Path(args.total_goals_odds).exists():
            print(f"Total-goals odds file not found: {args.total_goals_odds}")
        else:
            print(WORLD_CUP_ODDS_MISSING_MESSAGE)


if __name__ == "__main__":
    main()
