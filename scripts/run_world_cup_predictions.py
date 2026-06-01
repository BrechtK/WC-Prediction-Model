"""Generate recommendations for real upcoming World Cup matches."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.world_cup import (
    WORLD_CUP_ODDS_MISSING_MESSAGE,
    WorldCupPredictionSettings,
    run_and_print_world_cup_predictions,
)


def main() -> None:
    """Run the real-tournament prediction workflow and print a compact summary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--odds")
    parser.add_argument("--csv-output", default="data/processed/world_cup_recommendations.csv")
    parser.add_argument("--xlsx-output", default="data/processed/world_cup_recommendations.xlsx")
    args = parser.parse_args()

    try:
        run_and_print_world_cup_predictions(
            WorldCupPredictionSettings(
                input_path=Path(args.odds) if args.odds else None,
                csv_output_path=Path(args.csv_output),
                xlsx_output_path=Path(args.xlsx_output),
            )
        )
    except FileNotFoundError:
        print(WORLD_CUP_ODDS_MISSING_MESSAGE)


if __name__ == "__main__":
    main()
