"""Compare live recommendations across correct-score aggregation methods."""

from __future__ import annotations

import argparse

from wc_predictor.correct_score_comparison import (
    DEFAULT_CORRECT_SCORE_ODDS_PATH,
    DEFAULT_OUTPUT_PATH,
    compare_correct_score_aggregation_methods,
    export_correct_score_aggregation_comparison,
    format_correct_score_aggregation_comparison_summary,
)
from wc_predictor.world_cup import WORLD_CUP_ODDS_MISSING_MESSAGE


def main() -> None:
    """Run method comparison, export the workbook, and print sensitivity."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--odds")
    parser.add_argument("--correct-score-odds", default=str(DEFAULT_CORRECT_SCORE_ODDS_PATH))
    parser.add_argument("--correct-score-poisson-weight", type=float, default=0.85)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    try:
        comparison = compare_correct_score_aggregation_methods(
            args.odds,
            args.correct_score_odds,
            poisson_weight=args.correct_score_poisson_weight,
        )
        output_path = export_correct_score_aggregation_comparison(comparison, args.output)
        print(format_correct_score_aggregation_comparison_summary(comparison, output_path))
    except FileNotFoundError as error:
        print(str(error) if str(error) else WORLD_CUP_ODDS_MISSING_MESSAGE)


if __name__ == "__main__":
    main()
