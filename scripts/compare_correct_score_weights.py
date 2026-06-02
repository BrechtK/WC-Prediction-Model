"""Compare live recommendations across correct-score blend weights."""

from __future__ import annotations

import argparse

from wc_predictor.correct_score_weight_comparison import (
    DEFAULT_CORRECT_SCORE_ODDS_PATH,
    DEFAULT_CORRECT_SCORE_WEIGHTS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_WORLD_CUP_ODDS_PATH,
    compare_correct_score_weights,
    export_correct_score_weight_comparison,
    format_correct_score_weight_comparison_summary,
    parse_weight_grid,
)


def main() -> None:
    """Run live blend-weight sensitivity diagnostics and export the workbook."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--odds", default=str(DEFAULT_WORLD_CUP_ODDS_PATH))
    parser.add_argument("--correct-score-odds", default=str(DEFAULT_CORRECT_SCORE_ODDS_PATH))
    parser.add_argument("--total-goals-odds")
    parser.add_argument("--weights", help="Comma-separated Poisson weights, for example: 1,0.85,0.5,0")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    weights = parse_weight_grid(args.weights) if args.weights else DEFAULT_CORRECT_SCORE_WEIGHTS
    comparison = compare_correct_score_weights(
        args.odds,
        args.correct_score_odds,
        weights=weights,
        total_goals_odds_path=args.total_goals_odds,
    )
    output_path = export_correct_score_weight_comparison(comparison, args.output)
    print(format_correct_score_weight_comparison_summary(comparison, output_path))


if __name__ == "__main__":
    main()
