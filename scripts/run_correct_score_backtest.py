"""Rank optional correct-score Poisson blend weights using realised pool points."""

from __future__ import annotations

import argparse

from wc_predictor.correct_score_backtesting import backtest_correct_score_blend_weights
from wc_predictor.market_data import load_correct_score_odds, load_odds, load_results


def main() -> None:
    """Run the correct-score blend-weight sweep and print its ranking."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--odds", default="data/examples/example_odds.csv")
    parser.add_argument("--correct-score-odds", default="data/examples/example_correct_score_odds.csv")
    parser.add_argument("--results", default="data/examples/example_results.csv")
    parser.add_argument("--output", default="data/processed/correct_score_blend_backtest.csv")
    args = parser.parse_args()

    report = backtest_correct_score_blend_weights(
        load_odds(args.odds),
        load_correct_score_odds(args.correct_score_odds),
        load_results(args.results),
    )
    report.export_csv(args.output)
    print("Correct-Score Blend-Weight Ranking")
    print(report.summary.to_string(index=False))
    print(f"\nRanking written to {args.output}")


if __name__ == "__main__":
    main()

