"""Compare Dixon-Coles challenger recommendations across rho values."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.config import ProjectConfig
from wc_predictor.dixon_coles_rho_comparison import (
    DEFAULT_DIXON_COLES_RHOS,
    DEFAULT_OUTPUT_PATH,
    compare_live_dixon_coles_rhos,
    export_dixon_coles_rho_comparison,
    format_dixon_coles_rho_comparison_summary,
    parse_rho_grid,
)
from wc_predictor.live_prediction import LivePredictionError, LivePredictionSettings
from wc_predictor.oddsportal_core import DEFAULT_INPUT_FOLDER, DEFAULT_METADATA_ODDS_PATH


def main() -> None:
    """Parse the current live pastes, export rho sensitivity, and print a summary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder")
    parser.add_argument("--metadata-odds", default=str(DEFAULT_METADATA_ODDS_PATH))
    parser.add_argument("--match-id")
    parser.add_argument("--rhos", help="Comma-separated rho values, for example: -0.20,-0.10,0,0.10,0.20")
    parser.add_argument("--correct-score-poisson-weight", type=float, default=0.85)
    parser.add_argument("--correct-score-aggregation-method", default="auto")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    rhos = parse_rho_grid(args.rhos) if args.rhos else DEFAULT_DIXON_COLES_RHOS
    input_folder = Path(args.input_folder) if args.input_folder else DEFAULT_INPUT_FOLDER
    try:
        comparison = compare_live_dixon_coles_rhos(
            LivePredictionSettings(
                input_folder=input_folder,
                metadata_odds_path=Path(args.metadata_odds) if args.metadata_odds else None,
                match_id=args.match_id,
                strict=args.strict,
            ),
            rhos=rhos,
            config=ProjectConfig(
                correct_score_poisson_weight=args.correct_score_poisson_weight,
                correct_score_aggregation_method=args.correct_score_aggregation_method,
            ),
        )
        output_path = export_dixon_coles_rho_comparison(comparison, args.output)
    except (LivePredictionError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(format_dixon_coles_rho_comparison_summary(comparison, output_path))


if __name__ == "__main__":
    main()
