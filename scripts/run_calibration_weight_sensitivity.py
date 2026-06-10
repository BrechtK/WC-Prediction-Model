"""Run the pre-specified calibration-weight robustness sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.calibration_weight_study import (
    DEFAULT_TOURNAMENT_FOLDERS,
    ROBUSTNESS_PROFILES,
    SENSITIVITY_BY_TOURNAMENT_OUTPUT_PATH,
    SENSITIVITY_OUTPUT_PATH,
    run_calibration_weight_study,
    write_calibration_weight_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pre-specified robustness profiles for Poisson calibration weights."
    )
    parser.add_argument(
        "--tournament-folder",
        action="append",
        dest="tournament_folders",
        help="Historical tournament folder. Repeat to run several; defaults to wc2014/wc2018/wc2022.",
    )
    parser.add_argument("--cache-folder", default="cache/calibration_weight_sensitivity")
    parser.add_argument("--output", default=str(SENSITIVITY_OUTPUT_PATH))
    parser.add_argument("--by-tournament-output", default=str(SENSITIVITY_BY_TOURNAMENT_OUTPUT_PATH))
    parser.add_argument(
        "--no-market-consistent",
        action="store_true",
        help="Run only the baseline EV/modal profile configs; skip MC+AH diagnostics.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folders = (
        tuple(Path(folder) for folder in args.tournament_folders)
        if args.tournament_folders
        else DEFAULT_TOURNAMENT_FOLDERS
    )
    _, summary, by_tournament = run_calibration_weight_study(
        ROBUSTNESS_PROFILES,
        tournament_folders=folders,
        cache_folder=args.cache_folder,
        include_market_consistent=not args.no_market_consistent,
        progress=not args.quiet,
    )
    write_calibration_weight_outputs(
        summary,
        by_tournament,
        summary_path=args.output,
        by_tournament_path=args.by_tournament_output,
    )
    print("Calibration-weight robustness sweep complete")
    print(f"  Combined output     : {args.output}")
    print(f"  By-tournament output: {args.by_tournament_output}")
    if not summary.empty:
        columns = [
            "profile",
            "ev_default_points",
            "modal_points",
            "gated_mc_ah_points",
            "raw_mc_ah_points",
            "mean_expected_total_goals",
            "mean_actual_total_goals",
            "ev_picks_changed_vs_current_count",
        ]
        print(summary[[column for column in columns if column in summary]].to_string(index=False))


if __name__ == "__main__":
    main()
