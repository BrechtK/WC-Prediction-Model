"""Exploratory in-sample search over Poisson calibration weights.

This script is research-only. It searches historical realised pool points and
probability diagnostics in sample; results are likely overfit and must not be
used to change live defaults without separate out-of-sample validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.calibration_weight_study import (
    DEFAULT_TOURNAMENT_FOLDERS,
    ROBUSTNESS_PROFILES,
    SEARCH_LABEL,
    SEARCH_OUTPUT_PATH,
    SEARCH_TOP_PROFILES_OUTPUT_PATH,
    add_search_rankings,
    make_search_profiles,
    run_calibration_weight_study,
    write_calibration_weight_outputs,
)
from wc_predictor.utils import ensure_parent_directory


def _parse_grid(value: str | None, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return fallback
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an explicitly in-sample exploratory calibration-weight search."
    )
    parser.add_argument(
        "--tournament-folder",
        action="append",
        dest="tournament_folders",
        help="Historical tournament folder. Repeat to run several; defaults to wc2014/wc2018/wc2022.",
    )
    parser.add_argument("--cache-folder", default="cache/calibration_weight_search")
    parser.add_argument("--output", default=str(SEARCH_OUTPUT_PATH))
    parser.add_argument("--top-output", default=str(SEARCH_TOP_PROFILES_OUTPUT_PATH))
    parser.add_argument("--one-x-two-grid", help="Comma-separated grid, e.g. 0.75,1,1.25")
    parser.add_argument("--totals-grid", help="Comma-separated grid, e.g. 0.25,0.5,0.75")
    parser.add_argument("--btts-grid", help="Comma-separated grid, e.g. 0,0.25,0.5")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    from wc_predictor.calibration_weight_study import (
        SEARCH_BTTS_GRID,
        SEARCH_ONE_X_TWO_GRID,
        SEARCH_TOTALS_GRID,
    )

    args = parse_args()
    folders = (
        tuple(Path(folder) for folder in args.tournament_folders)
        if args.tournament_folders
        else DEFAULT_TOURNAMENT_FOLDERS
    )
    search_profiles = make_search_profiles(
        one_x_two_grid=_parse_grid(args.one_x_two_grid, SEARCH_ONE_X_TWO_GRID),
        totals_grid=_parse_grid(args.totals_grid, SEARCH_TOTALS_GRID),
        btts_grid=_parse_grid(args.btts_grid, SEARCH_BTTS_GRID),
    )
    profiles = (ROBUSTNESS_PROFILES[0], *search_profiles)
    _, summary, by_tournament = run_calibration_weight_study(
        profiles,
        tournament_folders=folders,
        cache_folder=args.cache_folder,
        include_market_consistent=False,
        progress=not args.quiet,
    )
    ranked = add_search_rankings(summary, by_tournament)
    write_calibration_weight_outputs(ranked, by_tournament, summary_path=args.output)
    top = ranked.head(max(1, args.top_n))
    ensure_parent_directory(Path(args.top_output))
    top.to_csv(args.top_output, index=False)

    print("Calibration-weight exploratory search complete")
    print(f"  Warning   : {SEARCH_LABEL}")
    print(f"  Full grid : {args.output}")
    print(f"  Top rows  : {args.top_output}")
    if not top.empty:
        columns = [
            "search_rank_by_points",
            "profile",
            "ev_default_points",
            "ev_points_delta_vs_current",
            "mean_log_loss_1x2",
            "ev_picks_changed_vs_current_count",
            "improves_or_matches_current_all_tournaments",
        ]
        print(top[[column for column in columns if column in top]].to_string(index=False))


if __name__ == "__main__":
    main()
