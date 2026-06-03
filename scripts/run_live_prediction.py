"""Run the complete live OddsPortal paste-to-submission workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.config import ProjectConfig
from wc_predictor.live_prediction import (
    LivePredictionError,
    LivePredictionSettings,
    format_live_prediction_summary,
    run_live_prediction,
)
from wc_predictor.oddsportal_combined import (
    CombinedOddsPortalPasteError,
    format_combined_oddsportal_split_summary,
    split_combined_oddsportal_pastes,
)
from wc_predictor.oddsportal_schedule import (
    DEFAULT_SCHEDULE_OUTPUT_PATH,
    DEFAULT_SCHEDULE_REPORT_PATH,
    OddsPortalScheduleParseError,
    format_oddsportal_schedule_parse_summary,
    prepare_schedule_metadata,
)
from wc_predictor.paths import (
    CACHE_SPLIT_PASTES_DIR,
    INPUT_ODDS_DIR,
    INPUT_SCHEDULE_PATH,
)

NO_ODDS_INPUT_MESSAGE = (
    "No odds input files found. Create files like input/odds/M001.txt with sections "
    "### 1X2, ### OVER_UNDER, ### BTTS, ### CORRECT_SCORE."
)


def _user_facing_live_error(error: Exception) -> str:
    """Translate internal cache errors into instructions for the live workflow."""

    message = str(error)
    if isinstance(error, LivePredictionError) and message.startswith(
        "No recognised OddsPortal paste files found below "
    ):
        return NO_ODDS_INPUT_MESSAGE
    return message


def main() -> None:
    """Parse live pastes, export recommendations, and print the final submission."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id")
    parser.add_argument("--correct-score-poisson-weight", type=float, default=0.85)
    parser.add_argument("--correct-score-aggregation-method", default="auto")
    parser.add_argument("--dixon-coles-rho", type=float, default=0.0)
    parser.add_argument("--skip-weight-sensitivity", action="store_true")
    parser.add_argument("--skip-combined-split", action="store_true")
    parser.add_argument("--overwrite-combined-split", action="store_true")
    parser.add_argument("--combined-input-folder")
    parser.add_argument("--paste-input-folder")
    parser.add_argument("--schedule")
    parser.add_argument("--schedule-output", default=str(DEFAULT_SCHEDULE_OUTPUT_PATH))
    parser.add_argument("--schedule-report-output", default=str(DEFAULT_SCHEDULE_REPORT_PATH))
    parser.add_argument("--skip-schedule-parse", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    try:
        schedule_path = Path(args.schedule) if args.schedule else INPUT_SCHEDULE_PATH
        combined_input_folder = (
            Path(args.combined_input_folder)
            if args.combined_input_folder
            else INPUT_ODDS_DIR
        )
        paste_input_folder = (
            Path(args.paste_input_folder)
            if args.paste_input_folder
            else CACHE_SPLIT_PASTES_DIR
        )
        if not schedule_path.exists():
            raise OddsPortalScheduleParseError(
                f"{schedule_path} is required. "
                "Paste the schedule there before running live predictions."
            )
        prepared_schedule = prepare_schedule_metadata(
            schedule_path,
            args.schedule_output,
            args.schedule_report_output,
            skip_parse=args.skip_schedule_parse,
        )
        if prepared_schedule.parse_result is not None:
            print(
                format_oddsportal_schedule_parse_summary(
                    prepared_schedule.parse_result,
                    args.schedule_output,
                    args.schedule_report_output,
                )
            )
            print()
        if not args.skip_combined_split:
            split_result = split_combined_oddsportal_pastes(
                combined_input_folder,
                paste_input_folder,
                overwrite=args.overwrite_combined_split or combined_input_folder == INPUT_ODDS_DIR,
                schedule=prepared_schedule.schedule,
            )
            if split_result.files_processed:
                print(format_combined_oddsportal_split_summary(split_result))
                print()
        result = run_live_prediction(
            LivePredictionSettings(
                input_folder=paste_input_folder,
                match_id=args.match_id,
                strict=args.strict,
                skip_weight_sensitivity=args.skip_weight_sensitivity,
                metadata_odds_path=prepared_schedule.metadata_path,
                schedule_input_path=schedule_path,
                schedule_output_path=args.schedule_output,
                schedule_parse_report_path=args.schedule_report_output,
                skip_schedule_parse=True,
            ),
            ProjectConfig(
                correct_score_poisson_weight=args.correct_score_poisson_weight,
                correct_score_aggregation_method=args.correct_score_aggregation_method,
                dixon_coles_rho=args.dixon_coles_rho,
            ),
        )
    except (CombinedOddsPortalPasteError, LivePredictionError, OddsPortalScheduleParseError, ValueError) as error:
        raise SystemExit(_user_facing_live_error(error)) from error
    print(format_live_prediction_summary(result))


if __name__ == "__main__":
    main()
