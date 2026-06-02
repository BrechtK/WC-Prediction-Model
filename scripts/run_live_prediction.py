"""Run the complete live OddsPortal paste-to-submission workflow."""

from __future__ import annotations

import argparse

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
    DEFAULT_SCHEDULE_INPUT_PATH,
    DEFAULT_SCHEDULE_OUTPUT_PATH,
    DEFAULT_SCHEDULE_REPORT_PATH,
    OddsPortalScheduleParseError,
    format_oddsportal_schedule_parse_summary,
    prepare_schedule_metadata,
)


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
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE_INPUT_PATH))
    parser.add_argument("--schedule-output", default=str(DEFAULT_SCHEDULE_OUTPUT_PATH))
    parser.add_argument("--schedule-report-output", default=str(DEFAULT_SCHEDULE_REPORT_PATH))
    parser.add_argument("--skip-schedule-parse", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    try:
        prepared_schedule = prepare_schedule_metadata(
            args.schedule,
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
                overwrite=args.overwrite_combined_split,
                schedule=prepared_schedule.schedule,
            )
            if split_result.files_processed:
                print(format_combined_oddsportal_split_summary(split_result))
                print()
        result = run_live_prediction(
            LivePredictionSettings(
                match_id=args.match_id,
                strict=args.strict,
                skip_weight_sensitivity=args.skip_weight_sensitivity,
                metadata_odds_path=prepared_schedule.metadata_path,
                schedule_input_path=args.schedule,
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
        raise SystemExit(str(error)) from error
    print(format_live_prediction_summary(result))


if __name__ == "__main__":
    main()
