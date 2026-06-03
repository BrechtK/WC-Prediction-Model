"""Parse one manually pasted OddsPortal-style fixture schedule."""

from __future__ import annotations

import argparse

from wc_predictor.oddsportal_schedule import (
    DEFAULT_EXISTING_METADATA_PATH,
    DEFAULT_SCHEDULE_INPUT_PATH,
    DEFAULT_SCHEDULE_OUTPUT_PATH,
    DEFAULT_SCHEDULE_REPORT_PATH,
    OddsPortalScheduleParseError,
    format_oddsportal_schedule_parse_summary,
    parse_oddsportal_schedule_file,
)


def main() -> None:
    """Parse the pasted schedule, export metadata, and print a compact summary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_SCHEDULE_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_SCHEDULE_OUTPUT_PATH))
    parser.add_argument("--report-output", default=str(DEFAULT_SCHEDULE_REPORT_PATH))
    parser.add_argument("--existing-metadata", default=str(DEFAULT_EXISTING_METADATA_PATH))
    args = parser.parse_args()

    try:
        result = parse_oddsportal_schedule_file(
            args.input,
            args.output,
            args.report_output,
            existing_metadata_path=args.existing_metadata,
        )
    except (FileNotFoundError, OddsPortalScheduleParseError) as error:
        raise SystemExit(str(error)) from error
    print(format_oddsportal_schedule_parse_summary(result, args.output, args.report_output))


if __name__ == "__main__":
    main()
