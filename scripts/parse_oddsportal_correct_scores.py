"""Convert manually pasted OddsPortal correct-score text into model-ready CSV."""

from __future__ import annotations

import argparse

from wc_predictor.oddsportal import (
    DEFAULT_INPUT_FOLDER,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REPORT_PATH,
    format_oddsportal_parse_summary,
    parse_oddsportal_correct_score_folder,
)


def main() -> None:
    """Parse all raw paste files and print a compact quality summary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", default=str(DEFAULT_INPUT_FOLDER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--odds-timestamp", default="")
    parser.add_argument("--odds-source-url", default="")
    parser.add_argument("--has-other-bucket", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    result = parse_oddsportal_correct_score_folder(
        args.input_folder,
        args.output,
        args.report_output,
        odds_timestamp=args.odds_timestamp,
        odds_source_url=args.odds_source_url,
        has_other_bucket=args.has_other_bucket,
        notes=args.notes,
    )
    print(format_oddsportal_parse_summary(result, args.output))
    print(f"- Parse report: {args.report_output}")


if __name__ == "__main__":
    main()
