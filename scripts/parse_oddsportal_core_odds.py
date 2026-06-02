"""Convert manually pasted OddsPortal core markets into model-ready CSV."""

from __future__ import annotations

import argparse

from wc_predictor.oddsportal_core import (
    DEFAULT_INPUT_FOLDER,
    DEFAULT_METADATA_ODDS_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_TOTAL_GOALS_OUTPUT_PATH,
    format_oddsportal_core_parse_summary,
    parse_oddsportal_core_odds_folder,
)


def main() -> None:
    """Parse pasted 1X2, BTTS, and O/U 2.5 files and print a compact summary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", default=str(DEFAULT_INPUT_FOLDER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--total-goals-output", default=str(DEFAULT_TOTAL_GOALS_OUTPUT_PATH))
    parser.add_argument("--metadata-odds", default=str(DEFAULT_METADATA_ODDS_PATH))
    parser.add_argument("--odds-timestamp", default="")
    parser.add_argument("--odds-source-url", default="")
    args = parser.parse_args()

    result = parse_oddsportal_core_odds_folder(
        args.input_folder,
        args.output,
        args.report_output,
        total_goals_output_path=args.total_goals_output,
        metadata_odds_path=args.metadata_odds,
        odds_timestamp=args.odds_timestamp,
        odds_source_url=args.odds_source_url,
    )
    print(format_oddsportal_core_parse_summary(result, args.output, args.total_goals_output))
    print(f"- Parse report: {args.report_output}")


if __name__ == "__main__":
    main()
