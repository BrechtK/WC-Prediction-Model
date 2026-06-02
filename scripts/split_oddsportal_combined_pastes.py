"""Split combined per-match OddsPortal pastes into existing parser input files."""

from __future__ import annotations

import argparse

from wc_predictor.oddsportal_combined import (
    DEFAULT_COMBINED_INPUT_FOLDER,
    DEFAULT_SPLIT_OUTPUT_FOLDER,
    CombinedOddsPortalPasteError,
    format_combined_oddsportal_split_summary,
    split_combined_oddsportal_pastes,
)


def main() -> None:
    """Split marked combined pastes and print a compact summary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", default=str(DEFAULT_COMBINED_INPUT_FOLDER))
    parser.add_argument("--output-folder", default=str(DEFAULT_SPLIT_OUTPUT_FOLDER))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        result = split_combined_oddsportal_pastes(
            args.input_folder,
            args.output_folder,
            overwrite=args.overwrite,
        )
    except CombinedOddsPortalPasteError as error:
        raise SystemExit(str(error)) from error
    print(format_combined_oddsportal_split_summary(result))


if __name__ == "__main__":
    main()
