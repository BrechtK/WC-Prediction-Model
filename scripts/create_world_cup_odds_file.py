"""Create a manual-entry World Cup odds workbook from the Excel template."""

from __future__ import annotations

import argparse

from wc_predictor.world_cup import create_world_cup_odds_file


def main() -> None:
    """Create the raw-data workbook while preserving existing manual entries."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/world_cup_odds.xlsx")
    parser.add_argument("--template", default="data/templates/world_cup_odds_template.xlsx")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    path, changed = create_world_cup_odds_file(args.output, args.template, args.overwrite)
    if changed:
        action = "Created" if not args.overwrite else "Created or replaced"
        print(f"{action} World Cup odds file: {path}")
    else:
        print(f"World Cup odds file already exists; left unchanged: {path}")
    print(f"Fill in {path}, then run scripts/run_world_cup_predictions.py.")


if __name__ == "__main__":
    main()
