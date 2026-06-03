"""Run the research historical World Cup backtest harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.backtesting import (
    HistoricalWorldCupCSVLoader,
    WorldCupResearchBacktestRunner,
    WorldCupResearchBacktestSettings,
)


DEFAULT_INPUT = Path("input/historical/2022/world_cup_matches.csv")
TEMPLATE = Path("templates/historical_world_cup_matches_template.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--summary-output", default="output/research/world_cup_backtest_summary.csv")
    parser.add_argument("--predictions-output", default="output/research/world_cup_backtest_predictions.csv")
    parser.add_argument("--skipped-output", default="output/research/world_cup_backtest_skipped.csv")
    parser.add_argument("--excel-output", default="output/research/world_cup_backtest.xlsx")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(
            "\n".join(
                [
                    f"Historical World Cup input not found: {input_path}",
                    "",
                    "Create a legally safe local CSV using the template:",
                    str(TEMPLATE),
                    "",
                    "Suggested 2022 path:",
                    str(DEFAULT_INPUT),
                ]
            )
        )
        return

    report = WorldCupResearchBacktestRunner(
        HistoricalWorldCupCSVLoader(input_path),
        settings=WorldCupResearchBacktestSettings(
            summary_output_path=Path(args.summary_output),
            predictions_output_path=Path(args.predictions_output),
            skipped_output_path=Path(args.skipped_output),
            excel_output_path=Path(args.excel_output),
        ),
    ).run()
    print(
        "\n".join(
            [
                "Historical World Cup Backtest",
                f"- strategy/config rows: {len(report.summary)}",
                f"- prediction rows: {len(report.predictions)}",
                f"- skipped diagnostics: {len(report.skipped)}",
                f"- summary: {args.summary_output}",
                f"- workbook: {args.excel_output}",
            ]
        )
    )


if __name__ == "__main__":
    main()
