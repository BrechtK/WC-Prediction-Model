"""Run the research historical World Cup backtest harness (CSV-odds version).

NOTE: There are two historical backtests. Pick the right one:

  * scripts/run_live_backtest.py  (RECOMMENDED)
      Uses the SAME pasted OddsPortal .txt files as the live tool, runs the
      full live pipeline (1X2, O/U, BTTS, correct score, Asian handicap,
      market-consistent challenger, Dixon-Coles / bivariate priors), and
      scores every strategy. This is the one to use if you collect odds by
      pasting from OddsPortal.

  * THIS script (run_historical_world_cup_backtest.py)
      A simpler, older harness that expects a single CSV with pre-extracted
      odds columns (odds_a_win, odds_draw, odds_b_win, ...). It does NOT use
      Asian handicap or the market-consistent challenger. Use it only if you
      already have a Football-Data-style odds CSV.

The two input formats are NOT interchangeable. This script needs a CSV whose
columns include odds_a_win/odds_draw/odds_b_win; it cannot read the paste-based
results.csv used by run_live_backtest.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.backtesting import (
    FootballDataWorldCupXLSXLoader,
    HistoricalWorldCupCSVLoader,
    NonFootballDataCSVError,
    WorldCupResearchBacktestRunner,
    WorldCupResearchBacktestSettings,
)


DEFAULT_INPUT = Path("input/historical/2022/world_cup_matches.csv")
TEMPLATE = Path("templates/historical_world_cup_matches_template.csv")
LIVE_BACKTEST_HINT = (
    "If you collect odds by pasting OddsPortal text into Mxxx.txt files, use\n"
    "scripts/run_live_backtest.py instead. This CSV harness needs a single file\n"
    "with odds columns (odds_a_win, odds_draw, odds_b_win, ...)."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--football-data-xlsx",
        default=None,
        help="Optional Football-Data World Cup XLSX workbook with one sheet per tournament.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Tournament years to load from --football-data-xlsx, e.g. --years 2010 2014 2018 2022.",
    )
    parser.add_argument(
        "--odds-source",
        default="avg",
        choices=["avg", "average", "betfair", "betfair_exchange", "exchange", "bet365", "b365"],
        help="1X2 odds source for Football-Data XLSX mode (default: avg).",
    )
    parser.add_argument(
        "--margin-method",
        default="normalised_inverse_odds",
        help="Margin-removal method for Football-Data XLSX mode (default: normalised_inverse_odds).",
    )
    parser.add_argument("--summary-output", default="output/research/world_cup_backtest_summary.csv")
    parser.add_argument("--predictions-output", default="output/research/world_cup_backtest_predictions.csv")
    parser.add_argument("--skipped-output", default="output/research/world_cup_backtest_skipped.csv")
    parser.add_argument("--yearly-summary-output", default="output/research/world_cup_backtest_summary_by_year.csv")
    parser.add_argument(
        "--probabilistic-summary-output",
        default="output/research/world_cup_backtest_probabilistic_summary.csv",
    )
    parser.add_argument("--excel-output", default="output/research/world_cup_backtest.xlsx")
    args = parser.parse_args()

    if args.football_data_xlsx:
        input_path = Path(args.football_data_xlsx)
        if not input_path.exists():
            print(f"Football-Data World Cup XLSX input not found: {input_path}")
            return
        loader = FootballDataWorldCupXLSXLoader(
            input_path,
            years=tuple(args.years or (2010, 2014, 2018, 2022)),
            odds_source=args.odds_source,
        )
        margin_methods = (args.margin_method,)
    else:
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
                        "",
                        LIVE_BACKTEST_HINT,
                        "",
                        "For the Football-Data multi-year workbook, pass:",
                        "  --football-data-xlsx path/to/world-cup.xlsx --years 2010 2014 2018 2022",
                    ]
                )
            )
            return
        loader = HistoricalWorldCupCSVLoader(input_path)
        margin_methods = ("normalised_inverse_odds", "power", "shin")

    try:
        report = WorldCupResearchBacktestRunner(
            loader,
            settings=WorldCupResearchBacktestSettings(
                summary_output_path=Path(args.summary_output),
                predictions_output_path=Path(args.predictions_output),
                skipped_output_path=Path(args.skipped_output),
                yearly_summary_output_path=Path(args.yearly_summary_output),
                probabilistic_summary_output_path=Path(args.probabilistic_summary_output),
                excel_output_path=Path(args.excel_output),
                margin_methods=margin_methods,
            ),
        ).run()
    except NonFootballDataCSVError as error:
        print(
            "\n".join(
                [
                    f"{input_path} is not a CSV-odds backtest file:",
                    f"  {error}",
                    "",
                    LIVE_BACKTEST_HINT,
                ]
            )
        )
        return
    print(
        "\n".join(
            [
                "Historical World Cup Backtest",
                f"- strategy/config rows: {len(report.summary)}",
                f"- yearly strategy rows: {len(report.yearly_summary)}",
                f"- probabilistic summary rows: {len(report.probabilistic_summary)}",
                f"- prediction rows: {len(report.predictions)}",
                f"- skipped diagnostics: {len(report.skipped)}",
                f"- summary: {args.summary_output}",
                f"- workbook: {args.excel_output}",
            ]
        )
    )


if __name__ == "__main__":
    main()
