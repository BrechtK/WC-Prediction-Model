"""Historical backtest using real OddsPortal paste files.

HOW TO USE
----------

1. Create your tournament folder:

       input/historical/wc2022/

2. Put a results CSV there (see template):

       input/historical/wc2022/results.csv

   Required columns:
       match_id, team_a, team_b, actual_score_a, actual_score_b
   Optional columns (defaults applied when missing):
       date, stage, group

3. For every match you want to backtest, create a combined paste file in the
   same format as the live tool:

       input/historical/wc2022/odds/M001.txt
       input/historical/wc2022/odds/M002.txt
       ...

   Each file must contain at least:

       ### MATCH
       Team A vs Team B

       ### 1X2
       <paste OddsPortal 1X2 table>

   Optional sections (used automatically when present):

       ### OVER_UNDER
       ### BTTS
       ### CORRECT_SCORE
       ### ASIAN_HANDICAP

4. Edit the USER SETTINGS below.

5. Press "Run Python File" in VS Code.

6. Open the Excel workbook:

       output/research/live_backtest.xlsx

   The "summary" sheet ranks every config x strategy combination by
   average realised pool points. The "group_stage_rounds" sheet groups totals
   by group-stage playing round: M001-M016, M017-M032, and M033-M048. The
   "predictions" sheet has per-match detail.

WHAT IS COMPARED
----------------
Each backtest config runs the full live pipeline with a different ProjectConfig.
Multiple strategies are extracted from each run and scored independently:

  ev_default         the main EV-optimal recommendation (always reported)
  baseline_poisson   the pure Poisson EV without correct-score blending
  most_likely        the modal (highest-probability) scoreline
  dixon_coles        the Dixon-Coles challenger recommendation
  market_consistent  the market-consistent KL projection (only when MC ran)

The key comparison is:
  baseline_ev / ev_default   vs   mc_with_ah / market_consistent

If mc_with_ah / market_consistent consistently outscores baseline_ev / ev_default
across many matches, the market-consistent challenger with AH is worth
considering as the live default recommendation.
"""

from __future__ import annotations

from pathlib import Path

from wc_predictor.live_backtest import (
    DEFAULT_BACKTEST_CONFIGS,
    QUICK_BACKTEST_CONFIGS,
    RESEARCH_BACKTEST_CONFIGS,
    BacktestConfigEntry,
    LiveBacktestSettings,
    run_live_backtest,
)
from wc_predictor.config import DevigConfig, MarketConsistentGroupWeights, ProjectConfig

# ============================================================
# USER SETTINGS
# ============================================================

# Path to the folder that contains results.csv and the odds/ subfolder.
TOURNAMENT_FOLDER = Path("input/historical/wc2022")

# Filename of the results CSV inside TOURNAMENT_FOLDER.
RESULTS_FILENAME = "results.csv"

# Subfolder (inside TOURNAMENT_FOLDER) that contains the combined paste files.
ODDS_SUBFOLDER = "odds"

# "standard" tests 5 configs (recommended for first run)
# "quick"    tests 2 configs: baseline + MC-with-AH (fastest)
# "research" adds blend-weight, modal/draw-threshold, and larger-grid sweeps
# "custom"   uses CUSTOM_CONFIGS below
BACKTEST_PROFILE = "standard"

# Output paths (relative to project root).
SUMMARY_OUTPUT = Path("output/research/live_backtest_summary.csv")
PREDICTIONS_OUTPUT = Path("output/research/live_backtest_predictions.csv")
EXCEL_OUTPUT = Path("output/research/live_backtest.xlsx")

# Cache folder for split paste files.
CACHE_FOLDER = Path("cache/live_backtest/split_pastes")

# Print per-match progress to the terminal while running.
SHOW_PROGRESS = True

# ---- Custom configs (used only when BACKTEST_PROFILE = "custom") ----------
# Uncomment and edit to build your own comparison.
# CUSTOM_CONFIGS = (
#     BacktestConfigEntry(
#         name="shin_devig",
#         config=ProjectConfig(
#             devig=DevigConfig(
#                 one_x_two_method="shin",
#                 btts_method="shin",
#                 total_goals_method="shin",
#                 correct_score_method="normalised_inverse_odds",
#             ),
#             enable_market_consistent_challenger=True,
#             enable_margin_method_comparison=False,
#         ),
#         description="Shin devig on 2-way markets + market-consistent with AH",
#         use_asian_handicap=True,
#     ),
# )

# ============================================================
# END OF USER SETTINGS
# ============================================================


def main() -> None:
    if BACKTEST_PROFILE == "standard":
        configs = DEFAULT_BACKTEST_CONFIGS
    elif BACKTEST_PROFILE == "quick":
        configs = QUICK_BACKTEST_CONFIGS
    elif BACKTEST_PROFILE == "research":
        configs = RESEARCH_BACKTEST_CONFIGS
    elif BACKTEST_PROFILE == "custom":
        try:
            configs = CUSTOM_CONFIGS  # type: ignore[name-defined]
        except NameError:
            raise ValueError(
                "BACKTEST_PROFILE = 'custom' but CUSTOM_CONFIGS is not defined. "
                "Uncomment and edit the CUSTOM_CONFIGS block above."
            ) from None
    else:
        raise ValueError(
            f"Unknown BACKTEST_PROFILE {BACKTEST_PROFILE!r}. "
            "Use 'standard', 'quick', 'research', or 'custom'."
        )

    odds_folder = TOURNAMENT_FOLDER / ODDS_SUBFOLDER
    results_path = TOURNAMENT_FOLDER / RESULTS_FILENAME

    if not results_path.exists():
        raise SystemExit(
            f"\n{results_path} not found.\n\n"
            "Create it from the template:\n"
            "  templates/historical_results_template.csv\n\n"
            "Required columns: match_id, team_a, team_b, actual_score_a, actual_score_b\n"
            "Optional columns: date, stage, group"
        )

    if not odds_folder.exists() or not any(odds_folder.glob("*.txt")):
        raise SystemExit(
            f"\nNo .txt paste files found in {odds_folder}.\n\n"
            "Create combined paste files using the same format as the live tool:\n"
            f"  {odds_folder}/M001.txt\n"
            f"  {odds_folder}/M002.txt\n\n"
            "Each file needs at minimum:\n"
            "  ### MATCH\n"
            "  Team A vs Team B\n\n"
            "  ### 1X2\n"
            "  <paste OddsPortal 1X2 table>\n\n"
            "Optional sections:\n"
            "  ### OVER_UNDER   ### BTTS   ### CORRECT_SCORE   ### ASIAN_HANDICAP"
        )

    settings = LiveBacktestSettings(
        historical_odds_folder=odds_folder,
        results_path=results_path,
        cache_folder=CACHE_FOLDER,
        summary_output_path=SUMMARY_OUTPUT,
        predictions_output_path=PREDICTIONS_OUTPUT,
        excel_output_path=EXCEL_OUTPUT,
        configs=configs,
    )

    print("=" * 60)
    print("Live-Pipeline Historical Backtest")
    print(f"  Tournament : {TOURNAMENT_FOLDER}")
    print(f"  Profile    : {BACKTEST_PROFILE} ({len(configs)} configs)")
    print(f"  Output     : {EXCEL_OUTPUT}")
    print("=" * 60)
    print()

    run_live_backtest(settings, export=True, progress=SHOW_PROGRESS)


if __name__ == "__main__":
    main()
