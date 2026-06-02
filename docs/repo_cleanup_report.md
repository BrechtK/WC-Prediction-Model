# Repository Cleanup Report

## Summary

This cleanup pass keeps the one-click live workflow as the primary user path:

```text
data/raw/oddsportal_pastes/*_1x2.txt
data/raw/oddsportal_pastes/*_over_under.txt
data/raw/oddsportal_pastes/*_btts.txt
data/raw/oddsportal_pastes/*_correct_score.txt
    -> scripts/run_live_prediction.py
    -> terminal recommendation and Excel reports
```

No model logic, scoring rules, calibration logic, correct-score aggregation,
or EV optimisation behavior was changed.

## File Classification

### Core Library Code

All files under `src/wc_predictor/` are retained. They contain parser,
probability, calibration, optimisation, reporting, live orchestration, and
backtesting code.

### Primary Live-Use Script

| Script | Purpose |
| --- | --- |
| `scripts/run_live_prediction.py` | Normal one-click OddsPortal paste-to-submission workflow |

### Advanced Live-Use Scripts

These remain useful for debugging, manual inspection, or running one stage
independently:

| Script | Purpose |
| --- | --- |
| `scripts/parse_oddsportal_core_odds.py` | Parse 1X2, BTTS, and total-goals ladder pastes |
| `scripts/parse_oddsportal_correct_scores.py` | Parse correct-score pastes |
| `scripts/run_world_cup_predictions.py` | Run recommendations from prepared CSV or Excel inputs |
| `scripts/compare_correct_score_weights.py` | Inspect live blend-weight sensitivity |
| `scripts/compare_correct_score_aggregation_methods.py` | Inspect live aggregation-method sensitivity |
| `scripts/create_world_cup_odds_file.py` | Create the manual workbook fallback |

### Research And Backtesting Scripts

These are intentionally retained:

| Script | Purpose |
| --- | --- |
| `scripts/run_backtest.py` | Generic historical backtest CLI |
| `scripts/run_backtest_all.py` | Run historical CSVs below `data/raw/` |
| `scripts/run_backtest_belgium.py` | Belgium historical folder runner |
| `scripts/run_backtest_england.py` | England historical folder runner |
| `scripts/run_correct_score_backtest.py` | Historical blend-weight evaluation |
| `scripts/inspect_synthetic_mismatches.py` | Extreme-favourite stress testing |

### Retained Generic Utilities

These are not required by the one-click live workflow, but remain useful for
examples, friend comparisons, realised standings, and manual analysis:

| Script | Purpose |
| --- | --- |
| `scripts/run_predictions.py` | Generic recommendation and friend-EV export |
| `scripts/inspect_predictions.py` | Compact generic model inspection |
| `scripts/update_standings.py` | Score submitted predictions and export standings |
| `scripts/generate_report.py` | Generate recommendation and standings bundles |

No scripts were deleted or moved. Moving scripts into subfolders would add
path churn without improving the VS Code workflow.

## Tracked Assets Kept Intentionally

- `data/templates/`: CSV and Excel input templates.
- `data/examples/`: dummy inputs for examples and automated tests.
- `tests/fixtures/`: parser regression fixtures.
- `docs/`: methodology, caveats, roadmap, and workflow documentation.
- `notebooks/exploratory_analysis.ipynb`: exploratory analysis placeholder.
- `prompts/initial_codex_prompt.md`: original project specification.
- `data/raw/.gitkeep` and `data/processed/.gitkeep`: directory placeholders.

## Local And Generated Files

The following are local or generated and are ignored by Git:

- `data/raw/oddsportal_pastes/`
- `data/raw/world_cup_odds.xlsx`
- `data/raw/world_cup_odds_from_pastes.csv`
- `data/raw/world_cup_total_goals_odds_from_pastes.csv`
- `data/raw/world_cup_correct_score_odds.csv`
- downloaded historical league CSVs below `data/raw/`
- all generated CSV and Excel reports below `data/processed/`
- virtual environments, Python caches, test caches, lint caches, package
  metadata, and temporary Excel lock files

`git ls-files` was inspected during cleanup. No raw live odds, downloaded
historical data, parsed odds outputs, or generated reports were tracked, so no
`git rm --cached` operation was needed.

## Remaining Cleanup TODOs

- Consider consolidating `docs/model_roadmap.md` and
  `docs/future_model_roadmap.md` after reviewing whether they serve distinct
  audiences.
- Revisit generic example scripts only if the friend-submission or standings
  workflow is retired.
- Keep the individual parser scripts because they are valuable when debugging
  live paste quality.
