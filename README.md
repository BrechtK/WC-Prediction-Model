# WC Predictor

## What This Project Does

`wc-predictor` turns pasted OddsPortal football markets into score predictions
for a private World Cup pool. It removes bookmaker margin, calibrates an
independent-Poisson baseline, optionally blends coherent correct-score odds,
and selects the scoreline with the highest expected pool points.

The live recommendation remains the established pure-EV model. Dixon-Coles and
public-field strategy outputs are diagnostic challengers and do not change the
submitted score unless you deliberately choose to use them outside the default
workflow.

## Quick Start: Live World Cup Prediction

The normal workflow uses only `input/`, `output/`, and one script.

1. Paste the complete fixture schedule into:

```text
input/schedule.txt
```

2. Create one combined odds file per match:

```text
input/odds/M001.txt
input/odds/M002.txt
input/odds/M003.txt
```

3. Run:

```powershell
python scripts/run_live_prediction.py
```

4. Read the terminal's `Final recommended submission` block and open:

```text
output/submission_sheet.xlsx
output/predictions.xlsx
```

The runner automatically parses the schedule, validates match IDs, splits the
combined odds files into `cache/split_pastes/`, writes parsed CSVs below
`cache/parsed/`, runs the model, and writes the live reports below `output/`.

## Running Daily From VS Code

During the tournament, the intended daily workflow is to edit the clearly
marked `USER SETTINGS` block at the top of:

```text
scripts/run_live_prediction.py
```

Then press `Run Python File` in VS Code.

1. Paste or update the full schedule in:

```text
input/schedule.txt
```

2. Set:

```python
RUN_MODE = "list_date"
DATE = "14-6"
```

Then run the file to list the fixtures and game numbers for that date.

3. Paste odds into the relevant combined odds file, for example:

```text
input/odds/M008.txt
```

4. Set:

```python
RUN_MODE = "date"
DATE = "14-6"
```

This runs every match on that date whose odds file exists in `input/odds/`.

To run just one match on the date, set:

```python
RUN_MODE = "single_match"
DATE = "14-6"
GAME_NUMBER = 3
MATCH_ID = None
```

If you already know the match ID, you can use:

```python
MATCH_ID = "M008"
```

5. Press `Run Python File`.

6. Read the terminal recommendation and open:

```text
output/submission_sheet.xlsx
```

Use `RUN_MODE = "all_available"` to process every valid odds file in
`input/odds/`, which preserves the original all-matches workflow. CLI arguments
still work for advanced use and override the editable settings, for example:

```powershell
python scripts/run_live_prediction.py --date 14-6 --game-number 3
```

## Input Format

Each `input/odds/M001.txt` file contains the pasted tables for one fixture.
Mark each market with a section header:

```text
### 1X2
<pasted 1X2 table>

### OVER_UNDER
<pasted over/under table>

### BTTS
<pasted both-teams-to-score table>

### CORRECT_SCORE
<pasted correct-score table>
```

`### 1X2` is required. Missing O/U, BTTS, or correct-score sections produce
warnings in the default non-strict mode.

Supported aliases include `### MATCH_ODDS`, `### FULL_TIME_RESULT`,
`### OVER UNDER`, `### O/U`, `### BOTH_TEAMS_TO_SCORE`,
`### BOTH TEAMS TO SCORE`, and `### CORRECT SCORE`.

The schedule paste is parsed in fixture order and assigned IDs `M001`, `M002`,
and so on. The file name for each odds paste must use the corresponding ID.
Unknown IDs fail clearly before model execution.

## Run Command

The default live command is:

```powershell
python scripts/run_live_prediction.py
```

Useful switches:

```powershell
python scripts/run_live_prediction.py --match-id M001
python scripts/run_live_prediction.py --strict
python scripts/run_live_prediction.py --skip-weight-sensitivity
python scripts/run_live_prediction.py --dixon-coles-rho -0.08
python scripts/run_live_prediction.py --strategy-mode public-ranking
```

`--dixon-coles-rho` changes only the diagnostic challenger. It never promotes
Dixon-Coles to the final live recommendation.

`--strategy-mode` controls the diagnostic public-field strategy layer:

- `ev` keeps pure expected-points optimisation and is the default;
- `balanced` and `public-ranking` allow small-EV-cost contrarian suggestions;
- `aggressive-public-ranking` allows a wider diagnostic EV-loss band.

The `Final recommended submission` block still uses `recommended_score`, the
pure-EV live recommendation. Public-ranking columns are for contest strategy
review, not automatic replacement.

Clean `input/odds/*.txt` files refresh their generated `cache/split_pastes/`
files automatically.

## Output Files

Normal live outputs:

| Path | Purpose |
| --- | --- |
| `output/submission_sheet.xlsx` | Concise entry-ready recommendations |
| `output/predictions.xlsx` | Full prediction and model diagnostics |
| `output/predictions.csv` | Machine-readable detailed recommendations |
| `output/correct_score_weight_sensitivity.xlsx` | Correct-score blend-weight comparison |
| `output/parse_reports/schedule_parse_report.csv` | Schedule parsing diagnostics |
| `output/parse_reports/odds_parse_report.csv` | 1X2, O/U, and BTTS parsing diagnostics |
| `output/parse_reports/correct_score_parse_report.csv` | Correct-score parsing diagnostics |

Generated parser intermediates live below `cache/`:

```text
cache/
    parsed/
        core_odds.csv
        total_goals_odds.csv
        correct_score_odds.csv
        schedule.csv
    split_pastes/
        M001_1x2.txt
        M001_over_under.txt
        M001_btts.txt
        M001_correct_score.txt
```

`output/predictions.xlsx` is the model-diagnostics workbook. Keeping one
detailed workbook avoids duplicating the same diagnostic tables under a second
name.

## Interpreting The Terminal Summary

For each match, the live runner prints:

- parsed schedule mapping and market coverage;
- fair 1X2 probabilities and calibrated Poisson lambdas;
- final EV-optimal live score and alternatives;
- baseline Poisson, correct-score blend, and final-live comparison;
- Dixon-Coles rho, top-five challenger EV predictions, and disagreement flag;
- estimated crowded public score, public-ranking diagnostic score, EV cost,
  public pick share, leverage score, and explanation;
- parse, calibration, sparse-market, and model-disagreement warnings.

The `Final recommended submission` block is the entry-ready answer. Challenger
diagnostics are there for validation, not for silent model replacement.

## Advanced Scripts

The one-click runner is the primary interface. These scripts remain useful for
debugging or research:

| Script | Purpose |
| --- | --- |
| `scripts/parse_oddsportal_schedule.py` | Parse only `input/schedule.txt` |
| `scripts/split_oddsportal_combined_pastes.py` | Split only `input/odds/*.txt` |
| `scripts/parse_oddsportal_core_odds.py` | Parse cached 1X2, O/U, and BTTS pastes |
| `scripts/parse_oddsportal_correct_scores.py` | Parse cached correct-score pastes |
| `scripts/compare_correct_score_weights.py` | Compare correct-score blend weights |
| `scripts/compare_correct_score_aggregation_methods.py` | Compare aggregation methods |
| `scripts/compare_dixon_coles_rho.py` | Run Dixon-Coles rho sensitivity |
| `scripts/simulate_public_contest.py` | Simulate pure EV vs public-ranking entries against a heuristic public field |
| `scripts/run_world_cup_predictions.py` | Run from prepared tabular odds files |

### Dixon-Coles Challenger Sensitivity

After a live run has prepared the cache, compare the diagnostic challenger over
the default rho grid:

```powershell
python scripts/compare_dixon_coles_rho.py
```

This writes:

```text
output/dixon_coles_rho_comparison.xlsx
```

The script reports recommendation changes across rho values without selecting
an optimal rho and without changing the live recommendation.

### Public-Field Strategy Diagnostics

Large public prediction contests can reward entries that are still high-EV but
less crowded than obvious public scorelines. The project now reports a
diagnostic public-ranking layer with:

- estimated most crowded public score;
- public-ranking score suggestion;
- EV cost versus the pure-EV recommendation;
- estimated public pick share and leverage score;
- optional friend-crowding diagnostic when friend predictions are available.

Run the live workflow with:

```powershell
python scripts/run_live_prediction.py --strategy-mode public-ranking
```

For a simulation-style stress test after a live run has prepared `cache/`:

```powershell
python scripts/simulate_public_contest.py
```

This writes:

```text
output/public_contest_simulation.xlsx
```

The simulation is a heuristic leaderboard diagnostic. It does not choose an
optimal strategy mode, alter the scoring rules, or change the default live
submission.

## Cleaning Generated Files

Remove generated reports and parser caches at any time:

```powershell
python scripts/clean_generated_outputs.py
```

This safely removes `output/`, `cache/`, and any obsolete
`data/processed/` folder. It also clears project-level Python and pytest
caches. It never removes source files, test files, documentation, or `input/`.
If retired paste files are found below `data/raw/`, the command warns before
cleanup and preserves those files for manual review.

## Backtesting And Research

Historical and synthetic workflows remain separate from live prediction:

| Script | Purpose |
| --- | --- |
| `scripts/run_backtest.py` | Generic historical CSV backtest |
| `scripts/run_backtest_all.py` | Recursive historical backtest |
| `scripts/run_backtest_belgium.py` | Belgium convenience runner |
| `scripts/run_backtest_england.py` | England convenience runner |
| `scripts/run_correct_score_backtest.py` | Historical blend-weight evaluation |
| `scripts/inspect_synthetic_mismatches.py` | Extreme-favourite stress tests |

Example:

```powershell
python scripts/run_backtest.py --input input/historical/England --fast
```

Downloaded historical league files can remain local below `input/historical/`.

## Project Structure

```text
input/                   user-created live pastes
output/                  user-facing generated reports
cache/                   generated parser intermediates
src/wc_predictor/        package code
scripts/                 live, debugging, and research entry points
tests/                   automated tests and tracked parser fixtures
data/examples/           tracked example datasets
data/templates/          tracked templates
docs/                    methodology and maintenance notes
```

## Methodology Links

- [Mathematical basis](docs/mathematical_basis.md)
- [Stylised facts](docs/stylised_facts.md)
- [Odds collection strategy](docs/odds_collection_strategy.md)
- [Model roadmap](docs/model_roadmap.md)
- [Repository cleanup report](docs/repo_cleanup_report.md)

## Install And Test

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```
