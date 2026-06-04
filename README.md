# WC Predictor

## What The Project Does

`wc-predictor` turns pasted OddsPortal football markets into World Cup
prediction-pool score recommendations. The live workflow parses a schedule,
splits one combined odds file per match, runs the established EV recommendation
logic, and writes a submission sheet plus a detailed diagnostics workbook.

The default live recommendation remains the existing pure-EV model. Research
diagnostics can challenge or explain a pick, but they do not silently change the
default submitted score.

## Quick Start / Matchday Use

1. Open `scripts/run_live_prediction.py`.
2. Keep `RUN_PROFILE = "live"` and `TERMINAL_VERBOSITY = "compact"`.
3. Set `RUN_MODE = "list_date"` and `DATE = "14-6"`.
4. Press **Run Python File** to see the game numbers.
5. Paste fresh odds into `input/odds/Mxxx.txt`.
6. Set `RUN_MODE = "single_match"`, keep `DATE`, and set `GAME_NUMBER`.
7. Press **Run Python File** again.
8. Read the `Final recommendations` block.

For the full step-by-step version, see
[docs/matchday_workflow.md](docs/matchday_workflow.md).

## Input Files

Create the schedule locally:

```text
input/schedule.txt
```

Create one odds file per match:

```text
input/odds/M008.txt
```

Use the tracked template:

```text
templates/odds_input_template.txt
```

The expected sections are:

```text
### MATCH
Team A vs Team B

### 1X2
<paste OddsPortal 1X2 table here>

### OVER_UNDER
<paste OddsPortal totals table here>

### BTTS
<paste OddsPortal BTTS table here>

### CORRECT_SCORE
<paste OddsPortal correct-score table here>

### ASIAN_HANDICAP
<optional: paste OddsPortal Asian handicap table here>
```

Live schedule and odds files are ignored by Git. Keep real tournament inputs in
`input/`, not in `data/raw/`.

Asian handicap is optional. When supplied, it is used by the
market-consistent challenger and diagnostics to improve goal-difference and
blowout-tail information; it does not override the default EV recommendation
automatically. The market roles are:

| Market | Main scoring information |
| --- | --- |
| 1X2 | Result tier |
| Correct score | Exact-score tier |
| Asian handicap | Goal-difference distribution |
| O/U totals | Total-goals environment |
| BTTS | Scoring dependence |

## Running From VS Code

The intended interface is the `USER SETTINGS` block in:

```text
scripts/run_live_prediction.py
```

Use these modes:

```python
RUN_MODE = "list_date"      # list games on DATE and exit
RUN_MODE = "single_match"   # run DATE + GAME_NUMBER
RUN_MODE = "date"           # run every game on DATE
RUN_MODE = "all_available"  # run every odds file in input/odds/
```

Command-line overrides still work for advanced use:

```powershell
python scripts/run_live_prediction.py --date 14-6 --game-number 3
python scripts/run_live_prediction.py --run-profile research --terminal-verbosity debug
```

## Output Files

Normal live outputs:

| Path | Purpose |
| --- | --- |
| `output/submission_sheet.xlsx` | Entry-ready recommendation sheet |
| `output/predictions.xlsx` | Detailed prediction and diagnostic workbook |
| `output/predictions.csv` | Machine-readable detailed recommendations |
| `output/parse_reports/` | Schedule and odds parser diagnostics |

Generated parser intermediates are written below `cache/`.

## Interpreting Results

`final recommendation` is the default score to submit.

`confidence` is a practical robustness label. High means the model is relatively
clear; medium means there is a reasonable alternative nearby; low or clustered
needs manual inspection.

`manual review` tells you whether to inspect the Excel files before submitting.
When it says `no`, the compact terminal output is usually enough.

`alternatives` show the main nearby score to consider if a match is flagged for
review.

## Research Mode / Advanced Diagnostics

Use research mode when you want slower diagnostic workbooks and fuller terminal
detail:

```python
RUN_PROFILE = "research"
TERMINAL_VERBOSITY = "debug"
```

Research mode enables margin-method comparison, market-consistent challenger
diagnostics, and correct-score weight sensitivity. These are for investigation;
they do not change the default live recommendation by themselves.

Public strategy remains diagnostic. Use `PUBLIC_STRATEGY_TARGET` /
`public_strategy_target` values such as `friends`, `balanced`, or `national`
to scale crowding/decorrelation diagnostics for different field sizes; pure
expected-points optimisation stays the default score.

Additional research diagnostics include optional Shin margin removal,
market-estimated Dixon-Coles rho from low correct-score cells, and a historical
World Cup backtest harness. To run the historical harness, create a legally safe
local CSV from:

```text
templates/historical_world_cup_matches_template.csv
```

Then run:

```powershell
python scripts/run_historical_world_cup_backtest.py --input input/historical/2022/world_cup_matches.csv
```

No real historical odds are tracked in the repository. Historical odds quality,
timing, and coverage matter, so treat results as validation evidence rather
than proof that a challenger should become the live default.

Longer technical notes live in:

- [docs/mathematical_basis.md](docs/mathematical_basis.md)
- [docs/model_roadmap.md](docs/model_roadmap.md)
- [docs/odds_collection_strategy.md](docs/odds_collection_strategy.md)
- [docs/backtesting.md](docs/backtesting.md)
- [docs/knockout_scoring_verification.md](docs/knockout_scoring_verification.md)

## Project Structure

```text
input/                   local live schedule and odds files
templates/               tracked paste templates
output/                  generated reports
cache/                   generated parser intermediates
src/wc_predictor/        package code
scripts/                 live, debugging, and research entry points
tests/fixtures/          tracked parser fixtures
docs/                    workflow and methodology notes
```

## Install And Test

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```
