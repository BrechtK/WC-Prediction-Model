# WC Predictor

`wc-predictor` is a market-implied football score prediction-pool optimiser. It
uses bookmaker odds as the primary Version 1 information source, translates those
odds into a coherent scoreline distribution, and selects the prediction that
maximises expected points under private competition rules.

The goal is not to beat bookmakers. The useful edge is valuing a custom payoff:
exact scores, goal differences, results, and knockout qualification points.

## Pipeline

```text
bookmaker odds
    -> raw implied probabilities
    -> margin removal
    -> aggregation across bookmakers
    -> calibrated Poisson scoreline distribution
    -> expected-points optimisation
    -> friend comparison and realised standings
```

Decimal odds `O_i` imply raw probabilities `q_i = 1 / O_i`. A bookmaker's
overround is `R = sum(q_i)`. Version 1 removes it proportionally:
`p_i = q_i / R`. Additive and power methods are also available behind the same
interface; Shin removal is an explicit future placeholder. Aggregation supports
mean, median, weighted, and selected sharp-bookmaker modes.

## Competition Rules

Group-stage predictions earn one mutually exclusive total:

| Outcome | Points |
| --- | ---: |
| Exact score | 10 |
| Correct goal difference, non-exact | 7 |
| Correct winner or draw, wrong difference | 5 |
| Submitted prediction otherwise | 1 |

Knockout scoring is configurable. The baseline interpretation is additive:
one participation point, plus 10 for the correct qualifier, plus 6 for an exact
score, plus 4 for the correct goal difference. Score and qualification are
separate because a drawn match can be decided on penalties.

## Scoreline Model

A full `P(X=x, Y=y)` distribution is needed because the raw modal scoreline is
not always the expected-points-optimal prediction. The transparent Version 1
baseline uses independent Poisson goals:

```text
X ~ Poisson(lambda_A)
Y ~ Poisson(lambda_B)
```

The lambdas are calibrated with `scipy.optimize` to fair 1X2 odds. Available
over/under 2.5 and both-teams-to-score odds become additional constraints.
Calibration uses full-distribution Poisson probabilities. Score-grid tail
probability is reported before renormalisation; EV optimisation uses the
renormalised finite grid and is conditional on its represented scores.

Correct-score odds can optionally be loaded and margin-adjusted in long format.
When supplied, they are aggregated into a direct scoreline market distribution
and blended with the Poisson matrix:

```text
P_final = w * P_poisson + (1 - w) * P_market
```

The default `w=1.0` preserves the Poisson-only baseline.

## Install

Create a fresh environment if the existing local environment is stale:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Run The Example

Dummy odds, predictions, and results live in `data/examples/`.

```powershell
python scripts/run_predictions.py
python scripts/inspect_predictions.py
python scripts/update_standings.py
python scripts/generate_report.py
python scripts/run_backtest.py
pytest
```

Generated CSV and Excel reports are written to `data/processed/`. Each script
accepts `--odds`, `--predictions`, `--results`, and `--output-dir` overrides as
appropriate.

`scripts/inspect_predictions.py` prints a compact per-match audit view with raw
and fair 1X2 probabilities, calibrated lambdas, fit error, score-grid tail mass,
the modal scoreline, and the top five EV predictions.

## Historical Backtesting

The first historical backtester evaluates group-stage-style pool scoring only:

```powershell
python scripts/run_backtest.py `
  --input data/examples/example_historical_matches.csv `
  --detailed-output data/processed/backtest_results_by_file.csv `
  --aggregate-output data/processed/backtest_results_aggregate.csv `
  --skipped-output data/processed/backtest_skipped_by_file.csv `
  --favourite-strength-output data/processed/backtest_favourite_strength.csv
```

`--input` may be one CSV file or a folder containing multiple CSV files. Folder
mode runs each CSV independently, retains `source_file` in per-file output, and
recomputes aggregate strategy metrics from the pooled match-level records.

The loader accepts Football-Data.co.uk-like columns: `HomeTeam`, `AwayTeam`,
`FTHG`, `FTAG`, optional `Date`, and average or bookmaker decimal odds such as
`AvgH`/`AvgD`/`AvgA` or `B365H`/`B365D`/`B365A`. Optional over/under columns
include `Avg>2.5`/`Avg<2.5` and `B365>2.5`/`B365<2.5`.

Football-Data formats vary by league and season. Older files may use `BbAv*`
aggregate columns, which are not currently consumed by the loader. See
[`docs/backtesting.md`](docs/backtesting.md) for the full column map, supported
columns, and the exact loader priority rule.

The exported strategy table compares fixed-score baselines, favourite-win
baselines, the modal Poisson scoreline, 1X2 EV optimisation, and 1X2 plus
over/under EV optimisation when that optional market is available. Missing
optional over/under odds fall back to the 1X2 calibration. Skipped rows and
their reasons are reported per strategy. A separate skip-diagnostic CSV groups
unique skipped matches by source file and reason.

The CLI prints a concise interpretation summary by default: scope, overall
ranking, baseline gaps, key conclusions, per-file winners, skips, and export
paths. Add `--verbose` to append the full raw per-file and aggregate tables.

### Favourite-Strength Diagnostics

Domestic league backtests may underrepresent extreme international mismatches.
The backtest therefore groups matches by the post-margin probability of the
stronger home or away team and exports a favourite-strength report. The console
summary shows whether EV optimisation improves on `favourite_1_0` as the market
favourite becomes stronger.

For a transparent stress test, run:

```powershell
python scripts/inspect_synthetic_mismatches.py
```

This calibrates synthetic fair 1X2 scenarios from moderate through extreme
favourites and prints lambdas, model-implied probabilities, modal scorelines,
EV-optimal scorelines, and the top five EV predictions.

## Running Real World Cup Predictions

The manual-entry workflow is designed to work directly from VS Code:

```powershell
python scripts/create_world_cup_odds_file.py
```

1. Run `scripts/create_world_cup_odds_file.py`.
2. Fill in `data/raw/world_cup_odds.xlsx`, using one row per bookmaker and
   upcoming match.
3. Run `scripts/run_world_cup_predictions.py`.
4. Read `data/processed/world_cup_submission_sheet.xlsx` for the concise
   entry-ready recommendations. Use `data/processed/world_cup_recommendations.xlsx`
   for the full diagnostic report.
5. Use recommended_score for predictions.

Both scripts can be opened in VS Code and launched with **Run Python File**.
The creation script preserves an existing manually edited workbook. Pass
`--overwrite` only when you intentionally want a fresh copy of the template.

Group-stage rows should include `group`; knockout rows may leave it blank.
Qualification odds are optional but recommended for knockout matches.
The optional fields `odds_timestamp`, `odds_source_url`, and `source_quality`
help track where manually collected prices came from and how fresh they are.

The prediction command is:

```powershell
python scripts/run_world_cup_predictions.py
```

It prefers `data/raw/world_cup_odds.xlsx` when available and falls back to
`data/raw/world_cup_odds.csv` for compatibility.

The script removes bookmaker margins, aggregates fair probabilities, calibrates
the Poisson score model, selects the EV-optimal pool prediction, and writes:

- `data/processed/world_cup_recommendations.csv`
- `data/processed/world_cup_recommendations.xlsx`
- `data/processed/world_cup_submission_sheet.xlsx`

Its console summary shows each match, stage and group, fair 1X2 probabilities,
calibrated lambdas, favourite-strength bucket, modal scoreline, EV-optimal
scoreline, and top five EV predictions. Use
`data/templates/friend_predictions_template.csv` to collect pool submissions.

For a dry run with the bundled dummy tournament file:

```powershell
python scripts/run_world_cup_predictions.py `
  --odds data/examples/example_world_cup_odds.csv
```

To add the optional correct-score enhancement:

```powershell
python scripts/run_world_cup_predictions.py `
  --odds data/examples/example_world_cup_odds.csv `
  --correct-score-odds data/examples/example_world_cup_correct_score_odds.csv `
  --correct-score-poisson-weight 0.75
```

Correct-score odds use the long-format template at
`data/templates/correct_score_odds_template.csv`. The full recommendations
report includes the top ten market and blended scorelines plus
`D_KL(P_market || P_poisson)`.

Backtest the standard blend-weight grid `{0, 0.25, 0.5, 0.75, 1}` with:

```powershell
python scripts/run_correct_score_backtest.py
```

### Run From VS Code

For the common historical folders, open one of these files in VS Code and press
**Run Python File**:

- `scripts/run_backtest_belgium.py`
- `scripts/run_backtest_england.py`
- `scripts/run_backtest_all.py`

The Belgium and England runners process their corresponding folders below
`data/raw/`. The all-data runner recursively processes every CSV below
`data/raw/`, including league subfolders. Each runner prints the same concise
summary as `scripts/run_backtest.py` and writes clearly named CSVs below
`data/processed/`.

## Add Your Data

Odds inputs may be CSV or Excel. Required columns are:

```text
match_id,date,stage,team_a,team_b,bookmaker,odds_a_win,odds_draw,odds_b_win
```

Optional columns include over/under 2.5, BTTS, and qualification decimal odds;
see `data/examples/example_odds.csv`. Missing optional markets are ignored.
Friend submissions and results follow the examples in `data/examples/`.

## Architecture

Core modules are deliberately separate:

| Module | Responsibility |
| --- | --- |
| `odds.py`, `margin.py`, `market_data.py` | Input, validation, fair probabilities, aggregation |
| `probabilities.py`, `score_models.py`, `calibration.py` | Score matrices and market calibration |
| `scoring_rules.py`, `optimiser.py` | Pure pool scoring and expected-points optimisation |
| `friends.py`, `results.py`, `reporting.py`, `workflow.py` | Analysis, standings, exports, orchestration |
| `strategies.py`, `backtesting.py` | Reusable strategies and group-stage historical backtesting |

## Limits And Roadmap

Independent Poisson is a translation layer, not a claim that football goals are
fully independent or exactly Poisson distributed. Qualification fallback without
qualification odds is intentionally marked as weak. Knockout score timing must be
checked against the competition app before live use.

The initial historical backtester covers group-stage-style scoring and common
Football-Data-like inputs. Knockout backtesting and richer provider adapters are
future work. Challenger models such as xG/Elo, Skellam-style, Dixon-Coles,
bivariate Poisson, and ML models are intentionally absent. Every later
challenger should be evaluated out of sample against the market-implied
baseline using realised pool points, not just model fit.

See `docs/` for formulas, assumptions, research notes, and the staged roadmap.
For tournament-specific caveats when interpreting domestic backtests, see
[`docs/stylised_facts.md`](docs/stylised_facts.md).

## Model Caveats And Stylised Facts

Version 1 is intentionally simple and market-implied. Domestic backtests are
useful for validating the workflow, but they are not perfect analogues for
neutral-site World Cup matches or extreme international mismatches. Review
[`docs/stylised_facts.md`](docs/stylised_facts.md) before live use and pay
particular attention to favourite-strength diagnostics.

## Odds Quality And Timestamps

Odds move over time, and not every source is equally informative. When entering
live prices, fill in `odds_timestamp`, `odds_source_url`, and `source_quality`
where possible. Suggested free-text quality labels include `sharp`,
`major_bookmaker`, `odds_comparison`, `recreational`, and `unknown`. Use the
latest available odds before the prediction deadline for final submissions.
The live report flags timestamps older than 24 hours as stale for manual review.

## Future Modelling Roadmap

The next major modelling improvements are robust comparison of margin-removal
methods and continued validation of optional correct-score blending. Richer O/U
and BTTS diagnostics, low-score corrections, strategic pool simulation, and
xG/Elo challengers follow after the market-implied baseline is measured. See
[`docs/model_roadmap.md`](docs/model_roadmap.md).
