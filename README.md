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
Calibration uses full-distribution Poisson probabilities and several bounded
starting points, keeping the best converged fit for robustness with extreme
favourites. Near-bound lambdas are flagged for review. Score-grid tail
probability is reported before renormalisation; EV optimisation uses the
renormalised finite grid and is conditional on its represented scores.

Correct-score odds can optionally be loaded and margin-adjusted in long format.
When supplied, margin removal happens separately within each bookmaker's full
available score grid. The resulting fair probabilities are aggregated into a
direct scoreline market distribution and blended with the Poisson matrix:

```text
P_final = w * P_poisson + (1 - w) * P_market
```

The default `w=1.0` preserves the Poisson-only baseline.

Correct-score decimal odds are never averaged directly. The default
`correct_score_aggregation_method=auto` uses a winsorized mean with at least
five bookmakers, a median with three or four bookmakers, and a mean with one
or two bookmakers. Scoreline-level outliers are flagged using log fair
probabilities and robust median/MAD diagnostics.

Correct-score blending is suppressed when fewer than
`min_scorelines_for_blend=10` usable scorelines are available for a match. The
partial market remains visible in diagnostics, but recommendations fall back
to pure Poisson and include a warning.

## Install

Create a fresh environment if the existing local environment is stale:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Primary Live Workflow

For normal live tournament use, fill the OddsPortal paste files and run one
script.

1. Paste the visible OddsPortal text into:

```text
data/raw/oddsportal_pastes/M001_1x2.txt
data/raw/oddsportal_pastes/M001_over_under.txt
data/raw/oddsportal_pastes/M001_btts.txt
data/raw/oddsportal_pastes/M001_correct_score.txt
```

2. Open `scripts/run_live_prediction.py`.
3. Press **Run Python File** in VS Code.
4. Read the `Final recommended submission` block in the terminal.
5. Open `data/processed/world_cup_recommendations.xlsx` for full diagnostics
   and `data/processed/correct_score_weight_comparison.xlsx` for blend-weight
   sensitivity when useful.

The runner parses all available pastes, uses the half-goal total-goals ladder,
runs the prediction model, writes the normal Excel outputs, and performs
correct-score blend sensitivity when correct-score odds are available.

`*_1x2.txt` is required. Missing O/U, BTTS, or correct-score pastes produce
warnings in the default non-strict mode. Use `--strict` to require all four
paste types, `--match-id M001` to process one match, or
`--skip-weight-sensitivity` for a faster run without the comparison workbook.

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

Backtest strategy summaries include `fraction_used_over_under_2_5`, showing
how often the O/U-enhanced strategy actually used that optional market instead
of its 1X2-only fallback.

The CLI prints a concise interpretation summary by default: scope, overall
ranking, baseline gaps, key conclusions, per-file winners, skips, and export
paths. It also prints file-level progress and elapsed time while a run is in
progress. Add `--verbose` to append the full raw per-file and aggregate tables.

Robust multi-start Poisson calibration makes full historical runs slower than
quick exploratory checks. Use `--fast` to run single-start calibration while
checking data coverage or iterating locally:

```powershell
python scripts/run_backtest.py --input data/raw/England --fast
```

Use the default full mode for final validation. Smoke tests can be bounded
without editing input folders:

```powershell
python scripts/run_backtest.py --input data/raw --fast --max-files 2 --max-matches 100
```

Folder mode skips CSV files that are not Football-Data-like historical match
files and reports them in the skipped-input diagnostics.

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

## Advanced Individual Scripts

The scripts below remain available for manual inspection, debugging, research,
and workflows that start from a hand-edited workbook. They are not required
for the normal one-click OddsPortal paste workflow.

### Manual Workbook Predictions

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

Compare live recommendations across all supported correct-score aggregation
methods with:

```powershell
python scripts/compare_correct_score_aggregation_methods.py
```

This uses `data/raw/world_cup_correct_score_odds.csv`, applies a default
Poisson weight of `0.85`, and writes
`data/processed/correct_score_aggregation_comparison.xlsx`. The workbook
contains detailed method runs and a pivot-style sensitivity sheet showing
whether each match's recommended score changes across aggregation methods.

### Correct-Score Blend-Weight Sensitivity

Compare live recommendations across Poisson blend weights from `1.00` down to
`0.00` in steps of `0.05` with:

```powershell
python scripts/compare_correct_score_weights.py
```

The script reads `data/raw/world_cup_odds.xlsx` and
`data/raw/world_cup_correct_score_odds.csv`, then writes
`data/processed/correct_score_weight_comparison.xlsx`. Its detail sheet shows
every match and weight. Its sensitivity sheet highlights recommendation
changes, the first departure from pure Poisson, and the stable weight interval
around `w=0.85`. A custom grid can be passed with
`--weights "1,0.85,0.75,0.5,0"`.

This is a live sensitivity diagnostic, not empirical proof of the best blend
weight. Historical validation is still needed before selecting a weight for
final submissions.

### OddsPortal Core Odds Paste Workflow

Core match odds can be collected by manually copying visible OddsPortal tables
into separate text files. This does not scrape OddsPortal or automate website
access. Use filenames such as:

```text
data/raw/oddsportal_pastes/M001_1x2.txt
data/raw/oddsportal_pastes/M001_btts.txt
data/raw/oddsportal_pastes/M001_over_under.txt
data/raw/oddsportal_pastes/M001_correct_score.txt
```

The core parser extracts bookmaker-specific 1X2 and BTTS rows plus the visible
over/under totals ladder. It keeps the familiar wide over/under `+2.5` columns,
ignores visible page noise and exchange sections, merges core markets by
`match_id` and bookmaker, and writes:

```powershell
python scripts/parse_oddsportal_core_odds.py
```

```text
data/raw/world_cup_odds_from_pastes.csv
data/raw/world_cup_total_goals_odds_from_pastes.csv
data/processed/oddsportal_core_odds_parse_report.csv
```

The parser joins match metadata from `data/raw/world_cup_odds.xlsx` when that
file is available. Review the compact console summary and parse report for
missing markets, sparse bookmaker coverage, suspicious prices, and ignored
exchange sections.

Parse optional correct-score text separately, then run predictions with the
generated CSV files. Pass the long totals ladder when you want multi-line
half-goal calibration:

```powershell
python scripts/parse_oddsportal_core_odds.py
python scripts/parse_oddsportal_correct_scores.py
python scripts/run_world_cup_predictions.py `
  --odds data/raw/world_cup_odds_from_pastes.csv `
  --total-goals-odds data/raw/world_cup_total_goals_odds_from_pastes.csv `
  --correct-score-odds data/raw/world_cup_correct_score_odds.csv `
  --correct-score-poisson-weight 0.85 `
  --correct-score-aggregation-method auto
```

Totals odds are stored in long format with one row per `match_id`, bookmaker,
and line. Margin removal happens within each bookmaker's two-way line before
fair over probabilities are aggregated across bookmakers. Half-goal lines such
as `1.5`, `2.5`, and `3.5` can constrain Poisson calibration. Integer and
quarter Asian lines are retained in diagnostics but skipped until explicit
push and half-stake settlement formulas are implemented.

The original baseline used only `+2.5` because it is a common, simple binary
market and is widely available. The full ladder adds information about the
shape of the total-goals distribution. Supplying no long ladder file preserves
the original `+2.5`-only behavior exactly.

Recommendation diagnostics make the calibration inputs auditable:

- `total_goals_lines_available` lists every parsed ladder line;
- `total_goals_lines_used_for_calibration` lists every half-goal constraint;
- `total_goals_lines_skipped` lists retained integer and quarter Asian lines;
- `multi_line_totals_used` confirms whether more than one half-goal line entered
  the fit;
- `total_goals_line_diagnostics` shows the fair market over probability, fitted
  model over probability, and signed error for each used line.

### OddsPortal Correct-Score Paste Workflow

Correct-score data can be collected manually without scraping or automating
website access:

1. Open the match's correct-score market on OddsPortal.
2. Expand the scorelines so the bookmaker-specific odds are visible.
3. Copy the visible text.
4. Paste it into a file such as
   `data/raw/oddsportal_pastes/M001_correct_score.txt`. The prefix before
   `_correct_score.txt` must match the World Cup odds workbook's `match_id`.
5. Parse every pasted file:

```powershell
python scripts/parse_oddsportal_correct_scores.py
```

6. Generate recommendations with the parsed correct-score file:

```powershell
python scripts/run_world_cup_predictions.py `
  --correct-score-odds data/raw/world_cup_correct_score_odds.csv `
  --correct-score-poisson-weight 0.85
```

7. Inspect the parser warnings, correct-score coverage diagnostics, and blend
   sensitivity before trusting a recommendation change.

The parser ignores page noise and displayed best-odds summaries, preferring
bookmaker-specific rows. It writes:

- `data/raw/world_cup_correct_score_odds.csv`
- `data/processed/oddsportal_correct_score_parse_report.csv`

The console summary highlights sparse match coverage, scorelines with missing
odds, and scorelines with fewer than three bookmaker prices. The recommendation
report adds correct-score scoreline and bookmaker counts, a sparse-market
warning, top-ten market and blended scorelines, KL divergence, and the selected
blend weight. It also reports the robust aggregation method, bookmaker
overround diagnostics, scoreline coverage warnings, and detected outliers.

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
`data/processed/`. These runners also accept `--fast`, `--max-files`, and
`--max-matches` when launched from a terminal.

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
| `oddsportal.py`, `oddsportal_core.py` | Parsing and validation for manually pasted OddsPortal markets |
| `probabilities.py`, `score_models.py`, `calibration.py` | Score matrices and market calibration |
| `scoring_rules.py`, `optimiser.py` | Pure pool scoring and expected-points optimisation |
| `friends.py`, `results.py`, `reporting.py`, `workflow.py` | Analysis, standings, exports, orchestration |
| `strategies.py`, `backtesting.py` | Reusable strategies and group-stage historical backtesting |

## Project Structure

| Path | Purpose |
| --- | --- |
| `src/wc_predictor/` | Package code for parsing, calibration, optimisation, reporting, and backtesting |
| `scripts/` | Runnable live-use, advanced, and research scripts |
| `data/templates/` | Tracked input templates |
| `data/examples/` | Tracked dummy data for examples and tests |
| `data/raw/` | Ignored local pastes, workbooks, and downloaded historical data |
| `data/processed/` | Ignored generated CSV and Excel reports |
| `tests/` | Automated tests and tracked parser fixtures |
| `docs/` | Mathematical basis, caveats, roadmap, and cleanup notes |

See [`docs/repo_cleanup_report.md`](docs/repo_cleanup_report.md) for the
repository hygiene classification and retained advanced scripts.

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
