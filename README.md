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
Score-grid tail probability is reported before optional renormalisation.

Correct-score odds can already be loaded and margin-adjusted in long format.
Using them as a direct score distribution or a configurable blend is reserved
for a later increment.

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
python scripts/update_standings.py
python scripts/generate_report.py
pytest
```

Generated CSV and Excel reports are written to `data/processed/`. Each script
accepts `--odds`, `--predictions`, `--results`, and `--output-dir` overrides as
appropriate.

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
| `strategies.py`, `backtesting.py` | Reusable strategy interface and Version 1.5 scaffold |

## Limits And Roadmap

Independent Poisson is a translation layer, not a claim that football goals are
fully independent or exactly Poisson distributed. Qualification fallback without
qualification odds is intentionally marked as weak. Knockout score timing must be
checked against the competition app before live use.

Historical backtesting is designed for but not implemented. Future challenger
models such as xG/Elo, Skellam-style, Dixon-Coles, bivariate Poisson, and ML
models are intentionally absent from Version 1. Every later challenger should be
evaluated out of sample against the market-implied baseline using realised pool
points, not just model fit.

See `docs/` for formulas, assumptions, research notes, and the staged roadmap.
