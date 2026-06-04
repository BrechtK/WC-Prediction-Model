# Model Roadmap

Version 1 intentionally remains a simple market-implied baseline. New modelling
work should preserve the existing scoring rules and be judged out of sample by
realised competition points, not only by probability calibration or betting
profit.

## Priority 1: Robust Market-Implied Probability Extraction

The live workflow default is `normalised_inverse_odds`. It also implements
diagnostic comparison for:

- normalised inverse-odds proportional normalisation;
- additive removal;
- power method;
- Shin method where numerically valid.

Still to evaluate or implement:

- favourite-longshot-bias-adjusted methods where justified.

The comparison sheet reports whether recommendations change across implemented
methods. Continue backtesting each approach under the pool scoring rules before
promoting any challenger. Record odds timing and source quality so early
recreational prices are not confused with sharper closing markets.
Shin, power, and additive methods are optional research diagnostics. They can
fail on sparse, low-overround, or unusually high-overround markets; live
processing falls back to `normalised_inverse_odds` per market and records a
warning rather than trusting invalid probabilities.

## Priority 2: Correct-Score Odds Validation

The project now accepts optional long-format inputs:

```text
match_id,bookmaker,score_a,score_b,decimal_odds
```

It removes margin within each correct-score market, converts results to a
scoreline probability matrix, and supports explicit blend weights:

```text
score_matrix_final
= w * model_implied_matrix
  + (1 - w) * correct_score_market_matrix
```

Continue validating blend weights across larger historical samples rather than
choosing one subjectively. Investigate incomplete correct-score market coverage,
liquidity, and bookmaker selection before treating the enhancement as a default.

## Priority 3: Improved Use Of O/U And BTTS

Review calibration weights and add fit diagnostics for over/under 2.5 and BTTS.
Compare:

- 1X2 only;
- 1X2 plus O/U;
- 1X2 plus O/U plus BTTS.

Report results across favourite-strength buckets because optional totals markets
matter most when translating strong favourites into scoreline predictions.

Asian handicap support is now available as an optional market-consistent
challenger input. The parser keeps the full pasted ladder for diagnostics, but
the optimiser uses only a selected stable subset. Near-the-money lines are most
informative; very deep lines are retained as tail diagnostics and skipped as
constraints when the finite score grid cannot represent the relevant tail.
Orientation warnings compare the AH-implied favourite direction with 1X2 prices
and are diagnostics for potentially reversed pasted tables.
This should be validated alongside totals because AH maps most directly to the
goal-difference tier in the scoring rule and is especially useful for strong
favourites and blowout-tail choices such as `3-0`, `4-0`, and `5-0`. It remains
diagnostic/challenger input, not a direct override of the default EV
recommendation.

Group-stage scoring constants are centralised as the assumed `10/7/5/1` rule:
exact score, correct goal difference, correct result, and participation. Margin
diagnostics derive their coefficients from those constants so they stay
consistent with the raw EV optimiser unless the confirmed rules change.

## Priority 4: Low-Score Correction / Dixon-Coles

The optional Dixon-Coles low-score adjustment for `0-0`, `1-0`, `0-1`, and
`1-1` is implemented as a diagnostic challenger. When correct-score market
data contains those cells, rho is estimated from market low-score probabilities
over a bounded grid; otherwise the configured rho is used with a warning.
Evaluate it against the independent-Poisson baseline using realised pool
points before considering any promotion to the default recommendation.

## Historical World Cup Backtesting

Use `scripts/run_historical_world_cup_backtest.py` with a local CSV based on
`templates/historical_world_cup_matches_template.csv` to evaluate 2022 or other
World Cup data. The harness compares margin-removal methods, the existing
simple baselines, and EV strategies, and records skipped diagnostics when
optional correct-score odds are unavailable. Do not commit proprietary or
scraped odds data; keep real historical datasets local unless they are legally
safe examples.

Backtesting should answer:

- whether blend weights below `w=1.0` improve realised points;
- whether Shin or power margin removal changes recommendations usefully;
- whether Asian handicap constraints improve margin-level decisions;
- whether market-consistent or Dixon-Coles challenger recommendations add value;
- whether decision-dashboard manual-review flags identify fragile matches.

## Knockout Scoring Verification

Knockout scoring currently defaults to `unverified`, preserving the existing
additive calculation while warning that Sporza rules must be confirmed. Use
`docs/knockout_scoring_verification.md` to compare additive and hierarchical
examples before changing the config for knockout EV.

## Priority 5: Strategic Prediction Layer

The project now includes a diagnostic public-field strategy layer. It estimates
crowded public scorelines, reports small-EV-cost contrarian alternatives,
surfaces friend-crowding diagnostics when friend predictions are available, and
can simulate pure-EV versus public-ranking entries against a heuristic public
field.

This layer is not a replacement for pure EV. Before promotion, validate:

- whether estimated public pick shares resemble actual contest entries;
- whether public-ranking suggestions improve realised leaderboard position;
- whether gains survive different field sizes and popularity assumptions;
- whether EV-loss caps remain conservative enough for small private pools.

The public-strategy target/field-size settings should be treated as diagnostic
scaling only:

- `friends` / field size around 12: low crowd/contrarian weight;
- `balanced` / field size around 100: moderate influence;
- `national` / field size around 30000: stronger decorrelation diagnostics.

The eventual objective may be probability of winning the pool rather than
expected points alone, but that requires out-of-sample leaderboard evidence.

## Final Decision Dashboard

The live workflow now includes a final decision dashboard that consolidates the
default EV recommendation, challenger model agreement, margin-removal
sensitivity, public-strategy suggestions, high-score clusters, plausible
alternatives, and warning flags.

This layer is reporting-only. `recommended_score` remains the pure default EV
recommendation, and `final_decision_score` defaults to that same score. The
dashboard highlights manual-review and override-candidate situations so a human
can make the final submission decision under tournament pressure.

## Priority 6: Own Challenger Models

Only after the market baseline is robust:

- build xG/Elo Poisson or Skellam-style models;
- test market-model blending;
- evaluate every challenger out of sample under the competition scoring rules.

No challenger should replace the market-implied baseline merely because it is
more elaborate.
