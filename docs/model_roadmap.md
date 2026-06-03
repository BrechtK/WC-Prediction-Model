# Model Roadmap

Version 1 intentionally remains a simple market-implied baseline. New modelling
work should preserve the existing scoring rules and be judged out of sample by
realised competition points, not only by probability calibration or betting
profit.

## Priority 1: Robust Market-Implied Probability Extraction

Compare margin-removal approaches:

- proportional normalisation;
- additive removal;
- power method;
- Shin method where feasible;
- favourite-longshot-bias-adjusted methods where justified.

Backtest each approach under the pool scoring rules. Record odds timing and
source quality so early recreational prices are not confused with sharper
closing markets.

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

## Priority 4: Low-Score Correction / Dixon-Coles

The optional Dixon-Coles low-score adjustment for `0-0`, `1-0`, `0-1`, and
`1-1` is implemented as a diagnostic challenger. Evaluate it against the
independent-Poisson baseline using realised pool points before considering any
promotion to the default recommendation.

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

The eventual objective may be probability of winning the pool rather than
expected points alone, but that requires out-of-sample leaderboard evidence.

## Priority 6: Own Challenger Models

Only after the market baseline is robust:

- build xG/Elo Poisson or Skellam-style models;
- test market-model blending;
- evaluate every challenger out of sample under the competition scoring rules.

No challenger should replace the market-implied baseline merely because it is
more elaborate.
