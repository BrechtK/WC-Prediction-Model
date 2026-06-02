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

## Priority 2: Correct-Score Odds Support

Accept optional long-format inputs:

```text
match_id,bookmaker,score_a,score_b,decimal_odds
```

Remove margin within each correct-score market and convert the results to a
scoreline probability matrix. Explore explicit blend weights:

```text
score_matrix_final
= w * model_implied_matrix
  + (1 - w) * correct_score_market_matrix
```

Backtest blend weights rather than choosing one subjectively.

## Priority 3: Improved Use Of O/U And BTTS

Review calibration weights and add fit diagnostics for over/under 2.5 and BTTS.
Compare:

- 1X2 only;
- 1X2 plus O/U;
- 1X2 plus O/U plus BTTS.

Report results across favourite-strength buckets because optional totals markets
matter most when translating strong favourites into scoreline predictions.

## Priority 4: Low-Score Correction / Dixon-Coles

Add an optional low-score adjustment for `0-0`, `1-0`, `0-1`, and `1-1`.
Evaluate Dixon-Coles or a related correction against the independent-Poisson
baseline using realised pool points.

## Priority 5: Strategic Prediction Layer

Compare model recommendations with friends' submissions, surface high-EV
contrarian alternatives, and later simulate leaderboard outcomes. The eventual
objective may be probability of winning the pool rather than expected points
alone.

## Priority 6: Own Challenger Models

Only after the market baseline is robust:

- build xG/Elo Poisson or Skellam-style models;
- test market-model blending;
- evaluate every challenger out of sample under the competition scoring rules.

No challenger should replace the market-implied baseline merely because it is
more elaborate.

