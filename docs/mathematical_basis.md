# Mathematical Basis

## Odds And Fair Probabilities

Decimal odds `O_i` imply raw probabilities:

```text
q_i = 1 / O_i
```

The bookmaker overround is:

```text
R = sum_i q_i
```

Version 1 removes the margin proportionally:

```text
p_i = q_i / R
```

Fair probabilities are calculated per bookmaker and market, then aggregated
across bookmakers.

## Independent Poisson Score Model

For expected goals `lambda_A` and `lambda_B`:

```text
P(X=x,Y=y)
= exp(-lambda_A) lambda_A^x / x!
  exp(-lambda_B) lambda_B^y / y!
```

The score matrix implies match outcomes:

```text
P(A wins) = sum_{x>y} P(X=x,Y=y)
P(draw)   = sum_{x=y} P(X=x,Y=y)
P(B wins) = sum_{x<y} P(X=x,Y=y)
```

Calibration evaluates these probabilities over the full independent-Poisson
distribution, not over the finite score grid used for reports and EV
optimisation. The implementation uses the Skellam distribution for the full
goal-difference probabilities. It also evaluates:

```text
P(over 2.5)  = P(X + Y >= 3), where X + Y ~ Poisson(lambda_A + lambda_B)
P(BTTS yes)  = (1 - exp(-lambda_A)) (1 - exp(-lambda_B))
```

When a long totals ladder is supplied, each bookmaker's two-way line is
margin-adjusted separately and fair over probabilities are aggregated across
bookmakers. The original baseline used only over/under `2.5`: it is a common,
simple binary market. The full ladder contains more information about the
shape of the total-goals distribution. Half-goal lines can add constraints:

```text
P(over L) = P(X + Y > L), for L in {0.5, 1.5, 2.5, ...}
```

Integer and quarter Asian totals are stored for diagnostics but do not yet
enter calibration. Integer lines require an explicit push treatment. Quarter
lines require their split-line half-stake settlement formula. Adding those
formulas is a deliberate future task rather than silently treating Asian
totals as ordinary binary markets.

The recommendation report lists all available ladder lines, the half-goal
lines actually used as targets, and the skipped Asian lines. For each used
half-goal line it also reports the aggregated fair market over probability,
the fitted Poisson over probability, and their signed difference.

Lambdas minimise squared calibration error. Calibration runs the bounded
optimiser from several starting points and keeps the best converged fit. This
improves robustness for extreme favourites without changing the loss function.
Near-bound lambdas are reported because they can indicate model degradation.
With optional market constraints:

```text
loss = w_1x2 * sum(outcome errors squared)
     + w_ou  * (P(X+Y >= 3) - P_market(over 2.5))^2
     + w_totals * sum_L (P(X+Y > L) - P_market(over L))^2
     + w_btts * (P(X>0 and Y>0) - P_market(BTTS yes))^2
```

## Expected Pool Points

For a group-stage score prediction `(a,b)`:

```text
EV(a,b) = sum_{x,y} P(X=x,Y=y) S_group(a,b;x,y)
```

Under the baseline additive knockout interpretation:

```text
EV(a,b,q)
 = 1
 + 10 P(q qualifies)
 + 6 P(exact score)
 + 4 P(correct goal difference)
```

Knockout score timing remains configurable because the competition app's exact
interpretation must be confirmed.

## Optional Correct-Score Market Blend

Correct-score odds may be supplied in long format:

```text
match_id,bookmaker,score_a,score_b,decimal_odds
```

Margin is removed within each bookmaker's full available correct-score market
before bookmaker aggregation. For bookmaker `b` and scoreline `s`:

```text
q_b(s) = 1 / O_b(s)
R_b    = sum_s q_b(s)
p_b(s) = q_b(s) / R_b
```

Decimal odds are never averaged directly. Doing so would mix prices with
different margins and exploit neither their probability scale nor their
coherent bookmaker-level market structure.

The fair bookmaker probabilities are then aggregated scoreline by scoreline to
produce:

```text
P_market(score)
```

Available aggregation methods are `mean`, `median`, `trimmed_mean`,
`winsorized_mean`, and `reliability_weighted_mean`. The default `auto` policy
uses:

```text
5+ bookmakers  -> winsorized_mean
3-4 bookmakers -> median
1-2 bookmakers -> mean
```

Outlier detection operates on `log(p_b(s))` within each scoreline using the
median and median absolute deviation. Observations with robust z-scores above
the configured threshold are reported. They are not deleted by default:
`winsorized_mean` caps their influence, while `reliability_weighted_mean`
downweights them.

When this optional market is supplied, optimisation uses:

```text
P_final(score)
= w P_poisson(score)
  + (1 - w) P_market(score)
```

where `0 <= w <= 1`. The default `w=1` exactly preserves the Poisson-only
workflow. Blending is suppressed for matches with fewer than the configured
`min_scorelines_for_blend` usable scorelines; those matches fall back to pure
Poisson while retaining direct-market diagnostics and a warning. Diagnostics
report the top market-implied and blended scorelines plus:

```text
D_KL(P_market || P_poisson)
```

The direct market matrix is conditional on the explicitly supplied scorelines.
Correct-score markets should therefore be collected consistently across
bookmakers and interpreted with care because their margins can be substantial.
Best odds for individual scorelines are not enough to recover fair
probabilities: the best prices may come from different bookmakers and do not
form one coherent market with a meaningful overround.

All supplied correct-score odds contribute to each bookmaker's overround and
fair probabilities, including long-shot scores beyond the configured finite
matrix. When the direct market is converted into the finite EV matrix,
out-of-grid scorelines are omitted, the represented grid is renormalised, and
the omitted scoreline labels are reported as a coverage warning.

## Finite Score Grid And Tail Mass

The score matrix contains scores from `0-0` through `max_goals-max_goals`.
Before renormalisation:

```text
tail mass = 1 - sum_{x=0..max_goals, y=0..max_goals} P(X=x,Y=y)
```

Version 1 reports this omitted tail mass. By default, the finite grid is
renormalised before EV optimisation, so optimisation is conditional on scores
inside the represented grid. This is a controlled approximation and is
negligible when the reported tail is small. Raw non-renormalised grids remain
available for diagnostics, but the optimiser rejects them because an incomplete
grid cannot produce internally consistent pool-point EV without modelling its
tail outcomes.
