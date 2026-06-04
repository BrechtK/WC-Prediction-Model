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

The default method removes the margin by normalised inverse odds:

```text
p_i = q_i / R
```

Fair probabilities are calculated per bookmaker and market, then aggregated
across bookmakers.

The live workflow can also compare diagnostic challenger methods. These do not
change the default recommendation unless `margin_removal_method` is explicitly
configured.

Power margin removal finds an exponent `alpha` such that:

```text
sum_i q_i^alpha = 1
p_i = q_i^alpha
```

Additive margin removal subtracts equal overround mass from each outcome:

```text
p_i = q_i - (R - 1) / n
```

where `n` is the number of outcomes. If this produces a non-positive
probability, the method is flagged as invalid for that market rather than being
silently trusted. Odds-ratio methods are not promoted until they are
implemented and tested with clear numerical safeguards.

Shin margin removal is implemented as an optional research method. It models
bookmaker overround as partly arising from insider-informed betting and solves
for a market-level parameter `z`. Larger `z` implies a stronger Shin correction.
The implementation returns `shin_z` diagnostics and rejects markets where a
valid positive probability vector cannot be found. Shin can be useful for
favourite-longshot bias diagnostics, but it remains a challenger rather than a
live default. It is especially uncertain for sparse correct-score markets,
where the listed scorelines may not represent a complete bookmaker market.

The `margin_methods` diagnostics compare calibrated lambdas, fit errors, EV
recommendations, and warning flags across implemented methods. Disagreement
between methods is a sensitivity signal, not an automatic model switch.

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

Integer and quarter Asian totals are stored for the market-consistent diagnostic
challenger and skipped from the default Poisson calibration. Integer lines use
explicit push treatment, and quarter lines split stake across adjacent integer
and half-goal lines when priced diagnostically.

The market-consistent challenger represents totals with expected-profit
constraints at fair decimal odds. For half-goal, integer Asian, and quarter
Asian total lines, two-way margin removal makes fair Under the complement of
fair Over. The Under expected-profit vector is therefore a scalar multiple of
the Over vector, including push and half-push outcomes. The optimiser keeps
one independent Over-side constraint per total line so a line is not silently
double-weighted. Both Over and Under odds are still parsed and stored for
diagnostics and fair-odds checks.

The recommendation report lists all available ladder lines, the half-goal
lines actually used as targets, and the skipped Asian lines. For each used
half-goal line it also reports the aggregated fair market over probability,
the fitted Poisson over probability, and their signed difference.

Asian handicap odds add direct market information about the goal-difference
distribution `X - Y`. For a team-A handicap `h`, a team-A stake at decimal odds
`O` settles from:

```text
adjusted_margin = X - Y + h
profit = O - 1    if adjusted_margin > 0
profit = 0        if adjusted_margin = 0
profit = -1       if adjusted_margin < 0
```

Team B uses the opposite side. Quarter lines split into two half-stake
components, for example `-3.25 = 0.5 * -3.0 + 0.5 * -3.5`.

Because integer and quarter handicaps include pushes or half-pushes, their
quoted sides are not simple binary event probabilities. The workflow removes
the two-way bookmaker margin to get fair decimal odds. It parses and reports
the full available handicap ladder, but it does not feed every parsed line into
the market-consistent challenger. Only a stable subset is used as soft
expected-profit constraints:

```text
E_P[profit(handicap side at fair odds)] ~= 0
```

Near-money handicap lines, better bookmaker coverage, and lower overround get
higher diagnostic weight. By default the challenger keeps only the strongest
small set of Asian handicap constraints, prioritising lines around the market
balance point. Lines with very high or very low cover probability are skipped
as constraints and retained for diagnostics.

Deep lines can be useful tail evidence, but they are unsafe when the finite
score grid cannot represent the tail needed to settle the line. For example,
with a `0..8` score grid, `Team A -7.5` can only cover as `8-0` inside the
grid, while the real market also prices `9-0`, `9-1`, `10-0`, and other
out-of-grid scores. The live challenger therefore marks such lines as
grid-boundary sensitive and skips them from the constraint set. The
`asian_handicap` workbook sheet records `selected_for_market_consistent`,
`selection_weight`, `selection_reason`, and `skipped_reason` for each line.
Asian handicap is optional and does not enter default Poisson calibration or
override the default EV recommendation automatically.

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

Independent Poisson remains the baseline scoreline model and the default input
to live recommendations unless the existing optional correct-score blending
policy applies.

## Dixon-Coles Low-Score Challenger

Dixon-Coles is an optional challenger to the independent-Poisson baseline. It
keeps the calibrated Poisson lambdas and applies a dependence correction only
to the four lowest scorelines:

```text
tau(0,0) = 1 - lambda_A lambda_B rho
tau(1,0) = 1 + lambda_B rho
tau(0,1) = 1 + lambda_A rho
tau(1,1) = 1 - rho
tau(x,y) = 1 otherwise

P_DC(x,y) = tau(x,y) P_poisson(x,y)
```

The finite matrix is normalised before EV optimisation. The configurable
parameter defaults to `rho=0.0`, which exactly reproduces independent Poisson.
The live workflow reports Dixon-Coles rho, its recommended scoreline, its EV
gap, its top five EV predictions, and whether it changes the final live
recommendation. When correct-score market probabilities contain the low-score
cells `0-0`, `1-0`, `0-1`, and `1-1`, the workflow estimates rho over a bounded
grid and reports `dixon_coles_rho_used`, `dixon_coles_rho_source`, and
`dixon_coles_rho_fit_error`. Missing or sparse correct-score data falls back to
the configured rho and emits a warning. Dixon-Coles remains diagnostic-only:
it must be validated out of sample before it can become a default
recommendation model.

## Public-Field Strategy Challenger

The default recommendation maximises expected pool points:

```text
score_EV = argmax_s EV(s)
```

Large public contests can also depend on how many other players choose the
same scoreline. The public-field strategy layer estimates a transparent public
pick distribution over scorelines using:

- model scoreline probabilities;
- common public scorelines such as `1-0`, `2-0`, `2-1`, `1-1`, and `0-0`;
- team popularity weights;
- favourite direction and strength;
- correct-score market top scorelines when available;
- friend-prediction crowding when supplied.

For a candidate scoreline `s`, the diagnostic ranking combines expected points,
estimated crowding, scoreline upside, and an explicit EV-cost penalty:

```text
public_ranking_score(s)
= EV(s)
  + alpha * (1 - public_pick_share(s))
  + beta  * upside(s)
  - gamma * max(EV(score_EV) - EV(s), 0)
```

Candidates must also pass configurable safety checks for maximum EV loss,
minimum exact-score probability, and minimum result probability. Severe model
warnings, such as high calibration error or high tail mass, force the public
strategy back to the pure-EV recommendation.

The default `strategy_mode=ev` disables all public-field switching and reports
the pure-EV score. Other modes are diagnostics for contest strategy review.
They must be validated against realised leaderboard outcomes before replacing
the default live submission.

Public strategy can be scaled by target field size: `friends` keeps
crowd/contrarian influence low, `balanced` is moderate, and `national` gives
stronger decorrelation diagnostics. This scaling affects only the public
strategy diagnostic score; it does not replace the default EV submission.

The public-pick power exponent, common-score multiplier, Belgium/public-team
bias, and alpha/beta/gamma ranking weights are heuristic and unvalidated. They
are retained as diagnostic-only parameters, not as public strategy features
that should override the final live decision.

## Final Decision Dashboard

The final decision dashboard is not a new probability model. It is a reporting
layer over existing quantities:

```text
recommended_score = argmax_s EV(s)
final_decision_score = recommended_score
```

It then classifies model consensus, EV-gap confidence, challenger disagreement,
margin-method sensitivity, public-strategy alternatives, high-score clusters,
and warning flags. Manual-review and override-candidate labels are decision
support only; they do not alter calibration, score probabilities, or EV
optimisation.

## Expected Pool Points

For a group-stage score prediction `(a,b)`:

```text
EV(a,b) = sum_{x,y} P(X=x,Y=y) S_group(a,b;x,y)
```

The margin diagnostics sheet rewrites the same group-stage EV by predicted
goal-difference margin `d = a - b`. For decisive margins:

```text
EV(d) = 1
      + 4 * P(correct result)
      + 2 * P(X - Y = d)
      + 3 * max_{a-b=d} P(a,b)
```

For draws:

```text
EV(0) = 1
      + 6 * P(draw)
      + 3 * max_{a=b} P(a,b)
```

This is algebraically equivalent to the raw scoreline EV optimiser for
group-stage scoring and is used only to explain why a margin and its best
representative scoreline win. The dashboard also reports the best draw score,
best decisive score, and the draw-vs-decisive EV gap for balanced matches.

The implemented group-stage scoring rule is:

```text
exact score                      -> 10 points
correct goal difference/result   -> 7 points
correct result                   -> 5 points
participation/otherwise          -> 1 point
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
interpretation must be confirmed. The default knockout scoring mode is
`unverified`, which preserves the existing additive calculation while warning
that the Sporza rule has not been confirmed. `additive` explicitly stacks score
components, while `hierarchical` lets an exact score supersede the
goal-difference score component. Before relying on knockout EV, verify the
actual Sporza knockout rules, especially score timing, qualifier treatment, and
whether score and qualifier points are additive.

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
workflow. A value such as `w=0.85` may be useful in research comparisons, but
it is not the conservative live default until validated by backtesting.
Blending is suppressed for matches with fewer than the configured
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
