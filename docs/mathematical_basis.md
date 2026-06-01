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

Lambdas minimise squared calibration error. With optional market constraints:

```text
loss = w_1x2 * sum(outcome errors squared)
     + w_ou  * (P(X+Y >= 3) - P_market(over 2.5))^2
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
