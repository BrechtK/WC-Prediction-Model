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

