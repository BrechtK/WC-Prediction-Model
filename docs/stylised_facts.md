# Stylised Facts For World Cup Predictions

Domestic-league backtests are useful for validating scoring logic, calibration,
and strategy behaviour. They are not a complete simulation of a World Cup. The
following caveats matter when interpreting historical results and live
recommendations.

## Modelling Caveats

### Domestic Leagues Underrepresent Extreme Mismatches

Domestic leagues contain many competitive matches and relatively few games
between teams at opposite ends of the international-strength distribution.
World Cup group stages can include much larger gaps. A heuristic such as
`favourite_1_0` may look strong in league backtests while becoming less
appropriate for extreme favourites.

### International Data Is Sparse And Heterogeneous

National teams play fewer matches than clubs. Opponent strength, tournament
context, qualification campaigns, friendlies, squad availability, travel, and
managerial changes vary substantially. A historical international sample is
therefore smaller and less uniform than a domestic-league sample.

### Group-Stage Incentives Vary

A team's incentives can change by matchday and current standings. Goal
difference, qualification scenarios, rotation, and whether a draw is sufficient
can affect the style and risk profile of a match. A market-implied baseline
absorbs some of this information only when the odds are collected late enough.

### Qualification And Score Predictions Are Different Objects

In knockout rounds, predicting the team that advances is not the same as
predicting the score before penalties. A match can be drawn while one team still
qualifies. The project therefore treats qualification probabilities and
scoreline probabilities separately.

### World Cup Matches Are Usually Neutral-Site

Domestic historical data is organised around home and away teams. World Cup
matches are generally played at neutral venues, even though the input schema
uses `team_a` and `team_b`. Domestic home advantage must not be interpreted as a
direct analogue for tournament matches.

### Odds Timing Matters

Closing odds generally incorporate more information than early prices:
confirmed line-ups, injuries, weather, tactical expectations, and market
liquidity. Early odds remain useful for planning, but recommendations should be
refreshed nearer kick-off when possible.

### Margin Removal Is A Modelling Choice

Bookmaker prices include overround. Proportional margin removal is transparent
and is the Version 1 default, but it is not the only possible method. Different
removal methods can shift the fair probabilities used for calibration,
especially when prices are asymmetric.

### Favourite-Longshot Bias May Affect Extreme Probabilities

Longshots and heavy favourites may not be priced symmetrically after margin.
Extreme probabilities deserve particular scrutiny because domestic data offers
less empirical support and simple proportional margin removal may not fully
capture market bias.

### Correct-Score Markets Contain Information But Carry High Margins

Correct-score odds can reveal how the market distributes probability across
specific outcomes such as `1-0`, `2-0`, and `3-0`. They are also fragmented and
often have higher margins than 1X2 markets. They should be treated as a useful
diagnostic or carefully validated extension, not as automatically superior
inputs.

### Independent Poisson Is A Transparent Approximation

Independent Poisson calibration translates market probabilities into a complete
score matrix. Real football scores are not perfectly independent Poisson draws.
The baseline may overstate or understate draws and low-scoring outcomes because
of tactical dependence, game state, and score correlation.

### O/U And BTTS Markets Add Important Context

The same 1X2 favourite probability can arise in different scoring
environments. Over/under 2.5 and both-teams-to-score odds help distinguish a
low-scoring favourite from a high-scoring mismatch, which can change the
EV-optimal pool prediction.

## Model Implications

| Caveat | Project implication |
| --- | --- |
| Extreme mismatches are underrepresented domestically | Use favourite-strength buckets and `scripts/inspect_synthetic_mismatches.py` before trusting league-wide averages for strong favourites. |
| International data is sparse and heterogeneous | Treat domestic backtests as baseline validation, not proof of tournament performance. Add international evaluation only with careful sample labelling. |
| Group-stage incentives vary | Refresh odds near kick-off and inspect match context, especially on the final group matchday. |
| Qualification differs from score prediction | Enter qualification odds for knockout rounds whenever available. The draw-split fallback is intentionally weaker. |
| World Cup matches are neutral-site | Interpret `team_a` and `team_b` as ordered participants, not literal domestic home and away sides. |
| Odds timing matters | Prefer closing or late prices for final submissions. Record collection timing in `notes` when manually entering odds. |
| Margin removal matters | Keep proportional removal as the transparent baseline and use alternative methods only as explicit sensitivity checks. |
| Favourite-longshot bias affects extremes | Manually inspect very high `p_fav` matches and compare modal, EV-optimal, and top-five scorelines. |
| Correct-score odds are informative but high-margin | Keep correct-score markets as a future validated extension or inspection source rather than blending them automatically. |
| Independent Poisson may miss score dependence | Monitor calibration error, tail mass, and low-score recommendations. Evaluate richer challenger models out of sample before adopting them. |
| O/U and BTTS add context | Collect these optional odds where possible because they improve scoreline calibration without changing the core baseline architecture. |

