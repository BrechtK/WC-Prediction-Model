# Stylised Facts For World Cup Predictions

Domestic-league backtests are useful for validating scoring logic, calibration,
and strategy behaviour. They are not a complete simulation of a World Cup.
Version 1 therefore uses bookmaker odds as a transparent baseline and treats
tournament-specific risks as explicit diagnostics.

## Modelling Caveats

### Domestic Leagues Underrepresent Extreme International Mismatches

Premier League and Belgian league seasons contain relatively few matches with
85%-95% favourites. World Cup groups can contain much larger strength gaps
between elite and weak international teams. As a result, `favourite_1_0` can
look unusually strong in domestic backtests while becoming less appropriate for
extreme international mismatches. The favourite-strength bucket reports and
`scripts/inspect_synthetic_mismatches.py` stress test are designed to expose
this issue.

### International Football Data Is Sparse And Heterogeneous

National teams play fewer matches than clubs. Friendlies, qualifiers, Nations
League matches, continental tournaments, and World Cup matches are not fully
comparable. Squads, managers, travel, and motivation also change over time.
This supports using bookmaker odds as the Version 1 baseline instead of
immediately fitting a small homemade team-strength model.

### Group-Stage Incentives Vary

Matchdays 1, 2, and 3 are strategically different. Already-qualified teams may
rotate. Some teams only need a draw, while others may need to win by multiple
goals. Late odds should partly incorporate these incentives, but early odds may
not. For now this belongs in `notes` and manual review rather than as a
hard-coded model input.

### Knockout Score And Qualification Are Different Objects

In knockout rounds, the 90-minute or 120-minute score and the team that advances
are different probability objects. A team can qualify after penalties even when
the match is drawn. The project intentionally keeps score and qualifier
predictions separate.

### Neutral-Site And Host Effects

Domestic data contains ordinary home advantage. World Cup matches are usually
neutral-site, except for host-country, crowd, travel, and climate effects.
Market odds should price much of this information, but domestic home-away
backtests are not a perfect analogue for tournament matches.

### Odds Timing Matters

Closing odds usually incorporate more information than early prices. Late
injuries, line-ups, rotation, and motivation can materially shift the market.
When the pool deadline is early, record `odds_timestamp` so the age of the odds
is visible. Final recommendations should use the latest available prices before
the deadline where possible.

### Margin-Removal Method Matters

Proportional normalisation is a clear baseline, but it is not the only approach.
Alternatives include additive removal, the power method, the Shin method, and
methods adjusted for favourite-longshot bias. Since every downstream estimate
depends on fair market probabilities, sensitivity analysis for margin removal
is a high-priority future improvement.

### Favourite-Longshot Bias And Public-Team Effects

Market-implied probabilities are not literal true probabilities. Famous teams,
longshots, and extreme favourites may be priced differently by bookmaker and
market. Aggregating multiple bookmakers and preferring sharper or closing odds
can reduce this risk, especially for very high `p_fav` matches.

### Correct-Score Markets May Contain Direct Scoreline Information

The current model infers scoreline probabilities from 1X2, over/under, and BTTS
odds where available. Correct-score odds are directly relevant to this pool,
but often carry larger margins and lower liquidity. They should be introduced
carefully through explicit calibration targets or validated blending.

### Independent Poisson And Low-Score Draw Issues

Independent Poisson calibration is transparent and useful, but real football
scores are not perfectly independent Poisson draws. The baseline may misstate
draws and low-scoring outcomes such as `0-0`, `1-0`, `0-1`, and `1-1`.
Dixon-Coles or related low-score corrections are natural future extensions.

### Total-Goals Environment Matters

1X2 odds alone cannot distinguish a low-total 70% favourite from a high-total
70% favourite. Over/under 2.5 and both-teams-to-score odds help separate
favourite strength from the total-goals environment. This is why the
`ev_optimal_1x2_over_under` strategy is an important backtest comparison.

## Model Implications

| Caveat | Why it matters | Current mitigation in the project | Future improvement |
| --- | --- | --- | --- |
| Extreme international mismatches | Domestic averages may overstate the usefulness of `favourite_1_0`. | Favourite-strength buckets and synthetic mismatch inspection. | Add carefully labelled international backtests and inspect extreme buckets separately. |
| Sparse, heterogeneous international data | Small samples can encourage fragile team-strength estimates. | Use market odds as the Version 1 baseline. | Add challenger models only after robust out-of-sample evaluation. |
| Variable group-stage incentives | Rotation and standings scenarios can shift score distributions. | Manual `notes` and late-odds collection. | Add contextual diagnostics before considering hard-coded features. |
| Score and qualification differ | A drawn knockout score can still produce a qualifier. | Separate qualifier odds and score optimisation. | Improve knockout backtesting and validate score-timing rules. |
| Neutral-site and host effects | Domestic home-away data is an imperfect tournament analogue. | Let the market price venue effects; interpret `team_a` and `team_b` as ordered sides. | Add explicit venue metadata to future challenger models. |
| Odds timing | Early prices may omit material late information. | Optional `odds_timestamp` plus stale-odds flags. | Compare early and closing prices in historical evaluation. |
| Margin-removal choice | Fair probabilities feed every later step. | Transparent proportional baseline. | Backtest additive, power, Shin, and bias-adjusted alternatives. |
| Favourite-longshot and public-team effects | Extreme prices may be systematically distorted. | Multiple-bookmaker aggregation, source-quality metadata, and extreme-favourite flags. | Prefer sharp closing sources and evaluate bias-adjusted methods. |
| Correct-score market information | Direct score prices may improve prediction-pool decisions but have high margins. | Loader support exists without automatic blending. | Build validated correct-score matrices and backtest blend weights. |
| Independent Poisson low-score limitations | Draw and low-score probabilities may be inaccurate. | Calibration warnings, tail diagnostics, and a transparent baseline. | Test Dixon-Coles against independent Poisson. |
| Total-goals environment | Identical 1X2 prices can imply different optimal scores. | Optional O/U and BTTS calibration targets with presence flags. | Review weights and compare market combinations by favourite bucket. |

