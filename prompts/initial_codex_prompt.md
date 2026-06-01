I want to build a serious Python project for a World Cup football/soccer score prediction competition with friends.

This should be treated as a quant-style market-implied prediction-pool optimiser.

## Core philosophy

The goal is NOT to beat bookmakers.

Assume that, for a high-liquidity tournament such as the 2026 FIFA World Cup, bookmaker odds are likely to contain better information about match outcome probabilities than a small homemade model based on limited international-football data.

Therefore, Version 1 of the project should use bookmaker odds as the primary source of probability information.

The project should follow this pipeline:

```
bookmaker odds
    -> margin removal / fair probabilities
    -> market-implied scoreline distribution
    -> expected-points optimisation under my private competition rules
    -> tracking of friends' predictions and realised standings
```

The main edge is not finding betting mispricings. The edge is translating market-implied probabilities into optimal predictions under a scoring system that rewards exact scores, goal differences, correct results, and knockout qualification.

This is similar to using market prices to infer an implied distribution, then valuing a custom payoff.

## Competition rules

The competition is about predicting football match scores.

GROUP STAGE SCORING:

* 10 points: exact score correctly predicted.
* 7 points: correct goal difference, but not the exact score.
  Example: prediction 3-1, actual 2-0 -> both goal difference +2.
* 5 points: correct result/winner/draw, but wrong exact score and wrong goal difference.
  Example: prediction 1-0, actual 3-0 -> correct winner, wrong goal difference.
* 1 point: submitting a prediction, regardless of correctness.

Interpretation:
For group-stage matches, the total score for a prediction should be one of:

* 10 if exact score
* 7 if correct goal difference but not exact
* 5 if correct result but wrong exact score and wrong goal difference
* 1 otherwise

KNOCKOUT STAGE SCORING:
From the round of 16 onward:

* +10 points: correctly predicting which country advances, including if the decision is after penalties.
* +6 points: exact score after 90 minutes, or after 120 minutes if extra time is played.
* +4 points: correct goal difference after 90 or 120 minutes.
* +1 point: submitting a prediction.

Important:
For knockout matches, qualification prediction and score prediction should be treated separately. A team can qualify after penalties even if the match is drawn after 120 minutes.

The knockout scoring logic should be configurable because the exact app interpretation may need to be checked later.

## Primary objective

Build a clean, modular Python project that can:

1. Read betting odds from different bookmakers.
2. Convert bookmaker odds into raw implied probabilities.
3. Remove the bookmaker margin/overround to obtain fair market-implied probabilities.
4. Aggregate fair probabilities across bookmakers.
5. Infer or calibrate a coherent scoreline probability distribution.
6. Optimise predicted scores under the competition scoring rules by maximising expected competition points.
7. Read predictions submitted by my friends.
8. Compare my friends' predictions with the model recommendations.
9. Track realised results and compute standings.
10. Generate clear reports, CSV/Excel outputs, and documentation.
11. Be extensible so that later we can add:

    * exact-score odds
    * over/under odds
    * Asian handicap odds
    * both-teams-to-score odds
    * qualification odds
    * Elo/xG features
    * Dixon-Coles correction
    * bivariate Poisson
    * negative binomial
    * ensemble methods
    * game-theoretic/leaderboard strategy
    * backtesting
    * challenger models against the market-implied baseline

This should be a serious project with good structure, documentation, tests, and clear separation between data, market-implied probabilities, scoreline modelling, optimisation, friend-prediction tracking, results, and reporting.

## Market data and odds philosophy

The model should treat bookmaker odds as the main information source.

The most important odds markets are:

1. 1X2 odds:

   * Team A wins
   * Draw
   * Team B wins

2. Over/under total-goals odds, especially:

   * over 2.5 goals
   * under 2.5 goals

3. Both-teams-to-score odds:

   * yes
   * no

4. Correct-score odds:

   * if available, these are extremely useful because they directly imply scoreline probabilities.

5. Knockout qualification odds:

   * Team A qualifies
   * Team B qualifies

6. Optional later:

   * Asian handicap
   * draw-no-bet
   * double chance
   * team totals

The initial implementation should work with only 1X2 odds, but the architecture should support additional markets later.

Important modelling hierarchy:

* If only 1X2 odds are available, calibrate a scoreline model to match fair win/draw/loss probabilities.
* If over/under odds are available, also calibrate to match the market-implied total-goals probability.
* If both-teams-to-score odds are available, also use them as calibration constraints.
* If correct-score odds are available, use them to infer or blend a direct market-implied scoreline distribution.
* If qualification odds are available in knockout matches, use them directly for the advancement component.

## Odds input format

Support CSV and preferably Excel.

Example odds input:

```
match_id,date,stage,team_a,team_b,bookmaker,odds_a_win,odds_draw,odds_b_win,odds_over_2_5,odds_under_2_5,odds_btts_yes,odds_btts_no,odds_a_qualifies,odds_b_qualifies
```

Correct-score odds may be in a separate long-format file:

```
match_id,bookmaker,score_a,score_b,decimal_odds
```

Example:

```
M001,Bookmaker1,0,0,8.50
M001,Bookmaker1,1,0,6.50
M001,Bookmaker1,1,1,6.00
M001,Bookmaker1,2,0,8.00
```

The code should handle missing columns and missing odds gracefully.

## Odds processing requirements

Implement functions to:

1. Convert decimal odds to raw implied probabilities:

   ```
   q_i = 1 / odds_i
   ```

2. Calculate overround:

   ```
   overround = sum(q_i)
   ```

3. Remove margin using at least proportional normalisation:

   ```
   p_i = q_i / sum(q_i)
   ```

4. Make margin-removal methods pluggable.

Create an interface for several margin-removal methods:

* proportional normalisation
* additive correction
* power method
* Shin method, optional/TODO if not implemented immediately

The initial fully working method should be proportional normalisation.

Implement aggregation across bookmakers:

* mean fair probability
* median fair probability
* optionally best/sharp bookmaker only
* optionally weighted average if weights are provided

Add validation:

* decimal odds must be greater than 1
* raw implied probabilities must be positive
* fair probabilities must sum to approximately 1
* warn if overround is suspiciously high or below 1
* handle missing odds cleanly

## Mathematical scoreline modelling

The project needs a full scoreline distribution:

```
P(X = x, Y = y)
```

where:

```
X = goals scored by team A
Y = goals scored by team B
```

This is necessary because the competition rewards exact scores and goal differences, not just match outcomes.

Baseline model:
Use an independent Poisson score model as the first transparent baseline:

```
X ~ Poisson(lambda_A)
Y ~ Poisson(lambda_B)

P(X=x,Y=y)
  = exp(-lambda_A) lambda_A^x / x!
    * exp(-lambda_B) lambda_B^y / y!
```

The project should generate a score-probability matrix from 0-0 to max_goals-max_goals, for example 6-6 or 8-8.

Tail probability beyond the grid should be:

* calculated if possible
* reported
* either ignored after renormalisation or included in a warning
* handled consistently

The independent Poisson model is a baseline, not necessarily the final truth.

Design the modelling layer so that alternative models can be added later:

* Dixon-Coles adjusted Poisson
* bivariate Poisson
* negative binomial / overdispersed count models
* direct correct-score market distribution
* market-blended model
* ensemble model
* xG/Elo-based model
* Skellam-style model
* ML-based lambda or scoreline model

Use a common model interface, for example:

```
class ScoreModel:
    def predict_score_matrix(self, match, max_goals: int) -> ScoreProbabilityMatrix:
        ...
```

The optimiser must be model-agnostic. It should only require a score-probability matrix and the relevant scoring rule.

## Calibration requirements

The main initial calibration problem is:

Given fair market-implied probabilities from odds, estimate lambda_A and lambda_B such that the Poisson score matrix approximately reproduces the market probabilities.

With only 1X2 odds:

```
P(A wins) = sum over x > y of P(X=x,Y=y)
P(draw)   = sum over x = y of P(X=x,Y=y)
P(B wins) = sum over x < y of P(X=x,Y=y)
```

Calibrate lambda_A and lambda_B by minimising squared errors versus market-implied probabilities.

Use scipy.optimize.

Objective example:

```
loss(lambda_A, lambda_B)
  = w_1x2 * [
        (P_model_A_win - P_market_A_win)^2
      + (P_model_draw  - P_market_draw)^2
      + (P_model_B_win - P_market_B_win)^2
    ]
```

Ensure lambda_A and lambda_B are positive. Use bounds.

If over/under 2.5 odds are available, include:

```
P(total goals > 2.5) = P(X + Y >= 3)
P(total goals < 2.5) = P(X + Y <= 2)
```

Add this to the calibration loss.

If both-teams-to-score odds are available, include:

```
P(BTTS yes) = P(X > 0 and Y > 0)
P(BTTS no)  = 1 - P(BTTS yes)
```

If correct-score odds are available:

* Convert correct-score odds into fair score probabilities after margin removal.
* Use these either as:

  1. a direct market-implied scoreline matrix, or
  2. an additional calibration target, or
  3. a blended model with Poisson-implied probabilities.
* Keep this modular/configurable.

Calibration output should include:

* lambda_A
* lambda_B
* model-implied 1X2 probabilities
* target market 1X2 probabilities
* model-implied over/under probability if used
* target market over/under probability if used
* calibration loss
* warnings if fit is poor

## Group-stage scoring and optimisation

Create a pure function:

```
score_group_prediction(pred_a, pred_b, actual_a, actual_b) -> int
```

Rules:

* exact score -> 10
* correct goal difference but not exact -> 7
* correct result/winner/draw but wrong exact and wrong goal difference -> 5
* otherwise -> 1

Define helper functions:

* goal_difference(a,b) = a - b
* result_sign(a,b) = sign(a-b), with draw = 0

Expected points for a predicted score (a,b):

```
EV(a,b) =
    sum over all possible actual scores (x,y):
        P(X=x,Y=y) * score_group_prediction(a,b,x,y)
```

The optimiser should evaluate all candidate predictions from 0-0 to max_candidate_goals-max_candidate_goals and return:

* best prediction
* expected points
* exact-score probability
* correct-goal-difference probability
* correct-result probability
* probability of getting only participation point
* top N alternative predictions ranked by expected points
* most likely raw scoreline
* note if EV-optimal score differs from most likely scoreline

Important:
The optimiser should not simply select the most likely scoreline. It should select the scoreline that maximises expected competition points.

## Knockout scoring and optimisation

Create a knockout scoring module, but keep it configurable.

Input:

* predicted score after regulation/extra-time convention
* predicted qualifier
* actual score after regulation or after extra time depending on rules
* actual qualifier

Base rule interpretation:

```
points = 1
       \+ 10 * I(predicted_qualifier == actual_qualifier)
       \+ 6  * I(predicted_score == actual_score)
       \+ 4  * I(predicted_goal_difference == actual_goal_difference)
```

However, because the competition app may have details about whether exact score is after 90 minutes or 120 minutes, implement a config object such as:

```
KnockoutScoringConfig:
    score_basis: "90min" | "120min_if_extra_time" | "final_before_penalties"
    exact_score_points: 6
    goal_difference_points: 4
    qualifier_points: 10
    participation_points: 1
    additive: True
```

Expected points for knockout prediction:

```
EV(score_a, score_b, predicted_qualifier)
   = 1
   \+ 10 * P(predicted_qualifier qualifies)
   \+ 6  * P(exact score)
   \+ 4  * P(correct goal difference)
```

For now, if detailed extra-time score probabilities are not available:

* use 90-minute 1X2 and scoreline odds for score component
* use qualification odds for qualifier component
* clearly document the approximation

If qualification odds are available, use them directly after margin removal.
If not available, approximate qualification probability from 90-minute probabilities, but flag this as a weaker approximation.

## Friend prediction tracking

Read friend predictions from CSV or Excel.

Example format:

```
match_id,player,predicted_team_a_goals,predicted_team_b_goals,predicted_qualifier
```

For group-stage matches, predicted_qualifier may be empty.

Implement functions to:

* calculate the model-implied expected points of each friend prediction before the match
* compare each friend prediction to the model recommendation
* rank friend predictions by expected value
* identify low-EV and high-EV predictions
* identify contrarian predictions relative to the model and other friends
* after results are available, calculate realised points

## Results input

Read match results from CSV or Excel.

Example:

```
match_id,team_a_goals_90,team_b_goals_90,went_to_extra_time,team_a_goals_120,team_b_goals_120,qualifier
```

For group-stage matches:

* use team_a_goals_90 and team_b_goals_90

For knockout matches:

* use the scoring config to decide whether to use 90-minute or 120-minute score
* use qualifier for the qualification points

## Backtesting requirement

The project should be designed so that the final model can be backtested on historical football odds and results.

Backtesting is not required to be fully implemented in the first version, but the architecture should make it easy to add.

The expected future data source is historical football odds/results data, for example Football-Data.co.uk, which provides historical match results and bookmaker odds for many leagues.

The backtesting module should eventually:

1. Load historical match odds and results.
2. Remove bookmaker margin from historical odds.
3. Calibrate the scoreline model from historical market-implied probabilities.
4. Generate predictions using multiple strategies:

   * naive always 1-1
   * naive always 0-0
   * favourite wins 1-0
   * favourite wins 2-0
   * most likely Poisson scoreline
   * expected-points optimal prediction
   * expected-points optimal prediction using 1X2 + over/under odds
   * direct correct-score market prediction, if correct-score odds are available
5. Score predictions using the competition's group-stage scoring rules.
6. Compare strategies using:

   * average realised points per match
   * exact-score hit rate
   * correct goal-difference hit rate
   * correct-result hit rate
   * variance of points
   * cumulative points over time
   * draw prediction performance
   * favourite/underdog split
7. Export backtest results to CSV/Excel and optionally plots.

Design implications:

* The optimiser should not depend on World Cup-specific data.
* Scoring rules should be reusable for historical matches.
* Odds loaders should be modular so that a Football-Data.co.uk loader can be added later.
* Strategy classes/functions should share a common interface.
* Results evaluation should be separated from prediction generation.
* Backtesting should evaluate realised competition points, not betting profit.

For the initial implementation, create only the structure/placeholders for:
src/wc_predictor/backtesting.py
scripts/run_backtest.py
tests/test_backtesting.py

Add clear TODOs and docstrings explaining the intended backtesting workflow.

## Future model-comparison roadmap

Version 1 should use bookmaker-implied probabilities as the primary source of information.

However, the project should be designed so that later versions can compare this market-implied baseline against independent challenger models.

The main future challenger model should be an xG/Elo-based Poisson or Skellam-style model:

```
lambda_A = f(team_a_xG_for, team_b_xG_against, Elo difference, venue, rest, etc.)
lambda_B = f(team_b_xG_for, team_a_xG_against, Elo difference, venue, rest, etc.)
```

This model should generate a full scoreline probability matrix and use the same expected-points optimiser as the market-implied model.

The project should also eventually support blending:

```
P_final = w * P_market + (1 - w) * P_model
```

or blending at the expected-goals level:

```
lambda_final = w * lambda_market + (1 - w) * lambda_model
```

Backtesting should be used to estimate whether challenger models add value relative to the market-implied baseline, and whether the optimal blending weight is close to 1.

Important:
Do not implement these challenger models in Version 1 unless the baseline is already clean and tested. For Version 1, only create the interfaces/placeholders needed to support them later.

Potential future models:

* xG/Elo Poisson model
* Skellam model for goal difference / W-D-L probabilities
* Dixon-Coles correction
* bivariate Poisson
* negative binomial / overdispersed count model
* machine-learning model estimating lambdas or scoreline probabilities
* market/model blending
* copula/HMM/tracking-data models as a later research extension only, not a core priority

## Reporting requirements

Create reporting functions that output:

1. Match-level recommendation report:

   * match_id
   * stage
   * team A vs team B
   * bookmaker probabilities after margin removal
   * aggregated fair market probabilities
   * calibrated lambda_A and lambda_B
   * model-implied 1X2 probabilities
   * model calibration error
   * best EV-optimal prediction
   * expected points of best prediction
   * exact-score probability
   * correct-goal-difference probability
   * correct-result probability
   * top 5 alternative predictions
   * most likely scoreline
   * whether the EV-optimal prediction differs from the most likely scoreline
   * warnings/notes

2. Player/friend pre-match EV report:

   * player
   * match_id
   * prediction
   * model-implied expected points
   * difference versus model-optimal EV
   * rank of the prediction among candidate predictions
   * whether the prediction is consensus/contrarian

3. Standings report after results:

   * player
   * total realised points
   * total expected points before matches
   * average EV per prediction
   * number of exact scores
   * number of correct goal differences
   * number of correct results
   * number of correct qualifiers
   * ranking

4. Export reports to:

   * CSV
   * Excel
   * optionally Markdown

## Project structure

Create a clean Python project structure, for example:

```
wc_prediction_model/
    README.md
    pyproject.toml or requirements.txt
    data/
        raw/
        processed/
        examples/
            example_odds.csv
            example_correct_score_odds.csv
            example_predictions.csv
            example_results.csv
    notebooks/
        exploratory_analysis.ipynb
    src/
        wc_predictor/
            __init__.py
            config.py
            odds.py
            margin.py
            probabilities.py
            score_models.py
            calibration.py
            scoring_rules.py
            optimiser.py
            friends.py
            results.py
            reporting.py
            market_data.py
            backtesting.py
            strategies.py
            utils.py
    tests/
        test_odds.py
        test_margin.py
        test_scoring_rules.py
        test_optimiser.py
        test_probabilities.py
        test_calibration.py
        test_results.py
        test_backtesting.py
        test_strategies.py
    scripts/
        run_predictions.py
        update_standings.py
        generate_report.py
        run_backtest.py
    docs/
        mathematical_basis.md
        research_notes.md
        assumptions.md
        future_model_roadmap.md
```

Use type hints, docstrings, and clear function names.

Use pandas, numpy, scipy and optionally matplotlib.

Do not hardcode absolute paths.

Use pathlib.

## Configuration

Create a central config file or dataclasses for:

* max_goals_score_matrix
* max_candidate_goals
* margin removal method
* bookmaker aggregation method
* calibration weights
* knockout scoring interpretation
* output paths
* warning thresholds
* whether to renormalise score matrix after truncation
* backtesting settings
* strategy settings

## Testing requirements

Write unit tests for:

1. Decimal odds conversion:

   * odds > 1 are accepted
   * invalid odds raise an error
   * raw implied probabilities are calculated correctly

2. Margin removal:

   * fair probabilities sum to 1
   * proportional normalisation works as expected
   * suspicious overround creates warning or flag

3. Group-stage scoring:

   * exact score gives 10
   * correct goal difference gives 7
   * correct winner but wrong goal difference gives 5
   * wrong result gives 1
   * draw handling works correctly
   * exact draw, non-exact draw, and wrong draw cases are tested

4. Knockout scoring:

   * correct qualifier adds 10
   * exact score adds 6
   * correct goal difference adds 4
   * participation point is always included
   * penalties/qualifier logic is separate from score logic

5. Poisson score grid:

   * probabilities are non-negative
   * probability mass approximately sums to 1 or tail mass is reported
   * very low and very high lambdas are handled

6. Calibration:

   * calibrated lambdas approximately reproduce target 1X2 probabilities
   * adding over/under constraints changes total-goals behaviour
   * bad inputs are handled gracefully

7. Optimiser:

   * for a manually defined score-probability matrix, expected points are calculated correctly
   * the optimiser returns the highest-EV prediction
   * the EV-optimal prediction can differ from the most likely scoreline

8. Friend prediction tracking:

   * expected value of friend predictions is calculated
   * realised standings are calculated correctly after results

9. Backtesting placeholders:

   * backtesting module imports successfully
   * strategy interface exists
   * TODOs/docstrings explain intended future workflow

## README requirements

Write a proper README explaining:

* Goal of the project
* Competition rules
* Core philosophy: market-implied probabilities, not trying to beat bookmakers
* How decimal odds imply raw probabilities
* How bookmaker margin/overround is removed
* How probabilities are aggregated across bookmakers
* Why a full scoreline distribution is needed
* Baseline Poisson score model
* Calibration from 1X2 odds
* Optional calibration from over/under, BTTS, and correct-score odds
* Expected-points optimisation
* Difference between most likely score and EV-optimal prediction
* Group-stage versus knockout-stage scoring
* How to run the scripts
* How to add odds, predictions, and results
* Backtesting roadmap
* Future challenger-model roadmap
* Limitations
* Future extensions

## Documentation requirements

Create docs/mathematical_basis.md with the formulas.

Include:

1. Decimal odds to implied probabilities:

   ```
   q_i = 1 / O_i
   ```

2. Overround:

   ```
   R = sum_i q_i
   ```

3. Proportional margin removal:

   ```
   p_i = q_i / R
   ```

4. Independent Poisson score model:

   ```
   P(X=x,Y=y)
   = exp(-lambda_A) lambda_A^x / x!
     exp(-lambda_B) lambda_B^y / y!
   ```

5. Market-implied outcome probabilities from score matrix:

   ```
   P(A wins) = sum_{x>y} P(X=x,Y=y)
   P(draw)   = sum_{x=y} P(X=x,Y=y)
   P(B wins) = sum_{x<y} P(X=x,Y=y)
   ```

6. Calibration loss.

7. Group-stage expected points:

   ```
   EV(a,b) =
       sum_{x,y} P(X=x,Y=y) S_group(a,b;x,y)
   ```

8. Knockout expected points:

   ```
   EV(a,b,q)
      = 1
      \+ 10 P(q qualifies)
      \+ 6 P(exact score)
      \+ 4 P(correct goal difference)
   ```

subject to configuration details.

Create docs/assumptions.md with:

* We assume bookmaker odds are the best available probability source for Version 1.
* We do not assume odds are literally true probabilities.
* We remove bookmaker margin to obtain fair market-implied probabilities.
* We use statistical score models mainly as a translation layer from outcome markets to scoreline distributions.
* The independent Poisson model is the first transparent baseline.
* The optimiser maximises private-pool expected points, not betting profit.
* Future own models must be compared against the market-implied baseline via backtesting.

Create docs/research_notes.md with:

* Dixon and Coles (1997) as the canonical football score-modelling reference.
* Explanation of why Poisson models are common for football scores.
* Known weaknesses of independent Poisson.
* What Dixon-Coles adjusts.
* Why bookmaker odds are a strong practical benchmark.
* Why exact-score prediction pools require scoreline distributions rather than only win/draw/loss probabilities.
* Why market-implied probabilities are the correct Version 1 baseline.
* Future extensions: bivariate Poisson, negative binomial, correct-score market extraction, Elo/xG, ensembles.

Create docs/future_model_roadmap.md with:

* Version 1: market-implied model.
* Version 1.5: backtesting of market-implied baseline.
* Version 2: xG/Elo Poisson or Skellam-style challenger model.
* Version 2.5: market/model blending.
* Version 3: Dixon-Coles and bivariate Poisson extensions.
* Later: machine learning, tracking-data models, copula/HMM models.
* State that every challenger model must be evaluated by realised competition points out-of-sample, not by elegance or in-sample fit.

## Important design philosophy

Prioritise:

* Correct mathematical logic
* Market-implied probability extraction
* Clean modular code
* Transparency
* Testability
* Extensibility
* Clear documentation

Avoid:

* Trying to build a black-box model that claims to beat bookmakers
* Overcomplicated ML at the start
* Hardcoded file paths
* Mixing data loading, modelling, optimisation, and reporting in one script
* Optimising for most likely score instead of expected competition points
* Implementing future challenger models before the market-implied baseline is clean and tested

## Version 1 priority

For the first version, prioritise:

1. odds processing;
2. margin removal;
3. market-implied probability aggregation;
4. Poisson scoreline calibration from market odds;
5. expected-points optimisation;
6. friend prediction tracking;
7. result scoring;
8. clean project structure and tests;
9. documentation of assumptions, mathematical basis, and future roadmap.

Backtesting should be designed for, but does not need to be fully implemented yet.

Independent xG/Elo/ML models should not be implemented in Version 1. Only create clean interfaces/placeholders so they can be added later and compared against the market-implied baseline.

## Deliverable

Start by generating the complete project skeleton and a working baseline implementation.

The project should be runnable immediately on dummy example data.

After generating the files, explain:

* what was created
* how to install dependencies
* how to run the example prediction script
* how to update odds/predictions/results
* what the next development steps should be
* which parts are intentionally placeholders for future development
