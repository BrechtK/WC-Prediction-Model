You are Claude Code and you are reviewing my GitHub project from scratch.

Please first audit the repository. Do not edit files yet unless I explicitly ask you to implement changes afterwards.

# Project context

This is a Python project for a World Cup / football exact-score prediction competition hosted by Sporza.

The user submits exact-score predictions for football matches. The project takes manually pasted OddsPortal market data, parses it, converts odds into fair probabilities, constructs score-probability matrices, computes expected points under the Sporza scoring rule, and outputs recommended scores plus diagnostics.

The project is both:

1. A live matchday tool:

   * I paste fresh odds shortly before a match.
   * I run one script from VS Code.
   * I get a compact recommendation and a submission sheet.

2. A research platform:

   * I can run challenger models, diagnostics, backtests, margin-method comparisons, market-consistent matrices, etc.

The main objective is to maximise expected tournament points, but I also care about strategic ranking performance against friends and in the public Sporza leaderboard.

# Current live workflow

The intended matchday workflow is:

1. Paste the tournament schedule into:

   input/schedule.txt

2. Paste fresh OddsPortal odds for one match into:

   input/odds/Mxxx.txt

3. The odds file can contain sections such as:

   ### MATCH

   Team A vs Team B

   ### 1X2

   <paste OddsPortal 1X2 odds>

   ### OVER_UNDER

   <paste OddsPortal total-goals odds>

   ### BTTS

   <paste both-teams-to-score odds>

   ### CORRECT_SCORE

   <paste correct-score odds>

   ### ASIAN_HANDICAP

   <paste Asian handicap odds>

4. Open:

   scripts/run_live_prediction.py

5. Edit the USER SETTINGS block at the top of the file.

6. Press “Run Python File” in VS Code.

7. Read the compact terminal output and/or open:

   output/submission_sheet.xlsx
   output/predictions.xlsx

Important:
I prefer editing settings inside `run_live_prediction.py` rather than using command-line arguments.

# Current model/features summary

Please verify these by reading the code rather than trusting this summary.

The project currently includes or should include:

* Schedule parsing from `input/schedule.txt`.
* Match selection by:

  * match ID, e.g. M009;
  * date and game number, e.g. DATE = "14/6", GAME_NUMBER = 4;
  * all available odds files;
  * list-date mode.
* Combined OddsPortal paste parsing from `input/odds/Mxxx.txt`.
* Parsers for:

  * 1X2;
  * Over/Under total goals;
  * BTTS;
  * correct score;
  * Asian handicap.
* Margin-removal methods:

  * normalised inverse odds / proportional;
  * power;
  * additive;
  * Shin, if safely implemented.
* Independent Poisson baseline.
* Calibration to 1X2, BTTS, and O/U constraints.
* Asian total-goals settlement:

  * half-goal lines;
  * integer lines with push;
  * quarter lines via half-stake splits.
* Asian handicap settlement:

  * half-handicap;
  * integer handicap with push;
  * quarter handicap via half-stake split;
  * team A / team B side symmetry.
* Correct-score market aggregation and blending.
* Dixon-Coles challenger, including estimated rho from low-score market cells if implemented.
* Market-consistent KL / entropy-pooling style score matrix challenger:

  * starts from a prior matrix;
  * uses soft constraints from 1X2, BTTS, O/U, Asian totals, Asian handicap, correct score;
  * diagnostic-only unless explicitly configured otherwise.
* Margin-level EV diagnostics:

  * score-margin probabilities;
  * best scoreline per margin;
  * EV decomposition by margin;
  * draw-vs-decisive diagnostics.
* Public-field strategy diagnostics:

  * friend pool / balanced / national leaderboard target;
  * public-field-size parameter;
  * crowding / contrarian diagnostics.
* Final decision dashboard:

  * default recommendation;
  * confidence;
  * manual review flag;
  * plausible alternatives;
  * strategic alternative;
  * risk notes.
* Compact live terminal mode.
* Research/debug mode.
* Historical World Cup backtesting scaffold, if implemented.
* Knockout scoring mode:

  * unverified;
  * additive;
  * hierarchical;
  * documentation for verifying Sporza knockout rules.

# Important files to read first

Please inspect the repository in this approximate order:

1. README.md
2. docs/matchday_workflow.md
3. docs/mathematical_basis.md
4. docs/model_roadmap.md
5. docs/odds_collection_strategy.md
6. docs/knockout_scoring_verification.md, if present
7. templates/odds_input_template.txt
8. scripts/run_live_prediction.py
9. scripts/run_historical_world_cup_backtest.py, if present
10. src/wc_predictor/
11. tests/

Focus especially on:

* src/wc_predictor/odds.py
* src/wc_predictor/margin.py
* src/wc_predictor/calibration.py
* src/wc_predictor/probabilities.py
* src/wc_predictor/scoring_rules.py
* src/wc_predictor/optimiser.py
* src/wc_predictor/workflow.py
* src/wc_predictor/live_prediction.py
* src/wc_predictor/reporting.py
* src/wc_predictor/correct_scores.py
* src/wc_predictor/asian_totals.py
* src/wc_predictor/asian_handicap.py
* src/wc_predictor/market_consistent.py
* src/wc_predictor/margin_diagnostics.py
* src/wc_predictor/score_models.py
* src/wc_predictor/public_strategy.py
* OddsPortal parser modules:

  * oddsportal.py
  * oddsportal_core.py
  * oddsportal_combined.py
  * oddsportal_schedule.py
  * oddsportal_asian_handicap.py, if present

# What I want from you

Please perform a full audit of the latest project state.

Do not just say “looks good”. Be critical, mathematical, practical, and specific.

I want:

1. A code audit.
2. A mathematical/model audit.
3. A live-workflow audit.
4. A test-coverage audit.
5. A documentation audit.
6. A Git hygiene audit.
7. Recommendations for further improvements.
8. Recommendations for possible new mathematical models or refinements.

Please clearly separate:

* must-fix bugs;
* model-risk issues;
* usability improvements;
* research ideas;
* overkill / not recommended ideas.

# Audit areas

1. Repository structure and usability

---

Review:

* input/
* output/
* cache/
* scripts/
* src/
* tests/
* docs/
* templates/
* data/ if still present

Questions:

* Is the project structure clean?
* Are live inputs separated from generated outputs?
* Are names intuitive?
* Are old/deprecated folders still present?
* Are generated files properly ignored?
* Is the main script friendly for someone who prefers VS Code over terminal?
* Are too many settings exposed in `run_live_prediction.py`?
* Is there enough separation between live mode and research mode?

2. Git hygiene and privacy

---

Check:

* .gitignore
* tracked files
* pycache / pyc files
* output files
* cache files
* input odds files
* local absolute paths
* large files

Questions:

* Are `input/schedule.txt` and `input/odds/*.txt` ignored?
* Are `output/` and `cache/` ignored?
* Are `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.pyc` ignored?
* Are there any real local data files accidentally tracked?
* Are there hardcoded local paths like `C:\Users\brech\...` in code/docs/tests?
* Are generated Excel/CSV files tracked accidentally?

3. Live matchday workflow

---

Review the actual matchday flow.

Questions:

* Is `RUN_MODE = "list_date"` easy to use?
* Does date/game-number selection work correctly?
* If DATE = "14/6" and GAME_NUMBER = 4, does it correctly resolve the 4th game on 14 June?
* Are test fixtures aligned with script defaults?
* Is compact output concise enough?
* Are warnings useful without overwhelming the user?
* Do stale-odds warnings behave properly?
* Does stale odds affect manual review only when intended?
* Is single-match mode fast?
* Does missing Asian handicap input remain optional?
* Is there a risk of stale odds or wrong match IDs?
* Are error messages actionable?

4. Odds parsing and input robustness

---

Review all parsers.

Questions:

* Are OddsPortal noisy pasted texts parsed robustly?
* Is the combined section splitter robust?
* Are section headers like `### OVER_UNDER`, `### OVER UNDER`, `### ASIAN_HANDICAP` handled correctly?
* Are bookmaker names handled correctly?
* Are duplicate bookmaker rows handled correctly?
* Are missing rows / missing odds warned about?
* Are exchange sections ignored safely?
* Are “Other” correct-score buckets handled correctly?
* Are Asian handicap lines parsed correctly from real-ish OddsPortal text?
* Is the parser too tailored to the current OddsPortal layout?
* Are there enough fixtures/tests with noisy realistic pastes?

5. Margin removal / devigging

---

Review the odds-to-probability conversion.

Questions:

* Is normalised inverse odds correct?
* Is the power method correct?
* Is additive method safe when probabilities go non-positive?
* Is Shin implemented correctly?
* Is Shin diagnostic-only unless explicitly chosen?
* Does Shin work for:

  * 2-way markets;
  * 3-way markets;
  * many-outcome correct-score markets?
* Does Shin fail gracefully when unsafe?
* Are margin-removal diagnostics written clearly?
* Are margin methods applied appropriately to:

  * 1X2;
  * BTTS;
  * O/U;
  * Asian totals;
  * Asian handicap;
  * correct score?
* Are high-overround correct-score markets treated carefully?

6. Asian total-goals maths

---

Review:

* half-goal totals;
* integer Asian totals with push;
* quarter totals via half-stake splits;
* expected-profit constraints;
* integration into the market-consistent matrix.

Questions:

* Are the payoff vectors correct?
* Are push probabilities handled correctly?
* Are over/under constraints redundant or double-weighted?
* Is the current chosen treatment mathematically defensible?
* Are there tests proving no redundant double weighting?
* Are near-the-money total lines weighted more heavily?
* Are far ITM/OTM lines handled safely?

7. Asian handicap maths

---

This is a recent important addition.

Review:

* src/wc_predictor/asian_handicap.py
* parser and aggregation logic
* market-consistent integration
* diagnostics sheets

Mathematical formulation:
For team A score (X), team B score (Y), handicap (h), define:

```
adjusted_margin = X - Y + h
```

For a unit stake on team A:

```
profit = odds - 1   if adjusted_margin > 0
profit = 0          if adjusted_margin = 0
profit = -1         if adjusted_margin < 0
```

Quarter lines split into two half-stake components.

Questions:

* Is the settlement math correct?
* Is team A / team B symmetry correct?
* Are integer pushes handled correctly?
* Are quarter lines handled correctly?
* Is devigging done appropriately?
* Are both sides of the handicap market used correctly?
* Are AH constraints integrated into KL projection correctly?
* Are deep AH lines filtered/downweighted/skipped to avoid grid-boundary artefacts?
* Are near-the-money AH lines selected sensibly?
* Are selected vs skipped lines clearly reported?
* Are skipped deep lines still useful as diagnostics?
* Does the model avoid artificial mass at the max score-grid boundary?
* Is the current selected-line logic robust for extreme favourites?

Please review specifically whether using AH lines like -3.0 to -4.5 for Germany vs Curaçao makes sense, while skipping -7.5/-7.75 due to grid-boundary risk.

8. Score probability models

---

Review:

* independent Poisson baseline;
* Dixon-Coles challenger;
* estimated Dixon-Coles rho;
* market-consistent KL challenger;
* correct-score blend.

Questions:

* Is the independent Poisson baseline implemented correctly?
* Does calibration use the full Poisson distribution where appropriate?
* Are lambdas bounded sensibly?
* Is score-grid truncation handled correctly?
* Are high-tail cases flagged?
* Is Dixon-Coles implemented correctly?
* Is rho estimation from low-score market cells sensible?
* Is rho fallback safe when correct-score data is missing?
* Is the KL / market-consistent matrix objective correct?
* Is the KL direction appropriate?
* Are soft constraints weighted sensibly?
* Are optimiser failures classified correctly?
* Is “not fully converged but fit acceptable” handled safely?
* Are AH/O-U/correct-score constraints overfitting noisy markets?
* Should any challenger become default, or remain diagnostic?

9. Expected-points optimisation and scoring rule

---

Review the core scoring / EV logic.

Questions:

* Is the Sporza group-stage scoring rule implemented correctly?
* Is it documented correctly?
* Does EV equal:
  [
  \sum_{x,y} \pi(x,y) g(p;x,y)
  ]
  for all candidate scorelines?
* Are exact score, goal difference, result, and participation handled correctly?
* Is the nested-event decomposition correct?
* Are plausible alternatives separated from raw EV alternatives?
* Are weird scores like 5-4 filtered properly from practical alternatives?
* Are raw top-EV diagnostics still available?
* Are margin-level EV diagnostics equivalent to the raw scoreline optimiser?
* Does the draw-vs-decisive diagnostic correctly identify close draw regimes?

10. Knockout scoring

---

Review knockout scoring support.

Questions:

* Is `knockout_scoring_mode = "unverified"` safe?
* Are additive and hierarchical modes implemented correctly?
* Are warnings shown if knockout scoring is unverified?
* Is the documentation clear that Sporza rules must be confirmed?
* Does knockout scoreline / qualifier optimisation decouple correctly if the rule is additive?
* Are tests sufficient?

11. Market-consistent matrix

---

This is one of the core advanced components.

Review:

* objective function;
* KL term;
* soft constraints;
* gradient;
* optimiser settings;
* diagnostics;
* selected constraints.

Questions:

* Is (D_{KL}(P||Q)) the right direction?
* Is softmax/logit parametrisation stable?
* Are gradients correct?
* Are constraints scaled/weighted sensibly?
* Does the optimiser overfit or overreact to noisy markets?
* Are AH constraints selected safely?
* Are total-goals constraints selected safely?
* Are correct-score constraints too sparse?
* Are “Other” buckets handled or ignored appropriately?
* Are fit errors meaningful and interpretable?
* Are warnings surfaced in compact output only when needed?

12. Margin-level EV diagnostics

---

Review the new margin-diagnostics layer.

Questions:

* Is the margin decomposition mathematically equivalent to raw score EV?
* Does it correctly compute:

  * margin probability;
  * correct-result probability;
  * best scoreline on margin;
  * representative scoreline EV?
* Does it correctly handle draws?
* Does it explain why `3-0` beats `4-0`, or why `1-0` beats `2-1`?
* Should this be used to speed up optimisation, or remain diagnostic?

13. Public strategy / game theory

---

Review public-field logic.

Context:
The competition may include both small friend groups and a large national Sporza leaderboard.

Questions:

* Is the public-strategy layer too heuristic?
* Does `PUBLIC_STRATEGY_TARGET = friends / balanced / national` make sense?
* Does `PUBLIC_FIELD_SIZE` enter meaningfully?
* Should the public strategy remain diagnostic?
* Should it ever affect `final_decision_score`?
* How would you model public picks more rigorously?
* How should friend-pool vs national-board objectives differ?
* How should strategy change if current standings are known?
* Is there a path to a principled field simulation?

14. Historical backtesting

---

Review the backtesting scaffold.

Questions:

* Is the historical template useful?
* Can the script run with synthetic data?
* Is the format realistic for 2022 WC odds?
* Does it compare meaningful strategies?
* Does it validate blend weight (w)?
* Does it handle missing optional markets gracefully?
* Is there any risk of using proprietary or legally problematic data?
* What minimum dataset would be useful?

15. Performance

---

Review runtime and tests.

Questions:

* Is live single-match mode fast enough?
* Is all-available mode acceptable?
* Are heavy diagnostics correctly disabled in live mode?
* Are research features expensive but gated?
* Is market-consistent optimisation efficient enough?
* Is the test suite too slow?
* Should tests be split into fast / slow?
* Are there avoidable recalculations?
* Is Excel writing a bottleneck?

16. Testing

---

Review test coverage.

Questions:

* Are mathematical functions tested thoroughly?
* Are parser tests realistic?
* Are edge cases covered?
* Are Asian handicap tests sufficient?
* Are market-consistent optimiser tests meaningful?
* Are integration tests robust?
* Do tests depend too much on current USER SETTINGS defaults?
* Should tests patch settings instead of relying on default DATE/GAME_NUMBER?
* Are slow tests marked?
* Is there a full live pipeline test with synthetic data?

17. Documentation

---

Review:

* README.md
* docs/matchday_workflow.md
* docs/mathematical_basis.md
* docs/model_roadmap.md
* docs/odds_collection_strategy.md
* docs/knockout_scoring_verification.md
* templates/odds_input_template.txt

Questions:

* Is the live workflow clear?
* Is the mathematical basis accurate?
* Are docs consistent with defaults?
* Is advanced/research functionality separated from matchday workflow?
* Are limitations honest?
* Is there outdated information?
* Are warnings about unverified components clear?

18. New mathematical model recommendations
    ==========================================
    After auditing the current state, propose further mathematical/statistical improvements.

Separate them into:

A. High-priority practical improvements
B. Medium-priority research extensions
C. Interesting but probably overkill
D. Not recommended / complexity theatre

For each proposal, explain:

* What problem it solves.
* How it fits into the current architecture.
* Required inputs.
* Whether it should be live default or diagnostic.
* How to validate it.
* Possible failure modes.
* Implementation complexity.

Please consider:

1. Better market inputs

   * Asian handicap refinements;
   * main line versus deep line selection;
   * Asian handicap tail diagnostics;
   * if/how to use AH lines in the default matrix.

2. Better score models

   * bivariate Poisson;
   * Dixon-Coles as prior;
   * copula score models;
   * Conway-Maxwell-Poisson;
   * Skellam/margin models;
   * draw-inflated models.

3. Better market-consistent matrix methods

   * entropy pooling refinements;
   * reliability-weighted constraints;
   * hard vs soft constraints;
   * constraint covariance / correlated market errors;
   * “Other” correct-score bucket handling;
   * high-score tail extrapolation.

4. Better devigging

   * Shin refinements;
   * odds-ratio method;
   * favourite-longshot bias;
   * market-type-specific devig.

5. Team-strength models

   * Elo;
   * SPI-like ratings;
   * hierarchical Bayesian attack/defence;
   * xG-based priors;
   * international-football-specific adjustments;
   * squad/player availability.
     But be critical: bookmaker odds are probably a stronger information source than any simple team-strength model.

6. Public-field strategy

   * public-pick distribution estimation;
   * friend-pool optimisation;
   * national-board optimisation;
   * dynamic strategy depending on standings;
   * correlation / decorrelation from field;
   * top-1%, top-0.1%, rank-1 probability.

7. Validation

   * historical backtesting;
   * calibration plots;
   * Brier/log score;
   * scoring-rule EV backtest;
   * ablation tests;
   * sensitivity analysis.

# Deliverable format

Please produce a structured audit report with:

1. Executive summary

   * Overall assessment
   * Biggest strengths
   * Biggest risks
   * Top 5 recommended next actions

2. Fix-by-fix / feature-by-feature assessment

   * What is solid
   * What is questionable
   * What is broken or risky

3. Architecture review

4. Mathematical model review

5. Odds parsing/data quality review

6. Live workflow review

7. Asian handicap and margin diagnostics review

8. Market-consistent matrix review

9. Public strategy/game-theory review

10. Performance review

11. Testing review

12. Documentation review

13. Specific code-level findings
    Include file names, functions, and line references where possible.

14. Recommended next actions
    Split into:

* must fix before merge;
* should fix soon;
* research roadmap;
* optional / later.

15. Proposed advanced mathematical improvements
    Prioritised and explained.

16. Questions for me
    Anything you need clarified.

# Important instructions

* Be critical and honest.
* Do not implement changes yet.
* Do not assume current code is correct.
* Point to exact files/functions when possible.
* Distinguish between live-use reliability and research-mode sophistication.
* Do not recommend complexity unless it solves a real problem.
* If something is mathematically wrong, explain clearly.
* If something is good and should not be touched, say so.
* Pay special attention to Asian handicap integration and margin-level diagnostics.
* Pay special attention to whether tests rely too much on current run-file defaults.
