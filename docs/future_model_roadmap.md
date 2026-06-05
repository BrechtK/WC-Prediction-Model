# Future Model Roadmap

This is the compact long-horizon roadmap. The active detailed roadmap is
`docs/model_roadmap.md`; keep both files consistent when modelling assumptions
change.

## Version 1

Use margin-adjusted bookmaker odds, aggregate bookmakers, calibrate an
independent-Poisson score distribution, optimise private-pool expected points,
and track friend submissions and standings. Live margin removal defaults to
`normalised_inverse_odds`; Shin is optional research diagnostics with safe
fallback warnings. Group-stage scoring assumes the central `10/7/5/1` config
unless confirmed rules differ.

## Version 1.5

Extend the initial group-stage Football-Data-like backtester with additional
provider adapters, cumulative-performance reporting, and richer historical
splits.

## Version 2

Add an xG/Elo Poisson or Skellam-style challenger behind the existing score-model
interface.

## Version 2.5

Evaluate market/model blending at the probability-distribution or expected-goals
level. Asian handicap ladders remain diagnostic inputs: parse broadly, constrain
only stable near-money lines, and treat orientation warnings as paste-quality
diagnostics.

## Version 3

Validate the diagnostic Dixon-Coles challenger and test bivariate-Poisson
extensions. Consider negative-binomial count models where overdispersion is
material.

Several of these are now implemented as diagnostic-first, opt-in options
(market-type-specific devig, dynamic larger grids for extreme favourites,
Dixon-Coles and bivariate-Poisson priors for the KL projection, a Skellam
margin model fitted to Asian handicap, and group-level constraint weights with a
correlation/double-counting scaffold). See `docs/model_roadmap.md` and
`docs/mathematical_basis.md`. The remaining work is out-of-sample validation
under the pool scoring rules and, eventually, a data-driven correlated-error
covariance model for market constraints.

## Later Research

Only after simpler baselines are measured: machine learning, richer ensembles,
tracking-data models, copulas, and hidden-Markov approaches.

Every challenger must be evaluated out of sample by realised competition points,
not by elegance or in-sample fit.
