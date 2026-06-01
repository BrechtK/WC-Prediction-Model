# Future Model Roadmap

## Version 1

Use margin-adjusted bookmaker odds, aggregate bookmakers, calibrate an
independent-Poisson score distribution, optimise private-pool expected points,
and track friend submissions and standings.

## Version 1.5

Implement historical backtesting adapters and compare market-implied strategies
using realised pool points, hit rates, variance, and cumulative performance.

## Version 2

Add an xG/Elo Poisson or Skellam-style challenger behind the existing score-model
interface.

## Version 2.5

Evaluate market/model blending at the probability-distribution or expected-goals
level.

## Version 3

Test Dixon-Coles and bivariate-Poisson extensions. Consider negative-binomial
count models where overdispersion is material.

## Later Research

Only after simpler baselines are measured: machine learning, richer ensembles,
tracking-data models, copulas, and hidden-Markov approaches.

Every challenger must be evaluated out of sample by realised competition points,
not by elegance or in-sample fit.

