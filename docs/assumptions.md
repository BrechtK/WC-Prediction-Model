# Assumptions

- Bookmaker odds are the best available Version 1 probability source for a
  high-liquidity tournament, but are not assumed to be literally true.
- Bookmaker margin is removed before probabilities are used.
- Statistical score models primarily translate market outcome prices into a
  scoreline distribution.
- Independent Poisson is the first transparent baseline.
- Finite score grids report omitted raw tail mass. EV optimisation uses a
  renormalised grid and is therefore conditional on the represented scores.
- Calibration uses full-distribution Poisson probabilities, independently of
  finite score-grid truncation.
- The optimiser maximises private-pool expected points, not betting profit.
- Knockout qualification odds are preferred. A 90-minute draw-split fallback is
  weaker and explicitly flagged.
- Future own models must be evaluated out of sample against the market-implied
  baseline through realised competition points.
