# Research Notes

Dixon and Coles (1997) is the canonical football score-modelling reference.
Poisson models are common because goals are non-negative counts and the model is
transparent, compact, and easy to calibrate.

Independent Poisson is imperfect. Team scores may not be independent, football
has low-score effects, and count variance may differ from a Poisson assumption.
Dixon-Coles adjusts probabilities around low-scoring outcomes. Later candidates
include bivariate Poisson, negative binomial models, and direct extraction from
correct-score markets.

Bookmaker prices are a strong practical benchmark because they combine broad
information, incentives, and market participation. Exact-score pools still need
a scoreline distribution: win/draw/loss probabilities alone cannot value exact
scores or goal differences.

Version 1 therefore starts with the market. Future work can compare xG/Elo
Poisson or Skellam-style challengers, score-market blends, and ensembles.

