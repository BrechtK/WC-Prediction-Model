"""Market-implied World Cup prediction-pool optimiser."""

from wc_predictor.config import ProjectConfig
from wc_predictor.optimiser import optimise_group_prediction
from wc_predictor.scoring_rules import score_group_prediction

__all__ = ["ProjectConfig", "optimise_group_prediction", "score_group_prediction"]

