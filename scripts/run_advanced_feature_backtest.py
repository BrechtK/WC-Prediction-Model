from wc_predictor.backtesting import HistoricalWorldCupCSVLoader, BacktestRunner
from wc_predictor.config import ProjectConfig, DevigConfig, MarketConsistentGroupWeights
from pathlib import Path
import pandas as pd

INPUT = Path("input/historical/2022/world_cup_matches.csv")
loader = HistoricalWorldCupCSVLoader(INPUT)

configs = {
    "baseline": ProjectConfig(),
    "power_devig": ProjectConfig(
        devig=DevigConfig(one_x_two_method="power", btts_method="power",
                         total_goals_method="power", correct_score_method="normalised_inverse_odds")
    ),
    "dc_prior_mc": ProjectConfig(
        market_consistent_prior="dixon_coles",
        enable_market_consistent_challenger=True,
    ),
    "bivariate_cov02": ProjectConfig(
        enable_bivariate_poisson_diagnostic=True,
        bivariate_poisson_covariance=0.2,
    ),
    "larger_grid": ProjectConfig(
        dynamic_grid_enabled=True,
        extreme_favourite_max_goals=15,
    ),
}

frames = []
for name, config in configs.items():
    report = BacktestRunner(loader, config=config, fast=True).run(export=False)
    summary = report.summary.copy()
    summary["config"] = name
    frames.append(summary)

result = pd.concat(frames, ignore_index=True)
result.to_csv("output/research/advanced_feature_comparison.csv", index=False)
print(result[["config", "strategy", "average_realised_points"]].to_string())