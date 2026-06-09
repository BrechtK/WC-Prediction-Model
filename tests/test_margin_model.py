import math

import pandas as pd
import pytest
from scipy.stats import skellam

from wc_predictor.margin_model import fit_skellam_margin_model


def _synthetic_asian_handicap(mu1: float, mu2: float) -> pd.DataFrame:
    rows = []
    for handicap in (-2.5, -1.5, -0.5, 0.5, 1.5):
        threshold = int(math.floor(-handicap) + 1)
        fair_team_a = float(skellam.sf(threshold - 1, mu1, mu2))
        rows.append({"handicap": handicap, "fair_team_a": fair_team_a, "fair_team_b": 1.0 - fair_team_a})
    return pd.DataFrame(rows)


def test_skellam_margin_model_recovers_plausible_margin_distribution() -> None:
    asian_handicap = _synthetic_asian_handicap(1.6, 0.9)

    model = fit_skellam_margin_model(asian_handicap, lambda_a=1.5, lambda_b=1.0)

    assert model is not None
    assert model.margin_model_type == "skellam"
    assert model.margin_fit_error < 1e-3
    assert model.fitted_margin_parameters["implied_mean_margin"] == pytest.approx(0.7, abs=0.1)
    assert "skellam=" in model.margin_distribution_comparison
    assert model.warnings == ""


def test_skellam_margin_model_returns_none_without_asian_handicap() -> None:
    assert fit_skellam_margin_model(None, lambda_a=1.3, lambda_b=1.1) is None
    assert fit_skellam_margin_model(pd.DataFrame(), lambda_a=1.3, lambda_b=1.1) is None


def test_skellam_margin_model_skips_with_too_few_half_goal_lines() -> None:
    asian_handicap = pd.DataFrame([{"handicap": -1.5, "fair_team_a": 0.55, "fair_team_b": 0.45}])

    model = fit_skellam_margin_model(asian_handicap, lambda_a=1.3, lambda_b=1.1)

    assert model is not None
    assert "insufficient_half_goal_lines" in model.warnings
