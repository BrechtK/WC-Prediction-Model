from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc_predictor.asian_totals import asian_total_profit_vector
from wc_predictor.market_consistent import MarketConsistentWeights, _build_constraints, fit_market_consistent_matrix
from wc_predictor.probabilities import poisson_score_matrix


def _market() -> dict[str, float]:
    return {"a_win": 0.50, "draw": 0.25, "b_win": 0.25, "btts_yes": 0.48}


def _total_row(line: float, fair_over: float, line_kind: str) -> dict[str, object]:
    return {
        "line": line,
        "fair_over": fair_over,
        "fair_under": 1.0 - fair_over,
        "line_kind": line_kind,
        "average_total_goals_overround": 1.03,
    }


def test_market_consistent_matrix_sums_to_one_and_remains_positive() -> None:
    prior = poisson_score_matrix(1.5, 1.0, max_goals=5)
    total_goals = pd.DataFrame(
        [
            {
                "line": 2.25,
                "fair_over": 0.52,
                "fair_under": 0.48,
                "line_kind": "quarter_asian",
                "average_total_goals_overround": 1.04,
            }
        ]
    )

    result = fit_market_consistent_matrix(prior, _market(), total_goals=total_goals)

    assert result.matrix.probabilities.sum() == pytest.approx(1.0)
    assert np.all(result.matrix.probabilities > 0)
    assert result.diagnostics["market_consistent_asian_totals_used"] == "2.25 quarter_asian"


def test_zero_penalty_weights_preserve_prior_approximately() -> None:
    prior = poisson_score_matrix(1.2, 1.1, max_goals=5)
    weights = MarketConsistentWeights(one_x_two=0.0, liquid_total_goals=0.0, btts=0.0, correct_score=0.0)

    result = fit_market_consistent_matrix(
        prior,
        {"a_win": 0.90, "draw": 0.05, "b_win": 0.05, "btts_yes": 0.20},
        weights=weights,
    )

    assert result.matrix.probabilities == pytest.approx(prior.probabilities)
    assert result.diagnostics["market_consistent_kl_divergence_vs_prior"] == pytest.approx(0.0)


def test_strong_constraints_move_posterior_toward_targets() -> None:
    prior = poisson_score_matrix(1.0, 1.0, max_goals=6)
    target_market = {"a_win": 0.70, "draw": 0.18, "b_win": 0.12, "btts_yes": 0.42}
    weights = MarketConsistentWeights(one_x_two=500.0, liquid_total_goals=0.0, btts=250.0, correct_score=0.0)

    result = fit_market_consistent_matrix(prior, target_market, weights=weights)

    prior_outcomes = prior.outcome_probabilities()
    posterior_outcomes = result.matrix.outcome_probabilities()
    assert abs(posterior_outcomes["a_win"] - target_market["a_win"]) < abs(
        prior_outcomes["a_win"] - target_market["a_win"]
    )
    assert abs(result.matrix.btts_yes_probability() - target_market["btts_yes"]) < abs(
        prior.btts_yes_probability() - target_market["btts_yes"]
    )


def test_total_goals_constraints_use_asian_expected_profit() -> None:
    prior = poisson_score_matrix(1.0, 0.8, max_goals=6)
    total_goals = pd.DataFrame(
        [
            {
                "line": 3.0,
                "fair_over": 0.55,
                "fair_under": 0.45,
                "line_kind": "integer_asian",
                "average_total_goals_overround": 1.03,
            },
            {
                "line": 3.25,
                "fair_over": 0.48,
                "fair_under": 0.52,
                "line_kind": "quarter_asian",
                "average_total_goals_overround": 1.03,
            },
        ]
    )

    result = fit_market_consistent_matrix(prior, _market(), total_goals=total_goals)

    assert "3 integer_asian" in result.diagnostics["market_consistent_asian_totals_used"]
    assert "3.25 quarter_asian" in result.diagnostics["market_consistent_asian_totals_used"]
    assert pd.notna(result.diagnostics["market_consistent_total_goals_fit_error"])


def test_half_goal_total_constraint_is_not_double_weighted() -> None:
    prior = poisson_score_matrix(1.0, 1.0, max_goals=5)
    constraints, totals_used = _build_constraints(
        prior,
        _market(),
        pd.DataFrame([_total_row(2.5, 0.54, "half_goal")]),
        None,
        MarketConsistentWeights(one_x_two=0.0, liquid_total_goals=140.0, btts=0.0, correct_score=0.0),
    )

    total_constraints = [constraint for constraint in constraints if constraint.group == "total_goals"]
    assert [constraint.name for constraint in total_constraints] == ["total_over_2.5"]
    assert total_constraints[0].weight == pytest.approx(140.0)
    assert totals_used == "2.5 half_goal"


@pytest.mark.parametrize(
    ("line", "fair_over", "line_kind"),
    [
        (3.0, 0.55, "integer_asian"),
        (2.25, 0.48, "quarter_asian"),
        (2.75, 0.52, "quarter_asian"),
    ],
)
def test_integer_and_quarter_asian_under_constraints_are_redundant(
    line: float,
    fair_over: float,
    line_kind: str,
) -> None:
    shape = (6, 6)
    fair_under = 1.0 - fair_over
    over_vector = asian_total_profit_vector(shape, line, "over", 1.0 / fair_over).reshape(-1)
    under_vector = asian_total_profit_vector(shape, line, "under", 1.0 / fair_under).reshape(-1)

    assert under_vector == pytest.approx((-fair_over / fair_under) * over_vector)

    prior = poisson_score_matrix(1.0, 1.0, max_goals=5)
    constraints, _ = _build_constraints(
        prior,
        _market(),
        pd.DataFrame([_total_row(line, fair_over, line_kind)]),
        None,
        MarketConsistentWeights(one_x_two=0.0, liquid_total_goals=140.0, btts=0.0, correct_score=0.0),
    )

    assert [constraint.name for constraint in constraints] == [f"total_over_{line:g}"]
