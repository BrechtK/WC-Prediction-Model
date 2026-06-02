from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wc_predictor.config import ProjectConfig
from wc_predictor.correct_score_backtesting import (
    DEFAULT_CORRECT_SCORE_BLEND_WEIGHTS,
    backtest_correct_score_blend_weights,
)
from wc_predictor.correct_scores import (
    blend_score_matrices,
    correct_score_market_matrix,
    format_top_scorelines,
    market_to_poisson_kl_divergence,
)
from wc_predictor.market_data import load_correct_score_odds, load_odds, load_results
from wc_predictor.odds import process_correct_score_odds
from wc_predictor.probabilities import ScoreProbabilityMatrix
from wc_predictor.workflow import run_prediction_workflow


EXAMPLES = Path("data/examples")


def _processed_correct_score_odds() -> pd.DataFrame:
    return process_correct_score_odds(
        pd.DataFrame(
            [
                {"match_id": "M1", "bookmaker": "A", "score_a": 0, "score_b": 0, "decimal_odds": 2.0},
                {"match_id": "M1", "bookmaker": "A", "score_a": 1, "score_b": 0, "decimal_odds": 4.0},
                {"match_id": "M1", "bookmaker": "B", "score_a": 0, "score_b": 0, "decimal_odds": 4.0},
                {"match_id": "M1", "bookmaker": "B", "score_a": 1, "score_b": 0, "decimal_odds": 2.0},
            ]
        )
    )


def test_correct_score_market_matrix_aggregates_margin_adjusted_bookmakers() -> None:
    matrix = correct_score_market_matrix(_processed_correct_score_odds(), "M1", max_goals=1)

    assert matrix.grid_probability == pytest.approx(1.0)
    assert matrix.probabilities[0, 0] == pytest.approx(0.5)
    assert matrix.probabilities[1, 0] == pytest.approx(0.5)
    assert matrix.probabilities[0, 1] == pytest.approx(0.0)


def test_correct_score_blend_respects_endpoint_weights_and_reports_kl() -> None:
    poisson = ScoreProbabilityMatrix(np.array([[0.4, 0.1], [0.3, 0.2]]))
    market = ScoreProbabilityMatrix(np.array([[0.1, 0.2], [0.6, 0.1]]))

    assert np.allclose(blend_score_matrices(poisson, market, 0.0).probabilities, market.probabilities)
    assert np.allclose(blend_score_matrices(poisson, market, 1.0).probabilities, poisson.probabilities)
    assert market_to_poisson_kl_divergence(poisson, market) > 0
    assert format_top_scorelines(market, top_n=2).startswith("1-0 (60.0000%); 0-1 (20.0000%)")


def test_prediction_workflow_optionally_uses_correct_score_market() -> None:
    odds = load_odds(EXAMPLES / "example_odds.csv")
    correct_score_odds = load_correct_score_odds(EXAMPLES / "example_correct_score_odds.csv")
    baseline = run_prediction_workflow(odds)
    blended = run_prediction_workflow(
        odds,
        config=ProjectConfig(correct_score_poisson_weight=0.0),
        correct_score_odds=correct_score_odds,
    )

    assert not baseline.match_report["has_correct_score_market"].any()
    report = blended.match_report.set_index("match_id")
    assert report.loc["M001", "has_correct_score_market"]
    assert report.loc["M001", "correct_score_poisson_weight"] == pytest.approx(0.0)
    assert report.loc["M001", "correct_score_market_top_10"]
    assert report.loc["M001", "correct_score_blended_top_10"]
    assert report.loc["M001", "correct_score_kl_divergence"] > 0
    assert np.allclose(
        blended.score_matrices["M001"].probabilities,
        blended.correct_score_market_matrices["M001"].probabilities,
    )
    assert not report.loc["M003", "has_correct_score_market"]


def test_poisson_workflow_is_unchanged_when_correct_score_market_is_absent() -> None:
    odds = load_odds(EXAMPLES / "example_odds.csv")
    baseline = run_prediction_workflow(odds)
    no_market = run_prediction_workflow(odds, config=ProjectConfig(correct_score_poisson_weight=0.0))

    for match_id in baseline.score_matrices:
        assert np.allclose(
            baseline.score_matrices[match_id].probabilities,
            no_market.score_matrices[match_id].probabilities,
        )
        assert baseline.match_report.set_index("match_id").loc[match_id, "recommended_score"] == (
            no_market.match_report.set_index("match_id").loc[match_id, "recommended_score"]
        )


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_correct_score_poisson_weight_must_be_between_zero_and_one(weight: float) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        ProjectConfig(correct_score_poisson_weight=weight)


def test_correct_score_blend_weight_backtest_produces_ranked_comparison() -> None:
    report = backtest_correct_score_blend_weights(
        load_odds(EXAMPLES / "example_odds.csv"),
        load_correct_score_odds(EXAMPLES / "example_correct_score_odds.csv"),
        load_results(EXAMPLES / "example_results.csv"),
    )

    assert set(report.summary["correct_score_poisson_weight"]) == set(DEFAULT_CORRECT_SCORE_BLEND_WEIGHTS)
    assert (report.summary["matches_used"] == 2).all()
    assert report.summary["rank"].notna().all()
    assert len(report.predictions) == 10


def test_correct_score_template_has_required_long_format_columns() -> None:
    template = pd.read_csv("data/templates/correct_score_odds_template.csv")
    assert template.columns.tolist() == ["match_id", "bookmaker", "score_a", "score_b", "decimal_odds"]
