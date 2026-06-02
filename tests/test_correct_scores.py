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
    aggregate_correct_score_market,
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


def test_one_bookmaker_full_grid_is_normalised_before_aggregation() -> None:
    processed = process_correct_score_odds(
        pd.DataFrame(
            [
                {"match_id": "M1", "bookmaker": "A", "score_a": 0, "score_b": 0, "decimal_odds": 2.0},
                {"match_id": "M1", "bookmaker": "A", "score_a": 1, "score_b": 0, "decimal_odds": 4.0},
            ]
        )
    )

    aggregation = aggregate_correct_score_market(processed, "M1", max_goals=1)

    assert aggregation.aggregation_method == "mean"
    assert aggregation.matrix.grid_probability == pytest.approx(1.0)
    assert aggregation.matrix.probabilities[0, 0] == pytest.approx(2 / 3)
    assert aggregation.matrix.probabilities[1, 0] == pytest.approx(1 / 3)


def test_two_bookmakers_aggregate_fair_probabilities_not_decimal_odds() -> None:
    processed = process_correct_score_odds(
        pd.DataFrame(
            [
                {"match_id": "M1", "bookmaker": "A", "score_a": 0, "score_b": 0, "decimal_odds": 2.0},
                {"match_id": "M1", "bookmaker": "A", "score_a": 1, "score_b": 0, "decimal_odds": 4.0},
                {"match_id": "M1", "bookmaker": "B", "score_a": 0, "score_b": 0, "decimal_odds": 4.0},
                {"match_id": "M1", "bookmaker": "B", "score_a": 1, "score_b": 0, "decimal_odds": 4.0},
            ]
        )
    )

    aggregation = aggregate_correct_score_market(processed, "M1", max_goals=1, method="mean")
    incorrectly_averaged_decimal_odds_probability = (1 / 3.0) / ((1 / 3.0) + (1 / 4.0))

    assert aggregation.matrix.probabilities[0, 0] == pytest.approx(((2 / 3) + 0.5) / 2)
    assert aggregation.matrix.probabilities[1, 0] == pytest.approx(((1 / 3) + 0.5) / 2)
    assert aggregation.matrix.probabilities[0, 0] != pytest.approx(incorrectly_averaged_decimal_odds_probability)


def test_out_of_grid_scorelines_contribute_to_margin_removal_and_are_reported() -> None:
    processed = process_correct_score_odds(
        pd.DataFrame(
            [
                {"match_id": "M1", "bookmaker": "A", "score_a": 0, "score_b": 0, "decimal_odds": 2.0},
                {"match_id": "M1", "bookmaker": "A", "score_a": 10, "score_b": 0, "decimal_odds": 4.0},
            ]
        )
    )

    aggregation = aggregate_correct_score_market(processed, "M1", max_goals=1)

    assert aggregation.bookmaker_diagnostics.loc[0, "correct_score_overround"] == pytest.approx(0.75)
    assert aggregation.matrix.probabilities[0, 0] == pytest.approx(1.0)
    assert aggregation.diagnostics["out_of_grid_scorelines_count"] == 1
    assert "out_of_grid_scorelines:10-0" in aggregation.diagnostics["scoreline_coverage_warning"]


def _processed_market_with_one_bad_bookmaker() -> pd.DataFrame:
    rows = []
    for bookmaker in ("A", "B", "C", "D"):
        rows.extend(
            [
                {"match_id": "M1", "bookmaker": bookmaker, "score_a": 0, "score_b": 0, "decimal_odds": 2.0},
                {"match_id": "M1", "bookmaker": bookmaker, "score_a": 1, "score_b": 0, "decimal_odds": 2.0},
            ]
        )
    rows.extend(
        [
            {"match_id": "M1", "bookmaker": "Outlier", "score_a": 0, "score_b": 0, "decimal_odds": 100.0},
            {"match_id": "M1", "bookmaker": "Outlier", "score_a": 1, "score_b": 0, "decimal_odds": 2.0},
        ]
    )
    return process_correct_score_odds(pd.DataFrame(rows))


def test_scoreline_log_probability_outliers_are_detected() -> None:
    aggregation = aggregate_correct_score_market(_processed_market_with_one_bad_bookmaker(), "M1", max_goals=1)

    assert aggregation.aggregation_method == "winsorized_mean"
    assert aggregation.diagnostics["outlier_count"] == 2
    assert "Outlier:0-0" in aggregation.diagnostics["top_outlier_examples"]


def test_winsorized_aggregation_reduces_outlier_impact() -> None:
    processed = _processed_market_with_one_bad_bookmaker()

    mean = aggregate_correct_score_market(processed, "M1", max_goals=1, method="mean").matrix
    winsorized = aggregate_correct_score_market(processed, "M1", max_goals=1, method="winsorized_mean").matrix

    assert abs(winsorized.probabilities[0, 0] - 0.5) < abs(mean.probabilities[0, 0] - 0.5)


def test_median_aggregation_is_robust_to_one_bad_bookmaker() -> None:
    median = aggregate_correct_score_market(
        _processed_market_with_one_bad_bookmaker(),
        "M1",
        max_goals=1,
        method="median",
    ).matrix

    assert median.probabilities[0, 0] == pytest.approx(0.5)
    assert median.probabilities[1, 0] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "method",
    ["mean", "median", "trimmed_mean", "winsorized_mean", "reliability_weighted_mean"],
)
def test_supported_correct_score_aggregation_methods_return_probability_matrix(method: str) -> None:
    aggregation = aggregate_correct_score_market(
        _processed_market_with_one_bad_bookmaker(),
        "M1",
        max_goals=1,
        method=method,
    )

    assert aggregation.aggregation_method == method
    assert aggregation.matrix.grid_probability == pytest.approx(1.0)


def test_correct_score_blend_respects_endpoint_weights_and_reports_kl() -> None:
    poisson = ScoreProbabilityMatrix(np.array([[0.4, 0.1], [0.3, 0.2]]))
    market = ScoreProbabilityMatrix(np.array([[0.1, 0.2], [0.6, 0.1]]))

    assert np.allclose(blend_score_matrices(poisson, market, 0.0).probabilities, market.probabilities)
    assert np.allclose(blend_score_matrices(poisson, market, 1.0).probabilities, poisson.probabilities)
    assert market_to_poisson_kl_divergence(poisson, market) > 0
    assert format_top_scorelines(market, top_n=2).startswith("1-0 (60.0000%); 0-1 (20.0000%)")


def test_sparse_correct_score_market_is_reported_but_blending_is_suppressed() -> None:
    odds = load_odds(EXAMPLES / "example_odds.csv")
    correct_score_odds = load_correct_score_odds(EXAMPLES / "example_correct_score_odds.csv")
    baseline = run_prediction_workflow(odds)
    suppressed = run_prediction_workflow(
        odds,
        config=ProjectConfig(correct_score_poisson_weight=0.0),
        correct_score_odds=correct_score_odds,
    )

    assert not baseline.match_report["has_correct_score_market"].any()
    report = suppressed.match_report.set_index("match_id")
    assert report.loc["M001", "has_correct_score_market"]
    assert report.loc["M001", "correct_score_poisson_weight"] == pytest.approx(0.0)
    assert report.loc["M001", "effective_correct_score_poisson_weight"] == pytest.approx(1.0)
    assert report.loc["M001", "correct_score_blend_suppressed"]
    assert not report.loc["M001", "correct_score_blend_applied"]
    assert "correct_score_blend_suppressed_sparse_market" in report.loc["M001", "warning_flags"]
    assert "below configured minimum 10" in report.loc["M001", "warnings"]
    assert report.loc["M001", "correct_score_market_top_10"]
    assert report.loc["M001", "correct_score_blended_top_10"]
    assert report.loc["M001", "correct_score_kl_divergence"] > 0
    assert np.allclose(
        suppressed.score_matrices["M001"].probabilities,
        baseline.score_matrices["M001"].probabilities,
    )
    assert not report.loc["M003", "has_correct_score_market"]


def test_non_sparse_correct_score_market_still_blends() -> None:
    odds = load_odds(EXAMPLES / "example_odds.csv")
    correct_score_odds = load_correct_score_odds(EXAMPLES / "example_correct_score_odds.csv")
    blended = run_prediction_workflow(
        odds,
        config=ProjectConfig(correct_score_poisson_weight=0.0, min_scorelines_for_blend=6),
        correct_score_odds=correct_score_odds,
    )

    report = blended.match_report.set_index("match_id")
    assert report.loc["M001", "correct_score_blend_applied"]
    assert not report.loc["M001", "correct_score_blend_suppressed"]
    assert report.loc["M001", "effective_correct_score_poisson_weight"] == pytest.approx(0.0)
    assert np.allclose(
        blended.score_matrices["M001"].probabilities,
        blended.correct_score_market_matrices["M001"].probabilities,
    )


def test_correct_score_weight_one_keeps_poisson_matrix_unchanged() -> None:
    odds = load_odds(EXAMPLES / "example_odds.csv")
    correct_score_odds = load_correct_score_odds(EXAMPLES / "example_correct_score_odds.csv")
    baseline = run_prediction_workflow(odds)
    endpoint = run_prediction_workflow(
        odds,
        config=ProjectConfig(correct_score_poisson_weight=1.0, min_scorelines_for_blend=6),
        correct_score_odds=correct_score_odds,
    )

    for match_id in baseline.score_matrices:
        assert np.allclose(
            endpoint.score_matrices[match_id].probabilities,
            baseline.score_matrices[match_id].probabilities,
        )


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


def test_correct_score_aggregation_method_must_be_supported() -> None:
    with pytest.raises(ValueError, match="correct_score_aggregation_method"):
        ProjectConfig(correct_score_aggregation_method="average_decimal_odds")


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_min_scorelines_for_blend_must_be_a_positive_integer(value) -> None:
    with pytest.raises(ValueError, match="min_scorelines_for_blend"):
        ProjectConfig(min_scorelines_for_blend=value)


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
