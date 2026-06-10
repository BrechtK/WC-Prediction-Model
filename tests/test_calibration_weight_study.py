from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_predictor.calibration_weight_study import (
    ROBUSTNESS_PROFILES,
    SEARCH_LABEL,
    CalibrationWeightProfile,
    add_search_rankings,
    build_profile_configs,
    make_search_profiles,
    summarise_calibration_weight_report,
    write_calibration_weight_outputs,
)
from wc_predictor.config import CalibrationWeights
from wc_predictor.live_backtest import LiveBacktestReport


def _prediction(
    *,
    profile: str,
    config: str,
    strategy: str,
    match_id: str,
    score: str,
    points: int,
    tournament: str = "wc_test",
    mc_acceptable: str = "no",
) -> dict[str, object]:
    return {
        "profile": profile,
        "config": f"{profile}__{config}",
        "tournament": tournament,
        "match_id": match_id,
        "strategy": strategy,
        "predicted_score": score,
        "realised_points": points,
        "matrix_expected_total_goals": 2.4,
        "actual_total_goals": 2,
        "predicted_probability_team_a_win": 0.50,
        "predicted_probability_draw": 0.30,
        "predicted_probability_team_b_win": 0.20,
        "market_a_win": 0.52,
        "market_draw": 0.29,
        "market_b_win": 0.19,
        "predicted_btts_yes_probability": 0.48,
        "market_fair_btts_yes_probability": 0.50,
        "brier_score_1x2": 0.20,
        "log_loss_1x2": 0.70,
        "rps_1x2": 0.10,
        "brier_score_btts": 0.25,
        "log_loss_btts": 0.69,
        "mc_ah_optimisation_acceptable": mc_acceptable,
    }


def _total_row(profile: str, config: str, match_id: str) -> dict[str, object]:
    return {
        "config": f"{profile}__{config}",
        "tournament": "wc_test",
        "match_id": match_id,
        "strategy": "ev_default",
        "predicted_over_probability": 0.55,
        "market_fair_over_probability": 0.60,
    }


def _report() -> LiveBacktestReport:
    predictions = pd.DataFrame(
        [
            _prediction(
                profile="current", config="baseline_ev", strategy="ev_default",
                match_id="M001", score="1-0", points=5,
            ),
            _prediction(
                profile="current", config="baseline_ev", strategy="most_likely",
                match_id="M001", score="1-1", points=1,
            ),
            _prediction(
                profile="current", config="mc_with_ah", strategy="ev_default",
                match_id="M001", score="1-0", points=5, mc_acceptable="yes",
            ),
            _prediction(
                profile="current", config="mc_with_ah", strategy="market_consistent",
                match_id="M001", score="2-0", points=7,
            ),
            _prediction(
                profile="alt", config="baseline_ev", strategy="ev_default",
                match_id="M001", score="2-0", points=7,
            ),
            _prediction(
                profile="alt", config="baseline_ev", strategy="most_likely",
                match_id="M001", score="2-0", points=7,
            ),
            _prediction(
                profile="alt", config="mc_with_ah", strategy="ev_default",
                match_id="M001", score="2-0", points=7, mc_acceptable="no",
            ),
            _prediction(
                profile="alt", config="mc_with_ah", strategy="market_consistent",
                match_id="M001", score="3-0", points=1,
            ),
        ]
    )
    totals = pd.DataFrame(
        [
            _total_row("current", "baseline_ev", "M001"),
            _total_row("alt", "baseline_ev", "M001"),
        ]
    )
    return LiveBacktestReport(
        summary=pd.DataFrame(),
        predictions=predictions,
        skipped=pd.DataFrame(),
        round_summary=pd.DataFrame(),
        total_goals_probability_diagnostics=totals,
    )


def test_current_profile_matches_live_calibration_defaults() -> None:
    current = ROBUSTNESS_PROFILES[0].weights()

    assert current == CalibrationWeights()


def test_build_profile_configs_does_not_change_live_defaults() -> None:
    configs = build_profile_configs(ROBUSTNESS_PROFILES[:1], include_market_consistent=True)

    assert [config.name for config in configs] == ["current__baseline_ev", "current__mc_with_ah"]
    assert CalibrationWeights() == ROBUSTNESS_PROFILES[0].weights()
    assert configs[0].config.calibration_weights == CalibrationWeights()
    assert configs[1].config.enable_market_consistent_challenger is True


def test_summarise_calibration_weight_report_has_expected_columns_and_gating() -> None:
    profiles = (
        ROBUSTNESS_PROFILES[0],
        CalibrationWeightProfile("alt", 1.0, 1.0, 1.0),
    )

    summary, by_tournament = summarise_calibration_weight_report(
        _report(),
        profiles,
        include_market_consistent=True,
    )

    assert {
        "profile",
        "ev_default_points",
        "modal_points",
        "gated_mc_ah_points",
        "raw_mc_ah_points",
        "mean_abs_model_market_1x2_error",
        "mean_abs_model_market_total_goals_error",
        "mean_abs_model_market_btts_error",
        "ev_picks_changed_vs_current_count",
    }.issubset(summary.columns)
    current = summary.set_index("profile").loc["current"]
    alt = summary.set_index("profile").loc["alt"]
    assert current["gated_mc_ah_points"] == 7
    assert alt["gated_mc_ah_points"] == 7
    assert alt["ev_picks_changed_vs_current_count"] == 1
    assert by_tournament["tournament"].tolist() == ["wc_test", "wc_test"]


def test_search_profiles_are_labelled_exploratory() -> None:
    profiles = make_search_profiles(one_x_two_grid=(1.0,), totals_grid=(0.75,), btts_grid=(0.5,))

    assert len(profiles) == 1
    assert profiles[0].study_type == "in_sample_exploratory"
    assert SEARCH_LABEL in profiles[0].note


def test_add_search_rankings_marks_in_sample_output() -> None:
    profiles = (
        ROBUSTNESS_PROFILES[0],
        CalibrationWeightProfile("alt", 1.0, 1.0, 1.0, "in_sample_exploratory", SEARCH_LABEL),
    )
    summary, by_tournament = summarise_calibration_weight_report(
        _report(),
        profiles,
        include_market_consistent=True,
    )

    ranked = add_search_rankings(summary, by_tournament)

    assert "search_rank_by_points" in ranked.columns
    assert SEARCH_LABEL in ranked.loc[ranked["profile"].eq("alt"), "research_warning"].iloc[0]
    assert ranked.loc[ranked["profile"].eq("alt"), "ev_points_delta_vs_current"].iloc[0] == 2


def test_write_calibration_weight_outputs(tmp_path: Path) -> None:
    summary = pd.DataFrame([{"profile": "current", "ev_default_points": 5}])
    by_tournament = pd.DataFrame([{"profile": "current", "tournament": "wc_test"}])
    summary_path = tmp_path / "summary.csv"
    by_tournament_path = tmp_path / "by_tournament.csv"

    write_calibration_weight_outputs(
        summary,
        by_tournament,
        summary_path=summary_path,
        by_tournament_path=by_tournament_path,
    )

    assert pd.read_csv(summary_path).columns.tolist() == ["profile", "ev_default_points"]
    assert pd.read_csv(by_tournament_path).columns.tolist() == ["profile", "tournament"]
