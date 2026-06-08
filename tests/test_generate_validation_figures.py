from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_validation_figures.py"

EXPECTED_OUTPUTS = {
    "calibration_1x2.png",
    "expected_vs_actual_goals.png",
    "points_vs_probability_quality.png",
    "calibration_1x2_summary.csv",
    "draw_summary.csv",
    "goals_layer_comparison.csv",
    "goals_summary.csv",
    "strategy_probability_summary.csv",
}

pytest.importorskip("matplotlib")


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_validation_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _prediction_rows() -> list[dict[str, object]]:
    base_matches = [
        ("M2014A", 2014, "team_a_win", 0.60, 0.25, 0.15, 2.0, 2, 1),
        ("M2014B", 2014, "draw", 0.45, 0.30, 0.25, 2.4, 1, 1),
        ("M2018A", 2018, "team_b_win", 0.35, 0.25, 0.40, 2.8, 0, 2),
        ("M2018B", 2018, "team_a_win", 0.55, 0.25, 0.20, 3.0, 3, 1),
        ("M2022A", 2022, "draw", 0.50, 0.28, 0.22, 2.2, 0, 0),
        ("M2022B", 2022, "team_b_win", 0.30, 0.27, 0.43, 2.6, 1, 2),
    ]
    strategies = [
        ("most_likely_poisson", 5),
        ("ev_optimal_1x2", 7),
        ("ev_optimal_1x2_over_under", 7),
    ]
    rows: list[dict[str, object]] = []
    for match_id, year, outcome, p_a, p_d, p_b, expected, actual_a, actual_b in base_matches:
        actual_index = {"team_a_win": 0, "draw": 1, "team_b_win": 2}[outcome]
        probs = [p_a, p_d, p_b]
        realised = [0, 0, 0]
        realised[actual_index] = 1
        brier = sum((prob - value) ** 2 for prob, value in zip(probs, realised))
        rps = (
            (probs[0] - realised[0]) ** 2
            + (probs[0] + probs[1] - realised[0] - realised[1]) ** 2
        ) / 2
        for strategy, points in strategies:
            rows.append(
                {
                    "strategy": strategy,
                    "match_id": match_id,
                    "year": year,
                    "tournament": f"wc{year}",
                    "realised_points": points,
                    "predicted_probability_team_a_win": p_a,
                    "predicted_probability_draw": p_d,
                    "predicted_probability_team_b_win": p_b,
                    "expected_total_goals_1x2": expected,
                    "actual_total_goals": actual_a + actual_b,
                    "realised_outcome": outcome,
                    "brier_score_1x2": brier,
                    "rps_1x2": rps,
                }
            )
    return rows


def _richer_prediction_rows() -> list[dict[str, object]]:
    return [
        {
            "config": "baseline_ev",
            "strategy": "baseline_poisson",
            "tournament": "wc2018",
            "match_id": "R2018A",
            "matrix_expected_total_goals": 2.5,
            "actual_total_goals": 3,
        },
        {
            "config": "baseline_ev",
            "strategy": "baseline_poisson",
            "tournament": "wc2022",
            "match_id": "R2022A",
            "matrix_expected_total_goals": 2.7,
            "actual_total_goals": 2,
        },
    ]


def _stage_predictions(root: Path) -> Path:
    dest = root / "output" / "research"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "world_cup_backtest_predictions.csv"
    pd.DataFrame(_prediction_rows()).to_csv(path, index=False)
    richer_dest = dest / "combined_backtest"
    richer_dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_richer_prediction_rows()).to_csv(richer_dest / "live_backtest_predictions.csv", index=False)
    return path


def test_generate_writes_expected_files(tmp_path: Path) -> None:
    _stage_predictions(tmp_path)
    module = _load_module()

    written = module.generate(tmp_path)

    figures_dir = tmp_path / "paper" / "figures"
    for name in EXPECTED_OUTPUTS:
        path = figures_dir / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0
    assert {p.name for p in written} == EXPECTED_OUTPUTS


def test_summary_tables_match_staged_predictions(tmp_path: Path) -> None:
    _stage_predictions(tmp_path)
    module = _load_module()
    module.generate(tmp_path)

    figures_dir = tmp_path / "paper" / "figures"
    goals = pd.read_csv(figures_dir / "goals_summary.csv")
    combined = goals.loc[goals["sample"] == "Combined"].iloc[0]
    assert int(combined["matches"]) == 6
    assert round(combined["expected_total_goals"], 3) == 2.500
    assert round(combined["actual_total_goals"], 3) == 2.333

    layers = pd.read_csv(figures_dir / "goals_layer_comparison.csv")
    assert set(layers["validation_layer"]) == {"Football-Data 1X2-only", "Richer live-paste + totals"}
    fd = layers.loc[layers["validation_layer"] == "Football-Data 1X2-only"].iloc[0]
    richer = layers.loc[layers["validation_layer"] == "Richer live-paste + totals"].iloc[0]
    assert int(fd["matches"]) == 4
    assert round(fd["expected_total_goals"], 3) == 2.650
    assert round(fd["actual_total_goals"], 3) == 2.250
    assert int(richer["matches"]) == 2
    assert round(richer["expected_total_goals"], 3) == 2.600
    assert round(richer["actual_total_goals"], 3) == 2.500

    calibration = pd.read_csv(figures_dir / "calibration_1x2_summary.csv")
    assert set(calibration["calibration_target"]) == {"favourite", "draw", "underdog"}

    draw = pd.read_csv(figures_dir / "draw_summary.csv")
    combined_draw = draw.loc[draw["sample"] == "Combined"].iloc[0]
    assert round(combined_draw["mean_predicted_draw_probability"], 3) == 0.267
    assert round(combined_draw["realised_draw_frequency"], 3) == 0.333


def test_generate_does_not_touch_source_outputs(tmp_path: Path) -> None:
    source = _stage_predictions(tmp_path)
    before = source.read_bytes()

    module = _load_module()
    module.generate(tmp_path)

    assert source.read_bytes() == before
