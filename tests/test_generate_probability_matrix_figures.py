from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_probability_matrix_figures.py"
BACKTEST_DIR = REPO_ROOT / "output" / "research" / "combined_backtest"
INPUT = BACKTEST_DIR / "live_backtest_predictions.csv"

EXPECTED_OUTPUTS = [
    "probability_matrix_extreme_favourite.png",
    "probability_matrix_balanced.png",
    "probability_matrix_examples.csv",
]

pytest.importorskip("matplotlib")

input_present = INPUT.exists()


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_probability_matrix_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(
    match_id: str,
    team_a: str,
    team_b: str,
    market_a_win: float,
    market_draw: float,
    market_b_win: float,
    lambda_a: float = 1.2,
    lambda_b: float = 1.1,
) -> dict[str, object]:
    return {
        "strategy": "baseline_poisson",
        "config": "baseline_ev",
        "tournament": "wc2099",
        "match_id": match_id,
        "team_a": team_a,
        "team_b": team_b,
        "market_a_win": market_a_win,
        "market_draw": market_draw,
        "market_b_win": market_b_win,
        "favourite_probability": max(market_a_win, market_b_win),
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "grid_max_goals_used": 8,
    }


def _stage_input(root: Path) -> None:
    dest = root / "output" / "research" / "combined_backtest"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "live_backtest_predictions.csv").write_bytes(INPUT.read_bytes())


def test_select_examples_prefers_highest_favourite_and_smallest_home_away_gap() -> None:
    module = _load_module()
    predictions = pd.DataFrame(
        [
            _row("M001", "Heavy", "Light", 0.82, 0.11, 0.07, lambda_a=2.6, lambda_b=0.4),
            _row("M002", "Even A", "Even B", 0.351, 0.299, 0.350),
            _row("M003", "Medium A", "Medium B", 0.48, 0.26, 0.26),
        ]
    )

    selected = module.select_example_rows(predictions)

    assert selected["extreme_favourite"]["match_id"] == "M001"
    assert selected["balanced"]["match_id"] == "M002"


@pytest.mark.skipif(not input_present, reason="combined backtest predictions not available")
def test_generate_writes_expected_files_and_summary(tmp_path: Path) -> None:
    _stage_input(tmp_path)
    module = _load_module()

    written = module.generate(tmp_path)

    figures_dir = tmp_path / "paper" / "figures"
    for name in EXPECTED_OUTPUTS:
        path = figures_dir / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0
    assert {path.name for path in written} == set(EXPECTED_OUTPUTS)

    summary = pd.read_csv(figures_dir / "probability_matrix_examples.csv")
    assert set(summary["example"]) == {"extreme_favourite", "balanced"}
    assert {
        "tournament",
        "match_id",
        "fixture",
        "favourite_probability",
        "draw_probability",
        "underdog_probability",
        "home_expected_goals",
        "away_expected_goals",
        "most_likely_scoreline",
        "ev_optimal_scoreline",
    }.issubset(summary.columns)


@pytest.mark.skipif(not input_present, reason="combined backtest predictions not available")
def test_generate_does_not_touch_source_predictions(tmp_path: Path) -> None:
    _stage_input(tmp_path)
    source = tmp_path / "output" / "research" / "combined_backtest" / "live_backtest_predictions.csv"
    before = source.read_bytes()
    module = _load_module()

    module.generate(tmp_path)

    assert source.read_bytes() == before
