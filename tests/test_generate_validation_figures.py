from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_validation_figures.py"
BACKTEST_DIR = REPO_ROOT / "output" / "research" / "combined_backtest"

REQUIRED_INPUTS = [
    "live_backtest_predictions.csv",
    "calibration_1x2.csv",
    "live_backtest_summary.csv",
]

EXPECTED_OUTPUTS = [
    "calibration_1x2.png",
    "expected_vs_actual_goals.png",
    "points_vs_probability_quality.png",
    "calibration_1x2_summary.csv",
    "goals_summary.csv",
    "strategy_probability_summary.csv",
]

pytest.importorskip("matplotlib")

inputs_present = all((BACKTEST_DIR / name).exists() for name in REQUIRED_INPUTS)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_validation_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _stage_inputs(root: Path) -> None:
    dest = root / "output" / "research" / "combined_backtest"
    dest.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_INPUTS:
        (dest / name).write_bytes((BACKTEST_DIR / name).read_bytes())


@pytest.mark.skipif(not inputs_present, reason="combined backtest outputs not available")
def test_generate_writes_expected_files(tmp_path: Path) -> None:
    _stage_inputs(tmp_path)
    module = _load_module()

    written = module.generate(tmp_path)

    figures_dir = tmp_path / "paper" / "figures"
    for name in EXPECTED_OUTPUTS:
        path = figures_dir / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0
    assert {p.name for p in written} == set(EXPECTED_OUTPUTS)


@pytest.mark.skipif(not inputs_present, reason="combined backtest outputs not available")
def test_summary_tables_match_known_values(tmp_path: Path) -> None:
    _stage_inputs(tmp_path)
    module = _load_module()
    module.generate(tmp_path)

    figures_dir = tmp_path / "paper" / "figures"
    goals = pd.read_csv(figures_dir / "goals_summary.csv")
    combined = goals.loc[goals["sample"] == "Combined"].iloc[0]
    # Reproduces the figures already reported in the paper, to 3 dp.
    assert round(combined["expected_total_goals"], 3) == 2.567
    assert round(combined["actual_total_goals"], 3) == 2.521
    assert int(combined["matches"]) == 96


@pytest.mark.skipif(not inputs_present, reason="combined backtest outputs not available")
def test_generate_does_not_touch_source_outputs(tmp_path: Path) -> None:
    """The generator must only write into paper/figures, never the backtest dir."""

    _stage_inputs(tmp_path)
    before = {
        name: (tmp_path / "output" / "research" / "combined_backtest" / name).read_bytes()
        for name in REQUIRED_INPUTS
    }
    module = _load_module()
    module.generate(tmp_path)
    for name, content in before.items():
        path = tmp_path / "output" / "research" / "combined_backtest" / name
        assert path.read_bytes() == content
