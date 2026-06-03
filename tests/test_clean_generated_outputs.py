from __future__ import annotations

from pathlib import Path
import runpy

CLEAN_SCRIPT = Path("scripts/clean_generated_outputs.py").resolve()


def test_cleanup_script_removes_only_generated_directories(tmp_path: Path, monkeypatch, capsys) -> None:
    generated = [
        tmp_path / "output/report.xlsx",
        tmp_path / "cache/parsed/core_odds.csv",
        tmp_path / "data/processed/legacy.csv",
        tmp_path / ".pytest_cache/state.txt",
        tmp_path / "src/wc_predictor.egg-info/PKG-INFO",
        tmp_path / "src/wc_predictor/__pycache__/module.pyc",
        tmp_path / "tests/__pycache__/test_module.pyc",
    ]
    protected = [
        tmp_path / "input/schedule.txt",
        tmp_path / "tests/fixture.txt",
        tmp_path / "src/module.py",
        tmp_path / "docs/notes.md",
        tmp_path / "data/raw/oddsportal_combined_pastes/M001_all_odds.txt",
        tmp_path / "data/raw/oddsportal_pastes/M001_1x2.txt",
    ]
    for path in [*generated, *protected]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(CLEAN_SCRIPT), run_name="__main__")

    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "data/processed").exists()
    assert not (tmp_path / ".pytest_cache").exists()
    assert not (tmp_path / "src/wc_predictor.egg-info").exists()
    assert not (tmp_path / "src/wc_predictor/__pycache__").exists()
    assert not (tmp_path / "tests/__pycache__").exists()
    assert all(path.exists() for path in protected)
    output = capsys.readouterr().out
    assert "WARNING: Legacy OddsPortal paste files found." in output
    assert "Move any files you still need into input/odds/" in output
    assert output.index("WARNING:") < output.index("Generated Output Cleanup")
