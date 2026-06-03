from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


RUN_SCRIPT = Path("scripts/run_live_prediction.py").resolve()


def _load_run_live_module():
    spec = importlib.util.spec_from_file_location("run_live_prediction_docs_test", RUN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_run_file_defaults_remain_fast_for_matchday() -> None:
    module = _load_run_live_module()

    assert module.RUN_PROFILE == "live"
    assert module.TERMINAL_VERBOSITY == "compact"
    assert module.ENABLE_MARGIN_METHOD_COMPARISON is False
    assert module.ENABLE_CORRECT_SCORE_WEIGHT_SENSITIVITY is False
    assert module.ENABLE_MARKET_CONSISTENT_CHALLENGER == "only_if_close"


def test_readme_and_matchday_docs_reference_live_paths() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    workflow = Path("docs/matchday_workflow.md").read_text(encoding="utf-8")
    template = Path("templates/odds_input_template.txt").read_text(encoding="utf-8")

    for text in (readme, workflow):
        assert "scripts/run_live_prediction.py" in text
        assert "input/schedule.txt" in text
        assert "input/odds/M" in text
        assert "templates/odds_input_template.txt" in text
        assert "output/submission_sheet.xlsx" in text
        assert "output/predictions.xlsx" in text

    assert "### MATCH" in template
    assert "<paste OddsPortal 1X2 table here>" in template
    assert "<paste OddsPortal correct-score table here>" in template


def test_gitignore_keeps_live_inputs_local_but_templates_tracked() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    lines = set(gitignore.splitlines())

    assert "input/schedule.txt" in gitignore
    assert "input/odds/*.txt" in gitignore
    assert "!input/.gitkeep" in gitignore
    assert "!input/odds/.gitkeep" in gitignore
    assert "output/" in gitignore
    assert "cache/" in gitignore
    assert "data/raw/" in gitignore
    assert "data/processed/" in gitignore
    assert ".venv/" in gitignore
    assert "**/__pycache__/" in gitignore
    assert "*.pyc" in gitignore
    assert ".pytest_cache/" in gitignore
    assert "~$*.xlsx" in gitignore
    assert "templates/" not in lines
    assert "docs/" not in lines
    assert "tests/fixtures/" not in lines
