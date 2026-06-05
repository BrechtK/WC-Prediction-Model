"""Tests for the live-pipeline historical backtest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wc_predictor.live_backtest import (
    DEFAULT_BACKTEST_CONFIGS,
    QUICK_BACKTEST_CONFIGS,
    HistoricalResult,
    LiveBacktestSettings,
    _parse_score,
    _score_strategies,
    load_historical_results,
    run_live_backtest,
)

CORE_FIXTURES = Path("tests/fixtures/oddsportal_core_pastes")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_results_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_combined_paste(folder: Path, match_id: str, *, include_ah: bool = False) -> None:
    """Write a minimal combined paste .txt file in live-tool format."""
    folder.mkdir(parents=True, exist_ok=True)

    one_x_two = CORE_FIXTURES / "MCORE_1x2.txt"
    over_under = CORE_FIXTURES / "MCORE_over_under.txt"
    btts = CORE_FIXTURES / "MCORE_btts.txt"

    sections = [
        ("MATCH", f"Team A vs Team B"),
        ("1X2", one_x_two.read_text(encoding="utf-8")),
        ("OVER_UNDER", over_under.read_text(encoding="utf-8")),
        ("BTTS", btts.read_text(encoding="utf-8")),
    ]
    if include_ah:
        ah_text = "\n".join([
            "Asian Handicap -1.5",
            "Pinnacle",
            "-1.5",
            "1.85",
            "1.97",
            "Bet365",
            "-1.5",
            "1.83",
            "1.95",
        ])
        sections.append(("ASIAN_HANDICAP", ah_text))

    (folder / f"{match_id}.txt").write_text(
        "\n\n".join(f"### {header}\n{content}" for header, content in sections),
        encoding="utf-8",
    )


def _simple_settings(
    tmp_path: Path,
    match_ids: list[str],
    *,
    include_ah: bool = False,
    configs=None,
) -> LiveBacktestSettings:
    odds_folder = tmp_path / "odds"
    results_path = tmp_path / "results.csv"
    cache_folder = tmp_path / "cache"

    for match_id in match_ids:
        _write_combined_paste(odds_folder, match_id, include_ah=include_ah)

    _write_results_csv(
        results_path,
        [
            {
                "match_id": mid,
                "date": "2022-11-20",
                "stage": "group stage",
                "group": "A",
                "team_a": "Team A",
                "team_b": "Team B",
                "actual_score_a": 1,
                "actual_score_b": 0,
            }
            for mid in match_ids
        ],
    )
    from wc_predictor.config import ProjectConfig

    if configs is None:
        configs = (
            DEFAULT_BACKTEST_CONFIGS[0],  # baseline_ev
        )

    return LiveBacktestSettings(
        historical_odds_folder=odds_folder,
        results_path=results_path,
        cache_folder=cache_folder,
        summary_output_path=tmp_path / "summary.csv",
        predictions_output_path=tmp_path / "predictions.csv",
        excel_output_path=tmp_path / "backtest.xlsx",
        configs=configs,
    )


# ---------------------------------------------------------------------------
# Unit tests: load_historical_results
# ---------------------------------------------------------------------------

def test_load_historical_results_parses_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results_csv(path, [
        {"match_id": "M001", "team_a": "Brazil", "team_b": "Serbia",
         "actual_score_a": 2, "actual_score_b": 0},
    ])

    results = load_historical_results(path)

    assert len(results) == 1
    assert results[0].match_id == "M001"
    assert results[0].team_a == "Brazil"
    assert results[0].actual_score_a == 2
    assert results[0].actual_score_b == 0


def test_load_historical_results_applies_optional_column_defaults(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results_csv(path, [
        {"match_id": "M001", "team_a": "A", "team_b": "B",
         "actual_score_a": 1, "actual_score_b": 1},
    ])

    results = load_historical_results(path)

    assert results[0].stage == "group stage"
    assert results[0].group == ""


def test_load_historical_results_raises_on_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results_csv(path, [
        {"match_id": "M001", "team_a": "A",
         "actual_score_a": 1, "actual_score_b": 0},  # missing team_b
    ])

    with pytest.raises(ValueError, match="team_b"):
        load_historical_results(path)


def test_load_historical_results_skips_rows_with_invalid_scores(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_results_csv(path, [
        {"match_id": "M001", "team_a": "A", "team_b": "B",
         "actual_score_a": "bad", "actual_score_b": 0},
        {"match_id": "M002", "team_a": "C", "team_b": "D",
         "actual_score_a": 1, "actual_score_b": 1},
    ])

    results = load_historical_results(path)

    assert len(results) == 1
    assert results[0].match_id == "M002"


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

def test_parse_score_parses_valid_label() -> None:
    assert _parse_score("2-1") == (2, 1)
    assert _parse_score("0-0") == (0, 0)


def test_parse_score_returns_none_for_invalid_input() -> None:
    assert _parse_score("") is None
    assert _parse_score(None) is None
    assert _parse_score("bad") is None


def test_score_strategies_extracts_ev_default() -> None:
    row = pd.Series({
        "recommended_score": "1-0",
        "baseline_poisson_recommended_score": "1-0",
        "most_likely_scoreline": "1-0",
        "dixon_coles_recommended_score": "1-0",
        "market_consistent_recommended_score": "1-0",
        "market_consistent_status": "skipped",
    })

    strategies = _score_strategies(row, actual_a=1, actual_b=0)

    names = {s["strategy"] for s in strategies}
    assert "ev_default" in names
    assert "market_consistent" not in names  # MC was skipped


def test_score_strategies_includes_market_consistent_when_mc_ran() -> None:
    row = pd.Series({
        "recommended_score": "1-0",
        "baseline_poisson_recommended_score": "1-0",
        "most_likely_scoreline": "1-0",
        "dixon_coles_recommended_score": "1-0",
        "market_consistent_recommended_score": "2-0",
        "market_consistent_status": "ok",
    })

    strategies = _score_strategies(row, actual_a=2, actual_b=0)

    names = {s["strategy"] for s in strategies}
    assert "market_consistent" in names
    mc = next(s for s in strategies if s["strategy"] == "market_consistent")
    assert mc["realised_points"] == 10  # exact score
    ev = next(s for s in strategies if s["strategy"] == "ev_default")
    assert ev["realised_points"] == 5  # correct result only (1-0 vs 2-0: different gd)


# ---------------------------------------------------------------------------
# Integration tests: run_live_backtest
# ---------------------------------------------------------------------------

def test_backtest_produces_summary_and_predictions_for_single_match(
    tmp_path: Path,
) -> None:
    settings = _simple_settings(tmp_path, ["M001"])

    report = run_live_backtest(settings, export=True, progress=False)

    assert not report.predictions.empty
    assert not report.summary.empty
    assert "config" in report.predictions.columns
    assert "strategy" in report.predictions.columns
    assert "realised_points" in report.predictions.columns
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "backtest.xlsx").exists()


def test_backtest_produces_output_for_multiple_matches(tmp_path: Path) -> None:
    settings = _simple_settings(tmp_path, ["M001", "M002", "M003"])

    report = run_live_backtest(settings, export=False, progress=False)

    assert report.predictions["match_id"].nunique() == 3
    assert not report.summary.empty


def test_backtest_records_skip_when_no_odds_file(tmp_path: Path) -> None:
    # M001 has odds; M999 has no paste file
    settings = _simple_settings(tmp_path, ["M001"])
    results_path = settings.results_path
    existing = pd.read_csv(results_path)
    extra = pd.DataFrame([{
        "match_id": "M999",
        "date": "2022-11-20",
        "stage": "group stage",
        "group": "A",
        "team_a": "X",
        "team_b": "Y",
        "actual_score_a": 1,
        "actual_score_b": 0,
    }])
    pd.concat([existing, extra]).to_csv(results_path, index=False)

    report = run_live_backtest(settings, export=False, progress=False)

    assert not report.skipped.empty
    assert "M999" in report.skipped["match_id"].values


def test_backtest_with_ah_runs_mc_challenger(tmp_path: Path) -> None:
    from wc_predictor.config import ProjectConfig

    mc_config = DEFAULT_BACKTEST_CONFIGS[3]  # mc_with_ah
    settings = _simple_settings(tmp_path, ["M001"], include_ah=True, configs=(mc_config,))

    report = run_live_backtest(settings, export=False, progress=False)

    assert not report.predictions.empty
    assert "mc_with_ah" in report.predictions["config"].values


def test_backtest_summary_ranks_configs_by_average_points(tmp_path: Path) -> None:
    settings = _simple_settings(tmp_path, ["M001", "M002"])

    report = run_live_backtest(settings, export=False, progress=False)

    assert "rank" in report.summary.columns
    assert report.summary["rank"].min() == 1


def test_quick_configs_are_a_subset_of_default_configs() -> None:
    default_names = {c.name for c in DEFAULT_BACKTEST_CONFIGS}
    quick_names = {c.name for c in QUICK_BACKTEST_CONFIGS}

    assert quick_names.issubset(default_names)
