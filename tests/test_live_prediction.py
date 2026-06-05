from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from wc_predictor.config import ProjectConfig
from wc_predictor.live_prediction import (
    LivePredictionError,
    LivePredictionSettings,
    format_live_prediction_summary,
    run_live_prediction,
)

CORE_FIXTURES = Path("tests/fixtures/oddsportal_core_pastes")
RUN_SCRIPT = Path("scripts/run_live_prediction.py").resolve()
TEST_WEIGHTS = (1.0, 0.85, 0.75, 0.50, 0.0)


def _write_metadata(path: Path, match_ids: tuple[str, ...] = ("M001",)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "match_id": match_id,
                "date": "2026-06-12",
                "stage": "group",
                "group": "A",
                "team_a": f"{match_id} Alpha",
                "team_b": f"{match_id} Beta",
                "bookmaker": "bet365",
            }
            for match_id in match_ids
        ]
    ).to_excel(path, index=False)


def _write_pastes(
    folder: Path,
    match_id: str = "M001",
    *,
    include_over_under: bool = True,
    include_btts: bool = True,
    include_correct_score: bool = True,
    include_1x2: bool = True,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "1x2": CORE_FIXTURES / "MCORE_1x2.txt",
        "over_under": CORE_FIXTURES / "MCORE_over_under.txt",
        "btts": CORE_FIXTURES / "MCORE_btts.txt",
    }
    included = {
        "1x2": include_1x2,
        "over_under": include_over_under,
        "btts": include_btts,
    }
    for market, source in fixtures.items():
        if included[market]:
            (folder / f"{match_id}_{market}.txt").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    if include_correct_score:
        scorelines = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (0, 2),
            (2, 1),
            (1, 2),
            (3, 0),
            (0, 3),
        ]
        blocks = []
        for index, (score_a, score_b) in enumerate(scorelines):
            book_a_odds = 5.0 + index
            book_b_odds = 5.2 + index
            blocks.append(
                "\n".join(
                    [
                        f"{score_a}:{score_b}",
                        "2",
                        f"{book_b_odds:.2f}",
                        "Book A",
                        "Book A",
                        f"{book_a_odds:.2f}",
                        "Book B",
                        "Book B",
                        f"{book_b_odds:.2f}",
                    ]
                )
            )
        (folder / f"{match_id}_correct_score.txt").write_text("\n\n".join(blocks), encoding="utf-8")


def _write_combined_paste(folder: Path, match_id: str = "M001", *, clean_filename: bool = False) -> None:
    source_folder = folder.parent / "combined_source"
    _write_pastes(source_folder, match_id)
    sections = [
        ("1X2", source_folder / f"{match_id}_1x2.txt"),
        ("OVER_UNDER", source_folder / f"{match_id}_over_under.txt"),
        ("BTTS", source_folder / f"{match_id}_btts.txt"),
        ("CORRECT_SCORE", source_folder / f"{match_id}_correct_score.txt"),
    ]
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{match_id}.txt" if clean_filename else f"{match_id}_all_odds.txt"
    (folder / filename).write_text(
        "\n\n".join(f"### {header}\n{path.read_text(encoding='utf-8')}" for header, path in sections),
        encoding="utf-8",
    )


def _write_schedule(path: Path, fixtures: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for date, team_a, team_b in fixtures:
        blocks.append(
            "\n".join(
                [
                    date,
                    "1",
                    "X",
                    "2",
                    "",
                    "21:00",
                    team_a,
                    team_a,
                    "-",
                    team_b,
                    team_b,
                    "1.50",
                    "4.00",
                    "7.00",
                ]
            )
        )
    path.write_text("\n\n".join(blocks), encoding="utf-8")


def _write_timed_schedule(path: Path, fixtures: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for date, time, team_a, team_b in fixtures:
        blocks.append(
            "\n".join(
                [
                    date,
                    "1",
                    "X",
                    "2",
                    "",
                    time,
                    team_a,
                    team_a,
                    "-",
                    team_b,
                    team_b,
                    "1.50",
                    "4.00",
                    "7.00",
                ]
            )
        )
    path.write_text("\n\n".join(blocks), encoding="utf-8")


def _load_run_live_module():
    spec = importlib.util.spec_from_file_location("run_live_prediction_for_test", RUN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _capture_script_weight_sensitivity(
    tmp_path: Path,
    monkeypatch,
    argv: list[str],
) -> dict[str, object]:
    module = _load_run_live_module()
    schedule_path = tmp_path / "input" / "schedule.txt"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text("schedule", encoding="utf-8")
    schedule = pd.DataFrame(
        [
            {
                "match_id": "M001",
                "date": "2026-06-12",
                "time": "21:00",
                "stage": "group",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
            }
        ]
    )
    prepared_schedule = SimpleNamespace(
        parse_result=None,
        schedule=schedule,
        metadata_path=tmp_path / "cache" / "parsed" / "schedule.csv",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(module, "prepare_schedule_metadata", lambda *args, **kwargs: prepared_schedule)
    monkeypatch.setattr(
        module,
        "_resolve_run_selection",
        lambda **kwargs: module.ResolvedRunSelection(
            "all_available",
            None,
            None,
            None,
            (),
            None,
            pd.DataFrame(),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(module, "_validate_selected_odds_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "split_combined_oddsportal_pastes",
        lambda *args, **kwargs: SimpleNamespace(files_processed=0),
    )

    def fake_run_live_prediction(settings, config):
        captured["skip_weight_sensitivity"] = settings.skip_weight_sensitivity
        captured["enable_margin_method_comparison"] = config.enable_margin_method_comparison
        captured["enable_market_consistent_challenger"] = config.enable_market_consistent_challenger
        captured["correct_score_poisson_weight"] = config.correct_score_poisson_weight
        captured["margin_removal_method"] = config.margin_removal_method
        captured["devig"] = config.devig
        captured["extra_warning_flags_by_match"] = settings.extra_warning_flags_by_match
        return SimpleNamespace(runtime_timings={})

    monkeypatch.setattr(module, "run_live_prediction", fake_run_live_prediction)
    monkeypatch.setattr(module, "format_live_prediction_summary", lambda *args, **kwargs: "summary")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_live_prediction.py", *argv])

    module.main()
    return captured


def _settings(tmp_path: Path, **overrides) -> LivePredictionSettings:
    raw = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    defaults = {
        "input_folder": raw / "oddsportal_pastes",
        "metadata_odds_path": raw / "world_cup_odds.xlsx",
        "schedule_input_path": raw / "oddsportal_schedule.txt",
        "schedule_output_path": raw / "world_cup_schedule_from_paste.csv",
        "schedule_parse_report_path": processed / "oddsportal_schedule_parse_report.csv",
        "core_odds_output_path": raw / "world_cup_odds_from_pastes.csv",
        "total_goals_output_path": raw / "world_cup_total_goals_odds_from_pastes.csv",
        "core_parse_report_path": processed / "oddsportal_core_odds_parse_report.csv",
        "correct_score_output_path": raw / "world_cup_correct_score_odds.csv",
        "correct_score_parse_report_path": processed / "oddsportal_correct_score_parse_report.csv",
        "recommendations_csv_output_path": processed / "world_cup_recommendations.csv",
        "recommendations_xlsx_output_path": processed / "world_cup_recommendations.xlsx",
        "submission_xlsx_output_path": processed / "world_cup_submission_sheet.xlsx",
        "weight_comparison_output_path": processed / "correct_score_weight_comparison.xlsx",
        "weight_sensitivity_weights": TEST_WEIGHTS,
    }
    defaults.update(overrides)
    return LivePredictionSettings(**defaults)


def test_live_runner_processes_all_pastes_writes_outputs_and_prints_submission(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder)

    result = run_live_prediction(settings)
    summary = format_live_prediction_summary(result)

    assert len(result.workflow.match_report) == 1
    assert result.weight_comparison is None
    assert settings.recommendations_xlsx_output_path.exists()
    assert settings.submission_xlsx_output_path.exists()
    assert not settings.weight_comparison_output_path.exists()
    assert settings.core_parse_report_path.exists()
    assert settings.correct_score_parse_report_path.exists()
    assert "Live Prediction Summary" in summary
    assert "Final recommendations:" in summary
    assert "M001 M001 Alpha vs M001 Beta:" in summary
    assert "Manual review:" in summary
    assert "review " in summary
    assert "Decision dashboard summary:" not in summary
    assert "Outputs:" in summary
    assert "Weight sensitivity:" not in summary
    assert "BTTS diagnostics:" not in summary
    assert "Margin-removal sensitivity:" not in summary
    assert str(settings.recommendations_xlsx_output_path) in summary
    assert result.workflow.margin_method_comparison.empty
    assert "Model comparison:" not in summary
    assert "Dixon-Coles challenger:" not in summary
    assert "Market data:" not in summary


def test_live_summary_debug_mode_prints_detailed_diagnostics(tmp_path: Path) -> None:
    settings = _settings(tmp_path, skip_weight_sensitivity=True)
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder)

    result = run_live_prediction(settings)
    summary = format_live_prediction_summary(result, terminal_verbosity="debug")

    assert "BTTS diagnostics:" in summary
    assert "EV explanation:" in summary
    assert "Decision aid:" in summary


def test_live_summary_compact_can_show_runtime_summary(tmp_path: Path) -> None:
    settings = _settings(tmp_path, skip_weight_sensitivity=True)
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder)

    result = run_live_prediction(settings)
    summary = format_live_prediction_summary(result, show_runtime_summary=True)

    assert "Runtime summary:" in summary
    assert "- total:" in summary
    assert "- main model run:" not in summary
    assert "- correct-score weight sensitivity:" not in summary
    assert "- other/unmeasured:" not in summary
    assert "- Excel write:" not in summary


def test_live_warning_flags_reach_dashboard_and_compact_summary(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        skip_weight_sensitivity=True,
        extra_warning_flags_by_match={"M001": ("stale_odds_file",)},
    )
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder)

    result = run_live_prediction(settings)
    summary = format_live_prediction_summary(result)
    dashboard = result.workflow.final_decision_dashboard.iloc[0]

    assert "stale_odds_file" in result.workflow.match_report.iloc[0]["warning_flags"]
    assert "stale_odds_file" in dashboard["risk_notes"]
    assert "Manual review:" in summary
    assert "stale_odds_file" in summary


def test_market_consistent_debug_details_are_hidden_in_compact_output(tmp_path: Path) -> None:
    settings = _settings(tmp_path, skip_weight_sensitivity=True)
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder)

    result = run_live_prediction(settings)
    report = result.workflow.match_report
    report.loc[0, "market_consistent_status"] = "failed_severe"
    report.loc[0, "market_consistent_optimisation_classification"] = "optimisation_failed_severe"
    report.loc[0, "market_consistent_optimisation_success"] = False
    report.loc[0, "market_consistent_optimisation_status_code"] = 1
    report.loc[0, "market_consistent_optimisation_message"] = "iteration limit reached"
    report.loc[0, "market_consistent_optimisation_iterations"] = 500
    report.loc[0, "market_consistent_final_objective_value"] = 1.25
    report.loc[0, "market_consistent_gradient_norm"] = 0.02
    report.loc[0, "market_consistent_max_constraint_error"] = 0.40
    report.loc[0, "market_consistent_constraint_count"] = 8
    report.loc[0, "market_consistent_1x2_constraint_count"] = 3
    report.loc[0, "market_consistent_btts_constraint_count"] = 1
    report.loc[0, "market_consistent_total_goals_constraint_count"] = 1
    report.loc[0, "market_consistent_correct_score_constraint_count"] = 3

    compact = format_live_prediction_summary(result)
    debug = format_live_prediction_summary(result, terminal_verbosity="debug")

    assert "Market-consistent optimiser:" not in compact
    assert "Market-consistent optimiser:" in debug
    assert "iteration limit reached" in debug
    assert "gradient inf-norm" in debug


def test_research_style_config_runs_margin_method_comparison(tmp_path: Path) -> None:
    settings = _settings(tmp_path, skip_weight_sensitivity=True)
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder)

    result = run_live_prediction(
        settings,
        ProjectConfig(
            correct_score_poisson_weight=0.85,
            enable_margin_method_comparison=True,
            enable_market_consistent_challenger=True,
        ),
    )

    assert not result.workflow.margin_method_comparison.empty


def test_live_runner_fails_clearly_without_required_one_x_two_paste(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder, include_1x2=False)

    with pytest.raises(LivePredictionError, match=r"missing required \*_1x2\.txt"):
        run_live_prediction(settings)


def test_live_runner_warns_and_continues_without_btts(tmp_path: Path) -> None:
    settings = _settings(tmp_path, skip_weight_sensitivity=True)
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder, include_btts=False)

    result = run_live_prediction(settings)

    assert "missing_optional_paste:btts" in result.paste_warnings["M001"]
    assert "no_btts_rows_parsed" in result.paste_warnings["M001"]
    assert "BTTS bookmakers:" not in format_live_prediction_summary(result)


def test_live_runner_warns_and_continues_without_correct_scores(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder, include_correct_score=False)

    result = run_live_prediction(settings)

    assert result.weight_comparison is None
    assert "missing_optional_paste:correct_score" in result.paste_warnings["M001"]
    assert "weight_sensitivity_skipped:no_correct_score_odds" not in result.paste_warnings["M001"]


def test_live_runner_match_id_filter_only_processes_selected_match(tmp_path: Path) -> None:
    settings = _settings(tmp_path, match_id="M002", skip_weight_sensitivity=True)
    _write_metadata(settings.metadata_odds_path, ("M001", "M002"))
    _write_pastes(settings.input_folder, "M001")
    _write_pastes(settings.input_folder, "M002")

    result = run_live_prediction(settings)

    assert result.workflow.match_report["match_id"].tolist() == ["M002"]
    assert set(result.core_parse.odds["match_id"]) == {"M002"}
    assert set(result.correct_score_parse.odds["match_id"]) == {"M002"}


def test_live_runner_skip_weight_sensitivity_does_not_write_comparison(tmp_path: Path) -> None:
    settings = _settings(tmp_path, skip_weight_sensitivity=True)
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder)

    result = run_live_prediction(settings)

    assert result.weight_comparison is None
    assert not settings.weight_comparison_output_path.exists()


def test_live_runner_weight_sensitivity_can_be_enabled_without_changing_recommendation(tmp_path: Path) -> None:
    fast_settings = _settings(tmp_path / "fast")
    _write_metadata(fast_settings.metadata_odds_path)
    _write_pastes(fast_settings.input_folder)
    fast_result = run_live_prediction(fast_settings)

    enabled_settings = _settings(tmp_path / "enabled", skip_weight_sensitivity=False)
    _write_metadata(enabled_settings.metadata_odds_path)
    _write_pastes(enabled_settings.input_folder)
    enabled_result = run_live_prediction(enabled_settings)
    summary = format_live_prediction_summary(enabled_result, show_runtime_summary=True)

    assert enabled_result.weight_comparison is not None
    assert enabled_settings.weight_comparison_output_path.exists()
    assert "Weight sensitivity:" in summary
    assert fast_result.workflow.match_report["recommended_score"].tolist() == enabled_result.workflow.match_report[
        "recommended_score"
    ].tolist()


def test_live_runner_warns_when_weight_sensitivity_enabled_without_correct_scores(tmp_path: Path) -> None:
    settings = _settings(tmp_path, skip_weight_sensitivity=False)
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder, include_correct_score=False)

    result = run_live_prediction(settings)

    assert "weight_sensitivity_skipped:no_correct_score_odds" in result.paste_warnings["M001"]


def test_script_live_profile_disables_weight_sensitivity_by_default(tmp_path: Path, monkeypatch) -> None:
    captured = _capture_script_weight_sensitivity(tmp_path, monkeypatch, [])

    assert captured["skip_weight_sensitivity"] is True
    assert captured["enable_margin_method_comparison"] is False
    assert captured["enable_market_consistent_challenger"] == "only_if_close"
    assert captured["correct_score_poisson_weight"] == pytest.approx(1.0)


def test_script_research_profile_enables_weight_sensitivity_by_default(tmp_path: Path, monkeypatch) -> None:
    captured = _capture_script_weight_sensitivity(tmp_path, monkeypatch, ["--run-profile", "research"])

    assert captured["skip_weight_sensitivity"] is False
    assert captured["enable_margin_method_comparison"] is True
    assert captured["enable_market_consistent_challenger"] is True


def test_script_cli_can_still_select_shin_margin_method(tmp_path: Path, monkeypatch) -> None:
    captured = _capture_script_weight_sensitivity(
        tmp_path,
        monkeypatch,
        ["--run-profile", "research", "--margin-removal-method", "shin"],
    )

    assert captured["margin_removal_method"] == "shin"


def test_script_devig_profile_defaults_to_global(tmp_path: Path, monkeypatch) -> None:
    captured = _capture_script_weight_sensitivity(tmp_path, monkeypatch, [])

    assert captured["devig"] is None


def test_script_cli_can_select_market_specific_devig(tmp_path: Path, monkeypatch) -> None:
    captured = _capture_script_weight_sensitivity(
        tmp_path,
        monkeypatch,
        ["--devig-profile", "market_specific"],
    )

    devig = captured["devig"]
    assert devig is not None
    assert devig.method_for("correct_score") == "normalised_inverse_odds"
    assert devig.method_for("1x2") == "power"


def test_script_flag_can_enable_weight_sensitivity_in_live_profile(tmp_path: Path, monkeypatch) -> None:
    captured = _capture_script_weight_sensitivity(tmp_path, monkeypatch, ["--enable-weight-sensitivity"])

    assert captured["skip_weight_sensitivity"] is False


def test_live_runner_strict_mode_fails_when_optional_paste_is_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path, strict=True)
    _write_metadata(settings.metadata_odds_path)
    _write_pastes(settings.input_folder, include_btts=False)

    with pytest.raises(LivePredictionError, match="Strict mode requires"):
        run_live_prediction(settings)


def test_live_runner_script_runs_with_explicit_vscode_style_launch_args(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_timed_schedule(
        tmp_path / "input/schedule.txt",
        [
            ("14 Jun 2026", "00:00", "Brazil", "Morocco"),
            ("14 Jun 2026", "03:00", "Haiti", "Scotland"),
            ("14 Jun 2026", "06:00", "Australia", "Turkey"),
            ("14 Jun 2026", "09:00", "Germany", "Curacao"),
            ("14 Jun 2026", "12:00", "Netherlands", "Japan"),
        ],
    )
    _write_combined_paste(tmp_path / "input/odds", "M004", clean_filename=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_live_prediction.py",
            "--run-mode",
            "single_match",
            "--date",
            "14-6",
            "--game-number",
            "4",
            "--skip-weight-sensitivity",
        ],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    assert "Final recommendations:" in output
    assert "M004 Germany vs Curacao:" in output
    assert (tmp_path / "output/predictions.xlsx").exists()


def test_live_runner_script_default_margin_method_is_normalised_inverse_odds() -> None:
    module = _load_run_live_module()

    assert module.MARGIN_REMOVAL_METHOD == "normalised_inverse_odds"
    assert "shin" in module.VALID_MARGIN_REMOVAL_METHODS


def test_live_runner_script_splits_combined_pastes_before_parsing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    _write_combined_paste(tmp_path / "input/odds", clean_filename=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_prediction.py", "--run-mode", "all_available", "--skip-weight-sensitivity"],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    assert "Combined OddsPortal Paste Split" not in output
    assert "Final recommendations:" in output
    assert (tmp_path / "cache/split_pastes/M001_1x2.txt").exists()
    assert (tmp_path / "output/predictions.xlsx").exists()


def test_live_runner_script_warns_when_selected_odds_file_is_stale(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    _write_combined_paste(tmp_path / "input/odds", clean_filename=True)
    stale_path = tmp_path / "input" / "odds" / "M001.txt"
    stale_mtime = time.time() - 25 * 60 * 60
    os.utime(stale_path, (stale_mtime, stale_mtime))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_prediction.py", "--run-mode", "all_available", "--skip-weight-sensitivity"],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    assert (
        "Warning: input/odds/M001.txt was last modified more than 24 hours ago. "
        "Check that odds are fresh."
    ) in output
    assert "Final recommendations:" in output
    report = pd.read_csv(tmp_path / "output" / "predictions.csv")
    assert "stale_odds_file" in report.loc[0, "warning_flags"]


def test_live_runner_script_can_keep_stale_odds_out_of_manual_review_flags(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    _write_combined_paste(tmp_path / "input/odds", clean_filename=True)
    stale_path = tmp_path / "input" / "odds" / "M001.txt"
    stale_mtime = time.time() - 25 * 60 * 60
    os.utime(stale_path, (stale_mtime, stale_mtime))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_live_prediction.py",
            "--run-mode",
            "all_available",
            "--skip-weight-sensitivity",
            "--no-stale-odds-affects-manual-review",
        ],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    assert "Check that odds are fresh." in output
    report = pd.read_csv(tmp_path / "output" / "predictions.csv")
    assert "stale_odds_file" not in str(report.loc[0, "warning_flags"])


def test_live_runner_compact_prints_split_summary_when_warnings_exist(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    source_folder = tmp_path / "combined_source"
    _write_pastes(source_folder, "M001", include_over_under=False, include_btts=False, include_correct_score=False)
    odds_folder = tmp_path / "input" / "odds"
    odds_folder.mkdir(parents=True, exist_ok=True)
    (odds_folder / "M001.txt").write_text(
        "### 1X2\n" + (source_folder / "M001_1x2.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_prediction.py", "--run-mode", "all_available", "--skip-weight-sensitivity"],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    assert "Combined OddsPortal Paste Split" in output
    assert "missing_optional_section:over_under" in output
    assert "Final recommendations:" in output


def test_live_runner_script_can_skip_combined_paste_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    _write_combined_paste(tmp_path / "input/odds", clean_filename=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_live_prediction.py",
            "--run-mode",
            "all_available",
            "--skip-combined-split",
            "--skip-weight-sensitivity",
        ],
    )

    with pytest.raises(
        SystemExit,
        match=r"No odds input files found\.",
    ):
        runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    assert not (tmp_path / "cache/split_pastes/M001_1x2.txt").exists()


def test_live_runner_script_reports_user_facing_error_when_no_odds_inputs_exist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_prediction.py", "--run-mode", "all_available", "--skip-weight-sensitivity"],
    )

    with pytest.raises(
        SystemExit,
        match=r"No odds input files found\.",
    ):
        runpy.run_path(str(RUN_SCRIPT), run_name="__main__")


def test_live_runner_script_uses_schedule_mapping_for_combined_paste(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    _write_combined_paste(tmp_path / "input/odds", clean_filename=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_prediction.py", "--run-mode", "all_available", "--skip-weight-sensitivity"],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    parsed_odds = pd.read_csv(tmp_path / "cache/parsed/core_odds.csv")
    assert "Schedule Paste Parse" not in output
    assert "Schedule mapping:" not in output
    assert "M001 Schedule Alpha vs Schedule Beta:" in output
    assert parsed_odds["team_a"].eq("Schedule Alpha").all()
    assert parsed_odds["team_b"].eq("Schedule Beta").all()


def test_live_runner_script_list_date_lists_fixtures_and_exits(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_timed_schedule(
        tmp_path / "input/schedule.txt",
        [
            ("14 Jun 2026", "00:00", "Brazil", "Morocco"),
            ("14 Jun 2026", "03:00", "Haiti", "Scotland"),
            ("14 Jun 2026", "06:00", "Australia", "Turkey"),
            ("14 Jun 2026", "09:00", "Germany", "Curacao"),
            ("14 Jun 2026", "12:00", "Netherlands", "Japan"),
        ],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_live_prediction.py", "--run-mode", "list_date", "--date", "14-6"])

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    assert "14 Jun 2026" in output
    assert "1. M001 | 00:00 | Brazil vs Morocco" in output
    assert "3. M003 | 06:00 | Australia vs Turkey" in output
    assert "4. M004 | 09:00 | Germany vs Curacao" in output
    assert "5. M005 | 12:00 | Netherlands vs Japan" in output
    assert "Final recommended submission:" not in output
    assert not (tmp_path / "output/predictions.xlsx").exists()


def test_live_runner_script_resolves_single_match_from_date_and_game_number(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_timed_schedule(
        tmp_path / "input/schedule.txt",
        [
            ("14 Jun 2026", "00:00", "Brazil", "Morocco"),
            ("14 Jun 2026", "03:00", "Haiti", "Scotland"),
            ("14 Jun 2026", "06:00", "Australia", "Turkey"),
        ],
    )
    _write_combined_paste(tmp_path / "input/odds", "M003", clean_filename=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_prediction.py", "--date", "14-6", "--game-number", "3", "--skip-weight-sensitivity"],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    parsed_odds = pd.read_csv(tmp_path / "cache/parsed/core_odds.csv")
    assert parsed_odds["match_id"].astype(str).unique().tolist() == ["M003"]
    assert "M003 Australia vs Turkey:" in output


def test_live_runner_script_runs_all_matches_on_selected_date(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_timed_schedule(
        tmp_path / "input/schedule.txt",
        [
            ("14 Jun 2026", "00:00", "Brazil", "Morocco"),
            ("14 Jun 2026", "03:00", "Haiti", "Scotland"),
            ("15 Jun 2026", "06:00", "Australia", "Turkey"),
        ],
    )
    _write_combined_paste(tmp_path / "input/odds", "M001", clean_filename=True)
    _write_combined_paste(tmp_path / "input/odds", "M002", clean_filename=True)
    _write_combined_paste(tmp_path / "input/odds", "M003", clean_filename=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_prediction.py", "--run-mode", "date", "--date", "14-6", "--skip-weight-sensitivity"],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    parsed_odds = pd.read_csv(tmp_path / "cache/parsed/core_odds.csv")
    assert parsed_odds["match_id"].astype(str).unique().tolist() == ["M001", "M002"]
    assert "M001 Brazil vs Morocco:" in output
    assert "M002 Haiti vs Scotland:" in output
    assert "M003 Australia vs Turkey:" not in output


def test_live_runner_script_date_mode_reports_missing_odds_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_timed_schedule(
        tmp_path / "input/schedule.txt",
        [
            ("14 Jun 2026", "00:00", "Brazil", "Morocco"),
            ("14 Jun 2026", "03:00", "Haiti", "Scotland"),
        ],
    )
    _write_combined_paste(tmp_path / "input/odds", "M001", clean_filename=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_live_prediction.py", "--run-mode", "date", "--date", "14-6"])

    with pytest.raises(SystemExit, match=r"Selected date fixtures with missing odds:"):
        runpy.run_path(str(RUN_SCRIPT), run_name="__main__")


def test_live_runner_script_match_id_overrides_date_and_game_number(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_timed_schedule(
        tmp_path / "input/schedule.txt",
        [
            ("14 Jun 2026", "00:00", "Brazil", "Morocco"),
            ("14 Jun 2026", "03:00", "Haiti", "Scotland"),
            ("14 Jun 2026", "06:00", "Australia", "Turkey"),
        ],
    )
    _write_combined_paste(tmp_path / "input/odds", "M003", clean_filename=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_live_prediction.py",
            "--run-mode",
            "single_match",
            "--date",
            "14-6",
            "--game-number",
            "1",
            "--match-id",
            "M003",
            "--skip-weight-sensitivity",
        ],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    assert "M001 Brazil vs Morocco:" not in output
    assert "M003 Australia vs Turkey:" in output


def test_live_runner_script_cli_overrides_user_settings_block(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    _write_combined_paste(tmp_path / "input/odds", clean_filename=True)
    module = _load_run_live_module()
    monkeypatch.setattr(module, "RUN_MODE", "list_date")
    monkeypatch.setattr(module, "DATE", "11-6")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_prediction.py", "--run-mode", "all_available", "--skip-weight-sensitivity"],
    )

    module.main()

    output = capsys.readouterr().out
    assert "Final recommendations:" in output
    assert (tmp_path / "output/predictions.xlsx").exists()


def test_live_runner_script_invalid_date_gives_helpful_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_live_prediction.py", "--run-mode", "list_date", "--date", "bad-date"])

    with pytest.raises(SystemExit, match="Invalid date 'bad-date'"):
        runpy.run_path(str(RUN_SCRIPT), run_name="__main__")


def test_live_runner_script_invalid_game_number_gives_helpful_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_timed_schedule(
        tmp_path / "input/schedule.txt",
        [
            ("14 Jun 2026", "00:00", "Brazil", "Morocco"),
            ("14 Jun 2026", "03:00", "Haiti", "Scotland"),
        ],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_live_prediction.py", "--date", "14-6", "--game-number", "3"])

    with pytest.raises(SystemExit, match="GAME_NUMBER 3 is out of range"):
        runpy.run_path(str(RUN_SCRIPT), run_name="__main__")


def test_live_runner_script_missing_selected_odds_file_gives_helpful_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_timed_schedule(
        tmp_path / "input/schedule.txt",
        [("14 Jun 2026", "06:00", "Australia", "Turkey")],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_live_prediction.py", "--date", "14-6", "--game-number", "1"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    message = str(exc_info.value)
    assert "Missing odds file:" in message
    assert "input\\odds\\M001.txt" in message or "input/odds/M001.txt" in message
    assert "templates\\odds_input_template.txt" in message or "templates/odds_input_template.txt" in message


def test_live_runner_script_rejects_combined_paste_unknown_to_schedule(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    _write_combined_paste(tmp_path / "input/odds", "M999", clean_filename=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_prediction.py", "--run-mode", "all_available", "--skip-weight-sensitivity"],
    )

    with pytest.raises(
        SystemExit,
        match=r"M999\.txt was found, but M999 is not present in the parsed schedule\.",
    ):
        runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    assert not (tmp_path / "cache/split_pastes/M999_1x2.txt").exists()


def test_live_runner_script_uses_clean_input_output_and_cache_structure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    _write_combined_paste(tmp_path / "input/odds", clean_filename=True)
    stale_split = tmp_path / "cache/split_pastes/M001_1x2.txt"
    stale_split.parent.mkdir(parents=True)
    stale_split.write_text("stale cache", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_prediction.py", "--run-mode", "all_available", "--skip-weight-sensitivity"],
    )

    runpy.run_path(str(RUN_SCRIPT), run_name="__main__")

    output = capsys.readouterr().out
    assert "M001 Schedule Alpha vs Schedule Beta:" in output
    assert (tmp_path / "cache/split_pastes/M001_1x2.txt").exists()
    assert stale_split.read_text(encoding="utf-8") != "stale cache"
    assert (tmp_path / "cache/parsed/core_odds.csv").exists()
    assert (tmp_path / "cache/parsed/total_goals_odds.csv").exists()
    assert (tmp_path / "cache/parsed/correct_score_odds.csv").exists()
    assert (tmp_path / "cache/parsed/schedule.csv").exists()
    assert (tmp_path / "output/predictions.xlsx").exists()
    assert (tmp_path / "output/submission_sheet.xlsx").exists()
    assert (tmp_path / "output/parse_reports/schedule_parse_report.csv").exists()
    assert (tmp_path / "output/parse_reports/odds_parse_report.csv").exists()
    assert (tmp_path / "output/parse_reports/correct_score_parse_report.csv").exists()


def test_live_runner_script_requires_clean_schedule_for_clean_odds_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_combined_paste(tmp_path / "input/odds", clean_filename=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_live_prediction.py", "--skip-weight-sensitivity"])

    with pytest.raises(SystemExit, match=r"input\\schedule\.txt is required"):
        runpy.run_path(str(RUN_SCRIPT), run_name="__main__")


def test_live_runner_script_rejects_clean_odds_match_unknown_to_schedule(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_schedule(tmp_path / "input/schedule.txt", [("11 Jun 2026", "Schedule Alpha", "Schedule Beta")])
    _write_combined_paste(tmp_path / "input/odds", "M999", clean_filename=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_prediction.py", "--run-mode", "all_available", "--skip-weight-sensitivity"],
    )

    with pytest.raises(
        SystemExit,
        match=r"M999\.txt was found, but M999 is not present in the parsed schedule\.",
    ):
        runpy.run_path(str(RUN_SCRIPT), run_name="__main__")
