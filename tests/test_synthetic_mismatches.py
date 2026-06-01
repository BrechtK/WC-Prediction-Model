import runpy

from wc_predictor.synthetic_mismatches import (
    DEFAULT_SYNTHETIC_MISMATCHES,
    format_synthetic_mismatch_report,
    inspect_synthetic_mismatches,
)


def test_synthetic_mismatch_inspection_runs_default_scenarios() -> None:
    results = inspect_synthetic_mismatches()
    assert len(results) == len(DEFAULT_SYNTHETIC_MISMATCHES) == 5
    assert results[0].favourite_probability == 0.55
    assert results[-1].favourite_bucket == "extreme_favourite"
    assert all(len(result.top_5_ev_predictions) == 5 for result in results)


def test_synthetic_mismatch_report_prints_calibration_and_ev_fields() -> None:
    report = format_synthetic_mismatch_report(inspect_synthetic_mismatches([(0.85, 0.10, 0.05)]))
    assert "Fair 1X2: A=85.00% D=10.00% B=5.00%" in report
    assert "Favourite strength: p_fav=85.00% (extreme_favourite)" in report
    assert "Calibrated lambdas:" in report
    assert "Model-implied 1X2:" in report
    assert "Most likely scoreline:" in report
    assert "EV-optimal scoreline:" in report
    assert "Top 5 EV predictions:" in report


def test_synthetic_mismatch_script_runs(capsys) -> None:
    runpy.run_path("scripts/inspect_synthetic_mismatches.py", run_name="__main__")
    report = capsys.readouterr().out
    assert "Fair 1X2: A=55.00% D=25.00% B=20.00%" in report
    assert "Fair 1X2: A=90.00% D=7.00% B=3.00%" in report
