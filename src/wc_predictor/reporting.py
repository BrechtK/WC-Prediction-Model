"""Report formatting and CSV/Excel exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_predictor.utils import ensure_parent_directory


def export_dataframe(frame: pd.DataFrame, path: str | Path) -> None:
    """Export a report as CSV or Excel according to the file suffix."""

    path = Path(path)
    ensure_parent_directory(path)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".xlsx":
        frame.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported report file type: {path.suffix}")


def export_report_bundle(
    match_report: pd.DataFrame,
    friend_report: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Write the standard pre-match reports to CSV and one Excel workbook."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    match_report.to_csv(output_dir / "match_recommendations.csv", index=False)
    friend_report.to_csv(output_dir / "friend_prediction_ev.csv", index=False)
    with pd.ExcelWriter(output_dir / "prediction_reports.xlsx") as writer:
        match_report.to_excel(writer, sheet_name="recommendations", index=False)
        friend_report.to_excel(writer, sheet_name="friend_ev", index=False)


def export_standings_bundle(
    scored_predictions: pd.DataFrame,
    standings: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Write realised match scoring and standings reports."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_predictions.to_csv(output_dir / "realised_prediction_scores.csv", index=False)
    standings.to_csv(output_dir / "standings.csv", index=False)
    with pd.ExcelWriter(output_dir / "standings_reports.xlsx") as writer:
        scored_predictions.to_excel(writer, sheet_name="prediction_scores", index=False)
        standings.to_excel(writer, sheet_name="standings", index=False)

