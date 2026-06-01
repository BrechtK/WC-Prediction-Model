"""Report formatting and CSV/Excel exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_predictor.utils import ensure_parent_directory


def format_model_inspection_report(match_report: pd.DataFrame) -> str:
    """Render a compact manual-inspection report from workflow match output."""

    sections: list[str] = []
    for _, row in match_report.iterrows():
        qualifier = f"; qualifier={row['recommended_qualifier']}" if row.get("recommended_qualifier") else ""
        warnings = f"\n  Warnings: {row['warnings']}" if row.get("warnings") else ""
        sections.append(
            "\n".join(
                [
                    f"{row['match_id']} | {row['stage']} | {row['team_a']} vs {row['team_b']}",
                    f"  Raw 1X2 by bookmaker: {row['bookmaker_raw_1x2']}",
                    f"  Fair 1X2 by bookmaker: {row['bookmaker_fair_1x2']}",
                    f"  Aggregated fair 1X2: A={row['market_a_win']:.4f} D={row['market_draw']:.4f} B={row['market_b_win']:.4f}",
                    f"  Calibrated lambdas: A={row['lambda_a']:.4f} B={row['lambda_b']:.4f}",
                    f"  Model-implied 1X2: A={row['model_a_win']:.4f} D={row['model_draw']:.4f} B={row['model_b_win']:.4f}",
                    f"  Calibration error: {row['calibration_loss']:.6f}",
                    f"  Score-matrix tail mass: {row['tail_probability_before_renormalisation']:.6%}",
                    f"  Most likely scoreline: {row['most_likely_scoreline']}",
                    f"  EV-optimal prediction: {row['recommended_score']}{qualifier} ({row['best_expected_points']:.3f} EV)",
                    f"  Top 5 EV predictions: {row['top_5_ev_predictions']}",
                ]
            )
            + warnings
        )
    return "\n\n".join(sections)


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
