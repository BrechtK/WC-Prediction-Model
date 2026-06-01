"""Report formatting and CSV/Excel exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from wc_predictor.backtesting import BatchBacktestReport, BatchBacktestSettings
from wc_predictor.utils import ensure_parent_directory


def _format_signed(value: float) -> str:
    return f"{value:+.3f}"


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _strategy_average(summary: pd.DataFrame, strategy: str) -> float:
    row = summary.loc[summary["strategy"] == strategy]
    if row.empty:
        raise ValueError(f"Missing strategy summary: {strategy}")
    return float(row.iloc[0]["average_realised_points"])


def _format_overall_strategy_ranking(aggregate_summary: pd.DataFrame) -> str:
    ranking = aggregate_summary.copy()
    favourite_average = _strategy_average(ranking, "favourite_1_0")
    modal_average = _strategy_average(ranking, "most_likely_poisson")
    ranking["gap_vs_favourite_1_0"] = ranking["average_realised_points"] - favourite_average
    ranking["gap_vs_most_likely_poisson"] = ranking["average_realised_points"] - modal_average
    ranking["rank"] = ranking["average_realised_points"].rank(method="min", ascending=False).astype(int)
    ranking = ranking.sort_values(
        ["average_realised_points", "total_points", "strategy"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    lines = [
        "Overall Strategy Ranking",
        "Rank  Strategy                       Avg pts  Total  Exact    Goal diff  Result   vs fav  vs modal",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"{row['rank']:>4}  {row['strategy']:<29} "
            f"{row['average_realised_points']:>7.3f}  "
            f"{int(row['total_points']):>5}  "
            f"{row['exact_score_hit_rate']:>6.1%}  "
            f"{row['correct_goal_difference_hit_rate']:>9.1%}  "
            f"{row['correct_result_hit_rate']:>6.1%}  "
            f"{_format_signed(row['gap_vs_favourite_1_0']):>7}  "
            f"{_format_signed(row['gap_vs_most_likely_poisson']):>8}"
        )
    return "\n".join(lines)


def _format_key_conclusions(aggregate_summary: pd.DataFrame) -> str:
    ranking = aggregate_summary.sort_values(
        ["average_realised_points", "total_points", "strategy"],
        ascending=[False, False, True],
    )
    best = ranking.iloc[0]
    tied_best = ranking[np.isclose(ranking["average_realised_points"], best["average_realised_points"])]
    best_strategies = ", ".join(sorted(tied_best["strategy"].astype(str)))
    favourite = _strategy_average(ranking, "favourite_1_0")
    modal = _strategy_average(ranking, "most_likely_poisson")
    ev_1x2 = _strategy_average(ranking, "ev_optimal_1x2")
    ev_over_under = _strategy_average(ranking, "ev_optimal_1x2_over_under")
    ev_beats_favourite = ev_1x2 > favourite
    ev_beats_modal = ev_1x2 > modal
    hypothesis_supported = ev_beats_favourite and ev_beats_modal
    return "\n".join(
        [
            "Key Conclusions",
            f"- Best strategy overall: {best_strategies} ({best['average_realised_points']:.3f} points per match)",
            f"- ev_optimal_1x2 beats favourite_1_0: {_yes_no(ev_beats_favourite)} ({_format_signed(ev_1x2 - favourite)} points per match)",
            f"- ev_optimal_1x2_over_under beats ev_optimal_1x2: {_yes_no(ev_over_under > ev_1x2)} ({_format_signed(ev_over_under - ev_1x2)} points per match)",
            f"- EV optimisation beats most_likely_poisson: {_yes_no(ev_beats_modal)} ({_format_signed(ev_1x2 - modal)} points per match)",
            f"- Results support the project hypothesis: {_yes_no(hypothesis_supported)}",
        ]
    )


def _per_file_winners(detailed_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    winners: list[dict[str, object]] = []
    first_place_counts: dict[str, int] = {}
    for source_file, group in detailed_summary.groupby("source_file", sort=True):
        ranked = group.sort_values(
            ["average_realised_points", "total_points", "strategy"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        best = ranked.iloc[0]
        tied_best = ranked[np.isclose(ranked["average_realised_points"], best["average_realised_points"])]
        tied_names = sorted(tied_best["strategy"].astype(str))
        for strategy in tied_names:
            first_place_counts[strategy] = first_place_counts.get(strategy, 0) + 1
        remaining = ranked[~np.isclose(ranked["average_realised_points"], best["average_realised_points"])]
        second = remaining.iloc[0] if not remaining.empty else None
        winners.append(
            {
                "source_file": source_file,
                "best_strategy": ", ".join(tied_names),
                "best_average_points": float(best["average_realised_points"]),
                "second_best_strategy": second["strategy"] if second is not None else "None",
                "performance_gap": (
                    float(best["average_realised_points"] - second["average_realised_points"])
                    if second is not None
                    else np.nan
                ),
            }
        )
    winners_frame = pd.DataFrame(winners)
    win_counts = pd.Series(first_place_counts, dtype=int).sort_index()
    return winners_frame, win_counts


def _format_per_file_winners(detailed_summary: pd.DataFrame) -> str:
    winners, win_counts = _per_file_winners(detailed_summary)
    lines = [
        "Per-File Winners",
        "Source file                    Best strategy                   Avg pts  Second-best strategy            Gap",
    ]
    for _, row in winners.iterrows():
        lines.append(
            f"{row['source_file']:<30} "
            f"{row['best_strategy']:<29} "
            f"{row['best_average_points']:>7.3f}  "
            f"{row['second_best_strategy']:<29} "
            f"{_format_signed(row['performance_gap']) if pd.notna(row['performance_gap']) else 'n/a':>7}"
        )
    lines.append("")
    lines.append("First-Place Counts")
    for strategy, count in win_counts.items():
        lines.append(f"- {strategy}: {count}")
    return "\n".join(lines)


def _format_skipped_matches(skipped_by_file_reason: pd.DataFrame) -> str:
    if skipped_by_file_reason.empty:
        return "Skipped Matches\nNo skipped matches."
    compact = (
        skipped_by_file_reason.groupby("reason", as_index=False)
        .agg(
            skipped_matches=("skipped_matches", "sum"),
            files_affected=("source_file", "nunique"),
        )
        .sort_values(["skipped_matches", "reason"], ascending=[False, True])
    )
    lines = ["Skipped Matches", "Reason                               Matches  Files"]
    for _, row in compact.iterrows():
        lines.append(f"{row['reason']:<36} {int(row['skipped_matches']):>7}  {int(row['files_affected']):>5}")
    return "\n".join(lines)


def _format_favourite_strength_analysis(favourite_strength_summary: pd.DataFrame) -> str:
    lines = [
        "Favourite-Strength Analysis",
        "---------------------------",
        "Bucket              Matches  Best strategy                   Avg pts  EV vs fav_1_0",
    ]
    for _, row in favourite_strength_summary.iterrows():
        if int(row["matches"]) == 0:
            lines.append(f"{row['favourite_bucket']:<19} {0:>7}  {'n/a':<29} {'n/a':>7}  {'n/a':>13}")
            continue
        lines.append(
            f"{row['favourite_bucket']:<19} "
            f"{int(row['matches']):>7}  "
            f"{row['best_strategy']:<29} "
            f"{row['best_average_points']:>7.3f}  "
            f"{_format_signed(row['best_ev_gap_vs_favourite_1_0']):>13}"
        )
    return "\n".join(lines)


def format_backtest_console_summary(
    report: BatchBacktestReport,
    input_path: str | Path,
    settings: BatchBacktestSettings,
    verbose: bool = False,
) -> str:
    """Render the default human-readable historical-backtest console report."""

    source_files = report.detailed_summary["source_file"].nunique()
    total_matches = (
        report.predictions[["source_file", "match_id"]].drop_duplicates().shape[0]
        if not report.predictions.empty
        else 0
    )
    total_skipped = (
        report.skipped[["source_file", "match_id"]].drop_duplicates().shape[0]
        if not report.skipped.empty
        else 0
    )
    sections = [
        "\n".join(
            [
                "Backtest Scope",
                f"- Source: {Path(input_path)}",
                f"- Files processed: {source_files}",
                f"- Total matches used: {total_matches}",
                f"- Total skipped matches: {total_skipped}",
            ]
        ),
        _format_overall_strategy_ranking(report.aggregate_summary),
        _format_key_conclusions(report.aggregate_summary),
        _format_favourite_strength_analysis(report.favourite_strength_summary),
        _format_per_file_winners(report.detailed_summary),
        _format_skipped_matches(report.skipped_by_file_reason),
        "\n".join(
            [
                "File Outputs",
                f"- Detailed per-file CSV: {settings.detailed_output_path}",
                f"- Aggregate strategy CSV: {settings.aggregate_output_path}",
                f"- Skipped-match CSV: {settings.skipped_output_path}",
                f"- Favourite-strength CSV: {settings.favourite_strength_output_path}",
            ]
        ),
    ]
    if verbose:
        sections.extend(
            [
                "Verbose Per-File Strategy Results\n" + report.detailed_summary.to_string(index=False),
                "Verbose Aggregate Strategy Results\n" + report.aggregate_summary.to_string(index=False),
                "Verbose Skipped Matches By File And Reason\n"
                + (
                    report.skipped_by_file_reason.to_string(index=False)
                    if not report.skipped_by_file_reason.empty
                    else "No skipped matches."
                ),
                "Verbose Favourite-Strength Results\n"
                + report.favourite_strength_summary.to_string(index=False),
            ]
        )
    return "\n\n".join(sections)


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
                    f"  Favourite strength: p_fav={row['favourite_probability']:.4f} ({row['favourite_bucket']})",
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


def format_world_cup_console_summary(
    match_report: pd.DataFrame,
    input_path: str | Path,
    csv_output_path: str | Path,
    xlsx_output_path: str | Path,
) -> str:
    """Render the real-tournament recommendation summary."""

    sections = [
        "\n".join(
            [
                "World Cup Predictions",
                f"- Source: {Path(input_path)}",
                f"- Matches: {len(match_report)}",
            ]
        )
    ]
    for _, row in match_report.iterrows():
        group = f" / {row['group']}" if pd.notna(row.get("group")) and str(row["group"]).strip() else ""
        qualifier = f"; qualifier={row['recommended_qualifier']}" if row.get("recommended_qualifier") else ""
        sections.append(
            "\n".join(
                [
                    f"{row['match_id']} | {row['team_a']} vs {row['team_b']}",
                    f"  Stage/group: {row['stage']}{group}",
                    f"  Fair 1X2: {row['team_a']}={row['market_a_win']:.2%} Draw={row['market_draw']:.2%} {row['team_b']}={row['market_b_win']:.2%}",
                    f"  Calibrated lambdas: {row['team_a']}={row['lambda_a']:.4f} {row['team_b']}={row['lambda_b']:.4f}",
                    f"  Favourite strength: p_fav={row['favourite_probability']:.2%} ({row['favourite_bucket']})",
                    f"  Modal scoreline: {row['most_likely_scoreline']}",
                    f"  EV-optimal scoreline: {row['recommended_score']}{qualifier}",
                    f"  Top 5 EV scorelines: {row['top_5_ev_predictions']}",
                ]
            )
        )
    sections.append(
        "\n".join(
            [
                "File Outputs",
                f"- CSV recommendations: {Path(csv_output_path)}",
                f"- Excel recommendations: {Path(xlsx_output_path)}",
            ]
        )
    )
    sections.append(format_world_cup_prediction_diagnostics(match_report))
    return "\n\n".join(sections)


def format_world_cup_prediction_diagnostics(match_report: pd.DataFrame) -> str:
    """Render compact post-generation diagnostics for real-tournament output."""

    score_counts = match_report["recommended_score"].value_counts().sort_index()
    bucket_counts = match_report["favourite_bucket"].value_counts().sort_index()
    differs_count = int(match_report["ev_optimal_differs_from_most_likely"].sum())
    lines = [
        "Prediction Output Diagnostics",
        "Recommended Score Counts",
        *(f"- {score}: {count}" for score, count in score_counts.items()),
        "",
        "Favourite-Strength Bucket Counts",
        *(f"- {bucket}: {count}" for bucket, count in bucket_counts.items()),
        "",
        f"- Average lambda_a: {match_report['lambda_a'].mean():.4f}",
        f"- Average lambda_b: {match_report['lambda_b'].mean():.4f}",
        f"- EV-optimal score differs from modal scoreline: {differs_count}",
        "",
        "Top 10 Matches By Favourite Probability",
    ]
    strongest_favourites = match_report.nlargest(10, "favourite_probability")
    for _, row in strongest_favourites.iterrows():
        lines.extend(
            [
                f"{row['match_id']} | {row['team_a']} vs {row['team_b']} | p_fav={row['favourite_probability']:.2%}",
                f"  Fair 1X2: market_a={row['market_a_win']:.2%} market_draw={row['market_draw']:.2%} market_b={row['market_b_win']:.2%}",
                f"  Lambdas: lambda_a={row['lambda_a']:.4f} lambda_b={row['lambda_b']:.4f}",
                f"  Scorelines: recommended={row['recommended_score']} modal={row['most_likely_scoreline']}",
                f"  Top 5 EV predictions: {row['top_5_ev_predictions']}",
            ]
        )
    return "\n".join(lines)


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


def export_world_cup_recommendations_excel(frame: pd.DataFrame, path: str | Path) -> None:
    """Export readable real-tournament recommendations with Excel formatting."""

    path = Path(path)
    ensure_parent_directory(path)
    aliases = {
        "market_a_win": "market_a",
        "market_b_win": "market_b",
        "ev_optimal_differs_from_most_likely": "ev_optimal_differs",
    }
    key_columns = [
        "match_id",
        "date",
        "stage",
        "group",
        "team_a",
        "team_b",
        "market_a_win",
        "market_draw",
        "market_b_win",
        "favourite_probability",
        "favourite_bucket",
        "lambda_a",
        "lambda_b",
        "recommended_score",
        "recommended_qualifier",
        "best_expected_points",
        "most_likely_scoreline",
        "ev_optimal_differs_from_most_likely",
        "top_5_ev_predictions",
        "warnings",
    ]
    ordered_columns = [column for column in key_columns if column in frame]
    ordered_columns.extend(column for column in frame.columns if column not in ordered_columns)
    display_frame = frame[ordered_columns].rename(columns=aliases)
    display_frame.to_excel(path, index=False, sheet_name="recommendations")

    probability_columns = {
        "market_a",
        "market_draw",
        "market_b",
        "favourite_probability",
        "model_a_win",
        "model_draw",
        "model_b_win",
        "exact_score_probability",
        "correct_goal_difference_probability",
        "correct_result_probability",
        "tail_probability_before_renormalisation",
    }
    decimal_columns = {"lambda_a", "lambda_b", "calibration_loss"}
    ev_columns = {"best_expected_points"}
    wrapped_columns = {"top_5_ev_predictions", "warnings"}
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    worksheet = workbook["recommendations"]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in worksheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for index, cell in enumerate(worksheet[1], start=1):
        column_name = str(cell.value)
        for value_cell in worksheet.iter_cols(min_col=index, max_col=index, min_row=2):
            for value in value_cell:
                if column_name in probability_columns:
                    value.number_format = "0.00%"
                elif column_name in decimal_columns:
                    value.number_format = "0.0000"
                elif column_name in ev_columns:
                    value.number_format = "0.000"
                if column_name in wrapped_columns:
                    value.alignment = Alignment(wrap_text=True, vertical="top")
        contents = [str(cell.value or "")]
        contents.extend(str(row[0].value or "") for row in worksheet.iter_cols(min_col=index, max_col=index, min_row=2))
        max_length = max(len(content) for content in contents)
        maximum_width = 60 if column_name in wrapped_columns else 32
        worksheet.column_dimensions[get_column_letter(index)].width = min(max(max_length + 2, 10), maximum_width)

    workbook.save(path)


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
