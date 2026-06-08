"""Generate validation figures from the World Cup backtest outputs.

The figures are read-only with respect to the model. They consume
``output/research/world_cup_backtest_predictions.csv`` plus, when available,
the richer 2018/2022 live-paste validation output in
``output/research/combined_backtest/live_backtest_predictions.csv``. The script
writes public-facing figures plus small summary CSVs into ``paper/figures/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPRESENTATIVE_STRATEGY = "ev_optimal_1x2"
STRATEGY_LABELS = {
    "most_likely_poisson": "Modal / most-likely",
    "ev_optimal_1x2": "EV 1X2",
    "ev_optimal_1x2_over_under": "EV 1X2 + O/U slot",
}


def _predictions_path(root: Path) -> Path:
    return root / "output" / "research" / "world_cup_backtest_predictions.csv"


def _richer_predictions_path(root: Path) -> Path:
    return root / "output" / "research" / "combined_backtest" / "live_backtest_predictions.csv"


def _figures_dir(root: Path) -> Path:
    return root / "paper" / "figures"


def _representative_matches(predictions: pd.DataFrame) -> pd.DataFrame:
    rep = predictions[predictions["strategy"].eq(REPRESENTATIVE_STRATEGY)].copy()
    if rep.empty:
        raise ValueError(f"No rows found for representative strategy {REPRESENTATIVE_STRATEGY!r}")
    return rep.drop_duplicates(["tournament", "match_id"]).reset_index(drop=True)


def _calibration_long(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in matches.iterrows():
        probs = {
            "team_a": float(row["predicted_probability_team_a_win"]),
            "draw": float(row["predicted_probability_draw"]),
            "team_b": float(row["predicted_probability_team_b_win"]),
        }
        favourite_side = "team_a" if probs["team_a"] >= probs["team_b"] else "team_b"
        underdog_side = "team_b" if favourite_side == "team_a" else "team_a"
        realised = str(row["realised_outcome"])
        rows.extend(
            [
                {
                    "calibration_target": "favourite",
                    "predicted_probability": probs[favourite_side],
                    "realised": int(realised == f"{favourite_side}_win"),
                },
                {
                    "calibration_target": "draw",
                    "predicted_probability": probs["draw"],
                    "realised": int(realised == "draw"),
                },
                {
                    "calibration_target": "underdog",
                    "predicted_probability": probs[underdog_side],
                    "realised": int(realised == f"{underdog_side}_win"),
                },
            ]
        )
    return pd.DataFrame(rows)


def build_calibration_summary(matches: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    """Build reliability buckets for favourite, draw, and underdog probabilities."""

    long = _calibration_long(matches)
    summaries: list[pd.DataFrame] = []
    for target, group in long.groupby("calibration_target", sort=False):
        sub = group.copy()
        unique_values = sub["predicted_probability"].nunique()
        q = max(1, min(buckets, unique_values, len(sub)))
        sub["probability_bucket"] = pd.qcut(
            sub["predicted_probability"],
            q=q,
            duplicates="drop",
        ).astype(str)
        summary = (
            sub.groupby("probability_bucket", sort=False)
            .agg(
                calibration_target=("calibration_target", "first"),
                count=("realised", "size"),
                predicted_probability_mean=("predicted_probability", "mean"),
                realised_frequency=("realised", "mean"),
            )
            .reset_index(drop=True)
        )
        summaries.append(summary)
    out = pd.concat(summaries, ignore_index=True)
    out["calibration_error"] = out["predicted_probability_mean"] - out["realised_frequency"]
    target_order = {"favourite": 0, "draw": 1, "underdog": 2}
    out["_order"] = out["calibration_target"].map(target_order)
    return out.sort_values(["_order", "predicted_probability_mean"]).drop(columns="_order").reset_index(drop=True)


def plot_calibration(summary: pd.DataFrame, out_path: Path) -> None:
    series = {
        "favourite": ("Favourite", "#1f77b4", "o"),
        "draw": ("Draw", "#ff7f0e", "s"),
        "underdog": ("Underdog", "#2ca02c", "^"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="Perfect calibration")
    for target, (label, color, marker) in series.items():
        sub = summary[summary["calibration_target"].eq(target)]
        if sub.empty:
            continue
        sizes = 30 + sub["count"].to_numpy(dtype=float) * 5.0
        ax.scatter(
            sub["predicted_probability_mean"],
            sub["realised_frequency"],
            s=sizes,
            color=color,
            marker=marker,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket mean)")
    ax.set_ylabel("Realised frequency")
    ax.set_title("1X2 calibration, Football-Data 2014-2022 group stages")
    ax.text(
        0.02,
        0.97,
        "Average 1X2 odds only; marker area scales with bucket size.\n"
        "Sample: 2014, 2018, 2022 group stages ($n=144$).",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="0.3",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_goals_summary(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, sub in [
        ("2014", matches[matches["year"].eq(2014)]),
        ("2018", matches[matches["year"].eq(2018)]),
        ("2022", matches[matches["year"].eq(2022)]),
        ("Combined", matches),
    ]:
        rows.append(
            {
                "sample": label,
                "matches": int(len(sub)),
                "expected_total_goals": float(sub["expected_total_goals_1x2"].mean()),
                "actual_total_goals": float(sub["actual_total_goals"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["bias"] = summary["expected_total_goals"] - summary["actual_total_goals"]
    summary["realised_minus_expected"] = summary["actual_total_goals"] - summary["expected_total_goals"]
    summary["relative_gap_vs_expected"] = summary["realised_minus_expected"] / summary["expected_total_goals"]
    return summary


def _football_data_2018_2022_summary(matches: pd.DataFrame) -> dict[str, object]:
    sub = matches[matches["year"].isin([2018, 2022])]
    return {
        "sample": "2018/2022 Football-Data 1X2-only",
        "validation_layer": "Football-Data 1X2-only",
        "matches": int(len(sub)),
        "expected_total_goals": float(sub["expected_total_goals_1x2"].mean()),
        "actual_total_goals": float(sub["actual_total_goals"].mean()),
    }


def _representative_richer_matches(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"config", "strategy", "tournament", "match_id", "matrix_expected_total_goals", "actual_total_goals"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Richer validation output is missing columns: {sorted(missing)}")
    rep = predictions[
        predictions["config"].eq("baseline_ev")
        & predictions["strategy"].eq("baseline_poisson")
        & predictions["tournament"].isin(["wc2018", "wc2022"])
    ].copy()
    if rep.empty:
        raise ValueError("No richer validation rows found for baseline_ev/baseline_poisson in wc2018/wc2022")
    return rep.drop_duplicates(["tournament", "match_id"]).reset_index(drop=True)


def build_goals_layer_comparison(matches: pd.DataFrame, richer_predictions: pd.DataFrame | None) -> pd.DataFrame:
    """Build the 2018/2022 apples-to-apples comparison across validation layers."""

    rows = [_football_data_2018_2022_summary(matches)]
    if richer_predictions is not None:
        richer = _representative_richer_matches(richer_predictions)
        rows.append(
            {
                "sample": "2018/2022 richer live-paste",
                "validation_layer": "Richer live-paste + totals",
                "matches": int(len(richer)),
                "expected_total_goals": float(richer["matrix_expected_total_goals"].mean()),
                "actual_total_goals": float(richer["actual_total_goals"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["bias"] = summary["expected_total_goals"] - summary["actual_total_goals"]
    summary["realised_minus_expected"] = summary["actual_total_goals"] - summary["expected_total_goals"]
    return summary


def build_draw_summary(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, sub in [
        ("2014", matches[matches["year"].eq(2014)]),
        ("2018", matches[matches["year"].eq(2018)]),
        ("2022", matches[matches["year"].eq(2022)]),
        ("Combined", matches),
    ]:
        predicted = float(sub["predicted_probability_draw"].mean())
        realised = float(sub["realised_outcome"].eq("draw").mean())
        rows.append(
            {
                "sample": label,
                "matches": int(len(sub)),
                "mean_predicted_draw_probability": predicted,
                "realised_draw_frequency": realised,
                "realised_minus_predicted": realised - predicted,
            }
        )
    return pd.DataFrame(rows)


def _annotate_bars(ax: plt.Axes, bars: object) -> None:
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_expected_actual_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    title: str,
    labels: list[str],
    show_ylabel: bool = True,
) -> None:
    x = np.arange(len(labels))
    width = 0.38
    bars_exp = ax.bar(
        x - width / 2,
        summary["expected_total_goals"],
        width,
        label="Expected, renormalised matrix",
        color="#1f77b4",
    )
    bars_act = ax.bar(
        x + width / 2,
        summary["actual_total_goals"],
        width,
        label="Realised",
        color="#9ecae1",
    )
    _annotate_bars(ax, bars_exp)
    _annotate_bars(ax, bars_act)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if show_ylabel:
        ax.set_ylabel("Mean total goals per match")
    ax.set_title(title)
    ax.margins(x=0.04)
    ax.grid(axis="y", color="0.9", linewidth=0.8)


def plot_expected_vs_actual_goals(
    football_data_summary: pd.DataFrame,
    layer_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=200)
    _plot_expected_actual_bars(
        axes[0],
        football_data_summary,
        title="Panel A: Football-Data 1X2-only",
        labels=football_data_summary["sample"].tolist(),
    )
    layer_labels = ["FD 1X2\n2018/2022", "Richer + totals\n2018/2022"][: len(layer_summary)]
    _plot_expected_actual_bars(
        axes[1],
        layer_summary,
        title="Panel B: same years, different inputs",
        labels=layer_labels,
        show_ylabel=False,
    )
    ymax = max(
        football_data_summary[["expected_total_goals", "actual_total_goals"]].max().max(),
        layer_summary[["expected_total_goals", "actual_total_goals"]].max().max(),
    )
    for ax in axes:
        ax.set_ylim(0, ymax * 1.20)
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_strategy_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    keep = predictions[predictions["strategy"].isin(STRATEGY_LABELS)].copy()
    return (
        keep.groupby("strategy", sort=False)
        .agg(
            matches=("match_id", "nunique"),
            total_points=("realised_points", "sum"),
            brier_1x2=("brier_score_1x2", "mean"),
            rps_1x2=("rps_1x2", "mean"),
        )
        .reset_index()
        .assign(label=lambda frame: frame["strategy"].map(STRATEGY_LABELS))
        .sort_values("total_points", ascending=False)
    )


def plot_points_vs_probability_quality(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    jitter = np.linspace(-0.0004, 0.0004, len(summary))
    x = summary["brier_1x2"].to_numpy(dtype=float) + jitter
    ax.scatter(x, summary["total_points"], s=70, color="#1f77b4", zorder=3, edgecolors="white")
    for x_value, (_, row) in zip(x, summary.iterrows()):
        ax.annotate(
            row["label"],
            (x_value, row["total_points"]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("1X2 Brier score (lower is better probability quality)")
    ax.set_ylabel("Realised pool points (144 matches)")
    ax.set_title("Realised points versus probability quality")
    ax.text(
        0.98,
        0.5,
        "Probability quality is identical across these\n"
        "scoreline-selection rules; realised points differ\n"
        "because the payoff-driven decision changes.",
        transform=ax.transAxes,
        va="center",
        ha="right",
        fontsize=8,
        color="0.3",
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate(root: Path) -> list[Path]:
    figures = _figures_dir(root)
    figures.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(_predictions_path(root))
    richer_path = _richer_predictions_path(root)
    richer_predictions = pd.read_csv(richer_path) if richer_path.exists() else None
    matches = _representative_matches(predictions)

    written: list[Path] = []

    calib_summary = build_calibration_summary(matches)
    calib_csv = figures / "calibration_1x2_summary.csv"
    calib_summary.to_csv(calib_csv, index=False)
    calib_png = figures / "calibration_1x2.png"
    plot_calibration(calib_summary, calib_png)
    written += [calib_csv, calib_png]

    goals_summary = build_goals_summary(matches)
    goals_csv = figures / "goals_summary.csv"
    goals_summary.to_csv(goals_csv, index=False)
    goals_layer_summary = build_goals_layer_comparison(matches, richer_predictions)
    goals_layer_csv = figures / "goals_layer_comparison.csv"
    goals_layer_summary.to_csv(goals_layer_csv, index=False)
    goals_png = figures / "expected_vs_actual_goals.png"
    plot_expected_vs_actual_goals(goals_summary, goals_layer_summary, goals_png)
    written += [goals_csv, goals_layer_csv, goals_png]

    draw_summary = build_draw_summary(matches)
    draw_csv = figures / "draw_summary.csv"
    draw_summary.to_csv(draw_csv, index=False)
    written.append(draw_csv)

    strategy_summary = build_strategy_summary(predictions)
    strategy_csv = figures / "strategy_probability_summary.csv"
    strategy_summary.to_csv(strategy_csv, index=False)
    strategy_png = figures / "points_vs_probability_quality.png"
    plot_points_vs_probability_quality(strategy_summary, strategy_png)
    written += [strategy_csv, strategy_png]

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".",
        help="Project root containing output/ and paper/ (default: current directory).",
    )
    args = parser.parse_args()
    written = generate(Path(args.root).resolve())
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
