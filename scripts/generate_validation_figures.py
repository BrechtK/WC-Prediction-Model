"""Generate public-facing validation figures and summary tables.

This script reads the combined probabilistic backtest outputs in
``output/research/combined_backtest/`` and produces a small set of clean
figures and CSV summaries for the GitHub README and the LaTeX paper.

It is read-only with respect to the model: it consumes existing backtest CSVs
and writes only into ``paper/figures/``. No model logic, live recommendation,
or backtest output is modified.

Figures produced
----------------
* ``calibration_1x2.png`` - reliability of favourite/draw/underdog probabilities.
* ``expected_vs_actual_goals.png`` - expected versus realised total goals.
* ``points_vs_probability_quality.png`` - realised pool points versus 1X2 Brier.

Summary tables produced
-----------------------
* ``calibration_1x2_summary.csv``
* ``goals_summary.csv``
* ``strategy_probability_summary.csv``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, reproducible rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The probability vectors are identical across the point-equivalent strategies,
# so a single representative config/strategy stands in for the probability
# backbone in the per-match and calibration views.
REPRESENTATIVE_CONFIG = "baseline_ev"
REPRESENTATIVE_STRATEGY = "baseline_poisson"

# Friendly labels for the strategy/probability-quality figure.
STRATEGY_LABELS = {
    ("baseline_ev", "ev_default"): "EV default",
    ("baseline_ev", "most_likely"): "Modal / most-likely",
    ("mc_with_ah", "market_consistent"): "MC+AH",
    ("mc_with_ah", "dixon_coles"): "Dixon-Coles",
}


def _input_dir(root: Path) -> Path:
    return root / "output" / "research" / "combined_backtest"


def _figures_dir(root: Path) -> Path:
    return root / "paper" / "figures"


def _weighted_mean(frame: pd.DataFrame, value: str, weight: str = "count") -> float:
    return float(np.average(frame[value], weights=frame[weight]))


def build_calibration_summary(calibration_1x2: pd.DataFrame) -> pd.DataFrame:
    """Pool the 1X2 calibration buckets across tournaments per outcome class."""

    rep = calibration_1x2[
        (calibration_1x2["config"] == REPRESENTATIVE_CONFIG)
        & (calibration_1x2["strategy"] == REPRESENTATIVE_STRATEGY)
    ]
    rows = []
    for (target, bucket), group in rep.groupby(["calibration_target", "probability_bucket"]):
        count = int(group["count"].sum())
        rows.append(
            {
                "calibration_target": target,
                "probability_bucket": bucket,
                "count": count,
                "predicted_probability_mean": _weighted_mean(group, "predicted_probability_mean"),
                "realised_frequency": _weighted_mean(group, "realised_frequency"),
            }
        )
    summary = pd.DataFrame(rows)
    summary["calibration_error"] = (
        summary["predicted_probability_mean"] - summary["realised_frequency"]
    )
    target_order = {"favourite": 0, "draw": 1, "underdog": 2}
    summary["_order"] = summary["calibration_target"].map(target_order).fillna(99)
    summary = summary.sort_values(["_order", "predicted_probability_mean"]).drop(columns="_order")
    return summary.reset_index(drop=True)


def plot_calibration(summary: pd.DataFrame, out_path: Path) -> None:
    """Reliability scatter of predicted probability versus realised frequency."""

    series = {
        "favourite": ("Favourite", "#1f77b4", "o"),
        "draw": ("Draw", "#ff7f0e", "s"),
        "underdog": ("Underdog", "#2ca02c", "^"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="Perfect calibration")

    for target, (label, color, marker) in series.items():
        sub = summary[summary["calibration_target"] == target]
        if sub.empty:
            continue
        # Marker area scaled by bucket sample size so thin buckets read as small.
        sizes = 25 + sub["count"].to_numpy(dtype=float) * 6.0
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
    ax.set_title("1X2 calibration, 2018 and 2022 group stages")
    ax.text(
        0.02,
        0.97,
        "Marker area scales with bucket sample size.\n"
        "Predicted probabilities are broadly aligned with realised\n"
        "frequencies, but sample size is limited ($n=96$).",
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


def build_goals_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Expected versus realised total goals by tournament and combined."""

    rep = predictions[
        (predictions["config"] == REPRESENTATIVE_CONFIG)
        & (predictions["strategy"] == REPRESENTATIVE_STRATEGY)
    ]
    rows = []
    samples = [("2018", "wc2018"), ("2022", "wc2022"), ("Combined", None)]
    for label, key in samples:
        sub = rep if key is None else rep[rep["tournament"] == key]
        rows.append(
            {
                "sample": label,
                "matches": int(len(sub)),
                "expected_total_goals": float(sub["expected_total_goals"].mean()),
                "actual_total_goals": float(sub["actual_total_goals"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["bias"] = summary["expected_total_goals"] - summary["actual_total_goals"]
    return summary


def plot_expected_vs_actual_goals(summary: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of expected versus realised total goals per sample."""

    labels = summary["sample"].tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=200)
    bars_exp = ax.bar(
        x - width / 2, summary["expected_total_goals"], width, label="Expected", color="#1f77b4"
    )
    bars_act = ax.bar(
        x + width / 2, summary["actual_total_goals"], width, label="Realised", color="#9ecae1"
    )

    for bars in (bars_exp, bars_act):
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

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean total goals per match")
    ax.set_ylim(0, max(summary[["expected_total_goals", "actual_total_goals"]].max()) * 1.18)
    ax.set_title("Expected versus realised total goals")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_strategy_summary(
    predictions: pd.DataFrame, live_summary: pd.DataFrame
) -> pd.DataFrame:
    """Realised pool points against 1X2 probability quality per strategy."""

    prob = (
        predictions.groupby(["config", "strategy"])
        .agg(
            matches=("brier_score_1x2", "size"),
            brier_1x2=("brier_score_1x2", "mean"),
            rps_1x2=("rps_1x2", "mean"),
        )
        .reset_index()
    )
    points = live_summary[["config", "strategy", "average_realised_points", "total_points"]]
    merged = prob.merge(points, on=["config", "strategy"], how="left")
    merged["label"] = [
        STRATEGY_LABELS.get((c, s), f"{c}/{s}")
        for c, s in zip(merged["config"], merged["strategy"])
    ]
    merged = merged.sort_values("total_points", ascending=False).reset_index(drop=True)
    return merged


def plot_points_vs_probability_quality(summary: pd.DataFrame, out_path: Path) -> None:
    """Scatter of realised points versus Brier for the labelled strategies."""

    # Keep the curated, interpretable strategies for the public figure.
    keep = summary[summary["label"].isin(STRATEGY_LABELS.values())].copy()
    keep = keep.sort_values("total_points", ascending=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    ax.scatter(
        keep["brier_1x2"], keep["total_points"], s=70, color="#1f77b4", zorder=3, edgecolors="white"
    )
    for _, row in keep.iterrows():
        ax.annotate(
            row["label"],
            (row["brier_1x2"], row["total_points"]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    span = keep["brier_1x2"].max() - keep["brier_1x2"].min()
    pad = span * 4 if span > 0 else 0.001
    ax.set_xlim(keep["brier_1x2"].min() - pad, keep["brier_1x2"].max() + pad * 2.5)
    ax.set_xlabel("1X2 Brier score (lower is better probability quality)")
    ax.set_ylabel("Realised pool points (96 matches)")
    ax.set_title("Realised points versus probability quality by strategy")
    ax.text(
        0.98,
        0.5,
        "Brier differences are tiny (4th decimal);\n"
        "points differences come mainly from the\n"
        "scoring-rule decision layer, not from\n"
        "materially different probability estimates.",
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
    """Build all figures and summary tables; return the written paths."""

    src = _input_dir(root)
    figures = _figures_dir(root)
    figures.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(src / "live_backtest_predictions.csv")
    calibration_1x2 = pd.read_csv(src / "calibration_1x2.csv")
    live_summary = pd.read_csv(src / "live_backtest_summary.csv")

    written: list[Path] = []

    # Figure 1 + summary: 1X2 calibration.
    calib_summary = build_calibration_summary(calibration_1x2)
    calib_csv = figures / "calibration_1x2_summary.csv"
    calib_summary.to_csv(calib_csv, index=False)
    calib_png = figures / "calibration_1x2.png"
    plot_calibration(calib_summary, calib_png)
    written += [calib_csv, calib_png]

    # Figure 2 + summary: expected versus realised goals.
    goals_summary = build_goals_summary(predictions)
    goals_csv = figures / "goals_summary.csv"
    goals_summary.to_csv(goals_csv, index=False)
    goals_png = figures / "expected_vs_actual_goals.png"
    plot_expected_vs_actual_goals(goals_summary, goals_png)
    written += [goals_csv, goals_png]

    # Figure 3 + summary: points versus probability quality.
    strategy_summary = build_strategy_summary(predictions, live_summary)
    strat_csv = figures / "strategy_probability_summary.csv"
    # Document strategies that were points-tested but absent from this run.
    note = (
        "Power devig and correct-score blend were points-tested in the paper but "
        "are absent from the probabilistic backtest outputs (margin method was "
        "normalised_inverse_odds throughout and correct-score blend weight was 1.0), "
        "so they have no probability-quality row here."
    )
    with strat_csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {note}\n")
        strategy_summary.to_csv(handle, index=False)
    strat_png = figures / "points_vs_probability_quality.png"
    plot_points_vs_probability_quality(strategy_summary, strat_png)
    written += [strat_csv, strat_png]

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
