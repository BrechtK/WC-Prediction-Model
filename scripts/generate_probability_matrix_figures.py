"""Generate publication heatmaps for representative exact-score matrices.

The script reads the combined historical live-backtest outputs and writes two
diagnostic figures into ``paper/figures/``:

* ``probability_matrix_extreme_favourite.png``
* ``probability_matrix_balanced.png``

It reconstructs the representative baseline Poisson matrix from the stored
``lambda_a`` and ``lambda_b`` values in
``output/research/combined_backtest/live_backtest_predictions.csv``. That is
the same transparent baseline probability backbone used by the paper's main
EV-default narrative. Model, calibration, optimisation and live recommendation
logic are not changed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from wc_predictor.optimiser import optimise_group_prediction
from wc_predictor.probabilities import ScoreProbabilityMatrix, poisson_score_matrix

REPRESENTATIVE_CONFIG = "baseline_ev"
REPRESENTATIVE_STRATEGY = "baseline_poisson"
DEFAULT_MAX_CANDIDATE_GOALS = 5


@dataclass(frozen=True)
class MatrixExample:
    """One selected match and its reconstructed baseline probability matrix."""

    label: str
    row: pd.Series
    matrix: ScoreProbabilityMatrix
    ev_score: tuple[int, int]
    modal_score: tuple[int, int]
    ev_expected_points: float


def _input_dir(root: Path) -> Path:
    return root / "output" / "research" / "combined_backtest"


def _figures_dir(root: Path) -> Path:
    return root / "paper" / "figures"


def _representative_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one row per match for the paper's representative baseline matrix."""

    rep = predictions[
        (predictions["config"] == REPRESENTATIVE_CONFIG)
        & (predictions["strategy"] == REPRESENTATIVE_STRATEGY)
    ].copy()
    required = {
        "tournament",
        "match_id",
        "team_a",
        "team_b",
        "market_a_win",
        "market_draw",
        "market_b_win",
        "favourite_probability",
        "lambda_a",
        "lambda_b",
        "grid_max_goals_used",
    }
    missing = sorted(required - set(rep.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")
    rep = rep.dropna(subset=list(required))
    rep = rep[rep["lambda_a"].astype(float).gt(0) & rep["lambda_b"].astype(float).gt(0)]
    rep = rep.sort_values(["tournament", "match_id"]).drop_duplicates(
        ["tournament", "match_id"], keep="first"
    )
    if rep.empty:
        raise ValueError("No representative baseline rows available for probability matrices")
    return rep.reset_index(drop=True)


def select_example_rows(predictions: pd.DataFrame) -> dict[str, pd.Series]:
    """Select an extreme favourite and a balanced match from historical outputs.

    Extreme favourite: highest market-implied favourite win probability.
    Balanced: smallest absolute gap between Team A and Team B win
    probabilities, with favourite probability as the deterministic tie-breaker.
    This chooses the most symmetric 1X2 profile available in the reconstructed
    baseline universe.
    """

    rep = _representative_rows(predictions)
    extreme = rep.sort_values(
        ["favourite_probability", "tournament", "match_id"],
        ascending=[False, True, True],
    ).iloc[0]

    balanced_candidates = rep.copy()
    balanced_candidates["team_a_team_b_gap"] = (
        balanced_candidates["market_a_win"].astype(float)
        - balanced_candidates["market_b_win"].astype(float)
    ).abs()
    balanced = balanced_candidates.sort_values(
        ["team_a_team_b_gap", "favourite_probability", "tournament", "match_id"],
        ascending=[True, True, True, True],
    ).iloc[0]

    return {"extreme_favourite": extreme, "balanced": balanced}


def _score_label(score: tuple[int, int]) -> str:
    return f"{score[0]}-{score[1]}"


def _tournament_label(value: str) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return f"World Cup {digits}" if digits else str(value)


def _build_example(label: str, row: pd.Series) -> MatrixExample:
    max_goals = int(row["grid_max_goals_used"])
    matrix = poisson_score_matrix(
        float(row["lambda_a"]),
        float(row["lambda_b"]),
        max_goals=max_goals,
        renormalise=True,
    )
    recommendation = optimise_group_prediction(
        matrix,
        max_candidate_goals=min(DEFAULT_MAX_CANDIDATE_GOALS, max_goals),
        top_n=5,
    )
    return MatrixExample(
        label=label,
        row=row,
        matrix=matrix,
        ev_score=recommendation.best.predicted_score,
        modal_score=recommendation.most_likely_scoreline,
        ev_expected_points=recommendation.best.expected_points,
    )


def build_examples(predictions: pd.DataFrame) -> dict[str, MatrixExample]:
    """Select and reconstruct the two example probability matrices."""

    return {
        label: _build_example(label, row)
        for label, row in select_example_rows(predictions).items()
    }


def _summary_text(example: MatrixExample) -> str:
    row = example.row
    return "\n".join(
        [
            f"P(Team A) {float(row['market_a_win']):.1%}   "
            f"P(draw) {float(row['market_draw']):.1%}   "
            f"P(Team B) {float(row['market_b_win']):.1%}",
            f"xG: {row['team_a']} {float(row['lambda_a']):.2f}, "
            f"{row['team_b']} {float(row['lambda_b']):.2f}",
            f"Modal: {_score_label(example.modal_score)}   "
            f"EV-optimal: {_score_label(example.ev_score)}",
        ]
    )


def _annotated_cells(probabilities: np.ndarray, highlights: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Return cells worth labelling without cluttering the heatmap."""

    flat = probabilities.ravel()
    top_count = min(10, flat.size)
    top_indices = np.argpartition(flat, -top_count)[-top_count:]
    cells = {
        tuple(int(v) for v in np.unravel_index(index, probabilities.shape))
        for index in top_indices
    }
    cells |= {
        tuple(int(v) for v in cell)
        for cell in zip(*np.where(probabilities >= 0.035))
    }
    cells |= highlights
    return cells


def _draw_panel(
    ax: plt.Axes,
    example: MatrixExample,
    norm: matplotlib.colors.Normalize,
    panel_label: str,
    display_max: int | None = None,
) -> matplotlib.image.AxesImage:
    """Draw one heatmap panel onto ax using a pre-built shared norm; return the image."""

    row = example.row
    probabilities = example.matrix.probabilities
    max_goals = probabilities.shape[0] - 1
    # visual crop: clip to display_max without altering probabilities
    show = min(display_max, max_goals) if display_max is not None else max_goals
    highlights = {example.modal_score, example.ev_score}

    image = ax.imshow(probabilities, origin="lower", cmap="YlGnBu", norm=norm)
    ax.set_xlim(-0.5, show + 0.5)
    ax.set_ylim(-0.5, show + 0.5)

    ax.set_xticks(np.arange(show + 1))
    ax.set_yticks(np.arange(show + 1))
    ax.set_xlabel("Team B goals", fontsize=9)
    ax.set_ylabel("Team A goals", fontsize=9)
    ax.set_title(
        f"({panel_label})  {row['team_a']} vs {row['team_b']}\n"
        f"{_tournament_label(str(row['tournament']))} ({row['match_id']})",
        fontsize=10,
        pad=8,
    )

    ax.set_xticks(np.arange(-0.5, show + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, show + 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75, alpha=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = norm.vmax * 0.50
    for team_a_goals, team_b_goals in sorted(_annotated_cells(probabilities, highlights)):
        if team_a_goals > show or team_b_goals > show:
            continue
        probability = probabilities[team_a_goals, team_b_goals]
        text_color = "white" if probability > threshold else "0.15"
        ax.text(
            team_b_goals,
            team_a_goals,
            f"{probability:.1%}",
            ha="center",
            va="center",
            fontsize=6.5,
            color=text_color,
        )

    if example.ev_score == example.modal_score:
        team_a_goals, team_b_goals = example.ev_score
        ax.add_patch(
            Rectangle(
                (team_b_goals - 0.5, team_a_goals - 0.5),
                1, 1, fill=False, edgecolor="#d62728", linewidth=2.5,
            )
        )
        legend_handles = [Patch(facecolor="none", edgecolor="#d62728", label="Modal and EV-optimal")]
    else:
        modal_team_a, modal_team_b = example.modal_score
        ev_team_a, ev_team_b = example.ev_score
        ax.add_patch(
            Rectangle(
                (modal_team_b - 0.5, modal_team_a - 0.5),
                1, 1, fill=False, edgecolor="#2ca02c", linewidth=2.2,
            )
        )
        ax.add_patch(
            Rectangle(
                (ev_team_b - 0.5, ev_team_a - 0.5),
                1, 1, fill=False, edgecolor="#d62728", linewidth=2.5,
            )
        )
        legend_handles = [
            Patch(facecolor="none", edgecolor="#2ca02c", label="Modal score"),
            Patch(facecolor="none", edgecolor="#d62728", label="EV-optimal score"),
        ]

    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=7.5)
    ax.text(
        0.02, 0.98,
        _summary_text(example),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=7.8, color="0.15",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )

    return image


def plot_combined_matrices(
    extreme: MatrixExample,
    balanced: MatrixExample,
    out_path: Path,
) -> None:
    """Plot both probability matrices side-by-side with a shared colorbar in the middle."""

    global_vmax = max(
        extreme.matrix.probabilities.max(),
        balanced.matrix.probabilities.max(),
    )
    norm = matplotlib.colors.Normalize(vmin=0, vmax=global_vmax)

    fig = plt.figure(figsize=(14.5, 6.0), dpi=220)
    gs = fig.add_gridspec(1, 3, width_ratios=[10, 0.45, 10], wspace=0.14)
    ax_left = fig.add_subplot(gs[0])
    ax_cbar = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2])

    _draw_panel(ax_left, extreme, norm, "A", display_max=8)
    image = _draw_panel(ax_right, balanced, norm, "B", display_max=5)

    cbar = fig.colorbar(image, cax=ax_cbar)
    # labels on the left so the right edge of the bar cleanly meets panel B
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label("Probability", fontsize=9, labelpad=6)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_matrix(example: MatrixExample, out_path: Path) -> None:
    """Plot one exact-score probability matrix heatmap."""

    norm = matplotlib.colors.Normalize(vmin=0, vmax=example.matrix.probabilities.max())
    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=220)
    image = _draw_panel(ax, example, norm, "")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    # remove the panel-label prefix from the title for the standalone version
    current_title = ax.get_title()
    ax.set_title(current_title.split("  ", 1)[-1], fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_summary(examples: dict[str, MatrixExample]) -> pd.DataFrame:
    """Build the compact summary CSV for the selected examples."""

    rows = []
    for label, example in examples.items():
        row = example.row
        team_a_win = float(row["market_a_win"])
        team_b_win = float(row["market_b_win"])
        favourite_side = "team_a" if team_a_win >= team_b_win else "team_b"
        underdog_probability = min(team_a_win, team_b_win)
        rows.append(
            {
                "example": label,
                "tournament": row["tournament"],
                "match_id": row["match_id"],
                "fixture": f"{row['team_a']} vs {row['team_b']}",
                "favourite_team": row["team_a"] if favourite_side == "team_a" else row["team_b"],
                "favourite_probability": max(team_a_win, team_b_win),
                "draw_probability": float(row["market_draw"]),
                "underdog_probability": underdog_probability,
                "team_a_expected_goals": float(row["lambda_a"]),
                "team_b_expected_goals": float(row["lambda_b"]),
                "most_likely_scoreline": _score_label(example.modal_score),
                "ev_optimal_scoreline": _score_label(example.ev_score),
                "ev_expected_points": example.ev_expected_points,
                "grid_max_goals": int(row["grid_max_goals_used"]),
                "matrix_source": "baseline_ev/baseline_poisson independent Poisson",
            }
        )
    return pd.DataFrame(rows)


def generate(root: Path) -> list[Path]:
    """Build probability matrix figures and summary table; return written paths."""

    src = _input_dir(root)
    figures = _figures_dir(root)
    figures.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(src / "live_backtest_predictions.csv")
    examples = build_examples(predictions)

    combined_path = figures / "probability_matrix_combined.png"
    plot_combined_matrices(examples["extreme_favourite"], examples["balanced"], combined_path)

    # also keep the individual files for reference
    individual_outputs = {
        "extreme_favourite": figures / "probability_matrix_extreme_favourite.png",
        "balanced": figures / "probability_matrix_balanced.png",
    }
    for label, out_path in individual_outputs.items():
        plot_matrix(examples[label], out_path)

    summary_path = figures / "probability_matrix_examples.csv"
    build_summary(examples).to_csv(summary_path, index=False)

    return [combined_path, *individual_outputs.values(), summary_path]


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
