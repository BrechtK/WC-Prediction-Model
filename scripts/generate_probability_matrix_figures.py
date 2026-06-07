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
    Balanced: smallest absolute gap between home/team-A and away/team-B win
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
    balanced_candidates["home_away_gap"] = (
        balanced_candidates["market_a_win"].astype(float)
        - balanced_candidates["market_b_win"].astype(float)
    ).abs()
    balanced = balanced_candidates.sort_values(
        ["home_away_gap", "favourite_probability", "tournament", "match_id"],
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
            f"P(home) {float(row['market_a_win']):.1%}   "
            f"P(draw) {float(row['market_draw']):.1%}   "
            f"P(away) {float(row['market_b_win']):.1%}",
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


def plot_matrix(example: MatrixExample, out_path: Path) -> None:
    """Plot one exact-score probability matrix heatmap."""

    row = example.row
    probabilities = example.matrix.probabilities
    max_goals = probabilities.shape[0] - 1
    highlights = {example.modal_score, example.ev_score}

    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=220)
    image = ax.imshow(probabilities, origin="lower", cmap="YlGnBu", vmin=0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.set_xticks(np.arange(max_goals + 1))
    ax.set_yticks(np.arange(max_goals + 1))
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_title(
        f"{row['team_a']} vs {row['team_b']} - {_tournament_label(str(row['tournament']))} "
        f"({row['match_id']})",
        fontsize=12,
        pad=12,
    )

    ax.set_xticks(np.arange(-0.5, max_goals + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, max_goals + 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75, alpha=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)

    for home_goals, away_goals in sorted(_annotated_cells(probabilities, highlights)):
        probability = probabilities[home_goals, away_goals]
        text_color = "white" if probability > probabilities.max() * 0.50 else "0.15"
        ax.text(
            away_goals,
            home_goals,
            f"{probability:.1%}",
            ha="center",
            va="center",
            fontsize=6.7,
            color=text_color,
        )

    if example.ev_score == example.modal_score:
        home_goals, away_goals = example.ev_score
        ax.add_patch(
            Rectangle(
                (away_goals - 0.5, home_goals - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#d62728",
                linewidth=2.5,
            )
        )
        legend_handles = [Patch(facecolor="none", edgecolor="#d62728", label="Modal and EV-optimal")]
    else:
        modal_home, modal_away = example.modal_score
        ev_home, ev_away = example.ev_score
        ax.add_patch(
            Rectangle(
                (modal_away - 0.5, modal_home - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#2ca02c",
                linewidth=2.2,
            )
        )
        ax.add_patch(
            Rectangle(
                (ev_away - 0.5, ev_home - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#d62728",
                linewidth=2.5,
            )
        )
        legend_handles = [
            Patch(facecolor="none", edgecolor="#2ca02c", label="Modal score"),
            Patch(facecolor="none", edgecolor="#d62728", label="EV-optimal score"),
        ]

    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)
    ax.text(
        0.02,
        0.98,
        _summary_text(example),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="0.15",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_summary(examples: dict[str, MatrixExample]) -> pd.DataFrame:
    """Build the compact summary CSV for the selected examples."""

    rows = []
    for label, example in examples.items():
        row = example.row
        home_win = float(row["market_a_win"])
        away_win = float(row["market_b_win"])
        favourite_side = "home" if home_win >= away_win else "away"
        underdog_probability = min(home_win, away_win)
        rows.append(
            {
                "example": label,
                "tournament": row["tournament"],
                "match_id": row["match_id"],
                "fixture": f"{row['team_a']} vs {row['team_b']}",
                "favourite_team": row["team_a"] if favourite_side == "home" else row["team_b"],
                "favourite_probability": max(home_win, away_win),
                "draw_probability": float(row["market_draw"]),
                "underdog_probability": underdog_probability,
                "home_expected_goals": float(row["lambda_a"]),
                "away_expected_goals": float(row["lambda_b"]),
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

    outputs = {
        "extreme_favourite": figures / "probability_matrix_extreme_favourite.png",
        "balanced": figures / "probability_matrix_balanced.png",
    }
    for label, out_path in outputs.items():
        plot_matrix(examples[label], out_path)

    summary_path = figures / "probability_matrix_examples.csv"
    build_summary(examples).to_csv(summary_path, index=False)

    return [outputs["extreme_favourite"], outputs["balanced"], summary_path]


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
