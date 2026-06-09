"""Paired-bootstrap significance test for the modal-versus-EV realised-points gap.

This is read-only with respect to the model. It consumes
``output/research/world_cup_backtest_predictions.csv`` (the Football-Data
1X2-only extension), pairs the per-match realised points of
``most_likely_poisson`` and ``ev_optimal_1x2`` by ``match_id``, and reports a
paired bootstrap confidence interval on the modal-minus-EV difference. These are
the numbers quoted next to Table~\ref{tab:results} in the paper: the observed
607-vs-594 gap is statistically indistinguishable from zero.

The seed is fixed so the reported interval is reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODAL_STRATEGY = "most_likely_poisson"
EV_STRATEGY = "ev_optimal_1x2"
DEFAULT_SEED = 20260608
DEFAULT_RESAMPLES = 100_000


def _predictions_path(root: Path) -> Path:
    return root / "output" / "research" / "world_cup_backtest_predictions.csv"


def _paired_points(predictions: pd.DataFrame, strategy: str) -> pd.Series:
    rows = predictions[predictions["strategy"] == strategy]
    return rows.set_index("match_id")["realised_points"].astype(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        default="output/research/modal_ev_bootstrap_summary.csv",
        help="Where to write the one-row bootstrap summary CSV.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    predictions = pd.read_csv(_predictions_path(root))

    modal = _paired_points(predictions, MODAL_STRATEGY)
    ev = _paired_points(predictions, EV_STRATEGY)
    match_ids = modal.index.intersection(ev.index)
    diff = (modal.loc[match_ids] - ev.loc[match_ids]).to_numpy()
    n = diff.size

    rng = np.random.default_rng(args.seed)
    sample = diff[rng.integers(0, n, size=(args.resamples, n))]
    boot_mean = sample.mean(axis=1)
    boot_total = sample.sum(axis=1)

    mean_lo, mean_hi = np.percentile(boot_mean, [2.5, 97.5])
    total_lo, total_hi = np.percentile(boot_total, [2.5, 97.5])
    prob_modal_not_ahead = float((boot_total <= 0).mean())

    summary = {
        "n_matches": n,
        "n_points_differ": int((diff != 0).sum()),
        "observed_total_diff": float(diff.sum()),
        "observed_mean_diff": float(diff.mean()),
        "ci95_mean_low": float(mean_lo),
        "ci95_mean_high": float(mean_hi),
        "ci95_total_low": float(total_lo),
        "ci95_total_high": float(total_hi),
        "prob_modal_not_ahead": prob_modal_not_ahead,
        "resamples": args.resamples,
        "seed": args.seed,
    }

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_path, index=False)

    print("Modal vs EV paired bootstrap (most_likely_poisson - ev_optimal_1x2)")
    print(f"- paired matches: {n}; points differ in {summary['n_points_differ']}")
    print(
        f"- observed gap: {summary['observed_total_diff']:.0f} points total, "
        f"{summary['observed_mean_diff']:.3f} per match"
    )
    print(f"- 95% CI mean diff/match: [{mean_lo:.3f}, {mean_hi:.3f}]")
    print(f"- 95% CI total diff over {n}: [{total_lo:.1f}, {total_hi:.1f}]")
    print(f"- P(modal not ahead): {prob_modal_not_ahead:.3f}")
    print(f"- summary written to {output_path}")


if __name__ == "__main__":
    main()
