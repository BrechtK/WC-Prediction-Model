"""Extract the richer-layer validation numbers reported in the paper/README.

Reads a combined live-backtest predictions CSV (the richer OddsPortal layer) and
prints the realised-points challenger table, the modal-vs-EV divergence table,
the expected-vs-actual total-goals comparison, and the MC+AH optimiser fit
breakdown. The strategy/config selection mirrors how the paper's richer
diagnostic table is constructed, so running it on the 2018/2022-only output
reproduces the previously reported totals, and running it on the combined
2014/2018/2022 output produces the updated richer-144 numbers.

Usage:
    python scripts/extract_richer_validation_numbers.py \
        --predictions output/research/combined_backtest/live_backtest_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# (config, strategy) selection for each reported challenger row.
CHALLENGER_ROWS = {
    "EV-optimal default": ("baseline_ev", "ev_default"),
    "Modal / most-likely": ("baseline_ev", "most_likely"),
    "Market-consistent + AH": ("mc_with_ah", "market_consistent"),
    "Power devig": ("power_devig_ev", "ev_default"),
    "Dixon-Coles": ("baseline_ev", "dixon_coles"),
    "Larger grid": ("larger_grid_15", "ev_default"),
}
CS_BLEND_CONFIGS = ["cs_weight_0_5", "cs_weight_0_75", "cs_weight_0_85"]


def _points(df: pd.DataFrame, config: str, strategy: str) -> int:
    sub = df[(df["config"] == config) & (df["strategy"] == strategy)]
    return int(sub["realised_points"].sum())


def challenger_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, (config, strategy) in CHALLENGER_ROWS.items():
        rows.append({"row": label, "points": _points(df, config, strategy)})
    # Correct-score blend sweep: best of the tested weights (ev_default rec).
    cs = {c: _points(df, c, "ev_default") for c in CS_BLEND_CONFIGS}
    best_cs = max(cs.values())
    rows.append({"row": f"Correct-score blend sweep (best of {cs})", "points": best_cs})
    return pd.DataFrame(rows)


def modal_ev_divergence(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["config"] == "baseline_ev"].drop_duplicates(["tournament", "match_id"]).copy()
    base["differs"] = base["modal_score"].astype(str) != base["ev_score"].astype(str)
    rows = []
    for tourn, sub in list(base.groupby("tournament")) + [("ALL", base)]:
        n = len(sub)
        diff = int(sub["differs"].sum())
        rows.append(
            {
                "sample": tourn,
                "matches": n,
                "modal_eq_ev": n - diff,
                "modal_ne_ev": diff,
                "divergence_pct": round(100 * diff / n, 1) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def goals_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rep = (
        df[(df["config"] == "baseline_ev") & (df["strategy"] == "baseline_poisson")]
        .drop_duplicates(["tournament", "match_id"])
        .copy()
    )
    col = "matrix_expected_total_goals" if "matrix_expected_total_goals" in rep.columns else "expected_total_goals"
    rep["actual_total"] = rep["actual_score"].apply(
        lambda s: sum(int(x) for x in str(s).replace(":", "-").split("-")[:2])
    ) if "actual_total_goals" not in rep.columns else rep.get("actual_total_goals")
    if "actual_total_goals" in rep.columns:
        rep["actual_total"] = rep["actual_total_goals"]
    rows = []
    for tourn, sub in list(rep.groupby("tournament")) + [("ALL", rep)]:
        rows.append(
            {
                "sample": tourn,
                "matches": len(sub),
                "expected_total_goals": round(float(sub[col].mean()), 3),
                "actual_total_goals": round(float(sub["actual_total"].mean()), 3),
            }
        )
    out = pd.DataFrame(rows)
    out["realised_minus_expected"] = (out["actual_total_goals"] - out["expected_total_goals"]).round(3)
    return out


ACCEPTABLE_FIT = "optimisation_not_fully_converged_but_fit_acceptable"


def gated_mc_ah_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Realised points of the *implemented* gated policy.

    Use the MC+AH (market-consistent) prediction only on matches whose optimiser
    fit status is acceptable; otherwise fall back to the EV-optimal default. This
    is what the live governance actually does, as opposed to the raw MC+AH
    diagnostic that uses MC+AH on every match.
    """

    cls = "market_consistent_optimisation_classification"
    cls = cls if cls in df.columns else "mc_status"
    ev = (
        df[(df["config"] == "baseline_ev") & (df["strategy"] == "ev_default")]
        .drop_duplicates(["tournament", "match_id"])[["tournament", "match_id", "realised_points"]]
        .rename(columns={"realised_points": "ev_pts"})
    )
    mc = (
        df[(df["config"] == "mc_with_ah") & (df["strategy"] == "market_consistent")]
        .drop_duplicates(["tournament", "match_id"])[["tournament", "match_id", "realised_points", cls]]
        .rename(columns={"realised_points": "mc_pts"})
    )
    m = ev.merge(mc, on=["tournament", "match_id"], how="left")
    m["acceptable"] = m[cls].eq(ACCEPTABLE_FIT)
    m["gated_pts"] = m.apply(
        lambda r: r["mc_pts"] if (r["acceptable"] and pd.notna(r["mc_pts"])) else r["ev_pts"],
        axis=1,
    )
    rows = []
    for tourn, sub in list(m.groupby("tournament")) + [("ALL", m)]:
        rows.append(
            {
                "sample": tourn,
                "matches": len(sub),
                "accepted": int(sub["acceptable"].sum()),
                "ev_default": int(sub["ev_pts"].sum()),
                "raw_mc_ah": int(sub["mc_pts"].sum()),
                "gated_policy": int(sub["gated_pts"].sum()),
                "gated_minus_ev": int(sub["gated_pts"].sum() - sub["ev_pts"].sum()),
            }
        )
    return pd.DataFrame(rows)


def mc_fit_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    mc = df[(df["config"] == "mc_with_ah") & (df["strategy"] == "market_consistent")].drop_duplicates(
        ["tournament", "match_id"]
    )
    col = "market_consistent_optimisation_classification"
    if col not in mc.columns:
        col = "mc_status"
    return mc[col].value_counts().rename_axis("classification").reset_index(name="matches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="output/research/combined_backtest/live_backtest_predictions.csv",
    )
    args = parser.parse_args()
    df = pd.read_csv(Path(args.predictions))
    n_matches = df.groupby("tournament")["match_id"].nunique().to_dict()
    total = sum(n_matches.values())
    print(f"\nSource: {args.predictions}")
    print(f"Matches per tournament: {n_matches}  (total {total})\n")

    print("=== Richer challenger realised-points table ===")
    print(challenger_table(df).to_string(index=False))
    print("\n=== Modal-vs-EV divergence ===")
    print(modal_ev_divergence(df).to_string(index=False))
    print("\n=== Expected vs actual total goals (baseline_ev / baseline_poisson) ===")
    print(goals_comparison(df).to_string(index=False))
    print("\n=== Gated MC+AH policy (implemented rule: MC+AH only when fit acceptable, else EV) ===")
    print(gated_mc_ah_policy(df).to_string(index=False))
    print("\n=== MC+AH optimiser fit classification ===")
    print(mc_fit_breakdown(df).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
