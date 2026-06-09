"""KL-regularised market-consistent score-matrix challenger."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc_predictor.asian_handicap import asian_handicap_profit_vector
from wc_predictor.asian_totals import asian_total_profit_vector
from wc_predictor.probabilities import ScoreProbabilityMatrix

ACCEPTABLE_MAX_CONSTRAINT_ERROR = 0.20
ACCEPTABLE_GROUP_FIT_ERROR = 0.05
POOR_MAX_CONSTRAINT_ERROR = 0.20
DEFAULT_MAX_ASIAN_HANDICAP_CONSTRAINTS = 7
ASIAN_HANDICAP_HARD_MIN_PROBABILITY = 0.10
ASIAN_HANDICAP_HARD_MAX_PROBABILITY = 0.90
ASIAN_HANDICAP_DEFAULT_MIN_PROBABILITY = 0.25
ASIAN_HANDICAP_DEFAULT_MAX_PROBABILITY = 0.75
ASIAN_HANDICAP_EXTREME_MIN_PROBABILITY = 0.15
ASIAN_HANDICAP_EXTREME_MAX_PROBABILITY = 0.85
EXTREME_FAVOURITE_AH_SELECTION_THRESHOLD = 0.75
GRID_BOUNDARY_MARGIN_BUFFER = 1.0


@dataclass(frozen=True)
class MarketConsistentWeights:
    """Reliability weights for soft market-consistency constraints."""

    one_x_two: float = 250.0
    liquid_total_goals: float = 140.0
    asian_handicap: float = 120.0
    btts: float = 80.0
    correct_score: float = 18.0
    max_correct_score_cells: int = 12


@dataclass(frozen=True)
class MarketConsistentMatrixResult:
    """Adjusted score matrix plus fit diagnostics."""

    matrix: ScoreProbabilityMatrix
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _Constraint:
    name: str
    group: str
    vector: np.ndarray
    target: float
    weight: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def _mask(shape: tuple[int, int], predicate: np.ndarray) -> np.ndarray:
    values = np.zeros(shape, dtype=float)
    values[predicate] = 1.0
    return values.reshape(-1)


def _constraint_rmse(probabilities: np.ndarray, constraints: list[_Constraint], group: str) -> float | object:
    values = [
        (float(np.dot(probabilities, constraint.vector)) - constraint.target) ** 2
        for constraint in constraints
        if constraint.group == group
    ]
    return float(np.sqrt(np.mean(values))) if values else pd.NA


def _constraint_errors(probabilities: np.ndarray, constraints: list[_Constraint]) -> list[float]:
    return [float(np.dot(probabilities, constraint.vector)) - constraint.target for constraint in constraints]


def _constraint_counts_by_group(constraints: list[_Constraint]) -> dict[str, int]:
    counts = {"1x2": 0, "btts": 0, "total_goals": 0, "asian_handicap": 0, "correct_score": 0}
    for constraint in constraints:
        counts[constraint.group] = counts.get(constraint.group, 0) + 1
    return counts


def _fit_is_acceptable(diagnostics: Mapping[str, object]) -> bool:
    max_error = diagnostics.get("market_consistent_max_constraint_error")
    if pd.isna(max_error) or float(max_error) > ACCEPTABLE_MAX_CONSTRAINT_ERROR:
        return False
    for key in (
        "market_consistent_1x2_fit_error",
        "market_consistent_btts_fit_error",
        "market_consistent_total_goals_fit_error",
        "market_consistent_asian_handicap_fit_error",
        "market_consistent_correct_score_fit_error",
    ):
        value = diagnostics.get(key, pd.NA)
        if pd.notna(value) and float(value) > ACCEPTABLE_GROUP_FIT_ERROR:
            return False
    return True


def _status_from_diagnostics(diagnostics: Mapping[str, object]) -> str:
    success = bool(diagnostics.get("market_consistent_optimisation_success", False))
    status_code = diagnostics.get("market_consistent_optimisation_status_code", pd.NA)
    status_ok = pd.notna(status_code) and int(status_code) == 0
    acceptable = _fit_is_acceptable(diagnostics)
    max_error = diagnostics.get("market_consistent_max_constraint_error", pd.NA)
    if success and status_ok and acceptable:
        return "ok"
    if success and status_ok and pd.notna(max_error) and float(max_error) > POOR_MAX_CONSTRAINT_ERROR:
        return "poor_fit"
    if success and status_ok:
        return "ok"
    if acceptable:
        return "not_fully_converged_fit_acceptable"
    return "failed_severe"


def _total_goals_weight(row: pd.Series, base_weight: float) -> float:
    fair_over = float(row["fair_over"])
    line_kind = str(row.get("line_kind", ""))
    near_money_factor = 1.0 if 0.40 <= fair_over <= 0.60 else 0.70
    kind_factor = 1.0 if line_kind == "half_goal" else 0.90
    overround = row.get("average_total_goals_overround", np.nan)
    overround_factor = 1.0
    if pd.notna(overround) and np.isfinite(float(overround)):
        overround_factor = min(1.0, 1.20 / max(float(overround), 1.0))
    return base_weight * near_money_factor * kind_factor * overround_factor


def _asian_handicap_weight(row: pd.Series, base_weight: float) -> float:
    fair_team_a = float(row["fair_team_a"])
    near_money_probability = min(fair_team_a, 1.0 - fair_team_a)
    near_money_factor = 1.0 if 0.40 <= fair_team_a <= 0.60 else max(0.35, near_money_probability / 0.40)
    kind_factor = 1.0 if str(row.get("line_kind", "")) == "half_goal" else 0.90
    bookmaker_factor = min(1.25, max(0.50, float(row.get("bookmakers_count", 1)) / 3.0))
    overround = row.get("average_asian_handicap_overround", np.nan)
    overround_factor = 1.0
    if pd.notna(overround) and np.isfinite(float(overround)):
        overround_factor = min(1.0, 1.20 / max(float(overround), 1.0))
    warning_factor = 0.75 if str(row.get("warnings", "")).strip() else 1.0
    return base_weight * near_money_factor * kind_factor * bookmaker_factor * overround_factor * warning_factor


def _asian_handicap_grid_boundary_diagnostics(handicap: float, shape: tuple[int, int]) -> tuple[bool, str]:
    threshold = -float(handicap)
    min_margin = -(shape[1] - 1)
    max_margin = shape[0] - 1
    if threshold >= max_margin - GRID_BOUNDARY_MARGIN_BUFFER:
        return True, (
            f"cover_threshold_near_upper_grid_boundary:{threshold:g}; "
            f"represented_margin_max={max_margin:g}"
        )
    if threshold <= min_margin + GRID_BOUNDARY_MARGIN_BUFFER:
        return True, (
            f"loss_threshold_near_lower_grid_boundary:{threshold:g}; "
            f"represented_margin_min={min_margin:g}"
        )
    return False, ""


def select_asian_handicap_constraints(
    asian_handicap: pd.DataFrame | None,
    shape: tuple[int, int],
    market_probabilities: Mapping[str, float] | None = None,
    *,
    max_lines: int = DEFAULT_MAX_ASIAN_HANDICAP_CONSTRAINTS,
) -> pd.DataFrame:
    """Annotate AH lines and select a stable subset for market-consistent fitting."""

    columns = [
        "selected_for_market_consistent",
        "selection_weight",
        "selection_reason",
        "skipped_reason",
        "handicap_line_near_grid_boundary",
        "skipped_due_to_grid_boundary",
    ]
    if asian_handicap is None or asian_handicap.empty:
        return pd.DataFrame(columns=[*(asian_handicap.columns if asian_handicap is not None else []), *columns])
    if max_lines <= 0:
        raise ValueError("max_lines must be positive")

    selected = asian_handicap.copy()
    for column in columns:
        selected[column] = "no" if column.startswith("selected") or column.startswith("skipped") or column == "handicap_line_near_grid_boundary" else ""
    selected["selection_weight"] = 0.0

    favourite_probability = 0.0
    if market_probabilities is not None:
        favourite_probability = max(
            float(market_probabilities.get("a_win", 0.0)),
            float(market_probabilities.get("b_win", 0.0)),
        )
    extreme_favourite = favourite_probability >= EXTREME_FAVOURITE_AH_SELECTION_THRESHOLD
    soft_min = ASIAN_HANDICAP_EXTREME_MIN_PROBABILITY if extreme_favourite else ASIAN_HANDICAP_DEFAULT_MIN_PROBABILITY
    soft_max = ASIAN_HANDICAP_EXTREME_MAX_PROBABILITY if extreme_favourite else ASIAN_HANDICAP_DEFAULT_MAX_PROBABILITY

    eligible_indexes: list[int] = []
    for index, row in selected.iterrows():
        reasons: list[str] = []
        skipped: list[str] = []
        line_kind = str(row.get("line_kind", ""))
        if line_kind not in {"half_goal", "integer_asian", "quarter_asian"}:
            skipped.append(f"unsupported_line_kind:{line_kind or 'missing'}")
        try:
            fair_team_a = float(row["fair_team_a"])
            fair_team_b = float(row["fair_team_b"])
            handicap = float(row["handicap"])
        except (KeyError, TypeError, ValueError):
            fair_team_a = np.nan
            fair_team_b = np.nan
            handicap = np.nan
            skipped.append("invalid_handicap_row")
        if not (np.isfinite(fair_team_a) and np.isfinite(fair_team_b) and 0 < fair_team_a < 1 and 0 < fair_team_b < 1):
            skipped.append("invalid_fair_probabilities")
        elif fair_team_a < ASIAN_HANDICAP_HARD_MIN_PROBABILITY or fair_team_a > ASIAN_HANDICAP_HARD_MAX_PROBABILITY:
            skipped.append(f"outside_hard_probability_range:{fair_team_a:.3f}")
        elif fair_team_a < soft_min or fair_team_a > soft_max:
            skipped.append(f"outside_stable_probability_range:{fair_team_a:.3f}")
        if np.isfinite(handicap):
            near_boundary, boundary_reason = _asian_handicap_grid_boundary_diagnostics(handicap, shape)
            selected.loc[index, "handicap_line_near_grid_boundary"] = "yes" if near_boundary else "no"
            selected.loc[index, "skipped_due_to_grid_boundary"] = "yes" if near_boundary else "no"
            if near_boundary:
                skipped.append(boundary_reason)
        weight = 0.0
        if not skipped:
            raw_weight = _asian_handicap_weight(row, 1.0)
            near_money_bonus = max(0.0, 1.0 - abs(fair_team_a - 0.5) / 0.5)
            boundary_buffer = min(
                float(shape[0] - 1) - (-handicap),
                (-handicap) - float(-(shape[1] - 1)),
            )
            boundary_factor = min(1.0, max(0.25, boundary_buffer / 3.0))
            weight = raw_weight * (0.50 + near_money_bonus) * boundary_factor
            reasons.append(
                f"eligible:p={fair_team_a:.3f};bookmakers={int(row.get('bookmakers_count', 0))};"
                f"overround={float(row.get('average_asian_handicap_overround', np.nan)):.3f}"
            )
            eligible_indexes.append(index)
        selected.loc[index, "selection_weight"] = float(weight)
        selected.loc[index, "selection_reason"] = "; ".join(reasons)
        selected.loc[index, "skipped_reason"] = "; ".join(skipped)

    ranked = selected.loc[eligible_indexes].sort_values(
        ["selection_weight", "bookmakers_count", "handicap"],
        ascending=[False, False, True],
        kind="stable",
    )
    chosen_indexes = set(ranked.head(max_lines).index)
    for index in eligible_indexes:
        if index in chosen_indexes:
            selected.loc[index, "selected_for_market_consistent"] = "yes"
        else:
            selected.loc[index, "skipped_reason"] = "lower_ranked_than_selected_constraint_limit"
    return selected


def _fair_decimal_odds(probability: float) -> float:
    probability = float(probability)
    if not np.isfinite(probability) or probability <= 0 or probability >= 1:
        raise ValueError("Fair probability must lie strictly between zero and one")
    return 1.0 / probability


def _build_constraints(
    prior: ScoreProbabilityMatrix,
    market_probabilities: Mapping[str, float],
    total_goals: pd.DataFrame | None,
    asian_handicap: pd.DataFrame | None,
    correct_score_matrix: ScoreProbabilityMatrix | MarketConsistentWeights | None = None,
    weights: MarketConsistentWeights | None = None,
) -> tuple[list[_Constraint], str, str]:
    legacy_return = False
    if isinstance(correct_score_matrix, MarketConsistentWeights) and weights is None:
        legacy_return = True
        weights = correct_score_matrix
        correct_score_matrix = None
    weights = weights or MarketConsistentWeights()
    shape = prior.probabilities.shape
    scores_a, scores_b = np.indices(shape)
    constraints: list[_Constraint] = []
    if weights.one_x_two > 0:
        constraints.extend(
            [
                _Constraint("1x2_a_win", "1x2", _mask(shape, scores_a > scores_b), float(market_probabilities["a_win"]), weights.one_x_two),
                _Constraint("1x2_draw", "1x2", _mask(shape, scores_a == scores_b), float(market_probabilities["draw"]), weights.one_x_two),
                _Constraint("1x2_b_win", "1x2", _mask(shape, scores_a < scores_b), float(market_probabilities["b_win"]), weights.one_x_two),
            ]
        )
    btts_yes = market_probabilities.get("btts_yes")
    if weights.btts > 0 and btts_yes is not None and pd.notna(btts_yes):
        constraints.append(
            _Constraint("btts_yes", "btts", _mask(shape, (scores_a > 0) & (scores_b > 0)), float(btts_yes), weights.btts)
        )

    used_totals: list[str] = []
    if total_goals is not None and not total_goals.empty and weights.liquid_total_goals > 0:
        for _, row in total_goals.iterrows():
            line_kind = str(row.get("line_kind", ""))
            if line_kind not in {"half_goal", "integer_asian", "quarter_asian"}:
                continue
            fair_over = float(row["fair_over"])
            fair_under = float(row["fair_under"])
            if not (0 < fair_over < 1 and 0 < fair_under < 1):
                continue
            line = float(row["line"])
            line_weight = _total_goals_weight(row, weights.liquid_total_goals)
            # After two-way margin removal, fair Under is the complement of fair
            # Over for half, integer, and quarter Asian totals. The Under
            # expected-profit vector is therefore a scalar multiple of the Over
            # vector, including push and half-push cases. Keep one independent
            # constraint per line so a totals market is not double-weighted.
            constraints.append(
                _Constraint(
                    f"total_over_{line:g}",
                    "total_goals",
                    asian_total_profit_vector(shape, line, "over", _fair_decimal_odds(fair_over)).reshape(-1),
                    0.0,
                    line_weight,
                )
            )
            used_totals.append(f"{line:g} {line_kind}")

    used_handicaps: list[str] = []
    if asian_handicap is not None and not asian_handicap.empty and weights.asian_handicap > 0:
        handicap_rows = asian_handicap
        if "selected_for_market_consistent" in handicap_rows:
            handicap_rows = handicap_rows[
                handicap_rows["selected_for_market_consistent"].astype(str).str.lower().eq("yes")
            ]
        for _, row in handicap_rows.iterrows():
            line_kind = str(row.get("line_kind", ""))
            if line_kind not in {"half_goal", "integer_asian", "quarter_asian"}:
                continue
            fair_team_a = float(row["fair_team_a"])
            fair_team_b = float(row["fair_team_b"])
            if not (0 < fair_team_a < 1 and 0 < fair_team_b < 1):
                continue
            handicap = float(row["handicap"])
            constraints.append(
                _Constraint(
                    f"asian_handicap_team_a_{handicap:g}",
                    "asian_handicap",
                    asian_handicap_profit_vector(
                        shape,
                        handicap,
                        "team_a",
                        _fair_decimal_odds(fair_team_a),
                    ).reshape(-1),
                    0.0,
                    _asian_handicap_weight(row, weights.asian_handicap),
                )
            )
            used_handicaps.append(f"{handicap:g} {line_kind}")

    if correct_score_matrix is not None and weights.correct_score > 0:
        if correct_score_matrix.probabilities.shape != shape:
            raise ValueError("Correct-score market matrix must match the prior matrix shape")
        market = correct_score_matrix.probabilities
        ranked = sorted(
            (
                ((int(score_a), int(score_b)), float(market[score_a, score_b]))
                for score_a, score_b in np.ndindex(shape)
                if market[score_a, score_b] > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        for (score_a, score_b), probability in ranked[: weights.max_correct_score_cells]:
            vector = np.zeros(shape, dtype=float)
            vector[score_a, score_b] = 1.0
            constraints.append(
                _Constraint(
                    f"correct_score_{score_a}_{score_b}",
                    "correct_score",
                    vector.reshape(-1),
                    probability,
                    weights.correct_score,
                )
            )

    if legacy_return:
        return constraints, "; ".join(used_totals)  # type: ignore[return-value]
    return constraints, "; ".join(used_totals), "; ".join(used_handicaps)


def _margin_distribution_summary(probabilities: np.ndarray, limit: int = 7) -> str:
    shape = probabilities.shape
    scores_a, scores_b = np.indices(shape)
    margins = scores_a - scores_b
    values: list[tuple[int, float]] = []
    for margin in range(-shape[1] + 1, shape[0]):
        values.append((margin, float(probabilities[margins == margin].sum())))
    ranked = sorted(values, key=lambda item: (-item[1], item[0]))[:limit]
    return "; ".join(f"{margin:+d}:{probability:.4f}" for margin, probability in ranked)


def _margin_distribution(probabilities: np.ndarray) -> dict[int, float]:
    scores_a, scores_b = np.indices(probabilities.shape)
    margins = scores_a - scores_b
    return {
        margin: float(probabilities[margins == margin].sum())
        for margin in range(-probabilities.shape[1] + 1, probabilities.shape[0])
    }


def _largest_margin_shift(before: np.ndarray, after: np.ndarray) -> tuple[int, float]:
    before_distribution = _margin_distribution(before)
    after_distribution = _margin_distribution(after)
    margins = sorted(set(before_distribution) | set(after_distribution))
    if not margins:
        return 0, 0.0
    margin = max(margins, key=lambda item: abs(after_distribution.get(item, 0.0) - before_distribution.get(item, 0.0)))
    return margin, after_distribution.get(margin, 0.0) - before_distribution.get(margin, 0.0)


def _asian_handicap_fit_rmse(
    probabilities: np.ndarray,
    asian_handicap: pd.DataFrame | None,
    *,
    selected_only: bool,
) -> float | object:
    if asian_handicap is None or asian_handicap.empty:
        return pd.NA
    rows = asian_handicap
    if selected_only and "selected_for_market_consistent" in rows:
        rows = rows[rows["selected_for_market_consistent"].astype(str).str.lower().eq("yes")]
    values: list[float] = []
    for _, row in rows.iterrows():
        if str(row.get("line_kind", "")) not in {"half_goal", "integer_asian", "quarter_asian"}:
            continue
        fair_team_a = float(row["fair_team_a"])
        if not 0 < fair_team_a < 1:
            continue
        vector = asian_handicap_profit_vector(
            probabilities.shape,
            float(row["handicap"]),
            "team_a",
            _fair_decimal_odds(fair_team_a),
        )
        values.append(float((probabilities * vector).sum()) ** 2)
    return float(np.sqrt(np.mean(values))) if values else pd.NA


def fit_market_consistent_matrix(
    prior: ScoreProbabilityMatrix,
    market_probabilities: Mapping[str, float],
    total_goals: pd.DataFrame | None = None,
    asian_handicap: pd.DataFrame | None = None,
    correct_score_matrix: ScoreProbabilityMatrix | None = None,
    weights: MarketConsistentWeights | None = None,
    max_iterations: int = 500,
) -> MarketConsistentMatrixResult:
    """Fit a positive score matrix close to the prior and market constraints."""

    weights = weights or MarketConsistentWeights()
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    q = np.asarray(prior.probabilities, dtype=float).reshape(-1)
    q = np.clip(q, 1e-15, None)
    q = q / q.sum()
    prior_probabilities = q.reshape(prior.probabilities.shape)
    selected_asian_handicap = select_asian_handicap_constraints(
        asian_handicap,
        prior.probabilities.shape,
        market_probabilities,
    )
    constraints, totals_used, handicaps_used = _build_constraints(
        prior,
        market_probabilities,
        total_goals,
        selected_asian_handicap,
        correct_score_matrix,
        weights,
    )
    constraint_counts = _constraint_counts_by_group(constraints)
    ah_available = 0 if asian_handicap is None else len(asian_handicap)
    ah_selected = int(
        selected_asian_handicap["selected_for_market_consistent"].astype(str).str.lower().eq("yes").sum()
    ) if not selected_asian_handicap.empty and "selected_for_market_consistent" in selected_asian_handicap else 0
    ah_skipped = max(0, ah_available - ah_selected)
    ah_skipped_lines = (
        "; ".join(
            f"{float(row['handicap']):+g}:{row.get('skipped_reason', '')}"
            for _, row in selected_asian_handicap.iterrows()
            if str(row.get("selected_for_market_consistent", "no")).lower() != "yes"
        )
        if not selected_asian_handicap.empty and "handicap" in selected_asian_handicap
        else ""
    )
    if not constraints:
        return MarketConsistentMatrixResult(
            ScoreProbabilityMatrix(
                q.reshape(prior.probabilities.shape),
                tail_probability=prior.tail_probability,
                lambda_a=prior.lambda_a,
                lambda_b=prior.lambda_b,
            ),
            {
                "market_consistent_status": "ok",
                "market_consistent_optimisation_classification": "success",
                "market_consistent_optimisation_success": True,
                "market_consistent_optimisation_status_code": 0,
                "market_consistent_optimisation_message": "no active constraints",
                "market_consistent_optimisation_iterations": 0,
                "market_consistent_final_objective_value": 0.0,
                "market_consistent_gradient_norm": 0.0,
                "market_consistent_max_constraint_error": pd.NA,
                "market_consistent_constraint_count": 0,
                "market_consistent_1x2_constraint_count": 0,
                "market_consistent_btts_constraint_count": 0,
                "market_consistent_total_goals_constraint_count": 0,
                "market_consistent_asian_handicap_constraint_count": 0,
                "market_consistent_correct_score_constraint_count": 0,
                "market_consistent_kl_divergence_vs_prior": 0.0,
                "market_consistent_1x2_fit_error": pd.NA,
                "market_consistent_btts_fit_error": pd.NA,
                "market_consistent_total_goals_fit_error": pd.NA,
                "market_consistent_asian_handicap_fit_error": pd.NA,
                "market_consistent_correct_score_fit_error": pd.NA,
                "market_consistent_asian_totals_used": totals_used,
                "market_consistent_asian_handicap_lines_available": ah_available,
                "market_consistent_asian_handicap_lines_selected": ah_selected,
                "market_consistent_asian_handicap_lines_skipped": ah_skipped,
                "market_consistent_asian_handicap_lines_used": handicaps_used,
                "market_consistent_asian_handicap_lines_skipped_detail": ah_skipped_lines,
                "market_consistent_asian_handicap_fit_error_selected": pd.NA,
                "market_consistent_asian_handicap_fit_error_all": _asian_handicap_fit_rmse(prior_probabilities, selected_asian_handicap, selected_only=False),
                "market_consistent_margin_distribution_before": _margin_distribution_summary(prior_probabilities),
                "market_consistent_margin_distribution_after": _margin_distribution_summary(prior_probabilities),
                "market_consistent_largest_margin_shift": pd.NA,
                "market_consistent_largest_margin_shift_value": 0.0,
            },
        )

    x0 = np.log(q)

    def objective_and_gradient(logits: np.ndarray) -> tuple[float, np.ndarray]:
        p = _softmax(logits)
        kl_gradient = np.log(p / q) + 1.0
        value = float(np.dot(p, np.log(p / q)))
        grad_p = kl_gradient.copy()
        for constraint in constraints:
            error = float(np.dot(p, constraint.vector)) - constraint.target
            value += constraint.weight * error**2
            grad_p += 2.0 * constraint.weight * error * constraint.vector
        grad_logits = p * (grad_p - float(np.dot(p, grad_p)))
        return value, grad_logits

    result = minimize(
        lambda logits: objective_and_gradient(logits),
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iterations, "ftol": 1e-9, "gtol": 1e-6, "maxls": 50},
    )
    p = _softmax(np.asarray(result.x, dtype=float))
    final_objective, final_gradient = objective_and_gradient(np.asarray(result.x, dtype=float))
    constraint_errors = _constraint_errors(p, constraints)
    max_constraint_error = max((abs(error) for error in constraint_errors), default=np.nan)
    probabilities = p.reshape(prior.probabilities.shape)
    posterior = ScoreProbabilityMatrix(
        probabilities,
        tail_probability=prior.tail_probability,
        lambda_a=prior.lambda_a,
        lambda_b=prior.lambda_b,
    )
    diagnostics = {
        "market_consistent_optimisation_success": bool(result.success),
        "market_consistent_optimisation_status_code": int(result.status),
        "market_consistent_optimisation_message": str(result.message),
        "market_consistent_optimisation_iterations": int(getattr(result, "nit", 0)),
        "market_consistent_final_objective_value": float(final_objective),
        "market_consistent_gradient_norm": float(np.linalg.norm(final_gradient, ord=np.inf)),
        "market_consistent_max_constraint_error": float(max_constraint_error),
        "market_consistent_constraint_count": len(constraints),
        "market_consistent_1x2_constraint_count": constraint_counts.get("1x2", 0),
        "market_consistent_btts_constraint_count": constraint_counts.get("btts", 0),
        "market_consistent_total_goals_constraint_count": constraint_counts.get("total_goals", 0),
        "market_consistent_asian_handicap_constraint_count": constraint_counts.get("asian_handicap", 0),
        "market_consistent_correct_score_constraint_count": constraint_counts.get("correct_score", 0),
        "market_consistent_kl_divergence_vs_prior": float(np.dot(p, np.log(p / q))),
        "market_consistent_1x2_fit_error": _constraint_rmse(p, constraints, "1x2"),
        "market_consistent_btts_fit_error": _constraint_rmse(p, constraints, "btts"),
        "market_consistent_total_goals_fit_error": _constraint_rmse(p, constraints, "total_goals"),
        "market_consistent_asian_handicap_fit_error": _constraint_rmse(p, constraints, "asian_handicap"),
        "market_consistent_correct_score_fit_error": _constraint_rmse(p, constraints, "correct_score"),
        "market_consistent_asian_totals_used": totals_used,
        "market_consistent_asian_handicap_lines_available": ah_available,
        "market_consistent_asian_handicap_lines_selected": ah_selected,
        "market_consistent_asian_handicap_lines_skipped": ah_skipped,
        "market_consistent_asian_handicap_lines_used": handicaps_used,
        "market_consistent_asian_handicap_lines_skipped_detail": ah_skipped_lines,
        "market_consistent_asian_handicap_fit_error_selected": _asian_handicap_fit_rmse(probabilities, selected_asian_handicap, selected_only=True),
        "market_consistent_asian_handicap_fit_error_all": _asian_handicap_fit_rmse(probabilities, selected_asian_handicap, selected_only=False),
        "market_consistent_margin_distribution_before": _margin_distribution_summary(prior_probabilities),
        "market_consistent_margin_distribution_after": _margin_distribution_summary(probabilities),
    }
    largest_shift_margin, largest_shift_value = _largest_margin_shift(prior_probabilities, probabilities)
    diagnostics["market_consistent_largest_margin_shift"] = int(largest_shift_margin)
    diagnostics["market_consistent_largest_margin_shift_value"] = float(largest_shift_value)
    status = _status_from_diagnostics(diagnostics)
    diagnostics["market_consistent_status"] = status
    diagnostics["market_consistent_optimisation_classification"] = (
        "success"
        if status == "ok"
        else "optimisation_not_fully_converged_but_fit_acceptable"
        if status == "not_fully_converged_fit_acceptable"
        else "constraint_fit_poor"
        if status == "poor_fit"
        else "optimisation_failed_severe"
    )
    return MarketConsistentMatrixResult(posterior, diagnostics)
