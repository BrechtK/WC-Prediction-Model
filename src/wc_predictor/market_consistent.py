"""KL-regularised market-consistent score-matrix challenger."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc_predictor.asian_totals import asian_total_profit_vector
from wc_predictor.probabilities import ScoreProbabilityMatrix


@dataclass(frozen=True)
class MarketConsistentWeights:
    """Reliability weights for soft market-consistency constraints."""

    one_x_two: float = 250.0
    liquid_total_goals: float = 140.0
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


def _fair_decimal_odds(probability: float) -> float:
    probability = float(probability)
    if not np.isfinite(probability) or probability <= 0 or probability >= 1:
        raise ValueError("Fair probability must lie strictly between zero and one")
    return 1.0 / probability


def _build_constraints(
    prior: ScoreProbabilityMatrix,
    market_probabilities: Mapping[str, float],
    total_goals: pd.DataFrame | None,
    correct_score_matrix: ScoreProbabilityMatrix | None,
    weights: MarketConsistentWeights,
) -> tuple[list[_Constraint], str]:
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
            constraints.append(
                _Constraint(
                    f"total_over_{line:g}",
                    "total_goals",
                    asian_total_profit_vector(shape, line, "over", _fair_decimal_odds(fair_over)).reshape(-1),
                    0.0,
                    line_weight,
                )
            )
            constraints.append(
                _Constraint(
                    f"total_under_{line:g}",
                    "total_goals",
                    asian_total_profit_vector(shape, line, "under", _fair_decimal_odds(fair_under)).reshape(-1),
                    0.0,
                    line_weight,
                )
            )
            used_totals.append(f"{line:g} {line_kind}")

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

    return constraints, "; ".join(used_totals)


def fit_market_consistent_matrix(
    prior: ScoreProbabilityMatrix,
    market_probabilities: Mapping[str, float],
    total_goals: pd.DataFrame | None = None,
    correct_score_matrix: ScoreProbabilityMatrix | None = None,
    weights: MarketConsistentWeights | None = None,
    max_iterations: int = 120,
) -> MarketConsistentMatrixResult:
    """Fit a positive score matrix close to the prior and market constraints."""

    weights = weights or MarketConsistentWeights()
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    q = np.asarray(prior.probabilities, dtype=float).reshape(-1)
    q = np.clip(q, 1e-15, None)
    q = q / q.sum()
    constraints, totals_used = _build_constraints(prior, market_probabilities, total_goals, correct_score_matrix, weights)
    if not constraints:
        return MarketConsistentMatrixResult(
            ScoreProbabilityMatrix(
                q.reshape(prior.probabilities.shape),
                tail_probability=prior.tail_probability,
                lambda_a=prior.lambda_a,
                lambda_b=prior.lambda_b,
            ),
            {
                "market_consistent_optimisation_success": True,
                "market_consistent_kl_divergence_vs_prior": 0.0,
                "market_consistent_1x2_fit_error": pd.NA,
                "market_consistent_btts_fit_error": pd.NA,
                "market_consistent_total_goals_fit_error": pd.NA,
                "market_consistent_correct_score_fit_error": pd.NA,
                "market_consistent_asian_totals_used": totals_used,
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
        options={"maxiter": max_iterations, "ftol": 1e-10},
    )
    p = _softmax(np.asarray(result.x, dtype=float))
    probabilities = p.reshape(prior.probabilities.shape)
    posterior = ScoreProbabilityMatrix(
        probabilities,
        tail_probability=prior.tail_probability,
        lambda_a=prior.lambda_a,
        lambda_b=prior.lambda_b,
    )
    diagnostics = {
        "market_consistent_optimisation_success": bool(result.success),
        "market_consistent_optimisation_message": str(result.message),
        "market_consistent_kl_divergence_vs_prior": float(np.dot(p, np.log(p / q))),
        "market_consistent_1x2_fit_error": _constraint_rmse(p, constraints, "1x2"),
        "market_consistent_btts_fit_error": _constraint_rmse(p, constraints, "btts"),
        "market_consistent_total_goals_fit_error": _constraint_rmse(p, constraints, "total_goals"),
        "market_consistent_correct_score_fit_error": _constraint_rmse(p, constraints, "correct_score"),
        "market_consistent_asian_totals_used": totals_used,
    }
    return MarketConsistentMatrixResult(posterior, diagnostics)
