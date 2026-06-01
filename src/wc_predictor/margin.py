"""Bookmaker margin-removal methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class MarginRemovalResult:
    """Fair probabilities plus diagnostics from margin removal."""

    raw_probabilities: np.ndarray
    fair_probabilities: np.ndarray
    overround: float
    method: str
    warnings: tuple[str, ...] = ()


class MarginRemover(Protocol):
    """Interface implemented by all margin-removal methods."""

    name: str

    def remove(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """Convert positive raw implied probabilities into probabilities summing to one."""


class ProportionalMarginRemover:
    """Remove margin by normalising each implied probability by the overround."""

    name = "proportional"

    def remove(self, raw_probabilities: np.ndarray) -> np.ndarray:
        return raw_probabilities / raw_probabilities.sum()


class AdditiveMarginRemover:
    """Remove equal probability mass from each outcome."""

    name = "additive"

    def remove(self, raw_probabilities: np.ndarray) -> np.ndarray:
        corrected = raw_probabilities - (raw_probabilities.sum() - 1.0) / len(raw_probabilities)
        if np.any(corrected <= 0):
            raise ValueError("Additive margin removal produced a non-positive probability")
        return corrected / corrected.sum()


class PowerMarginRemover:
    """Find a power k such that sum(q_i ** k) equals one."""

    name = "power"

    def remove(self, raw_probabilities: np.ndarray) -> np.ndarray:
        objective = lambda exponent: float(np.sum(raw_probabilities**exponent) - 1.0)
        exponent = brentq(objective, 0.01, 100.0)
        corrected = raw_probabilities**exponent
        return corrected / corrected.sum()


class ShinMarginRemover:
    """Placeholder for Shin's insider-trading margin-removal model."""

    name = "shin"

    def remove(self, raw_probabilities: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Shin margin removal is reserved for a later release")


def get_margin_remover(method: str) -> MarginRemover:
    """Return a configured margin remover."""

    methods: dict[str, MarginRemover] = {
        "proportional": ProportionalMarginRemover(),
        "additive": AdditiveMarginRemover(),
        "power": PowerMarginRemover(),
        "shin": ShinMarginRemover(),
    }
    try:
        return methods[method.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown margin-removal method: {method}") from exc


def remove_margin(
    raw_probabilities: Sequence[float],
    method: str = "proportional",
    suspicious_low: float = 1.0,
    suspicious_high: float = 1.20,
) -> MarginRemovalResult:
    """Remove bookmaker margin from a complete market and report diagnostics."""

    raw = np.asarray(raw_probabilities, dtype=float)
    if raw.ndim != 1 or len(raw) < 2:
        raise ValueError("A market must contain at least two outcomes")
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError("Raw implied probabilities must be finite and positive")

    overround = float(raw.sum())
    warnings: list[str] = []
    if overround < suspicious_low:
        warnings.append(f"Overround {overround:.4f} is below {suspicious_low:.4f}")
    if overround > suspicious_high:
        warnings.append(f"Overround {overround:.4f} exceeds {suspicious_high:.4f}")

    remover = get_margin_remover(method)
    fair = remover.remove(raw)
    if np.any(fair <= 0) or not np.isclose(fair.sum(), 1.0, atol=1e-9):
        raise ValueError("Margin removal must produce positive probabilities summing to one")
    return MarginRemovalResult(raw, fair, overround, remover.name, tuple(warnings))

