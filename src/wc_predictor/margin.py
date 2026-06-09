"""Bookmaker margin-removal methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from scipy.optimize import brentq

IMPLEMENTED_MARGIN_REMOVAL_METHODS = ("normalised_inverse_odds", "power", "additive", "shin")


@dataclass(frozen=True)
class MarginRemovalResult:
    """Fair probabilities plus diagnostics from margin removal."""

    raw_probabilities: np.ndarray
    fair_probabilities: np.ndarray
    overround: float
    method: str
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, float | str | bool] | None = None


class MarginRemover(Protocol):
    """Interface implemented by all margin-removal methods."""

    name: str

    def remove(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """Convert positive raw implied probabilities into probabilities summing to one."""

    def diagnostics(self) -> dict[str, float | str | bool]:
        """Return method-specific diagnostics from the most recent removal."""

        return {}


class NormalisedInverseOddsMarginRemover:
    """Remove margin by normalising each implied probability by the overround."""

    name = "normalised_inverse_odds"

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
    """Remove margin using Shin's insider-trading model."""

    name = "shin"

    def __init__(self) -> None:
        self._shin_z: float | None = None

    def remove(self, raw_probabilities: np.ndarray) -> np.ndarray:
        overround = float(raw_probabilities.sum())
        if len(raw_probabilities) < 2:
            raise ValueError("Shin margin removal requires at least two outcomes")
        if overround <= 1.0:
            raise ValueError("Shin margin removal requires an overround above one")
        if overround > 1.75:
            raise ValueError("Shin margin removal rejected unusually high overround")

        def probabilities_for_z(z: float) -> np.ndarray:
            denominator = 2.0 * (1.0 - z)
            if denominator <= 0:
                raise ValueError("Invalid Shin z denominator")
            adjusted = (
                np.sqrt(z * z + 4.0 * (1.0 - z) * (raw_probabilities * raw_probabilities) / overround)
                - z
            ) / denominator
            return adjusted

        def objective(z: float) -> float:
            return float(probabilities_for_z(z).sum() - 1.0)

        lower = 0.0
        upper = 1.0 - 1e-12
        lower_value = objective(lower)
        upper_value = objective(upper)
        if lower_value < 0 or upper_value > 0:
            raise ValueError("Shin margin removal could not bracket a valid z solution")
        z = brentq(objective, lower, upper, xtol=1e-12, rtol=1e-12, maxiter=100)
        fair = probabilities_for_z(z)
        if np.any(fair <= 0) or not np.all(np.isfinite(fair)):
            raise ValueError("Shin margin removal produced invalid probabilities")
        self._shin_z = float(z)
        return fair / fair.sum()

    def diagnostics(self) -> dict[str, float | str | bool]:
        return {"shin_z": self._shin_z} if self._shin_z is not None else {}


# TODO: Add odds-ratio margin removal only when numerical safeguards are clear.
def get_margin_remover(method: str) -> MarginRemover:
    """Return a configured margin remover."""

    normalised = NormalisedInverseOddsMarginRemover()
    methods: dict[str, MarginRemover] = {
        "normalised_inverse_odds": normalised,
        "proportional": normalised,
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
    diagnostics_method = getattr(remover, "diagnostics", None)
    diagnostics = diagnostics_method() if diagnostics_method is not None else {}
    return MarginRemovalResult(raw, fair, overround, remover.name, tuple(warnings), diagnostics)
