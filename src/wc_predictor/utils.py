"""Small shared helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

FAVOURITE_STRENGTH_BUCKETS = (
    "balanced",
    "slight_favourite",
    "clear_favourite",
    "strong_favourite",
    "huge_favourite",
    "extreme_favourite",
)


def result_sign(goals_a: int, goals_b: int) -> int:
    """Return 1 for an A win, 0 for a draw, and -1 for a B win."""

    return (goals_a > goals_b) - (goals_a < goals_b)


def goal_difference(goals_a: int, goals_b: int) -> int:
    """Return goals scored by A minus goals scored by B."""

    return goals_a - goals_b


def is_knockout_stage(stage: str) -> bool:
    """Return whether a stage label describes a knockout match."""

    normalised = str(stage).strip().lower().replace("_", " ").replace("-", " ")
    return normalised not in {"group", "group stage", "groups"}


def ensure_parent_directory(path: Path) -> None:
    """Create the parent directory for an output file when required."""

    path.parent.mkdir(parents=True, exist_ok=True)


def favourite_strength_bucket(favourite_probability: float) -> str:
    """Assign a post-margin home-or-away favourite probability to a diagnostic bucket."""

    if not np.isfinite(favourite_probability) or not 0 <= favourite_probability <= 1:
        raise ValueError("Favourite probability must lie between zero and one")
    if favourite_probability < 0.45:
        return "balanced"
    if favourite_probability < 0.55:
        return "slight_favourite"
    if favourite_probability < 0.65:
        return "clear_favourite"
    if favourite_probability < 0.75:
        return "strong_favourite"
    if favourite_probability < 0.85:
        return "huge_favourite"
    return "extreme_favourite"
