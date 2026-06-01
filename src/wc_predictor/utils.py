"""Small shared helpers."""

from __future__ import annotations

from pathlib import Path


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

