"""CSV and Excel input loaders for odds, friend predictions, and results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ODDS_REQUIRED_COLUMNS = {"match_id", "date", "stage", "team_a", "team_b", "bookmaker", "odds_a_win", "odds_draw", "odds_b_win"}
PREDICTION_REQUIRED_COLUMNS = {"match_id", "player", "predicted_team_a_goals", "predicted_team_b_goals"}
RESULT_REQUIRED_COLUMNS = {"match_id", "team_a_goals_90", "team_b_goals_90"}


def load_tabular_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel worksheet based on its suffix."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported tabular file type: {path.suffix}")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> pd.DataFrame:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} data is missing required columns: {sorted(missing)}")
    return frame


def load_odds(path: str | Path) -> pd.DataFrame:
    """Load bookmaker odds, preserving any optional markets that are present."""

    return _require_columns(load_tabular_data(path), ODDS_REQUIRED_COLUMNS, "Odds")


def load_correct_score_odds(path: str | Path) -> pd.DataFrame:
    """Load optional long-format correct-score odds for direct-market blending."""

    required = {"match_id", "bookmaker", "score_a", "score_b", "decimal_odds"}
    return _require_columns(load_tabular_data(path), required, "Correct-score odds")


def load_total_goals_odds(path: str | Path) -> pd.DataFrame:
    """Load optional long-format bookmaker totals ladders."""

    required = {"match_id", "bookmaker", "line", "odds_over", "odds_under"}
    return _require_columns(load_tabular_data(path), required, "Total-goals odds")


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load friends' predictions from CSV or Excel."""

    predictions = _require_columns(load_tabular_data(path), PREDICTION_REQUIRED_COLUMNS, "Prediction")
    if "predicted_qualifier" not in predictions:
        predictions["predicted_qualifier"] = pd.NA
    return predictions


def load_results(path: str | Path) -> pd.DataFrame:
    """Load realised results from CSV or Excel."""

    return _require_columns(load_tabular_data(path), RESULT_REQUIRED_COLUMNS, "Result")
