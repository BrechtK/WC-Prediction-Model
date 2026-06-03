"""Central paths for the live paste-to-prediction workflow."""

from __future__ import annotations

from pathlib import Path

INPUT_DIR = Path("input")
INPUT_ODDS_DIR = INPUT_DIR / "odds"
INPUT_SCHEDULE_PATH = INPUT_DIR / "schedule.txt"
INPUT_PREPARED_DIR = INPUT_DIR / "prepared"
INPUT_PREPARED_WORLD_CUP_ODDS_XLSX_PATH = INPUT_PREPARED_DIR / "world_cup_odds.xlsx"
INPUT_PREPARED_WORLD_CUP_ODDS_CSV_PATH = INPUT_PREPARED_DIR / "world_cup_odds.csv"
INPUT_HISTORICAL_DIR = INPUT_DIR / "historical"

OUTPUT_DIR = Path("output")
OUTPUT_PARSE_REPORTS_DIR = OUTPUT_DIR / "parse_reports"
OUTPUT_PREDICTIONS_CSV_PATH = OUTPUT_DIR / "predictions.csv"
OUTPUT_PREDICTIONS_XLSX_PATH = OUTPUT_DIR / "predictions.xlsx"
OUTPUT_SUBMISSION_XLSX_PATH = OUTPUT_DIR / "submission_sheet.xlsx"
OUTPUT_WEIGHT_SENSITIVITY_XLSX_PATH = OUTPUT_DIR / "correct_score_weight_sensitivity.xlsx"
OUTPUT_DIXON_COLES_SENSITIVITY_XLSX_PATH = OUTPUT_DIR / "dixon_coles_rho_comparison.xlsx"
OUTPUT_CORRECT_SCORE_AGGREGATION_XLSX_PATH = OUTPUT_DIR / "correct_score_aggregation_comparison.xlsx"
OUTPUT_SCHEDULE_PARSE_REPORT_PATH = OUTPUT_PARSE_REPORTS_DIR / "schedule_parse_report.csv"
OUTPUT_CORE_PARSE_REPORT_PATH = OUTPUT_PARSE_REPORTS_DIR / "odds_parse_report.csv"
OUTPUT_CORRECT_SCORE_PARSE_REPORT_PATH = OUTPUT_PARSE_REPORTS_DIR / "correct_score_parse_report.csv"
OUTPUT_RESEARCH_DIR = OUTPUT_DIR / "research"

CACHE_DIR = Path("cache")
CACHE_PARSED_DIR = CACHE_DIR / "parsed"
CACHE_SPLIT_PASTES_DIR = CACHE_DIR / "split_pastes"
CACHE_CORE_ODDS_PATH = CACHE_PARSED_DIR / "core_odds.csv"
CACHE_TOTAL_GOALS_ODDS_PATH = CACHE_PARSED_DIR / "total_goals_odds.csv"
CACHE_CORRECT_SCORE_ODDS_PATH = CACHE_PARSED_DIR / "correct_score_odds.csv"
CACHE_SCHEDULE_PATH = CACHE_PARSED_DIR / "schedule.csv"
