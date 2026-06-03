"""Run the complete live OddsPortal paste-to-submission workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import pandas as pd

from wc_predictor.config import ProjectConfig, PublicStrategyConfig
from wc_predictor.live_prediction import (
    LivePredictionError,
    LivePredictionSettings,
    format_live_prediction_summary,
    run_live_prediction,
)
from wc_predictor.oddsportal_combined import (
    CombinedOddsPortalPasteError,
    format_combined_oddsportal_split_summary,
    split_combined_oddsportal_pastes,
)
from wc_predictor.oddsportal_schedule import (
    DEFAULT_SCHEDULE_OUTPUT_PATH,
    DEFAULT_SCHEDULE_REPORT_PATH,
    OddsPortalScheduleParseError,
    format_oddsportal_schedule_parse_summary,
    prepare_schedule_metadata,
)
from wc_predictor.paths import (
    CACHE_SPLIT_PASTES_DIR,
    INPUT_ODDS_DIR,
    INPUT_SCHEDULE_PATH,
)

# ============================================================
# USER SETTINGS
# ============================================================

RUN_MODE = "all_available"
# Options:
# "all_available"  -> process all odds files in input/odds/
# "date"           -> process every match on DATE
# "single_match"   -> process one selected match
# "list_date"      -> list all games on DATE and exit

DATE = "13-6"
GAME_NUMBER = 3
MATCH_ID = None

STRATEGY_MODE = "ev"
# Options:
# "ev"
# "balanced"
# "public-ranking"
# "aggressive-public-ranking"

CORRECT_SCORE_POISSON_WEIGHT = 0.85
CORRECT_SCORE_AGGREGATION_METHOD = "auto"

MARGIN_REMOVAL_METHOD = "normalised_inverse_odds"

RUN_PROFILE = "live"
TERMINAL_VERBOSITY = "compact"
ENABLE_MARKET_CONSISTENT_CHALLENGER = "only_if_close"
ENABLE_MARGIN_METHOD_COMPARISON = False
ENABLE_CORRECT_SCORE_WEIGHT_SENSITIVITY = False

WRITE_DETAILED_EXCEL = True
WRITE_CACHE_OUTPUTS = True
SHOW_RUNTIME_SUMMARY = True

DIXON_COLES_RHO = 0.0

STRICT_INPUT_VALIDATION = False

NO_ODDS_INPUT_MESSAGE = (
    "No odds input files found. Create files like input/odds/M001.txt with sections "
    "### 1X2, ### OVER_UNDER, ### BTTS, ### CORRECT_SCORE."
)
VALID_RUN_MODES = {"all_available", "date", "single_match", "list_date"}
VALID_STRATEGY_MODES = ("ev", "balanced", "public-ranking", "aggressive-public-ranking")
VALID_MARGIN_REMOVAL_METHODS = ("normalised_inverse_odds", "power", "additive")
VALID_RUN_PROFILES = ("live", "research")
VALID_TERMINAL_VERBOSITIES = ("compact", "normal", "debug")


@dataclass(frozen=True)
class ResolvedRunSelection:
    run_mode: str
    date: str | None
    game_number: int | None
    match_id: str | None
    match_ids: tuple[str, ...]
    fixture: pd.Series | None
    date_rows: pd.DataFrame
    list_date_rows: pd.DataFrame


def _normalise_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped if stripped else None


def _parse_schedule_date(date_text: str | None, schedule: pd.DataFrame) -> str:
    if not date_text:
        raise ValueError("DATE is required for this run mode.")
    years = sorted({int(str(value)[:4]) for value in schedule["date"].dropna() if str(value).strip()})
    if not years:
        raise ValueError("Parsed schedule has no usable dates.")
    schedule_year = years[0]
    raw = str(date_text).strip()
    for fmt in ("%Y-%m-%d", "%d-%m", "%d/%m", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if fmt in {"%d-%m", "%d/%m"}:
            parsed = parsed.replace(year=schedule_year)
        return parsed.date().isoformat()
    raise ValueError(
        f"Invalid date {date_text!r}. Use formats like 14-6, 14/6, 14-06, or 2026-06-14."
    )


def _fixtures_on_date(schedule: pd.DataFrame, date_text: str | None) -> tuple[str, pd.DataFrame]:
    parsed_date = _parse_schedule_date(date_text, schedule)
    rows = schedule[schedule["date"].astype(str) == parsed_date].copy()
    if rows.empty:
        raise ValueError(f"No fixtures found on {date_text} ({parsed_date}) in the parsed schedule.")
    rows = rows.sort_values(["time", "match_id"], kind="stable").reset_index(drop=True)
    return parsed_date, rows


def _format_date_schedule(parsed_date: str, rows: pd.DataFrame) -> str:
    heading = datetime.strptime(parsed_date, "%Y-%m-%d").strftime("%d %b %Y")
    lines = [heading]
    for index, row in rows.iterrows():
        lines.append(
            f"{index + 1}. {row['match_id']} | {row['time']} | {row['team_a']} vs {row['team_b']}"
        )
    return "\n".join(lines)


def _format_fixture(row: pd.Series | None, include_datetime: bool = False) -> str:
    if row is None:
        return "all available odds files"
    if include_datetime:
        return f"{row['match_id']} | {row['date']} {row['time']} | {row['team_a']} vs {row['team_b']}"
    return f"{row['match_id']} | {row['team_a']} vs {row['team_b']}"


def _resolve_run_selection(
    *,
    run_mode: str,
    date: str | None,
    game_number: int | None,
    match_id: str | None,
    schedule: pd.DataFrame,
) -> ResolvedRunSelection:
    if run_mode not in VALID_RUN_MODES:
        raise ValueError(f"Invalid RUN_MODE {run_mode!r}. Use one of: {', '.join(sorted(VALID_RUN_MODES))}.")
    if run_mode == "all_available":
        return ResolvedRunSelection(run_mode, date, game_number, None, (), None, pd.DataFrame(), pd.DataFrame())

    if run_mode == "list_date":
        parsed_date, rows = _fixtures_on_date(schedule, date)
        return ResolvedRunSelection(run_mode, parsed_date, game_number, None, (), None, pd.DataFrame(), rows)

    if run_mode == "date":
        parsed_date, rows = _fixtures_on_date(schedule, date)
        match_ids = tuple(rows["match_id"].astype(str))
        return ResolvedRunSelection(run_mode, parsed_date, game_number, None, match_ids, None, rows, pd.DataFrame())

    selected_match_id = _normalise_optional_text(match_id)
    if selected_match_id:
        matches = schedule[schedule["match_id"].astype(str) == selected_match_id]
        if matches.empty:
            raise ValueError(f"Selected match ID {selected_match_id} was not found in the parsed schedule.")
        fixture = matches.iloc[0]
        return ResolvedRunSelection(
            run_mode,
            str(fixture["date"]),
            game_number,
            selected_match_id,
            (selected_match_id,),
            fixture,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    if game_number is None:
        raise ValueError("GAME_NUMBER is required when RUN_MODE is 'single_match' and MATCH_ID is not set.")
    parsed_date, rows = _fixtures_on_date(schedule, date)
    if game_number < 1 or game_number > len(rows):
        raise ValueError(
            f"GAME_NUMBER {game_number} is out of range for {date}. "
            f"Choose a number from 1 to {len(rows)}."
        )
    fixture = rows.iloc[game_number - 1]
    return ResolvedRunSelection(
        run_mode,
        parsed_date,
        game_number,
        str(fixture["match_id"]),
        (str(fixture["match_id"]),),
        fixture,
        pd.DataFrame(),
        pd.DataFrame(),
    )


def _validate_selected_odds_file(selection: ResolvedRunSelection, combined_input_folder: Path) -> None:
    if selection.run_mode not in {"single_match", "date"}:
        return
    missing_paths = [
        combined_input_folder / f"{match_id}.txt"
        for match_id in selection.match_ids
        if not (combined_input_folder / f"{match_id}.txt").exists()
    ]
    if not missing_paths:
        return
    if selection.run_mode == "date":
        fixtures = [
            _format_fixture(row, include_datetime=True)
            for _, row in selection.date_rows.iterrows()
            if combined_input_folder / f"{row['match_id']}.txt" in missing_paths
        ]
        raise ValueError(
            "\n".join(
                [
                    "Selected date fixtures with missing odds:",
                    *fixtures,
                    "",
                    "Missing odds files:",
                    *(str(path) for path in missing_paths),
                    "",
                    "Create each file with sections:",
                    "### 1X2",
                    "### OVER_UNDER",
                    "### BTTS",
                    "### CORRECT_SCORE",
                ]
            )
        )
    path = missing_paths[0]
    raise ValueError(
        "\n".join(
            [
                "Selected match:",
                _format_fixture(selection.fixture, include_datetime=True),
                "",
                "Missing odds file:",
                str(path),
                "",
                "Create this file with sections:",
                "### 1X2",
                "### OVER_UNDER",
                "### BTTS",
                "### CORRECT_SCORE",
            ]
        )
    )


def _format_run_configuration(
    selection: ResolvedRunSelection,
    strategy_mode: str,
    run_profile: str,
    terminal_verbosity: str,
    margin_removal_method: str,
    enable_margin_method_comparison: bool,
    enable_market_consistent_challenger: bool | str,
    enable_weight_sensitivity: bool,
) -> str:
    lines = [
        "Live Prediction Run Configuration",
        f"- run mode: {selection.run_mode}",
    ]
    if selection.date is not None:
        lines.append(f"- date: {selection.date}")
    if selection.game_number is not None:
        lines.append(f"- game number: {selection.game_number}")
    if selection.fixture is not None:
        lines.append(f"- resolved match: {_format_fixture(selection.fixture)}")
    elif selection.run_mode == "date":
        lines.append(f"- resolved matches: {', '.join(selection.match_ids)}")
    elif selection.run_mode == "all_available":
        lines.append("- resolved match: all available odds files")
    lines.append(f"- run profile: {run_profile}")
    lines.append(f"- terminal verbosity: {terminal_verbosity}")
    lines.append(f"- strategy mode: {strategy_mode}")
    lines.append(f"- margin removal method: {margin_removal_method}")
    lines.append(f"- margin method comparison: {'enabled' if enable_margin_method_comparison else 'disabled'}")
    lines.append(f"- market-consistent challenger: {enable_market_consistent_challenger}")
    lines.append(f"- correct-score weight sensitivity: {'enabled' if enable_weight_sensitivity else 'disabled'}")
    return "\n".join(lines)


def _user_facing_live_error(error: Exception) -> str:
    """Translate internal cache errors into instructions for the live workflow."""

    message = str(error)
    if isinstance(error, LivePredictionError) and message.startswith(
        "No recognised OddsPortal paste files found below "
    ):
        return NO_ODDS_INPUT_MESSAGE
    return message


def main() -> None:
    """Parse live pastes, export recommendations, and print the final submission."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mode", choices=tuple(sorted(VALID_RUN_MODES)))
    parser.add_argument("--date")
    parser.add_argument("--game-number", type=int)
    parser.add_argument("--match-id")
    parser.add_argument("--correct-score-poisson-weight", type=float)
    parser.add_argument("--correct-score-aggregation-method")
    parser.add_argument("--margin-removal-method", choices=VALID_MARGIN_REMOVAL_METHODS)
    parser.add_argument("--compare-margin-methods", action="store_true")
    parser.add_argument("--no-compare-margin-methods", action="store_true")
    parser.add_argument("--run-profile", choices=VALID_RUN_PROFILES)
    parser.add_argument("--terminal-verbosity", choices=VALID_TERMINAL_VERBOSITIES)
    parser.add_argument("--dixon-coles-rho", type=float)
    parser.add_argument(
        "--strategy-mode",
        choices=VALID_STRATEGY_MODES,
    )
    parser.add_argument("--skip-weight-sensitivity", action="store_true")
    parser.add_argument("--enable-weight-sensitivity", action="store_true")
    parser.add_argument("--disable-weight-sensitivity", action="store_true")
    parser.add_argument("--skip-combined-split", action="store_true")
    parser.add_argument("--overwrite-combined-split", action="store_true")
    parser.add_argument("--combined-input-folder")
    parser.add_argument("--paste-input-folder")
    parser.add_argument("--schedule")
    parser.add_argument("--schedule-output", default=str(DEFAULT_SCHEDULE_OUTPUT_PATH))
    parser.add_argument("--schedule-report-output", default=str(DEFAULT_SCHEDULE_REPORT_PATH))
    parser.add_argument("--skip-schedule-parse", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    try:
        run_mode = args.run_mode if args.run_mode is not None else RUN_MODE
        if args.run_mode is None and (args.match_id is not None or args.date is not None or args.game_number is not None):
            run_mode = "single_match"
        date = args.date if args.date is not None else DATE
        game_number = args.game_number if args.game_number is not None else GAME_NUMBER
        match_id = args.match_id if args.match_id is not None else MATCH_ID
        strategy_mode = args.strategy_mode if args.strategy_mode is not None else STRATEGY_MODE
        run_profile = args.run_profile if args.run_profile is not None else RUN_PROFILE
        terminal_verbosity = (
            args.terminal_verbosity
            if args.terminal_verbosity is not None
            else TERMINAL_VERBOSITY
            if run_profile == "live"
            else "normal"
        )
        correct_score_poisson_weight = (
            args.correct_score_poisson_weight
            if args.correct_score_poisson_weight is not None
            else CORRECT_SCORE_POISSON_WEIGHT
        )
        correct_score_aggregation_method = (
            args.correct_score_aggregation_method
            if args.correct_score_aggregation_method is not None
            else CORRECT_SCORE_AGGREGATION_METHOD
        )
        margin_removal_method = (
            args.margin_removal_method
            if args.margin_removal_method is not None
            else MARGIN_REMOVAL_METHOD
        )
        enable_market_consistent_challenger = ENABLE_MARKET_CONSISTENT_CHALLENGER
        enable_margin_method_comparison = ENABLE_MARGIN_METHOD_COMPARISON
        enable_weight_sensitivity = ENABLE_CORRECT_SCORE_WEIGHT_SENSITIVITY
        if run_profile == "research":
            enable_market_consistent_challenger = True
            enable_margin_method_comparison = True
            enable_weight_sensitivity = True
        if args.compare_margin_methods:
            enable_margin_method_comparison = True
        if args.no_compare_margin_methods:
            enable_margin_method_comparison = False
        if args.enable_weight_sensitivity:
            enable_weight_sensitivity = True
        if args.disable_weight_sensitivity or args.skip_weight_sensitivity:
            enable_weight_sensitivity = False
        dixon_coles_rho = args.dixon_coles_rho if args.dixon_coles_rho is not None else DIXON_COLES_RHO
        strict = args.strict or STRICT_INPUT_VALIDATION

        schedule_path = Path(args.schedule) if args.schedule else INPUT_SCHEDULE_PATH
        combined_input_folder = (
            Path(args.combined_input_folder)
            if args.combined_input_folder
            else INPUT_ODDS_DIR
        )
        paste_input_folder = (
            Path(args.paste_input_folder)
            if args.paste_input_folder
            else CACHE_SPLIT_PASTES_DIR
        )
        total_start = time.perf_counter()
        if not schedule_path.exists():
            raise OddsPortalScheduleParseError(
                f"{schedule_path} is required. "
                "Paste the schedule there before running live predictions."
            )
        schedule_start = time.perf_counter()
        prepared_schedule = prepare_schedule_metadata(
            schedule_path,
            args.schedule_output,
            args.schedule_report_output,
            skip_parse=args.skip_schedule_parse,
        )
        initial_runtime_timings = {
            "schedule parse": time.perf_counter() - schedule_start,
        }
        if prepared_schedule.parse_result is not None:
            print(
                format_oddsportal_schedule_parse_summary(
                    prepared_schedule.parse_result,
                    args.schedule_output,
                    args.schedule_report_output,
                )
            )
            print()
        selection = _resolve_run_selection(
            run_mode=run_mode,
            date=date,
            game_number=game_number,
            match_id=match_id,
            schedule=prepared_schedule.schedule,
        )
        if selection.run_mode == "list_date":
            print(_format_date_schedule(str(selection.date), selection.list_date_rows))
            return
        _validate_selected_odds_file(selection, combined_input_folder)
        print(
            _format_run_configuration(
                selection,
                strategy_mode,
                run_profile,
                terminal_verbosity,
                margin_removal_method,
                enable_margin_method_comparison,
                enable_market_consistent_challenger,
                enable_weight_sensitivity,
            )
        )
        print()
        if not args.skip_combined_split:
            split_start = time.perf_counter()
            split_result = split_combined_oddsportal_pastes(
                combined_input_folder,
                paste_input_folder,
                overwrite=args.overwrite_combined_split or combined_input_folder == INPUT_ODDS_DIR,
                schedule=prepared_schedule.schedule,
            )
            initial_runtime_timings["combined paste split"] = time.perf_counter() - split_start
            if split_result.files_processed:
                print(format_combined_oddsportal_split_summary(split_result))
                print()
        else:
            initial_runtime_timings["combined paste split"] = 0.0
        result = run_live_prediction(
            LivePredictionSettings(
                input_folder=paste_input_folder,
                match_id=selection.match_id,
                match_ids=selection.match_ids or None,
                strict=strict,
                skip_weight_sensitivity=not enable_weight_sensitivity,
                metadata_odds_path=prepared_schedule.metadata_path,
                schedule_input_path=schedule_path,
                schedule_output_path=args.schedule_output,
                schedule_parse_report_path=args.schedule_report_output,
                skip_schedule_parse=True,
                write_detailed_excel=WRITE_DETAILED_EXCEL,
                initial_runtime_timings=initial_runtime_timings,
            ),
            ProjectConfig(
                correct_score_poisson_weight=correct_score_poisson_weight,
                correct_score_aggregation_method=correct_score_aggregation_method,
                margin_removal_method=margin_removal_method,
                enable_margin_method_comparison=enable_margin_method_comparison,
                enable_market_consistent_challenger=enable_market_consistent_challenger,
                dixon_coles_rho=dixon_coles_rho,
                public_strategy=PublicStrategyConfig(mode=strategy_mode),
            ),
        )
        result.runtime_timings["total"] = time.perf_counter() - total_start
    except (CombinedOddsPortalPasteError, LivePredictionError, OddsPortalScheduleParseError, ValueError) as error:
        raise SystemExit(_user_facing_live_error(error)) from error
    print(
        format_live_prediction_summary(
            result,
            terminal_verbosity=terminal_verbosity,
            run_profile=run_profile,
            show_runtime_summary=SHOW_RUNTIME_SUMMARY,
        )
    )


if __name__ == "__main__":
    main()
