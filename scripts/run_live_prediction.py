"""Run the complete live OddsPortal paste-to-submission workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import pandas as pd

from wc_predictor.config import ProjectConfig, PublicStrategyConfig
from wc_predictor.live_prediction import ( LivePredictionError, LivePredictionSettings, format_live_prediction_summary, run_live_prediction, )
from wc_predictor.oddsportal_combined import ( CombinedOddsPortalPasteError, format_combined_oddsportal_split_summary, split_combined_oddsportal_pastes, )
from wc_predictor.oddsportal_schedule import ( DEFAULT_SCHEDULE_OUTPUT_PATH, DEFAULT_SCHEDULE_REPORT_PATH, OddsPortalScheduleParseError, format_oddsportal_schedule_parse_summary, prepare_schedule_metadata, )
from wc_predictor.paths import ( CACHE_SPLIT_PASTES_DIR, INPUT_ODDS_DIR, INPUT_SCHEDULE_PATH, )

# ============================================================
# USER SETTINGS
# ============================================================

# Normal matchday use:
# 1. Use RUN_MODE = "list_date" and set DATE to see the numbered games.
# 2. Paste fresh odds into input/odds/Mxxx.txt.
# 3. Use RUN_MODE = "single_match", keep the same DATE, and set GAME_NUMBER.
# 4. Press "Run Python File" in VS Code.

RUN_MODE = "all_available"
# RUN_MODE options:
# "list_date"      -> list the games on DATE and exit; use this first.
# "single_match"   -> run one game from DATE, selected by GAME_NUMBER.
# "date"           -> run every game on DATE; each game needs input/odds/Mxxx.txt.
# "all_available"  -> run every odds file currently in input/odds/.

# DATE accepts 14-6, 14/6, 14-06, or 2026-06-14.
DATE = "14/6"
# GAME_NUMBER comes from RUN_MODE = "list_date". It is 1 for the first listed game.
GAME_NUMBER = 4
# MATCH_ID is optional. Set it only if you already know the ID, for example "M008".
# When MATCH_ID is set, it overrides DATE and GAME_NUMBER in single-match mode.
MATCH_ID = None

STRATEGY_MODE = "ev"
# Options:
# "ev"
# "balanced"
# "public-ranking"
# "aggressive-public-ranking"
PUBLIC_STRATEGY_TARGET = "balanced"
# Options: "friends", "balanced", "national".
PUBLIC_FIELD_SIZE = 100

# Conservative live default: keep the Poisson-only score matrix until a
# blended correct-score default is validated. Use 0.85 only for research runs.
CORRECT_SCORE_POISSON_WEIGHT = 1.0
CORRECT_SCORE_AGGREGATION_METHOD = "auto"

MARGIN_REMOVAL_METHOD = "normalised_inverse_odds"
# Optional research choices: "power", "additive", or "shin". These methods are
# diagnostics and may fall back on sparse or unusual markets.

# Fast matchday defaults.
RUN_PROFILE = "live"
TERMINAL_VERBOSITY = "compact"
ENABLE_MARKET_CONSISTENT_CHALLENGER = "only_if_close"
ENABLE_MARGIN_METHOD_COMPARISON = False
ENABLE_CORRECT_SCORE_WEIGHT_SENSITIVITY = False
# RUN_PROFILE:
# "live"     -> fast tournament mode; skips slow diagnostics by default.
# "research" -> slower analysis mode; enables diagnostic comparisons.
#
# TERMINAL_VERBOSITY:
# "compact" -> short final recommendation dashboard for matchday.
# "normal"  -> adds parse summaries and model-disagreement summaries.
# "debug"   -> prints detailed per-match diagnostics for investigation.

WRITE_DETAILED_EXCEL = True
WRITE_CACHE_OUTPUTS = True
SHOW_RUNTIME_SUMMARY = True

# Set a small positive rho to increase the probability of low-scoring draws, or a small negative rho to decrease it. 
# The optimal value may differ between tournaments; 0.0 is a reasonable default for the World Cup.
DIXON_COLES_RHO = 0.0


STRICT_INPUT_VALIDATION = False
STALE_ODDS_WARNING_HOURS = 24
STALE_ODDS_AFFECTS_MANUAL_REVIEW = True

ODDS_TEMPLATE_PATH = Path("templates/odds_input_template.txt")
DEPRECATED_INPUT_FOLDERS = (
    Path("data/raw/oddsportal_combined_pastes"),
    Path("data/raw/oddsportal_pastes"),
)
NO_ODDS_INPUT_MESSAGE = (
    "No odds input files found.\n\n"
    "Create files like:\n"
    "input/odds/M001.txt\n\n"
    "Use the template:\n"
    f"{ODDS_TEMPLATE_PATH}\n\n"
    "Required section:\n"
    "### 1X2\n\n"
    "Recommended sections:\n"
    "### MATCH\n"
    "### OVER_UNDER\n"
    "### BTTS\n"
    "### CORRECT_SCORE\n"
    "Optional diagnostic section:\n"
    "### ASIAN_HANDICAP"
)
VALID_RUN_MODES = {"all_available", "date", "single_match", "list_date"}
VALID_STRATEGY_MODES = ("ev", "balanced", "public-ranking", "aggressive-public-ranking")
VALID_PUBLIC_STRATEGY_TARGETS = ("friends", "balanced", "national")
VALID_MARGIN_REMOVAL_METHODS = ("normalised_inverse_odds", "power", "additive", "shin")
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
        available_dates = ", ".join(sorted(schedule["date"].dropna().astype(str).unique()))
        raise ValueError(
            f"No fixtures found on {date_text} ({parsed_date}) in the parsed schedule.\n\n"
            "Set DATE to one of the dates in input/schedule.txt.\n"
            f"Available parsed dates: {available_dates or 'none'}"
        )
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
            raise ValueError(
                f"Selected match ID {selected_match_id} was not found in the parsed schedule.\n\n"
                "Set RUN_MODE = \"list_date\" and run again to see the correct match IDs."
            )
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
            f"Choose a number from 1 to {len(rows)}.\n\n"
            "Set RUN_MODE = \"list_date\" and run again to see the available game numbers."
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
                    "Create each file using:",
                    str(ODDS_TEMPLATE_PATH),
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
                "Create it using:",
                str(ODDS_TEMPLATE_PATH),
            ]
        )
    )


def _deprecated_input_warnings() -> tuple[str, ...]:
    warnings: list[str] = []
    for folder in DEPRECATED_INPUT_FOLDERS:
        if folder.exists() and any(folder.glob("*.txt")):
            warnings.append(
                f"{folder} contains old paste files. Move live odds to input/odds/Mxxx.txt."
            )
    return tuple(warnings)


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _selected_odds_file_stale_warnings(
    selection: ResolvedRunSelection,
    combined_input_folder: Path,
    threshold_hours: float,
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Return non-blocking warnings for selected combined odds files with old mtimes."""

    if threshold_hours <= 0:
        return {}, ()
    if selection.run_mode in {"single_match", "date"}:
        paths = tuple(combined_input_folder / f"{match_id}.txt" for match_id in selection.match_ids)
    else:
        paths = tuple(sorted(combined_input_folder.glob("*.txt")))
    threshold_seconds = threshold_hours * 60 * 60
    now = time.time()
    flags_by_match: dict[str, tuple[str, ...]] = {}
    messages: list[str] = []
    threshold_label = f"{threshold_hours:g}"
    for path in paths:
        if not path.exists():
            continue
        age_seconds = now - path.stat().st_mtime
        if age_seconds <= threshold_seconds:
            continue
        match_id = path.stem
        flags_by_match[match_id] = ("stale_odds_file",)
        messages.append(
            f"Warning: {_display_path(path)} was last modified more than {threshold_label} "
            "hours ago. Check that odds are fresh."
        )
    return flags_by_match, tuple(messages)


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
    if isinstance(error, CombinedOddsPortalPasteError) and "missing required ### 1X2 section" in message:
        return f"{message}\n\nAdd the missing section using:\n{ODDS_TEMPLATE_PATH}"
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
    parser.add_argument("--public-strategy-target", choices=VALID_PUBLIC_STRATEGY_TARGETS)
    parser.add_argument("--public-field-size", type=int)
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
    parser.add_argument("--stale-odds-warning-hours", type=float)
    parser.add_argument("--stale-odds-affects-manual-review", action="store_true")
    parser.add_argument("--no-stale-odds-affects-manual-review", action="store_true")
    args = parser.parse_args()

    try:
        run_mode = args.run_mode if args.run_mode is not None else RUN_MODE
        if args.run_mode is None and (args.match_id is not None or args.date is not None or args.game_number is not None):
            run_mode = "single_match"
        date = args.date if args.date is not None else DATE
        game_number = args.game_number if args.game_number is not None else GAME_NUMBER
        match_id = args.match_id if args.match_id is not None else MATCH_ID
        strategy_mode = args.strategy_mode if args.strategy_mode is not None else STRATEGY_MODE
        public_strategy_target = (
            args.public_strategy_target
            if args.public_strategy_target is not None
            else PUBLIC_STRATEGY_TARGET
        )
        public_field_size = args.public_field_size if args.public_field_size is not None else PUBLIC_FIELD_SIZE
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
        stale_odds_warning_hours = (
            args.stale_odds_warning_hours
            if args.stale_odds_warning_hours is not None
            else STALE_ODDS_WARNING_HOURS
        )
        stale_odds_affects_manual_review = STALE_ODDS_AFFECTS_MANUAL_REVIEW
        if args.stale_odds_affects_manual_review:
            stale_odds_affects_manual_review = True
        if args.no_stale_odds_affects_manual_review:
            stale_odds_affects_manual_review = False

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
        deprecated_warnings = _deprecated_input_warnings()
        if deprecated_warnings and not any(combined_input_folder.glob("*.txt")):
            raise ValueError(
                "\n".join(
                    [
                        "Old live input files were found in deprecated folders:",
                        *(f"- {warning}" for warning in deprecated_warnings),
                        "",
                        "Move the current match odds to:",
                        "input/odds/Mxxx.txt",
                        "",
                        "Use the template:",
                        str(ODDS_TEMPLATE_PATH),
                    ]
                )
            )
        total_start = time.perf_counter()
        if not schedule_path.exists():
            raise OddsPortalScheduleParseError(
                f"{schedule_path} is required.\n\n"
                "Create it by pasting the OddsPortal schedule into:\n"
                f"{schedule_path}\n\n"
                "Then run again with RUN_MODE = \"list_date\" to find the game number."
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
        if terminal_verbosity != "compact" and prepared_schedule.parse_result is not None:
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
        stale_warning_flags, stale_warning_messages = _selected_odds_file_stale_warnings(
            selection,
            combined_input_folder,
            stale_odds_warning_hours,
        )
        if terminal_verbosity != "compact":
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
            for warning in deprecated_warnings:
                print(f"Input warning: {warning}")
            if deprecated_warnings:
                print()
        for warning in stale_warning_messages:
            print(warning)
        if stale_warning_messages:
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
            if split_result.files_processed and (terminal_verbosity != "compact" or split_result.warnings):
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
                extra_warning_flags_by_match=stale_warning_flags if stale_odds_affects_manual_review else {},
            ),
            ProjectConfig(
                correct_score_poisson_weight=correct_score_poisson_weight,
                correct_score_aggregation_method=correct_score_aggregation_method,
                margin_removal_method=margin_removal_method,
                enable_margin_method_comparison=enable_margin_method_comparison,
                enable_market_consistent_challenger=enable_market_consistent_challenger,
                dixon_coles_rho=dixon_coles_rho,
                public_strategy=PublicStrategyConfig(
                    mode=strategy_mode,
                    public_strategy_target=public_strategy_target,
                    public_field_size=public_field_size,
                ),
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
