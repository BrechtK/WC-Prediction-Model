"""Historical group-stage-style strategy evaluation.

The first backtesting implementation deliberately reuses the Version 1 market
pipeline: proportional margin removal, bookmaker aggregation, independent
Poisson calibration, score-matrix generation, expected-points optimisation,
and the private-pool group scoring rule.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from wc_predictor.calibration import CalibrationTargets, calibrate_poisson_model
from wc_predictor.config import ProjectConfig
from wc_predictor.market_data import load_tabular_data
from wc_predictor.odds import aggregate_bookmaker_probabilities, process_bookmaker_odds
from wc_predictor.optimiser import optimise_group_prediction
from wc_predictor.probabilities import ScoreProbabilityMatrix
from wc_predictor.scoring_rules import score_group_prediction
from wc_predictor.strategies import FavouriteScoreStrategy, FixedScoreStrategy, MostLikelyScoreStrategy
from wc_predictor.utils import ensure_parent_directory, goal_difference, result_sign


@dataclass(frozen=True)
class OddsSourceColumns:
    """Football-Data-like source columns for one bookmaker or average market."""

    label: str
    home: str | None = None
    draw: str | None = None
    away: str | None = None
    over_2_5: str | None = None
    under_2_5: str | None = None
    is_average: bool = False


DEFAULT_ODDS_SOURCES = (
    OddsSourceColumns("Average", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5", True),
    OddsSourceColumns("Bet365", "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"),
    OddsSourceColumns("Betway", "BWH", "BWD", "BWA"),
    OddsSourceColumns("Interwetten", "IWH", "IWD", "IWA"),
    OddsSourceColumns("Pinnacle", "PSH", "PSD", "PSA", "P>2.5", "P<2.5"),
    OddsSourceColumns("WilliamHill", "WHH", "WHD", "WHA"),
    OddsSourceColumns("VCBet", "VCH", "VCD", "VCA"),
)


@dataclass(frozen=True)
class HistoricalBacktestData:
    """Historical matches mapped to the project's internal odds and result tables."""

    matches: pd.DataFrame
    odds: pd.DataFrame
    results: pd.DataFrame


class HistoricalOddsLoader(Protocol):
    """Interface for historical providers such as Football-Data.co.uk."""

    def load(self) -> HistoricalBacktestData:
        """Return historical matches in the project's internal formats."""


@dataclass(frozen=True)
class FootballDataCSVLoader:
    """Map a Football-Data.co.uk-like CSV or Excel file into internal tables.

    Average odds are preferred when present. If an average market is absent for
    a row, available bookmaker markets are retained and aggregated downstream.
    """

    path: str | Path
    odds_sources: tuple[OddsSourceColumns, ...] = DEFAULT_ODDS_SOURCES

    def load(self) -> HistoricalBacktestData:
        """Load and normalise historical home/away odds and full-time results."""

        source = load_tabular_data(self.path)
        required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        missing = required - set(source.columns)
        if missing:
            raise ValueError(f"Historical data is missing required columns: {sorted(missing)}")

        matches: list[dict[str, object]] = []
        results: list[dict[str, object]] = []
        odds: list[dict[str, object]] = []
        for index, row in source.iterrows():
            match_id = self._match_id(row, index)
            date = row.get("Date", "")
            matches.append(
                {
                    "match_id": match_id,
                    "date": date,
                    "stage": "group stage",
                    "team_a": row["HomeTeam"],
                    "team_b": row["AwayTeam"],
                }
            )
            results.append(
                {
                    "match_id": match_id,
                    "team_a_goals_90": row["FTHG"],
                    "team_b_goals_90": row["FTAG"],
                }
            )
            odds.extend(self._map_odds_rows(row, match_id))
        return HistoricalBacktestData(pd.DataFrame(matches), pd.DataFrame(odds), pd.DataFrame(results))

    @staticmethod
    def _match_id(row: pd.Series, index: object) -> str:
        for column in ("match_id", "MatchID", "MatchId"):
            if column in row and pd.notna(row[column]):
                return str(row[column])
        return f"H{int(index) + 1:06d}"

    def _map_odds_rows(self, row: pd.Series, match_id: str) -> list[dict[str, object]]:
        one_x_two_sources = self._select_sources(row, market="1x2")
        over_under_sources = self._select_sources(row, market="over_under_2_5")
        selected = {source.label: source for source in (*one_x_two_sources, *over_under_sources)}
        mapped: list[dict[str, object]] = []
        for source in selected.values():
            has_1x2 = source in one_x_two_sources
            has_over_under = source in over_under_sources
            mapped.append(
                {
                    "match_id": match_id,
                    "bookmaker": source.label,
                    "odds_a_win": row[source.home] if has_1x2 and source.home else np.nan,
                    "odds_draw": row[source.draw] if has_1x2 and source.draw else np.nan,
                    "odds_b_win": row[source.away] if has_1x2 and source.away else np.nan,
                    "odds_over_2_5": row[source.over_2_5] if has_over_under and source.over_2_5 else np.nan,
                    "odds_under_2_5": row[source.under_2_5] if has_over_under and source.under_2_5 else np.nan,
                }
            )
        return mapped

    def _select_sources(self, row: pd.Series, market: str) -> tuple[OddsSourceColumns, ...]:
        available = tuple(source for source in self.odds_sources if self._has_valid_market(row, source, market))
        average = tuple(source for source in available if source.is_average)
        return average[:1] if average else tuple(source for source in available if not source.is_average)

    @staticmethod
    def _has_valid_market(row: pd.Series, source: OddsSourceColumns, market: str) -> bool:
        columns = (
            (source.home, source.draw, source.away)
            if market == "1x2"
            else (source.over_2_5, source.under_2_5)
        )
        if any(column is None or column not in row for column in columns):
            return False
        values = pd.to_numeric(pd.Series([row[column] for column in columns]), errors="coerce")
        return bool(values.notna().all() and np.isfinite(values).all() and (values > 1.0).all())


@dataclass(frozen=True)
class BacktestSettings:
    """Settings for the first group-stage historical backtest."""

    output_path: Path = Path("data/processed/backtest_results.csv")


@dataclass(frozen=True)
class BacktestReport:
    """Summary metrics, match-level predictions, and skip diagnostics."""

    summary: pd.DataFrame
    predictions: pd.DataFrame
    skipped: pd.DataFrame

    def export_csv(self, path: str | Path) -> None:
        """Export strategy-level summary metrics to CSV."""

        path = Path(path)
        ensure_parent_directory(path)
        self.summary.to_csv(path, index=False)


@dataclass(frozen=True)
class BatchBacktestSettings:
    """Output paths for single-file or folder-level historical backtests."""

    detailed_output_path: Path = Path("data/processed/backtest_results_by_file.csv")
    aggregate_output_path: Path = Path("data/processed/backtest_results_aggregate.csv")
    skipped_output_path: Path = Path("data/processed/backtest_skipped_by_file.csv")


@dataclass(frozen=True)
class BatchBacktestReport:
    """Per-file, aggregate, and skip-diagnostic outputs for one or more CSVs."""

    detailed_summary: pd.DataFrame
    aggregate_summary: pd.DataFrame
    skipped_by_file_reason: pd.DataFrame
    predictions: pd.DataFrame
    skipped: pd.DataFrame

    def export_csvs(self, settings: BatchBacktestSettings) -> None:
        """Export per-file, aggregate, and skip-diagnostic tables."""

        for path in (
            settings.detailed_output_path,
            settings.aggregate_output_path,
            settings.skipped_output_path,
        ):
            ensure_parent_directory(path)
        self.detailed_summary.to_csv(settings.detailed_output_path, index=False)
        self.aggregate_summary.to_csv(settings.aggregate_output_path, index=False)
        self.skipped_by_file_reason.to_csv(settings.skipped_output_path, index=False)


@dataclass
class _StrategyAccumulator:
    """Mutable per-strategy records used while iterating through history."""

    name: str
    source_file: str
    predictions: list[dict[str, object]] = field(default_factory=list)
    skipped: list[dict[str, object]] = field(default_factory=list)

    def skip(self, match_id: str, reason: str) -> None:
        self.skipped.append(
            {
                "source_file": self.source_file,
                "strategy": self.name,
                "match_id": match_id,
                "reason": reason,
            }
        )


@dataclass(frozen=True)
class _MarketContext:
    """Fair probabilities and calibrated score matrices for one historical match."""

    fair_a_win: float
    fair_draw: float
    fair_b_win: float
    matrix_1x2: ScoreProbabilityMatrix
    matrix_1x2_over_under: ScoreProbabilityMatrix
    used_over_under: bool


class BacktestRunner:
    """Evaluate reusable strategies against historical group-stage-style scoring."""

    STRATEGY_NAMES = (
        "always_1_1",
        "always_0_0",
        "favourite_1_0",
        "favourite_2_0",
        "most_likely_poisson",
        "ev_optimal_1x2",
        "ev_optimal_1x2_over_under",
    )

    def __init__(
        self,
        loader: HistoricalOddsLoader,
        config: ProjectConfig | None = None,
        settings: BacktestSettings | None = None,
        source_file: str | None = None,
    ) -> None:
        self.loader = loader
        self.config = config or ProjectConfig()
        self.settings = settings or BacktestSettings()
        self.source_file = source_file or self._loader_source_file(loader)

    def run(self, export: bool = True) -> BacktestReport:
        """Run all group-stage strategies and optionally export summary metrics."""

        data = self.loader.load()
        accumulators = {
            name: _StrategyAccumulator(name, self.source_file) for name in self.STRATEGY_NAMES
        }
        odds_by_match = {
            str(match_id): group.copy()
            for match_id, group in data.odds.groupby("match_id", sort=False)
        } if not data.odds.empty else {}
        result_by_match = data.results.set_index("match_id")

        for _, match in data.matches.iterrows():
            match_id = str(match["match_id"])
            actual = self._actual_score(result_by_match, match_id)
            if actual is None:
                for accumulator in accumulators.values():
                    accumulator.skip(match_id, "missing or invalid full-time result")
                continue

            self._record_fixed_predictions(accumulators, match, actual)
            try:
                market = self._build_market_context(odds_by_match.get(match_id))
            except ValueError as exc:
                for name in self.STRATEGY_NAMES[2:]:
                    accumulators[name].skip(match_id, str(exc))
                continue
            self._record_market_predictions(accumulators, match, actual, market)

        predictions = pd.DataFrame(
            [prediction for accumulator in accumulators.values() for prediction in accumulator.predictions]
        )
        skipped = pd.DataFrame([skip for accumulator in accumulators.values() for skip in accumulator.skipped])
        summary = pd.DataFrame([self._summarise(accumulator) for accumulator in accumulators.values()])
        report = BacktestReport(summary, predictions, skipped)
        if export:
            report.export_csv(self.settings.output_path)
        return report

    def _build_market_context(self, match_odds: pd.DataFrame | None) -> _MarketContext:
        if match_odds is None or match_odds.empty:
            raise ValueError("missing or invalid 1X2 odds")
        bookmaker_probabilities = process_bookmaker_odds(
            match_odds,
            self.config.margin_removal_method,
            self.config.suspicious_overround_low,
            self.config.suspicious_overround_high,
        )
        aggregated = aggregate_bookmaker_probabilities(
            bookmaker_probabilities, self.config.bookmaker_aggregation_method
        )
        required = ("fair_a_win", "fair_draw", "fair_b_win")
        if aggregated.empty or any(column not in aggregated for column in required):
            raise ValueError("missing or invalid 1X2 odds")
        row = aggregated.iloc[0]
        if any(pd.isna(row[column]) for column in required):
            raise ValueError("missing or invalid 1X2 odds")
        targets_1x2 = CalibrationTargets(*(float(row[column]) for column in required))
        matrix_1x2 = self._calibrate(targets_1x2)
        used_over_under = "fair_over_2_5" in row and pd.notna(row["fair_over_2_5"])
        targets_with_over_under = CalibrationTargets(
            *targets_1x2_values(targets_1x2),
            over_2_5=float(row["fair_over_2_5"]) if used_over_under else None,
        )
        matrix_1x2_over_under = self._calibrate(targets_with_over_under) if used_over_under else matrix_1x2
        return _MarketContext(
            targets_1x2.a_win,
            targets_1x2.draw,
            targets_1x2.b_win,
            matrix_1x2,
            matrix_1x2_over_under,
            used_over_under,
        )

    def _calibrate(self, targets: CalibrationTargets) -> ScoreProbabilityMatrix:
        result = calibrate_poisson_model(
            targets,
            self.config.max_goals_score_matrix,
            self.config.calibration_weights,
            self.config.renormalise_score_matrix,
            self.config.poor_calibration_loss_threshold,
        )
        if not result.success:
            raise ValueError("Poisson calibration failed")
        return result.score_matrix

    @staticmethod
    def _actual_score(results: pd.DataFrame, match_id: str) -> tuple[int, int] | None:
        if match_id not in results.index:
            return None
        row = results.loc[match_id]
        try:
            goals_a = float(row["team_a_goals_90"])
            goals_b = float(row["team_b_goals_90"])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(goals_a) or not np.isfinite(goals_b):
            return None
        if goals_a < 0 or goals_b < 0 or not goals_a.is_integer() or not goals_b.is_integer():
            return None
        return int(goals_a), int(goals_b)

    def _record_fixed_predictions(
        self,
        accumulators: dict[str, _StrategyAccumulator],
        match: pd.Series,
        actual: tuple[int, int],
    ) -> None:
        self._record(accumulators["always_1_1"], match, actual, FixedScoreStrategy((1, 1)).predict(None))
        self._record(accumulators["always_0_0"], match, actual, FixedScoreStrategy((0, 0)).predict(None))

    def _record_market_predictions(
        self,
        accumulators: dict[str, _StrategyAccumulator],
        match: pd.Series,
        actual: tuple[int, int],
        market: _MarketContext,
    ) -> None:
        fair_probabilities = (market.fair_a_win, market.fair_draw, market.fair_b_win)
        self._record(
            accumulators["favourite_1_0"],
            match,
            actual,
            FavouriteScoreStrategy(1).predict(fair_probabilities),
        )
        self._record(
            accumulators["favourite_2_0"],
            match,
            actual,
            FavouriteScoreStrategy(2).predict(fair_probabilities),
        )
        self._record(
            accumulators["most_likely_poisson"],
            match,
            actual,
            MostLikelyScoreStrategy().predict(market.matrix_1x2),
        )
        self._record(
            accumulators["ev_optimal_1x2"],
            match,
            actual,
            optimise_group_prediction(market.matrix_1x2, self.config.max_candidate_goals).best.predicted_score,
        )
        self._record(
            accumulators["ev_optimal_1x2_over_under"],
            match,
            actual,
            optimise_group_prediction(
                market.matrix_1x2_over_under, self.config.max_candidate_goals
            ).best.predicted_score,
            used_over_under=market.used_over_under,
        )

    @staticmethod
    def _record(
        accumulator: _StrategyAccumulator,
        match: pd.Series,
        actual: tuple[int, int],
        prediction: tuple[int, int],
        used_over_under: bool = False,
    ) -> None:
        pred_a, pred_b = prediction
        actual_a, actual_b = actual
        accumulator.predictions.append(
            {
                "source_file": accumulator.source_file,
                "strategy": accumulator.name,
                "match_id": match["match_id"],
                "date": match["date"],
                "team_a": match["team_a"],
                "team_b": match["team_b"],
                "predicted_team_a_goals": pred_a,
                "predicted_team_b_goals": pred_b,
                "actual_team_a_goals": actual_a,
                "actual_team_b_goals": actual_b,
                "realised_points": score_group_prediction(pred_a, pred_b, actual_a, actual_b),
                "is_exact_score": (pred_a, pred_b) == (actual_a, actual_b),
                "is_correct_goal_difference": goal_difference(pred_a, pred_b) == goal_difference(actual_a, actual_b),
                "is_correct_result": result_sign(pred_a, pred_b) == result_sign(actual_a, actual_b),
                "used_over_under_2_5": used_over_under,
            }
        )

    @staticmethod
    def _summarise(accumulator: _StrategyAccumulator) -> dict[str, object]:
        predictions = pd.DataFrame(accumulator.predictions)
        skip_counts = Counter(skip["reason"] for skip in accumulator.skipped)
        skip_reasons = "; ".join(f"{reason}: {count}" for reason, count in sorted(skip_counts.items()))
        if predictions.empty:
            return {
                "source_file": accumulator.source_file,
                "strategy": accumulator.name,
                "average_realised_points": np.nan,
                "total_points": 0,
                "exact_score_hit_rate": np.nan,
                "correct_goal_difference_hit_rate": np.nan,
                "correct_result_hit_rate": np.nan,
                "points_variance": np.nan,
                "matches_used": 0,
                "skipped_matches": len(accumulator.skipped),
                "skip_reasons": skip_reasons,
            }
        return {
            "source_file": accumulator.source_file,
            "strategy": accumulator.name,
            "average_realised_points": float(predictions["realised_points"].mean()),
            "total_points": int(predictions["realised_points"].sum()),
            "exact_score_hit_rate": float(predictions["is_exact_score"].mean()),
            "correct_goal_difference_hit_rate": float(predictions["is_correct_goal_difference"].mean()),
            "correct_result_hit_rate": float(predictions["is_correct_result"].mean()),
            "points_variance": float(predictions["realised_points"].var(ddof=0)),
            "matches_used": len(predictions),
            "skipped_matches": len(accumulator.skipped),
            "skip_reasons": skip_reasons,
        }

    @staticmethod
    def _loader_source_file(loader: HistoricalOddsLoader) -> str:
        path = getattr(loader, "path", None)
        return Path(path).name if path is not None else "historical_data"


class BatchBacktestRunner:
    """Run the group-stage backtest against one CSV or a folder of CSV files."""

    def __init__(
        self,
        input_path: str | Path,
        config: ProjectConfig | None = None,
        settings: BatchBacktestSettings | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.config = config or ProjectConfig()
        self.settings = settings or BatchBacktestSettings()

    def run(self, export: bool = True) -> BatchBacktestReport:
        """Backtest every selected CSV and recompute aggregate metrics from match records."""

        reports = [
            BacktestRunner(
                FootballDataCSVLoader(path),
                config=self.config,
                source_file=path.name,
            ).run(export=False)
            for path in self._csv_paths()
        ]
        detailed_summary = pd.concat([report.summary for report in reports], ignore_index=True)
        predictions = self._concat_frames([report.predictions for report in reports])
        skipped = self._concat_frames([report.skipped for report in reports])
        aggregate_summary = self._aggregate_summary(predictions, skipped)
        skipped_by_file_reason = self._skipped_by_file_reason(skipped)
        report = BatchBacktestReport(
            detailed_summary,
            aggregate_summary,
            skipped_by_file_reason,
            predictions,
            skipped,
        )
        if export:
            report.export_csvs(self.settings)
        return report

    def _csv_paths(self) -> list[Path]:
        if self.input_path.is_file():
            if self.input_path.suffix.lower() != ".csv":
                raise ValueError(f"Historical input file must be CSV: {self.input_path}")
            return [self.input_path]
        if self.input_path.is_dir():
            paths = sorted(
                path for path in self.input_path.iterdir() if path.is_file() and path.suffix.lower() == ".csv"
            )
            if not paths:
                raise ValueError(f"Historical input folder contains no CSV files: {self.input_path}")
            return paths
        raise FileNotFoundError(self.input_path)

    @staticmethod
    def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        available = [frame for frame in frames if not frame.empty]
        return pd.concat(available, ignore_index=True) if available else pd.DataFrame()

    @staticmethod
    def _aggregate_summary(predictions: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for strategy in BacktestRunner.STRATEGY_NAMES:
            accumulator = _StrategyAccumulator(strategy, "ALL_FILES")
            if not predictions.empty:
                accumulator.predictions = predictions[predictions["strategy"] == strategy].to_dict("records")
            if not skipped.empty:
                accumulator.skipped = skipped[skipped["strategy"] == strategy].to_dict("records")
            rows.append(BacktestRunner._summarise(accumulator))
        return pd.DataFrame(rows)

    @staticmethod
    def _skipped_by_file_reason(skipped: pd.DataFrame) -> pd.DataFrame:
        columns = ("source_file", "reason", "skipped_matches", "affected_strategies")
        if skipped.empty:
            return pd.DataFrame(columns=columns)
        grouped = (
            skipped.groupby(["source_file", "reason"], as_index=False)
            .agg(
                skipped_matches=("match_id", "nunique"),
                affected_strategies=("strategy", lambda values: ", ".join(sorted(set(values)))),
            )
            .sort_values(["source_file", "reason"])
            .reset_index(drop=True)
        )
        return grouped[list(columns)]


def targets_1x2_values(targets: CalibrationTargets) -> tuple[float, float, float]:
    """Return the ordered 1X2 values from calibration targets."""

    return targets.a_win, targets.draw, targets.b_win
