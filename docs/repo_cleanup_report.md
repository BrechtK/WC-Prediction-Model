# Repository Cleanup Report

## Result

The normal live workflow exposes only:

```text
input/schedule.txt
input/odds/M001.txt
scripts/run_live_prediction.py
output/
```

Generated parser intermediates live below `cache/`. The obsolete visible raw
and processed trees were removed. No scoring, calibration, odds parsing,
correct-score aggregation, EV optimisation, or Dixon-Coles logic changed.

## Audit

The cleanup inspected `git ls-files`, `git status --ignored`, the full
repository tree, imports, and path references.

`git ls-files` found only two tracked files in the obsolete generated trees:

```text
data/raw/.gitkeep
data/processed/.gitkeep
```

Those placeholders were removed. No generated CSV, XLSX, or TXT artifact was
tracked, so no `git rm --cached` operation was needed.

## Retained Tracked Data

`data/` now contains only:

```text
data/examples/
data/templates/
```

These remain tracked because tests, example scripts, and the advanced prepared
workbook command use them. Parser regression fixtures remain under
`tests/fixtures/`.

## Generated Paths

The normal workflow creates only ignored paths:

```text
output/
cache/
```

Optional local advanced inputs use ignored paths:

```text
input/prepared/
input/historical/
```

The `.gitignore` file also ignores obsolete `data/raw/` and `data/processed/`
paths so stale folders cannot accidentally be committed if recreated locally.

## Cleanup Command

Run:

```powershell
python scripts/clean_generated_outputs.py
```

The script has an explicit allowlist and removes only:

```text
output/
cache/
data/processed/
.pytest_cache/
common project Python cache directories
```

It never removes source files, test files, documentation, or `input/`.
If retired OddsPortal paste files are found below `data/raw/`, it warns before
cleanup starts and preserves them for manual review.

## Script Classification

Primary live entry point:

```text
scripts/run_live_prediction.py
```

Advanced parser and sensitivity tools:

```text
scripts/parse_oddsportal_schedule.py
scripts/split_oddsportal_combined_pastes.py
scripts/parse_oddsportal_core_odds.py
scripts/parse_oddsportal_correct_scores.py
scripts/compare_correct_score_weights.py
scripts/compare_correct_score_aggregation_methods.py
scripts/compare_dixon_coles_rho.py
scripts/run_world_cup_predictions.py
```

Research and backtesting tools remain available and now write below
`output/research/`.

## Path Centralisation

`src/wc_predictor/paths.py` owns the live structure:

```text
INPUT_SCHEDULE_PATH
INPUT_ODDS_DIR
INPUT_PREPARED_DIR
INPUT_HISTORICAL_DIR
OUTPUT_DIR
OUTPUT_RESEARCH_DIR
OUTPUT_PARSE_REPORTS_DIR
CACHE_PARSED_DIR
CACHE_SPLIT_PASTES_DIR
```

## Final Structure

```text
input/
    .gitkeep
    odds/
        .gitkeep
data/
    examples/
    templates/
docs/
prompts/
scripts/
src/
tests/
notebooks/
README.md
pyproject.toml
.gitignore
```

`output/` and `cache/` appear only after generated work and can be removed with
the cleanup command.
