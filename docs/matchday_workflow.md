# Matchday Workflow

This is the beginner-friendly routine for using the project during the tournament.

## 1. Open the run file

Open:

```text
scripts/run_live_prediction.py
```

Use the `USER SETTINGS` block near the top of the file. You normally do not need
to edit anything else.

## 2. Use fast live settings

Make sure your local schedule paste exists at:

```text
input/schedule.txt
```

For matchday, keep:

```python
RUN_PROFILE = "live"
TERMINAL_VERBOSITY = "compact"
MARGIN_REMOVAL_METHOD = "normalised_inverse_odds"
```

`live` keeps the run fast. `compact` keeps the terminal focused on the final
choice and any warnings that need attention.
Shin, power, and additive margin removal remain available for explicit research
runs. If one of those diagnostic methods fails for a sparse or unusual market,
the live workflow falls back to `normalised_inverse_odds` for that market and
prints/report a warning.

## 3. Find the match number

Set:

```python
RUN_MODE = "list_date"
DATE = "14-6"
```

Then press **Run Python File** once in VS Code.

The script prints the games on that date, for example:

```text
14 Jun 2026
1. M006 | 00:00 | Brazil vs Morocco
2. M007 | 03:00 | Haiti vs Scotland
3. M008 | 06:00 | Australia vs Turkey
```

Remember the number of the game you want.

## 4. Paste the fresh odds

Create the matching odds file:

```text
input/odds/M008.txt
```

Use this template:

```text
templates/odds_input_template.txt
```

Paste the latest OddsPortal tables under the matching headers. Keep the file
name aligned with the match ID from the list-date output.

The `ASIAN_HANDICAP` section in the template is optional. Paste it when you
have time, especially for strong favourites where `3-0`, `4-0`, and `5-0`
can be close. Missing Asian handicap data is allowed and should not change the
normal live workflow. The parser keeps the full ladder for diagnostics, but the
market-consistent challenger uses only a stable near-money subset. If the AH
ladder appears reversed relative to 1X2 prices, the workbook and dashboard show
`asian_handicap_orientation_suspicious`.
AH under-cover patterns are monitor-only. The live workflow should not fade a
favourite automatically from AH diagnostics.

## 5. Run one selected match

Set:

```python
RUN_MODE = "single_match"
DATE = "14-6"
GAME_NUMBER = 3
```

Then press **Run Python File** again.

## 6. Read the compact terminal output

The normal output should be short:

```text
Live Prediction Summary
- run profile: live
- matches processed: 1

Final recommendations:
M008 Australia vs Turkey: 0-1 | medium | review no | alt 1-2

Manual review:
none

Runtime summary:
- total: 2.34s

Outputs:
- output/submission_sheet.xlsx
- output/predictions.xlsx
```

If warnings appear, fix those before submitting.

Compact output does not print full margin tables. It only calls out Asian
handicap when the market-consistent diagnostic score shifts or creates a
manual-review reason.
MC+AH remains a gated challenger only when optimiser status is acceptable; it
does not replace the final EV recommendation.

## 7. Inspect Excel only when needed

If `review yes` appears, open:

```text
output/submission_sheet.xlsx
output/predictions.xlsx
```

Use `submission_sheet.xlsx` for the clean entry-ready view. Use
`predictions.xlsx` when you want the supporting diagnostics.

The detailed workbook includes `asian_handicap` and `margin_diagnostics` sheets
when the relevant data is available. The margin sheet explains which
goal-difference margin wins under the same group-stage EV rule used by the
scoreline optimiser. Group-stage scoring is implemented as the assumed
`10/7/5/1` rule unless the Sporza rules are confirmed to differ.

## 8. Submit the final choice

Submit the score shown in the `Final recommendations` block unless the manual
review diagnostics give you a clear reason to choose the listed alternative.
Modal/draw, round-3 favourite fade, blowout risk, and BTTS conflict are
manual-review notes only. They are not automatic override rules.

## Useful Variations

To run every odds file currently in `input/odds/`:

```python
RUN_MODE = "all_available"
```

To run every game on one date:

```python
RUN_MODE = "date"
DATE = "14-6"
```

To run specific known match IDs:

```python
RUN_MODE = "matches"
MATCH_SELECTION = "M005-M008"
```

You can also combine separate picks and ranges:

```python
MATCH_SELECTION = "M005, M008-M010"
```

Every selected match needs a matching file such as `input/odds/M008.txt`.
