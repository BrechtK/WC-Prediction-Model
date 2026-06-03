# Historical Backtesting Data

The group-stage historical backtester accepts Football-Data.co.uk-like CSV
files. It maps home teams to internal team A, away teams to internal team B,
and full-time goals to the realised group-style score.

Folder inputs are scanned recursively. For local research data, use
`input/historical/` with league subfolders such as `Belgium/` and `England/`;
exported `source_file` values keep the relative nested path.

Football-Data column availability varies by league and season. Older files may
use `BbAv*` aggregate columns, while newer files may use `Avg*` columns. The
table below distinguishes common source columns from columns currently consumed
by the loader.

## Match And Result Columns

| Column | Meaning | Current loader usage |
| --- | --- | --- |
| `Div` | Division or league code | Informational only |
| `Date` | Match date | Loaded when present |
| `HomeTeam` | Home team | Required; mapped to internal team A |
| `AwayTeam` | Away team | Required; mapped to internal team B |
| `FTHG` | Full-time home goals | Required; mapped to team A goals |
| `FTAG` | Full-time away goals | Required; mapped to team B goals |
| `FTR` | Full-time result: home, draw, or away | Informational only; scoring uses `FTHG` and `FTAG` |

## Common 1X2 Odds Columns

`H`, `D`, and `A` mean home win, draw, and away win.

| Columns | Meaning | Current loader usage |
| --- | --- | --- |
| `B365H`, `B365D`, `B365A` | Bet365 opening 1X2 odds | Supported |
| `BWH`, `BWD`, `BWA` | Bwin 1X2 odds | Supported; represented internally by the legacy label `Betway` |
| `IWH`, `IWD`, `IWA` | Interwetten 1X2 odds | Supported |
| `LBH`, `LBD`, `LBA` | Ladbrokes 1X2 odds | Not currently consumed |
| `PSH`, `PSD`, `PSA` | Pinnacle / PS opening 1X2 odds | Supported |
| `WHH`, `WHD`, `WHA` | William Hill 1X2 odds | Supported |
| `VCH`, `VCD`, `VCA` | VC Bet 1X2 odds | Supported |
| `AvgH`, `AvgD`, `AvgA` | Newer-format average market 1X2 odds | Supported and preferred |
| `BbAvH`, `BbAvD`, `BbAvA` | Older-format average market 1X2 odds | Not currently consumed |
| `BbMxH`, `BbMxD`, `BbMxA` | Older-format maximum market 1X2 odds | Not currently consumed |
| `PSCH`, `PSCD`, `PSCA` | Pinnacle / PS closing 1X2 odds | Not currently consumed |

Maximum odds are not treated as fair average market prices. Closing odds are
also intentionally separate from the current opening-odds baseline.

## Common Over/Under 2.5 Columns

| Columns | Meaning | Current loader usage |
| --- | --- | --- |
| `B365>2.5`, `B365<2.5` | Bet365 over/under 2.5 odds | Supported |
| `P>2.5`, `P<2.5` | Pinnacle / PS over/under 2.5 odds | Supported |
| `Avg>2.5`, `Avg<2.5` | Newer-format average market over/under 2.5 odds | Supported and preferred |
| `BbAv>2.5`, `BbAv<2.5` | Older-format average market over/under 2.5 odds | Not currently consumed |
| `BbMx>2.5`, `BbMx<2.5` | Older-format maximum market over/under 2.5 odds | Not currently consumed |

## Current Loader Priority

The current implementation applies the following rule independently for 1X2
and over/under 2.5 markets:

1. If a complete valid newer-format average market exists, use only that market:
   `AvgH`, `AvgD`, `AvgA` for 1X2 or `Avg>2.5`, `Avg<2.5` for totals.
2. Otherwise retain every complete valid supported bookmaker market available
   for that row.
3. Aggregate retained bookmaker probabilities downstream using the configured
   bookmaker aggregation method. The default is the mean of fair probabilities
   after margin removal.
4. If over/under odds are unavailable, the `ev_optimal_1x2_over_under` strategy
   falls back to the 1X2-calibrated score matrix.
5. If full-time goals are missing or invalid, skip the match for every strategy
   and report `missing or invalid full-time result`.
6. If no supported valid 1X2 market is available, fixed-score strategies still
   run, while market-dependent strategies skip the match and report
   `missing or invalid 1X2 odds`.

A market is valid only when every required decimal odd is finite and greater
than `1.0`.

## Older Belgium-Style Files

Files such as `B1_2016_2017.csv` expose older aggregate columns including
`BbAvH`, `BbAvD`, `BbAvA`, `BbAv>2.5`, and `BbAv<2.5`. The current loader does
not consume those older aggregate aliases. It falls back to supported bookmaker
columns such as Bet365, Bwin, Interwetten, Pinnacle, William Hill, and VC Bet
when they are present.
