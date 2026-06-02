# Odds Collection Strategy

## Purpose

This document proposes a practical odds-collection workflow for the World Cup
prediction pool. It is a design note, not an implementation change.

The goal is to collect enough market information to make the existing model
useful without creating a manual process that is too slow or error-prone. The
main trade-off is between broad, fresh coverage of the most important markets
and deeper correct-score collection for selected matches.

The project should preserve a transparent Poisson baseline. Correct-score odds
are an optional enhancement because they are directly relevant to the pool
scoring rules, but they should only influence live predictions when the source
market is sufficiently coherent and complete.

## Executive Recommendation

Use a two-tier workflow:

1. **Default workflow for every match:** collect fresh 1X2, over/under 2.5, and
   BTTS odds from a small set of individually identified bookmakers. For
   knockout matches, also collect qualification odds. Record timestamps and
   source URLs. Run the existing Poisson workflow with `w=1.0`.
2. **High-effort workflow for selected matches:** add a full listed
   correct-score market from one coherent bookmaker, preferably a sharp or
   major source. Add a second coherent bookmaker only when the match matters
   enough to justify the extra time. Inspect blend sensitivity before using a
   weight below `w=1.0`.

Do not treat the best available OddsPortal price for each scoreline as if it
were one bookmaker market. It is a useful exploratory composite, but it is not
a coherent market from which a normal bookmaker margin can be removed.

Do not normalise a manually selected handful of plausible correct scores and
blend it into the live distribution. That would condition the market
distribution on the collector's choices and can make omitted scores appear
impossibly unlikely.

## 1. Markets To Collect For Each Match

### Required Core Markets

Collect these markets for every match:

| Market | Required fields | Why it matters |
| --- | --- | --- |
| 1X2 | Team A win, draw, Team B win | Provides the main strength and draw targets. |
| Over/under 2.5 | Over 2.5, under 2.5 | Helps distinguish a low-total favourite from a high-total favourite. |
| BTTS | Yes, no | Adds information about whether both teams are likely to score. |
| Qualification | Team A qualifies, Team B qualifies | Required for knockout recommendation logic when the market is available. |

Qualification odds are separate from 1X2 odds. A knockout match can be drawn at
the relevant score horizon while one team still qualifies after extra time or
penalties.

### Source Metadata

Record these fields whenever odds are collected:

| Field | Purpose |
| --- | --- |
| `bookmaker` | Keeps each coherent market vector attributable to one source. |
| `odds_timestamp` | Makes stale or early prices visible. |
| `odds_source_url` | Allows manual verification and later audit. |
| `source_quality` | Supports review of major, sharp, exchange-derived, average-market, and composite inputs. |
| `notes` | Captures line-up uncertainty, rotation risk, standings context, or collection exceptions. |

The current World Cup template already contains these metadata fields.

### Optional Correct-Score Market

Collect correct-score odds only when using the high-effort workflow or when
automation makes complete collection cheap. The current long format is:

```text
match_id,bookmaker,score_a,score_b,decimal_odds
```

For live blending, collect every listed scoreline from the chosen bookmaker,
including any explicit `other` bucket if the source provides one. The current
matrix format does not represent an `other` bucket, so its existence should be
recorded in collection notes until a future format is designed.

## 2. Source Strategy

### Preferred Core-Market Sources

The preferred default is a small panel of individually identified bookmakers:

- one sharper source where practical;
- two to four major bookmakers with reliable coverage;
- an exchange-derived market only if its meaning and liquidity are clear.

Calculate fair probabilities within each bookmaker market first, then aggregate
the fair probabilities across bookmakers. This preserves the logic already
used by the project.

### Average Odds And Best Odds

Average-market prices from a comparison site are acceptable as a fallback when
bookmaker-level rows are unavailable. They should be labelled as an aggregate
source and should not be mixed into the same aggregation as their constituent
bookmakers.

Best odds are useful for checking market dispersion but are a poor default
model input. The best home price, best draw price, and best away price may come
from different bookmakers. Together they form a synthetic market rather than a
bookmaker's priced opinion. Margin removal becomes ambiguous and may even
produce an underround.

### Timing

Prefer the latest reliable odds available before the prediction deadline.
Closing prices are generally more informative than early prices, but the useful
operational definition is "latest prices that can still be submitted safely."

Where time permits, keep an early snapshot and a late snapshot. Large movements
deserve manual inspection because they may reflect injuries, line-ups,
rotation, or group-stage incentives.

## 3. Correct-Score Collection Approaches

### Comparison

| Approach | Benefits | Problems | Recommended use |
| --- | --- | --- | --- |
| One coherent bookmaker, full listed market | Fastest defensible live input; margin removal has a clear interpretation; easy to audit. | One source may be noisy or high-margin; source-specific shading remains. | Default high-effort correct-score workflow. |
| Two coherent bookmakers, full listed markets | Reduces reliance on one source and permits comparison of distributions. | Roughly doubles manual effort; bookmaker score grids may differ. | Important matches when time permits. |
| All listed scorelines from many bookmakers | Richest manual approximation to market consensus. | Slow, error-prone, difficult to refresh close to deadline, and grids may differ. | Prefer only after automation. |
| OddsPortal best price for each scoreline | Quick exploratory view of plausible prices and market dispersion. | Each row may come from a different bookmaker; no coherent overround; optimistic synthetic distribution. | Diagnostic only unless separately modelled and validated. |
| OddsPortal average price for each scoreline | Easier summary of broad market pricing. | The averaging method and coverage can be opaque; missing scores still matter. | Diagnostic or carefully labelled fallback. |
| Only top Poisson or likely scorelines | Lowest manual effort. | Selection bias; omitted probability mass; normalisation overstates collected scores. | Diagnostic only, not live blending. |

### Why Coherence Matters

A bookmaker's correct-score probabilities can be margin-adjusted because all
listed odds belong to one priced market. A set of best prices across
bookmakers is different: it is a bettor-facing shopping basket, not a
bookmaker-facing probability vector.

For example, the best available `1-0` price may come from bookmaker A while the
best `2-0` price comes from bookmaker B. Summing inverse odds across those rows
does not measure either bookmaker's overround. Normalising that sum creates a
convenient distribution, but not an ordinary fair market estimate.

### Partial Markets

The existing implementation conditions the direct market matrix on explicitly
supplied scorelines. This is mathematically transparent but operationally
important: if only `0-0`, `1-0`, `1-1`, and `2-0` are entered, those four
outcomes receive all direct-market probability mass after normalisation.

Therefore:

- a partial correct-score sample should not drive live blending;
- top-score samples can still be printed as diagnostics;
- sparse inputs should eventually be flagged or rejected for non-trivial
  blending;
- any future `other` score bucket needs an explicit design before use.

## 4. Modelling Implications

### Core Markets And Poisson Calibration

1X2, over/under 2.5, and BTTS constrain different aspects of the score
distribution:

- 1X2 anchors relative team strength and draw probability;
- over/under 2.5 anchors the total-goal environment;
- BTTS helps distinguish clean-sheet-heavy distributions from matches where
  both sides are likely to score.

This makes the core workflow substantially more informative than 1X2 alone,
especially for strong favourites.

### Correct-Score Markets

Correct-score odds are directly aligned with the pool scoring rules. They may
capture low-score dependence, draw shading, public-team effects, and
distribution shape that independent Poisson misses.

They also carry risks:

- margins are often high;
- liquidity can be lower than for 1X2;
- score grids may be incomplete;
- bookmaker pricing may be rounded or copied;
- a comparison-site composite can look more precise than it is;
- late collection is slower, increasing stale-price risk.

The model should treat correct-score blending as an empirical challenger to the
Poisson baseline, not an automatic upgrade.

### Margin Removal

The current proportional method is a transparent Version 1 baseline. It should
be applied within a coherent bookmaker market before bookmaker aggregation.
For correct-score markets, the workflow converts each bookmaker's odds to raw
implied probabilities, removes that bookmaker's overround across its full
available score grid, and only then aggregates fair score probabilities across
bookmakers. It never averages decimal odds directly.

The default robust aggregation policy uses a winsorized mean for five or more
bookmakers, a median for three or four bookmakers, and a mean for one or two
bookmakers. Scoreline-level outliers are detected on log fair probabilities
using the median and median absolute deviation. They remain visible in
diagnostics rather than being silently discarded.

Correct-score markets are a good candidate for future sensitivity checks
because proportional normalisation may not fully address favourite-longshot
bias across a large score grid. Alternative margin-removal methods should be
backtested before they are used in live recommendations.

## 5. Blend-Weight Policy By Data Quality

The blend is:

```text
P_final(score)
= w * P_poisson(score)
  + (1 - w) * P_market(score)
```

The following weights are governance starting points, not empirically proven
optima. Backtesting should decide whether any lower `w` improves realised pool
points out of sample.

| Correct-score input quality | Suggested live policy | Reason |
| --- | --- | --- |
| No correct-score input | `w=1.0` | Preserve the Poisson baseline. |
| Hand-picked or sparse scorelines | `w=1.0` | Do not blend a distribution conditioned on collector selection. |
| Best-price composite across bookmakers | `w=1.0` by default | Use as a diagnostic until a composite-market method is validated. |
| Average-price comparison-site composite | `w=0.9` to `1.0` only after review | Treat cautiously because coverage and averaging may be opaque. |
| One coherent major or sharp bookmaker, full listed grid | Start around `w=0.85` | Adds direct evidence while retaining a strong Poisson anchor. |
| Two coherent bookmakers, full listed grids | Consider `w=0.7` to `0.85` | Better consensus can justify more market weight after inspection. |
| Many coherent bookmakers with automated, fresh coverage | Select by out-of-sample backtest | Automation reduces manual omissions, but validation still governs the weight. |

Never use `w=0.0` in live recommendations merely because correct-score odds are
available. A pure market matrix should earn that role through data-quality
checks and backtests.

## 6. Default Manual Workflow

Use this workflow for every tournament match:

1. Open the comparison page or bookmaker pages for the match.
2. Enter individually identified bookmaker rows for 1X2, over/under 2.5, and
   BTTS into `data/raw/world_cup_odds.xlsx`.
3. For knockout matches, enter qualification odds where available.
4. Record `odds_timestamp`, `odds_source_url`, `source_quality`, and any
   relevant `notes`.
5. Prefer three to five coherent bookmaker rows. Use fewer only when coverage
   is genuinely limited.
6. Refresh prices near the submission deadline when practical.
7. Run the existing World Cup prediction workflow with the Poisson baseline.
8. Review warnings, favourite strength, EV gaps, and the strongest-favourite
   diagnostics before submission.

### Expected Time

After the collector is familiar with the sites and workbook:

| Activity | Practical estimate per match |
| --- | --- |
| Core markets from three to five bookmaker rows | 4 to 8 minutes |
| Late refresh of core markets | 2 to 4 minutes |
| Add qualification odds for a knockout match | 1 to 3 minutes |

These are planning estimates. Time the first few real collection sessions and
adjust the workflow before the tournament becomes busy.

## 7. High-Effort Workflow

Use the high-effort workflow selectively:

1. Complete the default core collection and refresh it near the deadline.
2. Choose one coherent correct-score source, preferably sharp or major.
3. Enter every listed correct score for that source.
4. Record timestamp, URL, source quality, and whether an unmodelled `other`
   bucket was displayed.
5. For the most important matches, repeat with a second coherent bookmaker.
6. Inspect Poisson top scores, market top scores, blended top scores, KL
   divergence, and recommendation sensitivity across several `w` values.
7. Use a non-trivial blend only when the correct-score coverage is coherent and
   the recommendation change is understandable.

### When To Spend The Extra Time

Prioritise the high-effort workflow for:

- Belgium matches;
- knockout matches;
- extreme favourites and unusually asymmetric matches;
- matches with a small EV gap between the best alternatives;
- matches where the modal Poisson score and EV-optimal score differ;
- matches with material late odds movement;
- matches with uncertain line-ups, rotation, or matchday-three incentives;
- pool situations where one prediction has an outsized strategic impact.

### Expected Additional Time

| Activity | Additional estimate per match |
| --- | --- |
| Full listed correct-score grid from one bookmaker | 8 to 15 minutes |
| Full listed grid from a second bookmaker | 8 to 15 minutes |
| Manual many-bookmaker correct-score consensus | 30 to 60+ minutes |

The many-bookmaker version is difficult to refresh safely by hand and is not
recommended as a live default.

## 8. Diagnostics For Correct-Score Collection

The project already reports:

- top market-implied correct scores;
- top blended correct scores;
- `D_KL(P_market || P_poisson)`;
- favourite-strength bucket;
- modal and EV-optimal scorelines;
- core market presence;
- warning flags;
- timestamps and bookmaker counts.

Before correct-score blending becomes a routine live input, inspect:

| Diagnostic | Why it matters |
| --- | --- |
| Number of collected scorelines per bookmaker | Exposes sparse inputs. |
| Score-grid range and standard-score coverage | Detects missing common outcomes such as `0-0`, `1-0`, `0-1`, and `1-1`. |
| Correct-score overround per bookmaker | Reveals suspiciously incomplete or unusually expensive markets. |
| Aggregation method and outlier examples | Makes robust aggregation decisions auditable. |
| Presence of an unrepresented `other` bucket | Prevents silent omission of tail mass. |
| Coherent bookmaker versus composite-source label | Stops best-price baskets from being treated as ordinary markets. |
| Correct-score timestamp and age | Highlights stale direct-market data. |
| Recommendation by weight grid | Shows whether the submission is sensitive to subjective blend choice. |
| First weight at which the recommendation changes | Makes blend fragility easy to review. |
| Pairwise divergence across bookmakers | Helps detect one unusual source. |
| EV gap between the top two score predictions | Identifies matches where small input changes can alter the recommendation. |

The weight-sensitivity table is especially valuable. If the recommendation is
stable from `w=1.0` through `w=0.5`, the correct-score enhancement confirms the
baseline. If it flips repeatedly, manual review matters more than selecting one
apparently precise weight.

## 9. Recommended Templates

### Existing Core Workbook

Continue using:

```text
data/raw/world_cup_odds.xlsx
```

Its current template already includes core odds, source metadata, and notes.

### Future Correct-Score Workbook

Create a user-friendly Excel companion in a later implementation step:

```text
data/raw/world_cup_correct_score_odds.xlsx
```

Recommended columns:

```text
match_id
bookmaker
score_a
score_b
decimal_odds
odds_timestamp
odds_source_url
source_quality
market_kind
has_other_bucket
notes
```

Suggested `market_kind` values:

```text
coherent_bookmaker
comparison_site_average
comparison_site_best
partial_diagnostic
```

Only `coherent_bookmaker` should be eligible for ordinary live blending by
default.

### Future Collection Checklist

A small checklist workbook or sheet would reduce omissions:

```text
match_id
core_1x2_complete
over_under_complete
btts_complete
qualification_complete_if_needed
odds_timestamp_checked
source_url_checked
correct_score_tier
correct_score_complete
submission_reviewed
notes
```

## 10. Automated Workflow Later

Automation should use a licensed odds API or data provider rather than fragile
manual scraping. A future ingestion layer should:

1. Store raw odds snapshots append-only with event, market, bookmaker,
   timestamp, and source identifiers.
2. Preserve coherent bookmaker markets before any aggregation.
3. Distinguish ordinary bookmaker rows from average-price and best-price
   composites.
4. Validate required selections and flag incomplete market vectors.
5. Track correct-score grid coverage and explicit `other` buckets.
6. Retain both early and late snapshots for research.
7. Deduplicate repeated prices without losing timestamp provenance.
8. Produce the current workbook-shaped internal data so model logic does not
   depend on one provider.
9. Respect provider licensing, rate limits, and redistribution rules.

The provider adapter should be separate from calibration and optimisation.
That keeps the model testable and allows a manual workbook fallback.

## 11. Next Implementation Priorities

The recommended implementation order is:

1. Add the correct-score Excel companion template and collection checklist.
2. Extend correct-score input metadata with timestamp, source URL,
   `source_quality`, `market_kind`, and `has_other_bucket`.
3. Add validation that warns on sparse or suspicious correct-score markets and
   prevents accidental non-trivial blending of partial diagnostics.
4. Add per-match blend sensitivity reporting across
   `w in {0, 0.25, 0.5, 0.75, 1}`.
5. Build a larger historical correct-score dataset and compare weights using
   realised pool points out of sample.
6. Test alternative correct-score margin-removal methods and inspect results by
   favourite-strength bucket.
7. Add a provider adapter only after the manual workflow has clarified the
   fields and diagnostics that matter.

## Decision Summary

The most reliable manual strategy is broad and fresh core-market collection for
every match, with full correct-score collection from one coherent bookmaker for
selected high-value matches. A second correct-score bookmaker is worthwhile
when time permits.

Best-price and partial scoreline collections are still useful, but mainly as
diagnostics. Treating them as complete fair distributions would add confidence
faster than information.
