# Version 1 Mathematical Validation Notes

## Scope

The Version 1 baseline was reviewed before adding new features. The audit
covered scoring, odds processing, Poisson calibration, EV optimisation, and
finite-grid tail handling.

## Confirmed Rules

- Group scoring is mutually exclusive and ordered correctly: exact score `10`,
  non-exact correct goal difference `7`, correct result with wrong difference
  `5`, otherwise participation-only `1`.
- Draw predictions follow the same logic: exact draws earn `10`, non-exact draws
  earn `7`, and a predicted draw against a non-draw earns `1`.
- Knockout scoring keeps qualification separate from the score. Under the
  baseline additive config, participation always contributes `1`, the correct
  qualifier contributes `10`, exact score contributes `6`, and correct goal
  difference contributes `4`. An exact score also has the correct goal
  difference, so both score components apply.
- Group EV diagnostics are cumulative and reconcile algebraically:

```text
EV = 1
   + 4 P(correct result)
   + 2 P(correct goal difference)
   + 3 P(exact score)
```

- Knockout EV reconciles with the configured additive formula:

```text
EV = 1
   + 10 P(correct qualifier)
   + 6 P(exact score)
   + 4 P(correct goal difference)
```

## Corrections Made

Calibration previously evaluated fit against the truncated finite score grid.
That approximation is tiny for typical lambdas with an `8-8` grid, but it is
not mathematically exact. Calibration now uses full independent-Poisson market
summaries: Skellam outcome probabilities, Poisson total-goals probability, and
the closed-form BTTS probability. A regression test deliberately uses a small
grid with large tail mass and confirms that fitted lambdas remain correct.

Bookmaker aggregation now aggregates complete market vectors rather than
individual outcome columns. This prevents incomplete optional markets from
mixing probabilities from different bookmaker subsets.

EV evaluation now rejects raw truncated matrices. The default finite score grid
is renormalised before optimisation, and omitted raw tail mass is reported.
Raw matrices remain valid diagnostic objects when their grid mass plus tail mass
sums to one.

Knockout EV now validates that qualification probabilities are finite, bounded,
and sum to one.

## Test Coverage

The regression suite covers decimal-odds conversion, overround warnings,
proportional margin removal, market aggregation, scoring edge cases, algebraic
EV reconciliation, modal-score versus EV-optimal divergence, full-distribution
calibration, optional over/under and BTTS constraints, positive lambda bounds,
tail reporting, grid renormalisation semantics, and invalid qualifier inputs.
