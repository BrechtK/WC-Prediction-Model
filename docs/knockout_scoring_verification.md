# Knockout Scoring Verification

The group-stage scoring rule is implemented and documented separately. Knockout
scoring must be verified against the actual Sporza rules before relying on
knockout EV.

## What To Check

Confirm these points in the Sporza rules:

- Which score basis is used: 90 minutes, after extra time, or final score before penalties.
- Whether qualifier points are separate from scoreline points.
- Whether exact-score points stack with goal-difference points.
- Whether a correct qualifier can score when the predicted match score implies the other team advances.

## Distinguishing Examples

Use examples like these when reading the rules:

| Prediction | Actual | Qualifier | Additive | Hierarchical |
| --- | --- | --- | --- | --- |
| `1-1`, Team A | `1-1`, Team A | correct | participation + qualifier + exact + goal-difference | participation + qualifier + exact |
| `2-1`, Team A | `3-2`, Team A | correct | participation + qualifier + goal-difference | participation + qualifier + goal-difference |
| `1-0`, Team A | `0-0`, Team A | correct | participation + qualifier | participation + qualifier |

If exact score supersedes goal difference, use:

```python
KnockoutScoringConfig(knockout_scoring_mode="hierarchical")
```

If score components stack, use:

```python
KnockoutScoringConfig(knockout_scoring_mode="additive")
```

Until confirmed, keep the default:

```python
KnockoutScoringConfig(knockout_scoring_mode="unverified")
```

`unverified` preserves the existing additive EV calculation but emits knockout
warning flags so live and research reports do not imply the rule has been
confirmed.
