"""Inspect EV-optimal scorelines for synthetic international-style mismatches."""

from __future__ import annotations

from wc_predictor.synthetic_mismatches import format_synthetic_mismatch_report, inspect_synthetic_mismatches


def main() -> None:
    """Print calibrated Poisson and EV output for transparent fair 1X2 scenarios."""

    print(format_synthetic_mismatch_report(inspect_synthetic_mismatches()))


if __name__ == "__main__":
    main()

