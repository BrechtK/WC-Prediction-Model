"""Generate standalone bookmaker-margin stability diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc_predictor.overround_diagnostics import (
    DEFAULT_SOURCE_ROOT,
    DEFAULT_YEARS,
    build_overround_diagnostics,
    format_console_summary,
)

DEFAULT_OUTPUT_DIR = Path("output") / "research" / "bookmaker_margin_stability"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = build_overround_diagnostics(args.source_root, args.years)
    result.raw_summary.to_csv(output_dir / "overround_summary_by_year_market.csv", index=False)
    result.filtered_summary.to_csv(
        output_dir / "overround_summary_by_year_market_filtered.csv",
        index=False,
    )
    result.pairwise_year_differences.to_csv(
        output_dir / "overround_pairwise_year_differences.csv",
        index=False,
    )
    result.filter_audit.to_csv(output_dir / "overround_filter_audit.csv", index=False)

    print(format_console_summary(result))
    print("")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
