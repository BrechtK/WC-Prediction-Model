"""Backtesting command placeholder for Version 1.5."""

from __future__ import annotations

from wc_predictor.backtesting import BacktestRunner


def main() -> None:
    """Explain the intentionally deferred backtesting implementation."""

    try:
        BacktestRunner().run()
    except NotImplementedError as exc:
        print(exc)
        print("See src/wc_predictor/backtesting.py for the planned historical evaluation workflow.")


if __name__ == "__main__":
    main()

