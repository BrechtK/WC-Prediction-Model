import pytest

from wc_predictor.backtesting import BacktestRunner
from wc_predictor.strategies import PredictionStrategy


def test_backtesting_placeholder_imports_and_explains_deferred_work() -> None:
    assert "historical" in (BacktestRunner.__doc__ or "").lower()
    with pytest.raises(NotImplementedError, match="Version 1.5"):
        BacktestRunner().run()


def test_strategy_interface_exists() -> None:
    assert PredictionStrategy.__doc__

