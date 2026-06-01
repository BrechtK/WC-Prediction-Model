import numpy as np
import pytest

from wc_predictor.margin import remove_margin


def test_proportional_margin_removal_sums_to_one() -> None:
    result = remove_margin([0.55, 0.30, 0.25])
    assert result.overround == pytest.approx(1.10)
    assert np.allclose(result.fair_probabilities, np.array([0.55, 0.30, 0.25]) / 1.10)
    assert result.fair_probabilities.sum() == pytest.approx(1.0)


def test_suspicious_overround_creates_warning() -> None:
    result = remove_margin([0.70, 0.40, 0.20], suspicious_high=1.20)
    assert result.warnings


@pytest.mark.parametrize("method", ["additive", "power"])
def test_additional_margin_methods_are_pluggable(method: str) -> None:
    result = remove_margin([0.55, 0.30, 0.25], method)
    assert result.fair_probabilities.sum() == pytest.approx(1.0)

