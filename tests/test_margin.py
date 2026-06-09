import numpy as np
import pytest

from wc_predictor.margin import remove_margin


def test_proportional_margin_removal_sums_to_one() -> None:
    result = remove_margin([0.55, 0.30, 0.25])
    assert result.overround == pytest.approx(1.10)
    assert np.allclose(result.fair_probabilities, np.array([0.55, 0.30, 0.25]) / 1.10)
    assert result.fair_probabilities.sum() == pytest.approx(1.0)
    assert result.method == "normalised_inverse_odds"


def test_normalised_inverse_odds_reproduces_existing_proportional_method() -> None:
    raw = [0.55, 0.30, 0.25]
    current = remove_margin(raw, "normalised_inverse_odds")
    legacy = remove_margin(raw, "proportional")

    assert current.fair_probabilities == pytest.approx(np.array(raw) / sum(raw))
    assert legacy.fair_probabilities == pytest.approx(current.fair_probabilities)


def test_suspicious_overround_creates_warning() -> None:
    result = remove_margin([0.70, 0.40, 0.20], suspicious_high=1.20)
    assert result.warnings


def test_below_one_overround_creates_warning() -> None:
    result = remove_margin([0.40, 0.30, 0.20])
    assert result.warnings


@pytest.mark.parametrize("method", ["additive", "power"])
def test_additional_margin_methods_are_pluggable(method: str) -> None:
    result = remove_margin([0.55, 0.30, 0.25], method)
    assert result.fair_probabilities.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("raw", [[0.55, 0.55], [0.55, 0.30, 0.25]])
def test_shin_margin_removal_produces_positive_probabilities(raw: list[float]) -> None:
    result = remove_margin(raw, "shin")

    assert result.fair_probabilities.sum() == pytest.approx(1.0)
    assert (result.fair_probabilities > 0).all()
    assert result.method == "shin"
    assert result.diagnostics is not None
    assert 0 <= result.diagnostics["shin_z"] < 1


def test_shin_margin_removal_fails_gracefully_for_pathological_market() -> None:
    with pytest.raises(ValueError, match="high overround"):
        remove_margin([0.95, 0.95], "shin")


def test_power_method_probabilities_are_positive_for_two_and_three_way_markets() -> None:
    for raw in ([0.55, 0.55], [0.55, 0.30, 0.25]):
        result = remove_margin(raw, "power")
        assert result.fair_probabilities.sum() == pytest.approx(1.0)
        assert (result.fair_probabilities > 0).all()


def test_additive_method_works_for_normal_cases_and_fails_when_negative() -> None:
    result = remove_margin([0.55, 0.30, 0.25], "additive")
    expected = np.array([0.55, 0.30, 0.25]) - 0.10 / 3

    assert result.fair_probabilities == pytest.approx(expected)
    with pytest.raises(ValueError, match="non-positive"):
        remove_margin([1.20, 0.01, 0.01], "additive")
