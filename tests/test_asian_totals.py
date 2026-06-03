from __future__ import annotations

import pytest

from wc_predictor.asian_totals import (
    asian_total_components,
    asian_total_expected_profit_from_distribution,
    asian_total_profit,
    asian_total_profit_vector,
)


def test_half_goal_total_pricing_has_no_push() -> None:
    assert asian_total_components(2.5) == ((2.5, 1.0),)
    assert asian_total_profit(3, 2.5, "over", 2.0) == pytest.approx(1.0)
    assert asian_total_profit(2, 2.5, "over", 2.0) == pytest.approx(-1.0)
    assert asian_total_profit(2, 2.5, "under", 1.80) == pytest.approx(0.80)
    assert asian_total_profit(3, 2.5, "under", 1.80) == pytest.approx(-1.0)


def test_integer_asian_total_pricing_accounts_for_push() -> None:
    assert asian_total_components(3.0) == ((3.0, 1.0),)
    assert asian_total_profit(4, 3.0, "over", 1.95) == pytest.approx(0.95)
    assert asian_total_profit(3, 3.0, "over", 1.95) == pytest.approx(0.0)
    assert asian_total_profit(2, 3.0, "over", 1.95) == pytest.approx(-1.0)
    assert asian_total_profit(2, 3.0, "under", 1.95) == pytest.approx(0.95)
    assert asian_total_profit(3, 3.0, "under", 1.95) == pytest.approx(0.0)


def test_quarter_total_pricing_splits_stake_across_adjacent_lines() -> None:
    assert asian_total_components(2.25) == ((2.0, 0.5), (2.5, 0.5))
    assert asian_total_components(2.75) == ((2.5, 0.5), (3.0, 0.5))
    assert asian_total_profit(2, 2.25, "over", 2.0) == pytest.approx(-0.5)
    assert asian_total_profit(3, 2.25, "over", 2.0) == pytest.approx(1.0)
    assert asian_total_profit(3, 2.75, "under", 2.0) == pytest.approx(-0.5)
    assert asian_total_profit(2, 2.75, "under", 2.0) == pytest.approx(1.0)


def test_expected_profit_uses_profit_equation_not_binary_probability() -> None:
    # Total-goals distribution: P(0)=0.20, P(1)=0.20, P(2)=0.20, P(3)=0.20, P(4)=0.20.
    distribution = [0.20, 0.20, 0.20, 0.20, 0.20]
    # Over 2.0 at decimal 2.0 has wins at 3+, losses at 0-1, and a push at exactly 2.
    assert asian_total_expected_profit_from_distribution(distribution, 2.0, "over", 2.0) == pytest.approx(0.0)

    vector = asian_total_profit_vector((3, 3), 2.0, "over", 2.0)
    assert vector[2, 0] == pytest.approx(0.0)
    assert vector[2, 1] == pytest.approx(1.0)
    assert vector[0, 1] == pytest.approx(-1.0)


def test_invalid_total_line_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer, half-goal, or quarter-goal"):
        asian_total_components(2.1)
    with pytest.raises(ValueError, match="side"):
        asian_total_profit(3, 2.5, "yes", 2.0)
    with pytest.raises(ValueError, match="shape"):
        asian_total_profit_vector((0, 3), 2.5, "over", 2.0)
