import pytest

from latedeliveries.evaluation import choose_threshold, order_cost


def test_cost_of_a_small_example():
    y_true = [1, 0, 0, 1, 0]
    flagged = [True, True, False, False, False]
    # two vouchers (2 x 4) and one missed late order (1 x 15)
    assert order_cost(y_true, flagged) == 2 * 4 + 1 * 15


def test_flagging_nobody_costs_15_per_late_order():
    assert order_cost([1, 1, 0, 0], [False] * 4) == 2 * 15


def test_flagging_everybody_costs_4_per_order():
    assert order_cost([1, 1, 0, 0], [True] * 4) == 4 * 4


def test_threshold_in_the_middle():
    # Paper check: flag the top two (cost 2 x 4 = 8). Flag all = 16. Flag none = 30.
    y_true = [1, 1, 0, 0]
    proba = [0.9, 0.6, 0.3, 0.1]
    threshold, cost = choose_threshold(y_true, proba)
    assert cost == 8
    assert 0.3 < threshold <= 0.6


def test_threshold_can_flag_everybody():
    # Everyone is late, so flagging all (4 x 4 = 16) beats flagging none (4 x 15 = 60).
    threshold, cost = choose_threshold([1, 1, 1, 1], [0.5, 0.6, 0.7, 0.8])
    assert cost == 16
    assert threshold <= 0.5


def test_threshold_can_flag_nobody():
    # Nobody is late, so flagging nobody costs 0.
    threshold, cost = choose_threshold([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
    assert cost == 0
    assert threshold > 0.4


@pytest.mark.parametrize(
    "y_true, flagged, expected",
    [
        ([1], [True], 4),
        ([1], [False], 15),
        ([0], [True], 4),
        ([0], [False], 0),
    ],
)
def test_each_of_the_four_cases(y_true, flagged, expected):
    assert order_cost(y_true, flagged) == expected
