"""Swath coverage on a grid (no Qt)."""
import coverage as C


def test_a_straight_pass_covers_its_swath():
    cells = C.covered_cells([(0.0, 0.0), (0.0, 10.0)], swath_m=4.0, cell_m=0.5)
    assert 0.9 * 4.0 * 14.0 / 0.25 < len(cells) < 1.2 * 4.0 * 14.0 / 0.25            # ~ (10 m + the round ends) x 4 m


def test_following_the_plan_covers_all_of_it_and_a_shortcut_does_not():
    plan = [(0.0, 0.0), (0.0, 10.0), (6.0, 10.0), (6.0, 0.0)]
    assert C.coverage_fraction(plan, plan, 3.0) == 1.0
    shortcut = [(0.0, 0.0), (0.0, 10.0), (6.0, 0.0)]
    assert 0.5 < C.coverage_fraction(plan, shortcut, 3.0) < 0.95
    assert C.coverage_fraction(plan, [], 3.0) == 0.0


def test_empty_plan_is_fully_covered_and_zero_swath_covers_nothing():
    assert C.coverage_fraction([], [(0.0, 0.0)], 3.0) == 1.0
    assert C.covered_cells([(0.0, 0.0)], 0.0) == set()
