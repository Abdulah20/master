import numpy as np

from tools.m100_reaudit.cruise_condition_scores import date_bootstrap_weights


def test_paired_bootstrap_resamples_whole_dates_and_weights_each_observed_flight():
    rows = [{"date": d, "speed_mps": 4} for d in ("a", "a", "b")]
    point, draws = date_bootstrap_weights(rows, "flight")
    errors = np.array([10., 10., -10.])
    np.testing.assert_allclose(point @ errors, 10./3)
    np.testing.assert_equal(draws[:, 0], draws[:, 1])
    np.testing.assert_allclose(np.unique(np.round(draws @ errors, 6)), [-10., 3.333333, 10.])


def test_balanced_speed_draws_have_all_three_speeds_at_fixed_weights():
    rows = [{"date": d, "speed_mps": s} for d, s in (("a", 4), ("a", 4), ("b", 6), ("c", 8))]
    point, draws = date_bootstrap_weights(rows, "equal_speed")
    np.testing.assert_allclose(point, [1/6, 1/6, 1/3, 1/3])
    assert 0 < len(draws) < 10000
    np.testing.assert_allclose(draws[:, :2].sum(axis=1), 1/3)
    np.testing.assert_allclose(draws[:, 2:], 1/3)


def test_one_date_yields_a_point_without_a_date_interval():
    point, draws = date_bootstrap_weights([{"date": "a", "speed_mps": 6}], "flight")
    np.testing.assert_array_equal(point, [1.0])
    assert not len(draws)
