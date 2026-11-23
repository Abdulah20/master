import pytest

from tools.m100_reaudit.cruise_native_inputs import reference_grid


def test_native_grid_uses_commanded_altitude_and_frozen_residual():
    cells = reference_grid()
    supported = cells[.25, 50, 6]
    assert supported["density_kg_m3"] == pytest.approx(1.2191306315968917)
    assert supported["regression_mass_kg"] == pytest.approx(3.965)
    assert supported["baseline_W"] == pytest.approx(509.47314230612324)
    assert supported["legacy_W"] == pytest.approx(491.10375730241384)
    assert cells[.25, 100, 6]["density_kg_m3"] < supported["density_kg_m3"]
    assert cells[.25, 100, 6]["baseline_W"] > supported["baseline_W"]
    assert cells[.5, 50, 6]["baseline_W"] > supported["baseline_W"]
    for speed in (5, 10, 12, 15, 20):
        outside = cells[.25, 50, speed]
        assert outside["legacy_adjustment_W"] is None
        assert outside["legacy_W"] == outside["baseline_W"]


def test_route_duration_gate_and_hover_scope():
    cells = reference_grid(payloads=(.25,), altitudes=(50,), speeds=(6, 8), route_length_m=150)
    assert cells[.25, 50, 6]["legacy_adjustment_W"] is not None
    assert cells[.25, 50, 8]["legacy_adjustment_W"] is None
    with pytest.raises(ValueError, match="positive commanded speed"):
        reference_grid(speeds=(0,))
