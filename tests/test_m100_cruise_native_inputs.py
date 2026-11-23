import pytest

from tools.m100_reaudit.cruise_native_inputs import reference_grid
from tools.m100_reaudit.mission_campaign import active_cruise_config


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


def test_acquisition_pins_the_profile_selected_by_the_target_checkout(tmp_path, monkeypatch):
    module = tmp_path / "drone_nav"
    module.mkdir()
    (module / "__init__.py").touch()
    selector = module / "m100_stable_cruise_speed.py"
    selector.write_text("def profile_config_name(profile):\n    return profile + '_from_target.yaml'\n")
    monkeypatch.delenv("M100_STABLE_CRUISE_PROFILE", raising=False)
    assert active_cruise_config(tmp_path) == tmp_path / "config/native_from_target.yaml"
    monkeypatch.setenv("M100_STABLE_CRUISE_PROFILE", "legacy")
    assert active_cruise_config(tmp_path) == tmp_path / "config/legacy_from_target.yaml"
    selector.write_text("# Historical checkout with one supported profile.\n")
    assert active_cruise_config(tmp_path) == tmp_path / "config/m100_stable_cruise_speed_v1.yaml"
