"""Regression checks for the M100 envelope and actual-height power calculation."""
import importlib.util
import math
import shutil
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.m100_reaudit.physics import (
    Model, PATCH, PATCH_HASHES, PORTABLE_RUNTIME, air_velocity_enu, ensure_runtime_patch,
    relative_wind, wind_enu,
)


@pytest.fixture(scope="module")
def model():
    return Model.load()


def test_physical_mass_and_regression_feature_are_separate(model):
    result = model.assess(0.25, 10, 0, 100, 15)
    assert result.physical_mass_kg == pytest.approx(1.831 + 0.600 + 0.136 + 1.113 + 0.25)
    assert result.regression_mass_kg == pytest.approx(3.080 + 0.635 + 0.25)
    assert result.runtime_eligible and result.stock_reasons == ("ABOVE_DJI_MAXIMUM_TAKEOFF_MASS",)
    assert "B_M100_SYSTEM_LEVEL_EMPIRICAL_PUBLISHED_MODEL" in result.evidence_flags


def test_spec_limits_do_not_censor_or_relabel_experimental_inputs(model):
    result = model.assess(1.25, 20, 12, 200, 35)
    assert result.runtime_eligible
    assert "ABOVE_DJI_MAXIMUM_WIND" in result.stock_reasons
    assert "ABOVE_DJI_GPS_UNLOADED_CALM_REFERENCE" in result.evidence_flags
    assert "C_NOT_CALIBRATED" in result.evidence_flags
    assert "ATMOSPHERIC_SENSITIVITY_NO_BATTERY_THERMAL_CALIBRATION" in result.evidence_flags
    bad = model.assess(0.10, 6, 10, 25, 15)
    assert not bad.runtime_eligible and "payload" in bad.runtime_rejection
    assert model.assess(0, 5, 0, 100, 15).evidence_flags[-1] == "INTERPOLATED_SPEED_BETWEEN_PUBLISHED_LEVELS"
    assert "LIMITED_EMPIRICAL_PROXY" in model.assess(0, 0, 0, 4, 15, hover_s=15).evidence_flags


@pytest.mark.parametrize("course, direction, along, left", [(0, 0, 3, 0), (0, 180, -3, 0),
    (0, 90, 0, 3), (90, 0, 0, -3), (180, 180, 3, 0)])
def test_wind_components_relative_to_course(course, direction, along, left):
    assert relative_wind(course, 3, direction) == pytest.approx((along, left), abs=1e-12)
    wind = wind_enu(3, direction)
    assert air_velocity_enu((10, 4, 2), wind) == pytest.approx((10 - wind[0], 4 - wind[1], 2))


def test_true_phase_average_not_arbitrary_motor_losses(model):
    density = model.atmosphere(15, 100).density_kg_m3
    for phase, slope, intercept in (("takeoff", 80.4, 13.8), ("cruise", 68.9, 16.8), ("landing", 71.5, -24.3)):
        assert model.phase_power(phase, 0, density) == pytest.approx(slope * 3.715**1.5 / math.sqrt(density) + intercept)
    assert model.phase_power("HOVER", 0, density) is None
    assert model.atmosphere(35, 100).density_kg_m3 < density
    assert model.atmosphere(15, 100).pressure_pa < model.atmosphere(15, 0).pressure_pa


def test_nameplate_exhaustion_remains_visible_on_partial_energy(model):
    assert model.budget(10).status == "PARTIAL_ENERGY_UPPER_BOUND_ON_REMAINING"
    assert model.budget(10, complete_coverage=True).status == "WITHIN_NOMINAL_BUDGET"
    assert model.budget(91).status == "BELOW_PLANNED_RESERVE"
    over = model.budget(101)
    assert over.status == "NAMEPLATE_EXHAUSTED" and over.nominal_remaining_wh == pytest.approx(-1.1)
    assert over.planned_reserve_wh == pytest.approx(9.99)
    assert model.settings["battery"]["measured_usable_energy_wh"] is None


def test_version_guard_reapplies_only_to_exact_original(tmp_path):
    # The bundled, post-density fixture remains exact after later flight-guidance revisions.
    source = PORTABLE_RUNTIME
    for name in PATCH_HASHES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    ensure_runtime_patch(tmp_path)
    subprocess.run(["git", "-C", str(tmp_path), "apply", "-R", str(PATCH)], check=True)
    assert ensure_runtime_patch(tmp_path) == "APPLIED"
    assert ensure_runtime_patch(tmp_path) == "ALREADY_APPLIED"
    manager_name = "ros2_ws/src/drone_nav/drone_nav/m100_experiment_manager.py"
    subprocess.run(["git", "-C", str(tmp_path), "apply", "--include=" + manager_name,
                    str(PATCH.with_name("braking.patch"))], check=True)
    assert ensure_runtime_patch(tmp_path) == "ALREADY_APPLIED"
    with (tmp_path / "ros2_ws/src/drone_nav/drone_nav/m100_published_energy_v2.py").open("a") as stream:
        stream.write("\n# local work\n")
    with pytest.raises(RuntimeError, match="unexpected runtime source hashes"):
        ensure_runtime_patch(tmp_path)


@pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="source the ROS Jazzy environment to test the live manager callback")
def test_live_entrypoint_tracks_actual_height_and_breaks_invalid_intervals(model):
    from drone_nav.m100_experiment_manager import M100ExperimentManager
    from drone_nav.m100_published_energy_v2 import M100PublishedEnergyModel

    manager = object.__new__(M100ExperimentManager)
    manager._config = model.contract.build_config(15, 0, 0, 0, 10, 100, 500, 60)
    manager._applied_atmosphere = model.atmosphere_module.compare_density_to_15c_reference(15, 100)
    manager._state = "CLIMB"
    manager._recovering_landing = False
    manager._active_runtime_model = None
    manager._measurement_active = False
    manager._recording = True
    manager._mav = Mock()
    manager._active_payload = manager._config.payload
    manager._distance_m = 0
    manager._home_ned = (0, 0, 0)
    manager._runtime_energy = Mock()
    manager._runtime_energy.optional_wind_power_delta.return_value = None
    manager._transient_power = Mock()
    manager._transient_power.estimate.return_value = None
    manager._energy_config = model.energy
    manager._energy_model = M100PublishedEnergyModel(model.energy)
    manager._energy_model.begin_bms_session()
    manager._power_position_stamp = (None, 0.0)
    manager._propulsion = SimpleNamespace(thrust_constant_n_per_rad_s2=2.45e-5, yaw_moment_ratio_m=0.01)
    manager._campaign_sample_observer = Mock()

    def sample(t, altitude, wall_age=0, boot_s=None):
        manager._mav.snapshot.return_value = {
            "fresh": True, "armed": True, "landed": False,
            "boot_time_s": t if boot_s is None else boot_s,
            "last_local_position_monotonic": time.monotonic() - wall_age,
            "local": dict(x=0, y=0, z=-altitude, vx=0, vy=0, vz=0),
        }
        msg = SimpleNamespace(name=[], velocity=[], header=SimpleNamespace(stamp=SimpleNamespace(
            sec=int(t), nanosec=round((t - int(t)) * 1e9))))
        manager._process_rotor_joint_state(msg)
        snap, _, _, _ = manager._campaign_sample_observer.call_args.args
        return snap

    for i, z in enumerate((0, 50, 100)):
        snap = sample(1 + 0.1 * i, z)
        rho = model.atmosphere(15, z).density_kg_m3
        assert snap.air_density_kg_m3 == rho
        assert snap.electrical_power_w == pytest.approx(80.4 * 3.715**1.5 / math.sqrt(rho) + 13.8)
    expected_wh = snap.cumulative_energy_wh
    stale = sample(1.3, 100, wall_age=1)
    assert stale.electrical_power_w is None and stale.air_density_kg_m3 is None
    assert "POWER_DENSITY_POSITION_UNAVAILABLE" in stale.warnings
    resuming = sample(1.4, 99)
    assert resuming.cumulative_energy_wh == expected_wh
    assert sample(1.5, -0.1).air_density_kg_m3 == model.atmosphere(15, 0).density_kg_m3
    assert sample(1.6, -1.1).air_density_kg_m3 is None
    assert sample(2, 2, boot_s=1.6).air_density_kg_m3 is not None
    assert sample(2.2, 2, boot_s=1.6).air_density_kg_m3 is None
