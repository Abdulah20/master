"""Exercise the linked SITL manager's measurement clock independently of wall speed."""
import ast
import hashlib
import importlib.util
import math
import sys
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest

from tools.m100_reaudit.braking import MANAGER, MANAGER_AFTER, ensure_braking_patch
from tools.m100_reaudit.physics import Model, PORTABLE_RUNTIME


@pytest.fixture(scope="module")
def patched_manager(tmp_path_factory):
    # The committed portable freeze predates this patch. Exercise the actual
    # manager code without requiring the ROS server, Gazebo or an active vehicle.
    root = tmp_path_factory.mktemp("px4-flight-clock")
    manager = root / MANAGER
    manager.parent.mkdir(parents=True)
    shutil.copy2(PORTABLE_RUNTIME / MANAGER, manager)
    assert ensure_braking_patch(root) == "APPLIED"
    assert ensure_braking_patch(root) == "ALREADY_APPLIED"
    assert hashlib.sha256(manager.read_bytes()).hexdigest() == MANAGER_AFTER
    return manager


def test_unknown_sources_are_preserved(tmp_path, patched_manager):
    manager = tmp_path / MANAGER
    manager.parent.mkdir(parents=True)
    manager.write_bytes(patched_manager.read_bytes() + b"# local change\n")
    before = manager.read_bytes()
    with pytest.raises(ValueError, match="Unexpected source"):
        ensure_braking_patch(tmp_path)
    assert manager.read_bytes() == before


class SimulatedFlight:
    def __init__(self, factor, speed=6., stay_moving=False):
        self.factor = factor
        self.wall = 20.
        self.sim = 50.
        self.start_sim = self.sim
        self.progress = True
        self.speed = speed
        self.stay_moving = stay_moving
        self.command = (speed, 0.)
        self.zero_seen = None
        self.commands = []
        self.positions = []
        self.age = 0.
        self.abort = False
        self.boot_override = None

    def sleep(self, period):
        self.wall += period
        if self.progress:
            self.sim += self.factor * period

    def snapshot(self):
        # Under a sustained residual, the stop command alone cannot end braking.
        speed = .25 if self.stay_moving and self.zero_seen is not None else self.command[0]
        return dict(fresh=True, heartbeat_fresh=True, mode="OFFBOARD",
                    local=dict(vx=speed, vy=self.command[1], x=7., y=9.),
                    last_local_position_monotonic=self.wall - self.age,
                    boot_time_s=self.sim if self.boot_override is None else self.boot_override(self))

    def velocity_setpoint(self, vx, vy, z, yaw):
        self.command = vx, vy
        if self.command == (0., 0.) and self.zero_seen is None:
            self.zero_seen = self.sim
        self.commands.append((self.sim, vx, vy, z, yaw))

    def position_setpoint(self, x, y, z, yaw):
        self.positions.append((x, y, z, yaw))


def bind_manager(source, flight):
    # Compile the exact patched helper/methods with their production constants.
    # The separate linked-project pytest run imports the real ROS manager.
    tree = ast.parse(source.read_text())
    methods = {"_hold_measured_hover", "_brake_before_return"}
    keep = {"_LocalPositionClock", "_ramp_return_velocity_ned"}
    clock_constants = {"M100_RETURN_HORIZONTAL_ACCEL_MPS2", "M100_RETURN_SETPOINT_PERIOD_S",
                       "M100_HOME_CENTER_SPEED_MPS", "M100_RETURN_STOP_SETTLE_S"}
    body = []
    for item in tree.body:
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name in keep:
            body.append(item)
        if isinstance(item, ast.ClassDef) and item.name == "M100ExperimentManager":
            workflow = next(m for m in item.body if getattr(m, "name", "") == "_run_full_test")
            assert any(isinstance(n, ast.Attribute) and n.attr == "_hold_measured_hover" for n in ast.walk(workflow))
            item.bases = []
            item.body = [m for m in item.body if getattr(m, "name", "") in methods]
            body.append(item)
        if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id in clock_constants for t in item.targets):
            body.append(item)
    ns = dict(math=math, M100_HOME_CENTER_SPEED_MPS=Model.load().contract.M100_HOME_CENTER_SPEED_MPS, time=SimpleNamespace(monotonic=lambda: flight.wall, sleep=flight.sleep))
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), ns)
    m = ns["M100ExperimentManager"]()
    m._mav = flight
    m._set_state = lambda *args: None

    def check_abort():
        if flight.abort:
            raise RuntimeError("operator abort")

    m._check_abort = check_abort
    return m


def test_actual_ros_manager_methods_under_slow_physics(patched_manager, monkeypatch):
    pytest.importorskip("rclpy", reason="linked ROS environment is needed for the real manager import")
    pytest.importorskip("drone_nav", reason="linked drone_nav package is needed for manager dependencies")
    module_name = "drone_nav._m100_control_clock_candidate"
    spec = importlib.util.spec_from_file_location(module_name, patched_manager)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    flight = SimulatedFlight(.30)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: flight.wall, sleep=flight.sleep))
    manager = object.__new__(module.M100ExperimentManager)
    manager._abort = threading.Event()
    manager._mav = flight
    manager._set_state = lambda *args: None
    manager._hold_measured_hover(12., -25., .5)
    assert 12. <= flight.sim - flight.start_sim < 12.1
    manager._brake_before_return(-25., .5)
    assert flight.sim - flight.zero_seen >= module.M100_RETURN_STOP_SETTLE_S


@pytest.mark.parametrize("factor", [.30, 1., 1.8])
def test_braking_slew_and_dwell_use_physics_time(patched_manager, factor):
    flight = SimulatedFlight(factor)
    manager = bind_manager(patched_manager, flight)
    manager._brake_before_return(-25., .5)
    assert flight.zero_seen is not None and flight.zero_seen - flight.start_sim >= 4. - 1e-9
    assert flight.sim - flight.zero_seen >= .75
    assert flight.sim - flight.start_sim <= 5.35
    for (old_t, old_v, *_), (new_t, new_v, _, down, yaw) in zip(flight.commands, flight.commands[1:]):
        assert 0 <= old_v - new_v <= 1.5 * min(new_t - old_t, .2) + 1e-10
        assert (down, yaw) == (-25., .5)


@pytest.mark.parametrize("factor", [.30, 1.8])
def test_homeward_leg_still_rejects_unsettled_motion(patched_manager, factor):
    flight = SimulatedFlight(factor, stay_moving=True)
    m = bind_manager(patched_manager, flight)
    with pytest.raises(RuntimeError, match="^horizontal motion did not settle before the homeward leg$"):
        m._brake_before_return(-25., 0.)
    assert 18. <= flight.sim - flight.start_sim < 18.2


@pytest.mark.parametrize("factor", [.30, 1., 1.8])
def test_hover_energy_window_follows_requested_sim_duration(patched_manager, factor):
    flight = SimulatedFlight(factor)
    m = bind_manager(patched_manager, flight)
    m._hold_measured_hover(12., -25., .25)
    assert 12. <= flight.sim - flight.start_sim < 12.2
    assert set(flight.positions) == {(7., 9., -25., .25)}


@pytest.mark.parametrize("which", ["hover", "brake"])
@pytest.mark.parametrize("failure", ["stalled", "backward", "stale", "absent", "nonfinite", "operator"])
def test_invalid_clock_or_operator_abort_never_completes(patched_manager, which, failure):
    f = SimulatedFlight(.4, speed=0.)
    m = bind_manager(patched_manager, f)
    if failure == "stalled":
        f.progress = False
    elif failure == "backward":
        f.boot_override = lambda x: x.sim if x.wall < 20.5 else x.start_sim - 1.
    elif failure == "stale":
        f.age = 2.1
    elif failure == "operator":
        f.abort = True
    else:
        f.boot_override = lambda _: None if failure == "absent" else float("nan")
    message = {"stalled": "clock stalled", "backward": "clock moved backward", "operator": "operator abort"}.get(failure, "telemetry unavailable")
    with pytest.raises(RuntimeError, match=message):
        if which == "hover":
            m._hold_measured_hover(12., -4., 0.)
        else:
            m._brake_before_return(-4., 0.)
    assert f.wall - 20. <= 2.11
