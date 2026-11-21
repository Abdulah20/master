"""ROS 2 backend for the existing Gazebo UAV Experiment dashboard.

The node keeps the GUI transport contract stable while replacing the legacy
SJTU vehicle backend with PX4 SITL and the physically distinct M100 models.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Any, Callable

from ament_index_python.packages import get_package_share_directory
import pymavlink
from pymavlink import mavutil
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .m100_contract import (
    CSV_FIELDS,
    CSV_SCHEMA,
    CSV_SAMPLE_INTERVAL_S,
    PAYLOAD_MODELS,
    PX4_ROOT,
    STATIC_THRUST_RESERVE_BASIS,
    STATIC_THRUST_RESERVE_DENSITY_BASIS,
    STATIC_THRUST_RESERVE_DENSITY_STATUS,
    STATIC_THRUST_RESERVE_STATUS,
    WORLD_NAME,
    ExperimentConfig,
    PayloadModel,
    M100_FINAL_LANDING_TOLERANCE_M,
    M100_HOME_CENTER_SETTLE_S,
    M100_HOME_CENTER_SPEED_MPS,
    M100_HOME_CENTER_TOLERANCE_M,
    M100_OFFBOARD_DESCENT_TARGET_M,
    landing_timeout_limit,
    m100_v2_energy_assessment,
    measurement_energy_per_distance,
    parse_apply_command,
    static_thrust_reserve,
)
from .m100_atmosphere import (
    AtmosphereDensityComparison,
    DryAirState,
    MAX_ALTITUDE_M_MSL,
    compare_density_to_15c_reference,
    dry_air_state,
    is_reference_temperature_c,
)
from .m100_published_energy_v2 import (
    M100PublishedEnergyModel,
    PublishedEnergyConfig,
    ROTOR_JOINT_NAMES,
)
from .m100_propulsion_v2 import M100PropulsionV2
from .m100_thevenin import TheveninProfile
from .m100_transient_power import (
    MODEL_ID as TRANSIENT_POWER_MODEL_ID,
    STATUS as TRANSIENT_POWER_STATUS,
    WARNING as TRANSIENT_POWER_WARNING,
    TransientPowerEstimator,
    TransientPowerFit,
)
from .m100_wind_simulation import (
    MODEL_ID as WIND_SIMULATION_MODEL_ID,
    WARNING as WIND_SIMULATION_WARNING,
    WindSimulation,
)
from .m100_stable_cruise_speed import (
    MODEL_STATUS as STABLE_CRUISE_MODEL_STATUS,
    MODEL_WARNING as STABLE_CRUISE_MODEL_WARNING,
    StableCruiseSpeedFit,
)
from .m100_temperature_sdf import (
    GeneratedTemperatureModel,
    generate_temperature_model,
)
from .m100_runtime_energy_v2 import (
    M100RuntimeEnergyV2,
    WIND_DELTA_STATUS,
)


PROJECT_ROOT = Path("/home/abdulah/Master/Drone/project")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CSV_ROOT = OUTPUT_ROOT / "m100_csv"
LOG_ROOT = OUTPUT_ROOT / "m100_logs"
TEMPERATURE_MODEL_ROOT = OUTPUT_ROOT / "m100_temperature_models"
PX4_BINARY = PX4_ROOT / "build/px4_sitl_default/bin/px4"
PX4_ROOTFS = PX4_ROOT / "build/px4_sitl_default/rootfs"
PX4_MODELS_ROOT = PX4_ROOT / "Tools/simulation/gz/models"
MODEL_VALIDATOR = (
    PX4_ROOT / "Tools/simulation/gz/models/m100_cmu_v2/validate_m100_cmu_v2.py"
)
_SOURCE_ENERGY_CONFIG = (
    Path(__file__).resolve().parents[1] / "config" / "m100_energy_model_v2.yaml"
)
ENERGY_CONFIG = (
    _SOURCE_ENERGY_CONFIG if _SOURCE_ENERGY_CONFIG.is_file()
    else Path(get_package_share_directory("drone_nav"))
    / "config"
    / "m100_energy_model_v2.yaml"
)
_SOURCE_GONG_CONFIG = (
    Path(__file__).resolve().parents[1] / "config" / "m100_gong_power_model.yaml"
)
GONG_CONFIG = (
    _SOURCE_GONG_CONFIG if _SOURCE_GONG_CONFIG.is_file()
    else Path(get_package_share_directory("drone_nav")) / "config" / "m100_gong_power_model.yaml"
)
_STABLE_CRUISE_CONFIG_NAME = "m100_stable_cruise_speed_v1.yaml"
_SOURCE_STABLE_CRUISE_CONFIG = (
    Path(__file__).resolve().parents[1] / "config" / _STABLE_CRUISE_CONFIG_NAME
)
STABLE_CRUISE_ENERGY_CONFIG = (
    _SOURCE_STABLE_CRUISE_CONFIG
    if _SOURCE_STABLE_CRUISE_CONFIG.is_file()
    else Path(get_package_share_directory("drone_nav")) / "config" / _STABLE_CRUISE_CONFIG_NAME
)

STATES = {
    "IDLE", "CONFIGURING", "SPAWNING", "PX4_CONNECTING", "READY",
    "ARMING", "TAKEOFF", "CLIMB", "STABILIZING", "MEASURING",
    "CRUISE", "HOVER", "RETURN_BRAKING", "RETURNING", "LANDING", "COMPLETE", "RESETTING",
    "ABORTED", "ERROR",
}
BUSY_CONFIGURATION_STATES = {"CONFIGURING", "SPAWNING", "PX4_CONNECTING"}
FLIGHT_STATES = {
    "ARMING", "TAKEOFF", "CLIMB", "STABILIZING", "MEASURING",
    "CRUISE", "HOVER", "RETURN_BRAKING", "RETURNING", "LANDING",
}
TEMPERATURE_EFFECT_SCOPE = "ROTOR_DENSITY_FIXED_ALTITUDE_APPROXIMATION"
BMS_ENERGY_SCOPE = "RUN_RELATIVE_SUPPORTED_PHASES_ONLY"
BMS_SUPPORTED_PHASES = (
    "TAKEOFF",
    "CLIMB",
    "CRUISE",
    "RETURNING",
    "LANDING",
    "STABILIZING",
    "RETURN_BRAKING",
    "HOVER",
)
BMS_LIMITED_HOVER_POLICY = "PUBLISHED_MEAN_IN_ITS_ENVELOPE_OR_CMU_KINEMATIC_SIMULATION_TRANSFER"
BMS_UNSUPPORTED_PHASE_POLICY = "NOT_INTEGRATED_AND_CONTINUITY_IS_BROKEN"
M100_RETURN_SPEED_CAP_MPS = 4.0
M100_RETURN_DEFAULT_SPEED_MPS = 2.0
M100_RETURN_CAPTURE_RADIUS_M = 0.75
M100_RETURN_SLOWDOWN_HORIZON_S = 2.0
# Supervisory setpoint limits. PX4 remains responsible for vehicle control.
M100_RETURN_HORIZONTAL_ACCEL_MPS2 = 1.5
M100_RETURN_SPEED_HEADROOM = 0.90
M100_RETURN_STOP_SETTLE_S = 0.75
M100_RETURN_SETPOINT_PERIOD_S = 0.1


def _safe_return_speed_limit_mps(commanded_speed_mps: float) -> float:
    """Resolve a bounded return speed without changing experiment inputs."""
    speed = float(commanded_speed_mps)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("return speed must be finite and non-negative")
    requested = speed if speed > 0.0 else M100_RETURN_DEFAULT_SPEED_MPS
    return min(requested, M100_RETURN_SPEED_CAP_MPS)


def _limited_return_velocity_ned(
    current_north_m: float,
    current_east_m: float,
    home_north_m: float,
    home_east_m: float,
    speed_limit_mps: float,
) -> tuple[float, float]:
    """Return an XY NED velocity command that cannot exceed ``speed_limit_mps``."""
    values = (
        current_north_m,
        current_east_m,
        home_north_m,
        home_east_m,
        speed_limit_mps,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("return guidance inputs must be finite")
    if speed_limit_mps <= 0.0:
        raise ValueError("return speed limit must be positive")

    north_error = home_north_m - current_north_m
    east_error = home_east_m - current_east_m
    distance = math.hypot(north_error, east_error)
    if distance <= M100_RETURN_CAPTURE_RADIUS_M:
        return 0.0, 0.0

    # Reduce the velocity target before the position-controller capture zone.
    # The final position step is therefore below 0.75 m, not the full route.
    speed = min(speed_limit_mps, distance / M100_RETURN_SLOWDOWN_HORIZON_S)
    return speed * north_error / distance, speed * east_error / distance


def _ramp_return_velocity_ned(
    current: tuple[float, float],
    target: tuple[float, float],
    elapsed_s: float,
) -> tuple[float, float]:
    """Limit the vector setpoint change, including across both horizontal axes."""
    north_change = target[0] - current[0]
    east_change = target[1] - current[1]
    change = math.hypot(north_change, east_change)
    # Avoid a large setpoint step after an unusually late control tick.
    dt = min(elapsed_s, 2.0 * M100_RETURN_SETPOINT_PERIOD_S)
    max_change = M100_RETURN_HORIZONTAL_ACCEL_MPS2 * dt
    if change <= max_change:
        return target
    fraction = max_change / change
    return current[0] + fraction * north_change, current[1] + fraction * east_change


def _battery_budget_summary(model_energy: Any) -> dict[str, Any]:
    """Return a JSON-safe, scientifically bounded battery-budget record."""
    return {
        "scope": BMS_ENERGY_SCOPE,
        "is_full_flight_energy": False,
        "supported_phases": list(BMS_SUPPORTED_PHASES),
        "limited_hover_policy": BMS_LIMITED_HOVER_POLICY,
        "unsupported_phase_policy": BMS_UNSUPPORTED_PHASE_POLICY,
        "state": model_energy.bms_state,
        "safety_level": model_energy.bms_safety_level,
        "budget_quality": model_energy.bms_data_quality,
        "estimator_mode": model_energy.battery_estimator_mode,
        "power_source": model_energy.bms_power_source,
        "soc_source": model_energy.bms_soc_source,
        "consumed_supported_phase_energy_wh": (
            model_energy.battery_consumed_energy_wh
        ),
        "consumed_capacity_ah": model_energy.consumed_capacity_ah,
        "remaining_nameplate_energy_wh": (
            model_energy.battery_nameplate_remaining_wh
        ),
        "remaining_capacity_ah": model_energy.battery_remaining_capacity_ah,
        "run_relative_soc_percent": model_energy.soc_percent,
        "endurance_min": model_energy.endurance_min,
        "dynamic_model_enabled": model_energy.battery_dynamic_model_enabled,
        "advisory_only": model_energy.bms_advisory_only,
        "terminal_voltage_in_envelope": (
            model_energy.battery_terminal_voltage_in_envelope
        ),
        "pack_identity_status": model_energy.battery_pack_identity_status,
        "warnings": list(model_energy.warnings),
    }


@dataclass(frozen=True)
class TemperatureRuntimeSelection:
    """One canonical or density-scaled model selected for an APPLY case."""

    payload: PayloadModel
    px4_target: str
    gazebo_model: str
    atmosphere: AtmosphereDensityComparison
    generated_model: GeneratedTemperatureModel | None

    @property
    def density_ratio(self) -> float:
        return self.atmosphere.density_ratio


class AbortRequested(RuntimeError):
    pass


class MavlinkBackend:
    """Thread-safe MAVLink telemetry receiver and PX4 command adapter."""

    REQUIRED_PYMAVLINK_VERSION = "2.4.50"

    MESSAGE_INTERVALS_US = {
        "LOCAL_POSITION_NED": 50_000,
        "ATTITUDE": 50_000,
        "EXTENDED_SYS_STATE": 100_000,
        "SYS_STATUS": 200_000,
        "ACTUATOR_OUTPUT_STATUS": 50_000,
        "ESC_STATUS": 100_000,
    }

    def __init__(self) -> None:
        self._connection = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._telemetry: dict[str, Any] = {
            "connected": False,
            "last_message_monotonic": 0.0,
            "last_heartbeat_monotonic": 0.0,
            "last_local_position_monotonic": None,
            "heartbeat_generation": 0,
            "armed": False,
            "landed": True,
            "mode": "UNKNOWN",
            "local": None,
            "attitude": None,
            "voltage_v": None,
            "current_a": None,
            "battery_pct": None,
            "actuator": [None] * 4,
            "motor": [None] * 4,
        }

    def connect(self, timeout_s: float = 45.0) -> None:
        if pymavlink.__version__ != self.REQUIRED_PYMAVLINK_VERSION:
            raise RuntimeError(
                "M100 requires pymavlink "
                f"{self.REQUIRED_PYMAVLINK_VERSION} for authoritative PX4 "
                f"custom-mode decoding; found {pymavlink.__version__}"
            )
        self.close()
        self._stop.clear()
        connection = mavutil.mavlink_connection(
            "udpin:0.0.0.0:14540",
            source_system=255,
            source_component=mavutil.mavlink.MAV_COMP_ID_MISSIONPLANNER,
        )
        heartbeat = connection.wait_heartbeat(timeout=timeout_s)
        if heartbeat is None:
            connection.close()
            raise RuntimeError("PX4 MAVLink heartbeat was not received")
        connection.target_component = 1
        with self._lock:
            self._connection = connection
            self._telemetry["connected"] = True
            self._update_message(heartbeat)
        self._request_intervals()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def close(self) -> None:
        self._stop.set()
        reader = self._reader
        if reader and reader.is_alive() and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        self._reader = None
        with self._lock:
            connection = self._connection
            self._connection = None
            self._telemetry["connected"] = False
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def _request_intervals(self) -> None:
        connection = self._require_connection()
        for name, interval_us in self.MESSAGE_INTERVALS_US.items():
            message_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}", None)
            if message_id is None:
                continue
            connection.mav.command_long_send(
                connection.target_system,
                1,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                message_id,
                interval_us,
                0, 0, 0, 0, 0,
            )

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                connection = self._require_connection()
                message = connection.recv_match(blocking=True, timeout=0.2)
                if message is not None and message.get_type() != "BAD_DATA":
                    with self._lock:
                        self._update_message(message)
            except Exception:
                if not self._stop.is_set():
                    time.sleep(0.1)

    def _update_message(self, message: Any) -> None:
        kind = message.get_type()
        if (
            kind == "HEARTBEAT"
            and int(getattr(message, "autopilot", -1))
            != mavutil.mavlink.MAV_AUTOPILOT_PX4
        ):
            # The endpoint is intended for one PX4 autopilot. Ignore GCS or
            # companion heartbeats so they cannot overwrite flight mode/armed.
            return
        self._telemetry["last_message_monotonic"] = time.monotonic()
        if kind == "HEARTBEAT":
            self._telemetry["last_heartbeat_monotonic"] = time.monotonic()
            self._telemetry["heartbeat_generation"] += 1
            self._telemetry["armed"] = bool(
                message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            self._telemetry["mode"] = mavutil.mode_string_v10(message)
        elif kind == "EXTENDED_SYS_STATE":
            self._telemetry["landed"] = (
                message.landed_state == mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND
            )
        elif kind == "LOCAL_POSITION_NED":
            self._telemetry["last_local_position_monotonic"] = time.monotonic()
            self._telemetry["local"] = {
                key: float(getattr(message, key))
                for key in ("x", "y", "z", "vx", "vy", "vz")
            }
            self._telemetry["boot_time_s"] = float(message.time_boot_ms) / 1000.0
        elif kind == "ATTITUDE":
            self._telemetry["attitude"] = {
                key: float(getattr(message, key))
                for key in ("roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed")
            }
        elif kind == "SYS_STATUS":
            voltage = int(message.voltage_battery)
            current = int(message.current_battery)
            remaining = int(message.battery_remaining)
            self._telemetry["voltage_v"] = None if voltage in (-1, 65535) else voltage / 1000.0
            self._telemetry["current_a"] = None if current == -1 else current / 100.0
            self._telemetry["battery_pct"] = None if remaining < 0 else float(remaining)
        elif kind == "ACTUATOR_OUTPUT_STATUS":
            values = list(message.actuator)
            self._telemetry["actuator"] = [
                float(values[index]) if index < len(values) else None for index in range(4)
            ]
        elif kind == "ESC_STATUS":
            values = list(getattr(message, "rpm", []))
            index = int(getattr(message, "index", 0))
            motors = list(self._telemetry["motor"])
            for offset, value in enumerate(values):
                target = index + offset
                if target < 4:
                    motors[target] = float(value)
            self._telemetry["motor"] = motors

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._telemetry)
            result["local"] = dict(result["local"]) if result["local"] else None
            result["attitude"] = (
                dict(result["attitude"]) if result["attitude"] else None
            )
            result["actuator"] = list(result["actuator"])
            result["motor"] = list(result["motor"])
            result["fresh"] = (
                result["connected"]
                and time.monotonic() - result["last_message_monotonic"] < 2.0
            )
            result["heartbeat_fresh"] = (
                result["connected"]
                and time.monotonic() - result["last_heartbeat_monotonic"] < 2.0
            )
            return result

    def _require_connection(self):
        connection = self._connection
        if connection is None:
            raise RuntimeError("PX4 MAVLink is not connected")
        return connection

    def send_gcs_heartbeat(self) -> None:
        connection = self._require_connection()
        connection.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, mavutil.mavlink.MAV_STATE_ACTIVE,
        )

    def command(self, command: int, *params: float) -> None:
        connection = self._require_connection()
        values = list(params) + [0.0] * (7 - len(params))
        connection.mav.command_long_send(
            connection.target_system, 1, command, 0, *values[:7]
        )

    def set_mode(self, name: str) -> None:
        connection = self._require_connection()
        mapping = connection.mode_mapping() or {}
        if name.upper() not in mapping:
            raise RuntimeError(f"PX4 mode {name} is unavailable")
        # PX4's map value is a (base_mode, main_mode, sub_mode) tuple. Let
        # pymavlink encode all three fields instead of passing that tuple to
        # SET_MODE as though it were a single integer.
        connection.set_mode(name.upper())

    def arm(self) -> None:
        self.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0)

    def takeoff(self, altitude_m: float) -> None:
        nan = float("nan")
        self.command(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0.0, 0.0, 0.0, nan, nan, nan, altitude_m,
        )

    def land(self) -> None:
        # Setting PX4 LAND mode proved deterministic in SITL, while a bare
        # NAV_LAND command without a global destination can remain in LOITER.
        self.set_mode("LAND")

    def position_setpoint(self, north: float, east: float, down: float, yaw: float) -> None:
        connection = self._require_connection()
        mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system, 1,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, mask,
            north, east, down, 0, 0, 0, 0, 0, 0, yaw, 0,
        )

    def velocity_setpoint(
        self, north_mps: float, east_mps: float, down: float, yaw: float
    ) -> None:
        connection = self._require_connection()
        mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system, 1,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, mask,
            0, 0, down, north_mps, east_mps, 0, 0, 0, 0, yaw, 0,
        )


class M100ExperimentManager(Node):
    def __init__(self) -> None:
        super().__init__("m100_experiment_manager")
        self.declare_parameter("auto_start_stack", True)
        self.declare_parameter("gz_partition", "m100_experiment")
        self.declare_parameter("csv_root", str(CSV_ROOT))
        self.declare_parameter("energy_config", str(ENERGY_CONFIG))
        self.declare_parameter("battery_research_profile", "")
        self.declare_parameter("gong_energy_config", str(GONG_CONFIG))
        self.declare_parameter("stable_cruise_energy_config", str(STABLE_CRUISE_ENERGY_CONFIG))
        self._partition = str(self.get_parameter("gz_partition").value)
        self._csv_root = Path(str(self.get_parameter("csv_root").value))
        self._csv_root.mkdir(parents=True, exist_ok=True)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        energy_config_path = Path(str(self.get_parameter("energy_config").value))
        self._energy_config_path = energy_config_path.resolve()
        energy_config = PublishedEnergyConfig.load(energy_config_path)
        self._energy_config = energy_config
        research_profile_path = str(self.get_parameter("battery_research_profile").value)
        research_profile = (
            TheveninProfile.load(research_profile_path) if research_profile_path else None
        )
        self._energy_model = M100PublishedEnergyModel(energy_config, research_profile)
        self._stable_cruise_config_path = Path(
            str(self.get_parameter("stable_cruise_energy_config").value)
        ).resolve()
        self._stable_cruise_fit = StableCruiseSpeedFit.load(self._stable_cruise_config_path)
        self._transient_config_path = STABLE_CRUISE_ENERGY_CONFIG.with_name("m100_transient_power_v1.yaml")
        self._transient_power = TransientPowerEstimator(
            TransientPowerFit.load(self._transient_config_path), energy_config.maximum_valid_gap_s
        )
        self._wind_config_path = STABLE_CRUISE_ENERGY_CONFIG.with_name("m100_wind_simulation_v1.yaml")
        self._wind_simulation = WindSimulation.load(self._wind_config_path)
        self._wind_environment_application = None
        self._propulsion = M100PropulsionV2.load()
        gong_config_path = Path(str(self.get_parameter("gong_energy_config").value))
        self._runtime_energy = M100RuntimeEnergyV2.load(
            gong_config_path,
            maximum_valid_gap_s=energy_config.maximum_valid_gap_s,
        )

        self._nav_pub = self.create_publisher(String, "/drone_nav/status", 10)
        self._energy_pub = self.create_publisher(String, "/drone_nav/energy", 10)
        self.create_subscription(String, "/drone_nav/experiment_cmd", self._on_apply, 10)
        self.create_subscription(String, "/drone_nav/cmd", self._on_nav_command, 10)
        self.create_subscription(String, "/drone_nav/energy_cmd", self._on_energy_command, 10)
        self.create_subscription(
            JointState, "/m100/rotor_joint_states", self._on_rotor_joint_state, 20
        )
        self._gazebo_sim_time_s: float | None = None
        self.create_subscription(
            Clock, "/clock", self._on_sim_clock, qos_profile_sensor_data
        )

        self._lock = threading.RLock()
        self._mav = MavlinkBackend()
        self._state = "IDLE"
        self._status = "M100 backend starting"
        self._config: ExperimentConfig | None = None
        self._active_payload: PayloadModel | None = None
        self._active_runtime_model: str | None = None
        self._applied_atmosphere: AtmosphereDensityComparison | None = None
        self._power_position_stamp: tuple[float | None, float] = (None, 0.0)
        self._power_air_state: DryAirState | None = None
        self._generated_temperature_model: GeneratedTemperatureModel | None = None
        self._applied_density_ratio = 1.0
        self._apply_seq = 0
        self._auto_running = False
        self._job: threading.Thread | None = None
        self._abort = threading.Event()
        self._px4_process: subprocess.Popen | None = None
        self._px4_log_handle = None
        self._home_ned: tuple[float, float, float] | None = None
        self._manual_hold: tuple[float, float, float, float] | None = None
        self._manual_speed_mps = 2.0
        self._last_heartbeat_tx = 0.0

        self._recording = False
        self._csv_handle = None
        self._csv_writer = None
        self._csv_path: Path | None = None
        self._run_id = ""
        self._run_provenance_context: dict[str, Any] | None = None
        self._last_run_provenance_path: Path | None = None
        self._sample_count = 0
        self._last_csv_energy_timestamp_s: float | None = None
        self._cached_energy_csv_rows_skipped = 0
        self._reset_csv_sampling()
        self._distance_m = 0.0
        self._last_position_enu: tuple[float, float, float] | None = None
        self._rotor_telemetry_monotonic = 0.0
        self._measurement_start_energy_wh: float | None = None
        self._measurement_energy_wh: float | None = None
        self._measurement_distance_m: float | None = None
        self._measurement_energy_per_distance_wh_per_km: float | None = None
        self._measurement_active = False
        self._energy_assessment = m100_v2_energy_assessment(None, "IDLE")
        self._reset_seq = 0

        self.create_timer(0.1, self._tick)
        if bool(self.get_parameter("auto_start_stack").value):
            self._start_job("startup", lambda: self._startup_stack(PAYLOAD_MODELS[0.0]))

    def _energy_phase(self) -> str:
        if self._state == "MEASURING":
            return "HOVER" if self._config and self._config.cruise_speed_mps == 0.0 else "CRUISE"
        return self._state

    def _on_sim_clock(self, message: Clock) -> None:
        # Use the Gazebo world clock for the dashboard even before a case is
        # applied. A newly spawned world can legitimately restart at zero.
        self._gazebo_sim_time_s = (
            message.clock.sec + message.clock.nanosec * 1e-9
        )

    def _air_state_for_power(self, sample, altitude_agl_m, timestamp_s):
        """Sample the dry atmosphere at the PX4 height above the captured home."""
        age_limit = self._energy_config.maximum_valid_gap_s
        boot_s = sample.get("boot_time_s")
        arrival_s = sample.get("last_local_position_monotonic")
        if (not sample["fresh"] or self._home_ned is None or boot_s is None
                or arrival_s is None or time.monotonic() - arrival_s > age_limit
                or altitude_agl_m is None or not math.isfinite(altitude_agl_m)):
            return None
        last_boot_s, first_seen_s = self._power_position_stamp
        if boot_s != last_boot_s or timestamp_s < first_seen_s:
            self._power_position_stamp = (boot_s, timestamp_s)
        elif timestamp_s - first_seen_s > age_limit:
            return None
        # Small negative EKF excursions at the ground use the surface density.
        # Larger excursions or missing heights leave power unavailable.
        if not -0.5 <= altitude_agl_m <= MAX_ALTITUDE_M_MSL:
            return None
        return dry_air_state(self._config.temperature_c, max(0.0, altitude_agl_m))

    def _on_rotor_joint_state(self, message: JointState) -> None:
        # Serialize the manager lifecycle flag with the energy-model lifecycle.
        # This prevents a CSV/status sample from pairing recording=true with an
        # INACTIVE budget (or recording=false with an ACTIVE budget).
        with self._lock:
            self._process_rotor_joint_state(message)

    def _process_rotor_joint_state(self, message: JointState) -> None:
        if self._state in BUSY_CONFIGURATION_STATES:
            # The active SDF and density ratio are not a committed pair while
            # APPLY is replacing PX4/Gazebo.  Drop these transient samples so
            # neither status nor a CSV can expose identity-kT calculations for
            # a derived-temperature model (or the reverse).
            self._rotor_telemetry_monotonic = 0.0
            return
        by_name = {
            name: float(message.velocity[index])
            for index, name in enumerate(message.name)
            if index < len(message.velocity)
        }
        velocities = [by_name.get(name) for name in ROTOR_JOINT_NAMES]
        timestamp_s = float(message.header.stamp.sec) + message.header.stamp.nanosec * 1e-9
        energy_phase = self._energy_phase()
        calibration = m100_v2_energy_assessment(
            self._config,
            energy_phase,
            measurement_active=self._measurement_active,
        )
        sample = self._mav.snapshot()
        enu = self._enu_values(sample)
        local = sample.get("local")
        payload = self._active_payload or (
            self._config.payload if self._config is not None else None
        )
        atmosphere = self._applied_atmosphere
        if payload is None or atmosphere is None:
            self._rotor_telemetry_monotonic = 0.0
            return
        self._power_air_state = self._air_state_for_power(sample, enu["z"], timestamp_s)
        power_density = (self._power_air_state.density_kg_m3
                         if self._power_air_state is not None else None)
        horizontal_speed = None
        altitude_asl = None
        wind_delta = None
        if local is not None:
            horizontal_speed = math.hypot(enu["vx"], enu["vy"])
            # The M100 Gazebo world datum has elevation=0 m.  Relative ENU
            # altitude is therefore also the Gong ISA height above that datum.
            home_down = self._home_ned[2] if self._home_ned is not None else 0.0
            altitude_asl = home_down - local["z"]
        if (
            self._config is not None
            and payload is not None
            and altitude_asl is not None
            and enu["vx"] is not None
            and enu["vy"] is not None
        ):
            wind_delta = self._runtime_energy.optional_wind_power_delta(
                total_mass_kg=payload.total_mass_kg,
                ground_velocity_enu_mps=(enu["vx"], enu["vy"]),
                wind_velocity_enu_mps=(
                    self._config.wind_vector_enu_mps[0],
                    self._config.wind_vector_enu_mps[1],
                ),
                altitude_asl_m=altitude_asl,
            )
        # The flight-level identifiability study rejected the Gong airspeed
        # delta as an operational M100 wind transfer.  Keep it only as a
        # research channel; it never changes V2 power or the battery budget.
        speed_adjustment_w = (
            self._stable_cruise_fit.adjustment_w(self._config)
            if self._state == "CRUISE" and self._measurement_active else None
        )
        transient_estimate = None
        if power_density is not None:
            transient_estimate = self._transient_power.estimate(
                phase=energy_phase, timestamp_s=sample.get("boot_time_s"),
                velocity_enu_mps=(enu["vx"], enu["vy"], enu["vz"]),
                payload_kg=payload.payload_kg, density_kg_m3=power_density,
            )
        if calibration.use_limited_hover_mean:
            transient_estimate = None
        wind_adjustment_w = None
        if self._config is not None and self._config.wind_speed_mps > 0.0 and power_density is not None:
            wind_adjustment_w = self._wind_simulation.power_delta_w(
                self._transient_power.fit, payload_kg=payload.payload_kg,
                density_kg_m3=power_density,
                velocity_enu_mps=(enu["vx"], enu["vy"], enu["vz"]),
                wind_enu_mps=self._config.wind_vector_enu_mps,
            )
        model_energy = self._energy_model.update(
            timestamp_s,
            velocities,
            energy_phase,
            self._distance_m,
            physical_total_mass_kg=payload.total_mass_kg,
            payload_mass_kg=payload.payload_kg,
            air_density_kg_m3=power_density,
            use_limited_hover_mean=calibration.use_limited_hover_mean,
            air_relative_speed_mps=(
                wind_delta.air_relative_speed_mps if wind_delta is not None else None
            ),
            stable_cruise_speed_adjustment_w=speed_adjustment_w,
            transient_power_estimate=transient_estimate,
            airborne=bool(sample["armed"] and not sample["landed"]),
            wind_power_adjustment_w=wind_adjustment_w,
            research_rotor_torque_coefficient=(
                self._propulsion.thrust_constant_n_per_rad_s2
                * self._propulsion.yaw_moment_ratio_m
                * atmosphere.density_ratio
            ),
        )
        # Keep the scientific applicability label atomic with the exact model
        # snapshot that used it.  Status/CSV readers run under the same manager
        # lock and therefore cannot pair a new lifecycle flag with an older
        # power sample during begin/freeze transitions.
        if model_energy.stable_cruise_speed_adjustment_w is not None:
            calibration = replace(
                calibration,
                status=STABLE_CRUISE_MODEL_STATUS,
                warnings=(*calibration.warnings, STABLE_CRUISE_MODEL_WARNING),
                note=(
                    "Stable outbound speed correction evaluated on separate "
                    "flights from the CMU campaign."
                ),
            )
        if model_energy.model_id == TRANSIENT_POWER_MODEL_ID:
            calibration = replace(
                calibration, status=TRANSIENT_POWER_STATUS,
                warnings=(*calibration.warnings, TRANSIENT_POWER_WARNING),
                note="CMU kinematic power transferred to simulator transients with extrapolation uncertainty.",
            )
        if model_energy.model_id == WIND_SIMULATION_MODEL_ID:
            calibration = replace(
                calibration, status="C_NOT_CALIBRATED",
                warnings=(*calibration.warnings, WIND_SIMULATION_WARNING),
                note="Theoretical commanded-wind power transfer for simulation sensitivity analysis.",
            )
        self._energy_assessment = calibration
        self._runtime_energy.update(
            timestamp_s=timestamp_s,
            operational_power_w=model_energy.operational_power_w,
            total_mass_kg=payload.total_mass_kg,
            actual_horizontal_speed_mps=horizontal_speed,
            altitude_asl_m=altitude_asl,
            air_relative_speed_mps=(
                wind_delta.air_relative_speed_mps if wind_delta is not None else None
            ),
            gong_applicable=(
                self._measurement_active
                and self._energy_phase() in {"HOVER", "CRUISE"}
            ),
        )
        self._rotor_telemetry_monotonic = time.monotonic()
        observer = getattr(self, "_campaign_sample_observer", None)
        if observer is not None and self._recording:
            observer(model_energy, sample, enu, self._measurement_active)

    # ------------------------- state and process lifecycle -----------------
    def _set_state(self, state: str, status: str | None = None) -> None:
        if state not in STATES:
            raise ValueError(f"invalid experiment state {state}")
        with self._lock:
            self._state = state
            if status is not None:
                self._status = status
        self.get_logger().info(f"state={state}: {self._status}")

    def _start_job(
        self,
        label: str,
        function: Callable[[], None],
        *,
        reserved_state: str | None = None,
        reserved_status: str | None = None,
        require_recording_idle: bool = False,
    ) -> bool:
        with self._lock:
            if self._job and self._job.is_alive():
                self._status = f"REJECTED — {label}: another operation is active"
                return False
            if require_recording_idle and (
                self._recording or self._measurement_active or self._auto_running
            ):
                self._status = (
                    f"REJECTED — {label}: stop recording and finish the active run"
                )
                return False
            if reserved_state is not None:
                if reserved_state not in STATES:
                    raise ValueError(f"invalid reserved state {reserved_state}")
                self._state = reserved_state
                if reserved_status is not None:
                    self._status = reserved_status

            def wrapped() -> None:
                try:
                    function()
                except AbortRequested:
                    self._safe_land_after_abort()
                except Exception as error:
                    self.get_logger().error(f"{label} failed: {error}")
                    self._set_state("ERROR", f"REJECTED — {error}")
                    # A full-test finally block clears _auto_running before
                    # its exception reaches this wrapper. Preserve recovery
                    # for that case even after recording has been finalized.
                    if label == "full-test" or self._auto_running:
                        self._safe_land_after_abort(error=True)

            self._job = threading.Thread(target=wrapped, name=f"m100-{label}", daemon=True)
            self._job.start()
            return True

    def _startup_stack(self, payload: PayloadModel) -> None:
        self._set_state("SPAWNING", "Starting M100/PX4/Gazebo")
        self._ensure_stack(payload)
        self._set_state("IDLE", "M100 connected — enter inputs and press APPLY INPUTS")

    def _temperature_runtime_selection(
        self, config: ExperimentConfig
    ) -> TemperatureRuntimeSelection:
        """Resolve the exact Gazebo/PX4 model for one commanded case.

        The canonical 15 C case is deliberately an identity branch: it does
        not write or launch a derived SDF.  Non-reference cases use one
        immutable, content-addressed artifact whose rotor coefficients and
        energy-model ratio share the same atmosphere comparison object.
        """

        atmosphere = compare_density_to_15c_reference(
            config.temperature_c,
            config.altitude_m,
        )
        if is_reference_temperature_c(config.temperature_c):
            return TemperatureRuntimeSelection(
                payload=config.payload,
                px4_target=config.payload.target,
                gazebo_model=config.payload.model,
                atmosphere=atmosphere,
                generated_model=None,
            )
        generated = generate_temperature_model(
            px4_models_root=PX4_MODELS_ROOT,
            output_root=TEMPERATURE_MODEL_ROOT,
            payload_model_name=config.payload.model,
            temperature_c=config.temperature_c,
            reference_altitude_m=config.altitude_m,
            density_ratio=atmosphere.density_ratio,
        )
        return TemperatureRuntimeSelection(
            payload=config.payload,
            px4_target=f"gz_{generated.model_name}",
            gazebo_model=generated.model_name,
            atmosphere=atmosphere,
            generated_model=generated,
        )

    def _external_px4_pids(self) -> list[int]:
        own_pid = self._px4_process.pid if self._px4_process else None
        matches = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / "cmdline").read_bytes().split(b"\0",1)[0].decode()
            except (OSError, UnicodeDecodeError):
                continue
            if command == str(PX4_BINARY) and int(entry.name) != own_pid:
                matches.append(int(entry.name))
        return matches

    def _ensure_stack(
        self,
        payload: PayloadModel,
        *,
        runtime_target: str | None = None,
        runtime_model: str | None = None,
        force_clean_session: bool = False,
    ) -> None:
        target = runtime_target or payload.target
        model = runtime_model or payload.model
        current = self._mav.snapshot()
        if (
            not force_clean_session
            and self._active_payload == payload
            and self._active_runtime_model == model
            and self._px4_process is not None
            and self._px4_process.poll() is None
            and current["fresh"]
            and current["local"] is not None
        ):
            return
        self._stop_owned_stack(require_landed=True, remove_runtime_model=True)
        external = self._external_px4_pids()
        if external:
            raise RuntimeError(
                "an unrelated PX4 SITL process is already running; stop it before M100 mode"
            )
        # A prior manager may have been interrupted after Gazebo was re-parented
        # to the user service manager.  In that case this new manager has no
        # in-memory ``_active_runtime_model`` but the exact ``<model>_0`` entity
        # can still occupy the persistent world and PX4's fail-closed spawn
        # (allow_renaming=false) will block.  Remove only the exact target when
        # the server is already running; if no server exists PX4 will create it.
        if self._gazebo_server_running():
            self._remove_runtime_model(model)
        if not PX4_BINARY.is_file():
            raise RuntimeError(f"PX4 binary missing: {PX4_BINARY}")
        validation = subprocess.run(
            ["python3", str(MODEL_VALIDATOR)],
            cwd=PX4_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if validation.returncode != 0 or "PASS: masses" not in validation.stdout:
            raise RuntimeError("M100 static mass/model validation failed")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_ROOT / f"px4_{model}_{stamp}.log"
        self._px4_log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "HEADLESS": "1",
            "GZ_PARTITION": self._partition,
            "PX4_SYS_AUTOSTART": str(payload.autostart),
            "PX4_SIM_MODEL": target,
            "PX4_GZ_WORLD": WORLD_NAME,
            # PX4's default FOLLOW camera mode locks manual orbit / wheel
            # zoom.  Keep the M100 GUI in the normal interactive camera mode;
            # CameraTracking remains loaded for optional user-initiated follow.
            "PX4_GZ_NO_FOLLOW": "1",
            "GZ_IP": "127.0.0.1",
        })
        plugin_directory = Path(get_package_share_directory("m100_gazebo_gui")).parents[1] / "lib/m100_gazebo_gui"
        env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = os.pathsep.join(filter(None, [
            str(plugin_directory), env.get("GZ_SIM_SYSTEM_PLUGIN_PATH", ""),
        ]))
        self._px4_process = subprocess.Popen(
            [str(PX4_BINARY)],
            cwd=PX4_ROOTFS,
            env=env,
            # Keep stdin open. PX4's shell redraws its prompt continuously when
            # attached to /dev/null, which can create multi-gigabyte junk logs.
            stdin=subprocess.PIPE,
            stdout=self._px4_log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._set_state("PX4_CONNECTING", f"Connecting PX4 to {model}")
        self._mav.connect(timeout_s=60.0)
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if self._px4_process.poll() is not None:
                raise RuntimeError(f"PX4 exited; inspect {log_path}")
            sample = self._mav.snapshot()
            if sample["fresh"] and sample["local"] is not None:
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("PX4 connected but local position is unavailable")
        if not self._runtime_model_present(model):
            raise RuntimeError(f"Gazebo did not report expected model {model}_0")
        self._active_payload = payload
        self._active_runtime_model = model
        self._capture_home()

    def _remove_runtime_model(self, model: str | None) -> None:
        """Remove exactly the Gazebo entity owned by this PX4 instance.

        PX4's Gazebo startup requests ``allow_renaming: false``.  A clean PX4
        restart therefore cannot spawn ``<model>_0`` while the prior entity is
        still present in the persistent world.  Queue removal after the prior
        PX4 process has exited, then allow Gazebo to process the next world
        update; never remove by wildcard or by model family prefix.  Gazebo's
        ``remove/blocking`` endpoint can itself wait indefinitely when bridge
        components are being torn down, so the acknowledged non-blocking
        endpoint is deliberately used here.
        """
        if not model or not self._gazebo_server_running():
            return
        entity_name = f"{model}_0"
        command = [
            "gz", "service",
            "-s", f"/world/{WORLD_NAME}/remove",
            "--reqtype", "gz.msgs.Entity",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "10000",
            "--req", f'name: "{entity_name}", type: MODEL',
        ]
        try:
            result = subprocess.run(
                command,
                env=self._gz_environment(),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            detail = str(error)
        else:
            if result.returncode == 0 and "data: true" in result.stdout.lower():
                time.sleep(0.5)
                return
            detail = result.stderr.strip() or result.stdout.strip()
        # Gazebo may belong to the old PX4 process group and exit after PX4.
        # In that case the old entity is already gone with its entire world.
        if self._gazebo_server_running():
            raise RuntimeError(
                f"failed to remove owned Gazebo model {entity_name}: {detail}"
            )

    @staticmethod
    def _gazebo_server_running() -> bool:
        """Return whether a headless Gazebo server process owns the world."""
        world_file = f"{WORLD_NAME}.sdf"
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command = (
                    (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
                )
            except (OSError, UnicodeDecodeError):
                continue
            if "gz sim" in command and " -s " in f" {command} " and world_file in command:
                return True
        return False

    @staticmethod
    def _live_owned_group_members(pgid: int) -> list[int]:
        """Live members of the exact session created for this PX4 process."""
        members = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != os.getuid():
                    continue
                raw = (entry/'stat').read_text()
                fields = raw[raw.rfind(')')+2:].split()
                if fields[0] != 'Z' and int(fields[2]) == pgid and int(fields[3]) == pgid:
                    members.append(int(entry.name))
            except (OSError,ValueError,IndexError):
                continue
        return members

    def _stop_owned_process_group(self, process, graces=(10.0,5.0,5.0)) -> None:
        """Do not reuse a world while its old Gazebo descendants survive."""
        leader_exited = process.poll() is not None
        for requested_signal, grace in zip((signal.SIGINT,signal.SIGTERM,signal.SIGKILL),graces):
            if leader_exited and not self._live_owned_group_members(process.pid):
                return
            deadline = time.monotonic()+grace
            try:
                os.killpg(process.pid,requested_signal)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=grace)
                leader_exited = True
            except subprocess.TimeoutExpired:
                leader_exited = False
            while leader_exited:
                if not self._live_owned_group_members(process.pid):
                    return
                if time.monotonic() >= deadline:
                    break
                time.sleep(.1)
        raise RuntimeError('owned PX4/Gazebo process group did not exit; refusing a new session')

    def _stop_owned_stack(
        self,
        require_landed: bool,
        *,
        remove_runtime_model: bool = False,
    ) -> None:
        process = self._px4_process
        runtime_model = self._active_runtime_model
        if process is None:
            self._mav.close()
            if remove_runtime_model:
                self._remove_runtime_model(runtime_model)
                self._active_payload = None
                self._active_runtime_model = None
            return
        sample = self._mav.snapshot()
        if require_landed and (sample["armed"] or not sample["landed"]):
            raise RuntimeError("payload/model switching is forbidden while armed or airborne")
        self._mav.close()
        self._stop_owned_process_group(process)
        self._px4_process = None
        if self._px4_log_handle:
            self._px4_log_handle.close()
            self._px4_log_handle = None
        if remove_runtime_model:
            self._remove_runtime_model(runtime_model)
            self._active_payload = None
            self._active_runtime_model = None
        time.sleep(1.0)

    def _gz_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["GZ_PARTITION"] = self._partition
        return env

    def _runtime_model_present(self, model: str) -> bool:
        # The park world contains many visual models, so Gazebo needs longer
        # than the plain M100 world to serialize the model list.  A timeout is
        # a transient readiness condition here, not a fatal experiment error.
        deadline = time.monotonic() + 45.0
        expected = f"{model}_0"
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    ["gz", "model", "--list"],
                    env=self._gz_environment(), capture_output=True, text=True,
                    timeout=15, check=False,
                )
            except subprocess.TimeoutExpired:
                time.sleep(0.5)
                continue
            if result.returncode == 0 and expected in result.stdout:
                return True
            time.sleep(0.5)
        return False

    def _configure_wind(self, config: ExperimentConfig) -> None:
        self._wind_environment_application = None
        self._configure_wind_vector(config.wind_vector_enu_mps)
        density = compare_density_to_15c_reference(config.temperature_c, config.altitude_m).case.density_kg_m3
        requested = [density, *config.wind_vector_enu_mps]
        result = subprocess.run(
            ["gz", "service", "-s", "/m100/aerodynamics/configure",
             "--reqtype", "gz.msgs.Double_V", "--reptype", "gz.msgs.Double_V",
             "--timeout", "10000", "--req", " ".join(f"data: {v:.17g}" for v in requested)],
            env=self._gz_environment(), capture_output=True, text=True, timeout=15, check=False,
        )
        try:
            applied = [float(value) for value in re.findall(r"\bdata:\s*(\S+)", result.stdout)]
        except ValueError:
            applied = []
        if (result.returncode != 0 or len(applied) != 4
                or not all(math.isfinite(value) and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)
                           for value, expected in zip(applied, requested))):
            raise RuntimeError("Gazebo environment parameters were not confirmed by the force plugin")
        self._wind_environment_application = {
            "method": "APPLIED_GAZEBO_FORCE_PARAMETERS_SERVICE_RESPONSE",
            "verified": True,
            "density_kg_m3": applied[0],
            "wind_vector_enu_mps": applied[1:],
        }

    def _configure_wind_vector(
        self, wind_vector_enu_mps: tuple[float, float, float]
    ) -> None:
        x, y, z = wind_vector_enu_mps
        payload = (
            f"linear_velocity: {{x: {x:g}, y: {y:g}, z: {z:g}}}, "
            "enable_wind: true"
        )
        result = subprocess.run(
            ["gz", "topic", "-t", f"/world/{WORLD_NAME}/wind",
             "-m", "gz.msgs.Wind", "-p", payload],
            env=self._gz_environment(), capture_output=True, text=True,
            timeout=10, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Gazebo wind command failed: {result.stderr.strip()}")

    # ------------------------- GUI command callbacks ----------------------
    def _on_apply(self, message: String) -> None:
        try:
            config = parse_apply_command(message.data)
        except (ValueError, TypeError) as error:
            self._set_state("ERROR", f"REJECTED — {error}")
            return
        with self._lock:
            sample = self._mav.snapshot()
            if self._recording or self._measurement_active or self._auto_running:
                self._status = (
                    "REJECTED — stop recording and finish the active run "
                    "before applying inputs"
                )
                return
            if (
                sample["armed"]
                or not sample["landed"]
                or self._state in FLIGHT_STATES
            ):
                self._status = "REJECTED — land and disarm before applying inputs"
                return

        def apply() -> None:
            with self._lock:
                # Fail closed while a new case is being installed: a prior
                # successful derived model must never be reported as the
                # current case if spawn, wind, or telemetry verification fails.
                self._config = None
                self._applied_atmosphere = None
                self._generated_temperature_model = None
                self._applied_density_ratio = 1.0
            selection = self._temperature_runtime_selection(config)
            # Every APPLY begins an independent experiment case.  Respawn the
            # owned PX4 instance even when the payload is unchanged so a prior
            # LAND / disarm cycle cannot leave the next case in a non-armable
            # flight mode.  The airborne guard above makes this safe.
            self._ensure_stack(
                config.payload,
                runtime_target=selection.px4_target,
                runtime_model=selection.gazebo_model,
                force_clean_session=True,
            )
            self._configure_wind(config)
            sample_after = self._mav.snapshot()
            if not sample_after["fresh"] or sample_after["local"] is None:
                raise RuntimeError("PX4 telemetry verification failed")
            with self._lock:
                # Commit the configuration, atmosphere and model as one
                # epoch.  Reset all energy/freshness state before accepting a
                # sample from the newly spawned model.
                self._energy_model.reset()
                self._runtime_energy.reset()
                self._transient_power.reset()
                self._energy_assessment = m100_v2_energy_assessment(None, "IDLE")
                self._clear_measurement_metrics()
                self._rotor_telemetry_monotonic = 0.0
                self._config = config
                self._applied_atmosphere = selection.atmosphere
                self._generated_temperature_model = selection.generated_model
                self._applied_density_ratio = selection.density_ratio
                self._apply_seq += 1
            note = "READY — payload verified. Gazebo wind command sent."
            if selection.generated_model is None:
                note += " Rotor density uses the 15 °C reference."
            else:
                note += (
                    " Rotor density adjusted at target altitude "
                    f"(rho/rho15={selection.density_ratio:.6f})."
                )
            self._set_state("READY", note)
            self._refresh_gui_after_world_restart()

        self._start_job(
            "apply",
            apply,
            reserved_state="CONFIGURING",
            reserved_status="Validating M100 experiment inputs",
            require_recording_idle=True,
        )

    @staticmethod
    def _refresh_gui_after_world_restart() -> None:
        """Reconnect a launcher-owned GUI to the new clean Gazebo world."""
        subprocess.Popen(
            [str(PROJECT_ROOT / "start_m100_simulation.sh"), "--refresh-gui"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )

    def _on_nav_command(self, message: String) -> None:
        fields = message.data.strip().lower().split()
        if not fields:
            return
        command = fields[0]
        if command == "runexperiment":
            if self._state != "READY" or self._config is None:
                self._status = "REJECTED — APPLY INPUTS and wait for READY first"
                return
            self._abort.clear()
            self._start_job("full-test", self._run_full_test, require_recording_idle=True)
        elif command == "abortexperiment":
            self._abort.set()
            self._status = "ABORT requested — commanding safe landing"
            try:
                if self._mav.snapshot()["fresh"]:
                    self._mav.land()
            except Exception as error:
                self._status = f"ABORT requested; land command failed: {error}"
        elif command == "resetexperiment":
            sample = self._mav.snapshot()
            if (
                sample["armed"] or not sample["landed"] or self._auto_running
                or self._state in FLIGHT_STATES or self._recording
            ):
                self._status = "REJECTED — RESET is allowed only after landing and disarm"
                return
            self._abort.clear()
            self._start_job("reset", self._reset_experiment)
        elif command in {"square", "circle", "figure8", "spiral", "scan", "path"}:
            self._status = "REJECTED — predefined SJTU paths are disabled in M100 mode"
        elif command == "takeoff":
            altitude = float(fields[1]) if len(fields) > 1 else 2.0
            self._start_job("manual-takeoff", lambda: self._manual_takeoff(altitude))
        elif command == "land":
            self._start_job("manual-land", self._manual_land)
        elif command == "hover" or command == "stop":
            self._start_job("manual-hold", self._hold_current_position)
        elif command == "home":
            self._start_job("manual-home", self._manual_home_and_land)
        elif command == "goto" and len(fields) == 5:
            x, y, z, yaw_deg = map(float, fields[1:])
            self._start_job("manual-goto", lambda: self._manual_goto(x, y, z, yaw_deg))
        elif command == "setvel" and len(fields) == 2:
            value = float(fields[1])
            if not 0.2 <= value <= 20.0:
                self._status = "REJECTED — manual speed must be in [0.2, 20] m/s"
            else:
                self._manual_speed_mps = value
                self._status = f"Manual maximum speed set to {value:g} m/s"
        else:
            self._status = f"REJECTED — unsupported M100 command: {message.data}"

    def _on_energy_command(self, message: String) -> None:
        command = message.data.strip().lower()
        try:
            if command.startswith("start"):
                with self._lock:
                    if self._auto_running:
                        self._status = "REJECTED — the automatic test owns its recording"
                        return
                    self._start_recording("manual")
                    self._status = "CSV recording started"
            elif command == "stop":
                with self._lock:
                    if self._auto_running:
                        self._status = "REJECTED — the automatic test owns its recording"
                        return
                    path = self._stop_recording()
                    self._status = f"CSV saved: {path}" if path else "No active CSV recording"
            elif command == "export":
                self._status = (
                    f"CSV ready: {self._csv_path}" if self._csv_path else "No CSV available"
                )
        except Exception as error:
            self._status = f"REJECTED — CSV: {error}"

    # ------------------------- flight operations --------------------------
    def _capture_home(self) -> None:
        local = self._mav.snapshot()["local"]
        if local is None:
            raise RuntimeError("cannot capture home without local position")
        self._home_ned = (local["x"], local["y"], local["z"])
        self._power_position_stamp = (None, 0.0)

    def _check_abort(self) -> None:
        if self._abort.is_set():
            raise AbortRequested()

    def _clear_measurement_metrics(self) -> None:
        self._measurement_active = False
        self._measurement_start_energy_wh = None
        self._measurement_energy_wh = None
        self._measurement_distance_m = None
        self._measurement_energy_per_distance_wh_per_km = None

    def _begin_measurement(self, has_distance: bool) -> None:
        snapshot = self._energy_model.snapshot()
        self._measurement_start_energy_wh = snapshot.cumulative_energy_wh
        self._measurement_active = True
        try:
            self._runtime_energy.begin_measurement()
        except Exception:
            self._measurement_active = False
            raise
        self._measurement_energy_wh = None
        self._measurement_distance_m = 0.0 if has_distance else None
        self._measurement_energy_per_distance_wh_per_km = None

    def _freeze_measurement(self, distance_m: float | None) -> None:
        try:
            runtime_energy = self._runtime_energy.freeze_measurement()
        finally:
            self._measurement_active = False
        energy_wh = runtime_energy.e_operational_wh
        self._measurement_energy_wh = (
            energy_wh if energy_wh is not None and energy_wh > 0.0 else None
        )
        self._measurement_distance_m = distance_m
        self._measurement_energy_per_distance_wh_per_km = (
            measurement_energy_per_distance(
                self._measurement_energy_wh,
                self._measurement_distance_m,
            )
        )

    def _live_measurement_energy_per_distance(self) -> float | None:
        if self._measurement_energy_per_distance_wh_per_km is not None:
            return self._measurement_energy_per_distance_wh_per_km
        runtime_energy = self._runtime_energy.snapshot()
        if runtime_energy.e_operational_wh is None:
            return None
        return measurement_energy_per_distance(
            runtime_energy.e_operational_wh,
            self._measurement_distance_m,
        )

    def _reset_experiment(self) -> None:
        """Return the experiment to a fresh, landed, zeroed M100 session."""
        self._set_state("RESETTING", "RESETTING — restoring the default M100")
        self._manual_hold = None
        self._stop_owned_stack(require_landed=True, remove_runtime_model=True)
        self._active_payload = None
        self._active_runtime_model = None
        with self._lock:
            # Canonical respawn may publish rotor samples before readiness;
            # restore the identity ratio before those samples can be consumed.
            self._config = None
            self._applied_atmosphere = None
            self._generated_temperature_model = None
            self._applied_density_ratio = 1.0
        self._ensure_stack(PAYLOAD_MODELS[0.0])
        # The world persists between PX4 respawns; explicitly remove any wind
        # left by the preceding experiment.
        self._configure_wind_vector((0.0, 0.0, 0.0))
        with self._lock:
            self._apply_seq = 0
            self._auto_running = False
            self._run_id = ""
            self._sample_count = 0
            self._last_csv_energy_timestamp_s = None
            self._cached_energy_csv_rows_skipped = 0
            self._reset_csv_sampling()
            self._distance_m = 0.0
            self._last_position_enu = None
            self._csv_path = None
            self._rotor_telemetry_monotonic = 0.0
            self._clear_measurement_metrics()
            self._energy_model.reset()
            self._runtime_energy.reset()
            self._transient_power.reset()
            self._energy_assessment = m100_v2_energy_assessment(None, "IDLE")
            self._reset_seq += 1
        self._set_state(
            "IDLE",
            "RESET COMPLETE — default M100 restored; enter inputs and press APPLY INPUTS",
        )
        self._refresh_gui_after_world_restart()

    def _wait_until(
        self, predicate: Callable[[dict[str, Any]], bool], timeout_s: float, label: str
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._check_abort()
            sample = self._mav.snapshot()
            if sample["fresh"] and predicate(sample):
                return sample
            time.sleep(0.1)
        raise RuntimeError(f"timeout while waiting for {label}")

    def _arm_and_takeoff(self, altitude_m: float) -> None:
        self._capture_home()
        self._set_state("ARMING", "Arming PX4")
        self._mav.arm()
        self._wait_until(lambda value: value["armed"], 15.0, "arming")
        self._set_state("TAKEOFF", f"Takeoff to {altitude_m:g} m")
        self._mav.takeoff(altitude_m)
        self._set_state("CLIMB", f"Climbing to {altitude_m:g} m")
        self._wait_until(
            lambda value: self._relative_altitude(value) >= max(0.8, altitude_m - 0.5),
            max(40.0, altitude_m * 3.0),
            "commanded altitude",
        )

    def _enter_offboard_with_setpoint(
        self, publish_setpoint: Callable[[], None]
    ) -> None:
        for _ in range(15):
            self._check_abort()
            publish_setpoint()
            time.sleep(0.1)
        heartbeat_before_request = int(
            self._mav.snapshot().get("heartbeat_generation", 0)
        )
        self._mav.set_mode("OFFBOARD")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self._check_abort()
            sample = self._mav.snapshot()
            heartbeat_after_request = int(sample.get("heartbeat_generation", 0))
            if (
                sample.get("heartbeat_fresh") is True
                and heartbeat_after_request > heartbeat_before_request
                and sample["mode"] == "OFFBOARD"
            ):
                return
            # PX4 exits OFFBOARD if the external setpoint stream stops while
            # we wait for the confirming heartbeat.
            publish_setpoint()
            time.sleep(0.1)
        raise RuntimeError(
            "PX4 did not confirm OFFBOARD mode within 5 s after SET_MODE"
        )

    def _enter_offboard_hold(
        self, north: float, east: float, down: float, yaw: float
    ) -> None:
        self._enter_offboard_with_setpoint(
            lambda: self._mav.position_setpoint(north, east, down, yaw)
        )

    def _enter_offboard_velocity(
        self,
        north_mps: float,
        east_mps: float,
        down: float,
        yaw: float,
    ) -> None:
        self._enter_offboard_with_setpoint(
            lambda: self._mav.velocity_setpoint(
                north_mps, east_mps, down, yaw
            )
        )

    def _run_full_test(self) -> None:
        config = self._config
        if config is None:
            raise RuntimeError("no applied experiment configuration")
        with self._lock:
            self._start_recording("m100")
            self._auto_running = True
        outcome = "ERROR"
        try:
            self._arm_and_takeoff(config.altitude_m)
            home = self._home_ned
            if home is None:
                raise RuntimeError("home position is unavailable")
            target_down = home[2] - config.altitude_m
            current = self._mav.snapshot()
            yaw = (current["attitude"] or {}).get("yaw", 0.0)
            local = current["local"]
            self._set_state("STABILIZING", "Holding experiment altitude before measurement")
            self._enter_offboard_hold(local["x"], local["y"], target_down, yaw)
            stabilize_until = time.monotonic() + 3.0
            while time.monotonic() < stabilize_until:
                self._check_abort()
                self._mav.position_setpoint(local["x"], local["y"], target_down, yaw)
                time.sleep(0.1)

            if config.cruise_speed_mps == 0.0:
                self._set_state("MEASURING", "Hover measurement started")
                measurement_start = self._mav.snapshot()["local"]
                self._begin_measurement(False)
                self._set_state("HOVER", f"Timed hover for {config.hover_duration_s:g} s")
                deadline = time.monotonic() + config.hover_duration_s
                while time.monotonic() < deadline:
                    self._check_abort()
                    self._mav.position_setpoint(
                        measurement_start["x"], measurement_start["y"], target_down, yaw
                    )
                    time.sleep(0.1)
                self._freeze_measurement(None)
            else:
                cruise_speed = config.cruise_speed_mps

                # Initial operational tolerances; require later validation.
                speed_tolerance = max(0.3, 0.05 * cruise_speed)
                lateral_tolerance = max(0.3, 0.05 * cruise_speed)
                altitude_tolerance = 0.5
                required_stable_s = 2.0

                self._set_state(
                    "STABILIZING", "Accelerating before cruise measurement"
                )
                acceleration_deadline = time.monotonic() + max(
                    30.0,
                    2.0 * config.route_length_m / max(cruise_speed, 0.1),
                )
                stable_since = None

                while time.monotonic() < acceleration_deadline:
                    self._check_abort()
                    sample = self._mav.snapshot()
                    local = sample.get("local")

                    if not sample["fresh"] or local is None:
                        raise RuntimeError(
                            "PX4 telemetry unavailable during acceleration"
                        )

                    stable = (
                        abs(local["vy"] - cruise_speed) <= speed_tolerance
                        and abs(local["vx"]) <= lateral_tolerance
                        and abs(local["z"] - target_down) <= altitude_tolerance
                    )

                    if stable:
                        if stable_since is None:
                            stable_since = time.monotonic()
                        elif time.monotonic() - stable_since >= required_stable_s:
                            break
                    else:
                        stable_since = None

                    self._mav.velocity_setpoint(
                        0.0, cruise_speed, target_down, yaw
                    )
                    time.sleep(0.1)
                else:
                    self._precision_return_and_land(
                        home,
                        target_down,
                        yaw,
                        config.route_length_m,
                        config.cruise_speed_mps,
                    )
                    raise RuntimeError(
                        "Cruise did not stabilize; returned to home"
                    )

                measurement_start = self._mav.snapshot()["local"]
                self._begin_measurement(True)

                self._set_state(
                    "CRUISE",
                    "Outbound +X ENU for "
                    f"{config.route_length_m:g} m at "
                    f"{config.cruise_speed_mps:g} m/s",
                )
                timeout = max(
                    30.0,
                    config.route_length_m
                    / max(config.cruise_speed_mps, 0.1)
                    * 4.0,
                )
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    self._check_abort()
                    sample = self._mav.snapshot()
                    if not sample["fresh"] or sample["local"] is None:
                        raise RuntimeError("PX4 position telemetry became unavailable")
                    # PX4 local NED east is Gazebo world ENU +X.
                    traveled_plus_x = sample["local"]["y"] - measurement_start["y"]
                    self._measurement_distance_m = max(0.0, traveled_plus_x)
                    if traveled_plus_x >= config.route_length_m:
                        break
                    self._mav.velocity_setpoint(0.0, config.cruise_speed_mps, target_down, yaw)
                    time.sleep(0.1)
                else:
                    raise RuntimeError("cruise route did not complete within its safety timeout")
                self._freeze_measurement(self._measurement_distance_m)

            self._precision_return_and_land(
                home,
                target_down,
                yaw,
                config.route_length_m,
                config.cruise_speed_mps,
            )
            path = self._stop_recording(outcome="COMPLETE")
            self._set_state("COMPLETE", f"COMPLETE — CSV saved: {path}")
        except AbortRequested:
            outcome = "ABORTED"
            raise
        finally:
            if self._measurement_active:
                try:
                    self._runtime_energy.freeze_measurement()
                except Exception as error:
                    self.get_logger().error(
                        f"measurement freeze during cleanup failed: {error}"
                    )
                finally:
                    self._measurement_active = False
            with self._lock:
                try:
                    if self._recording:
                        self._stop_recording(outcome=outcome)
                finally:
                    self._auto_running = False
                    self._manual_hold = None

    def _precision_return_and_land(
        self,
        home: tuple[float, float, float],
        target_down: float,
        yaw: float,
        route_length_m: float,
        commanded_speed_mps: float,
    ) -> None:
        """Center over home, descend while holding XY, then use PX4 LAND."""
        speed_limit_mps = _safe_return_speed_limit_mps(commanded_speed_mps)
        initial = self._mav.snapshot()
        local = initial.get("local")
        if not initial["fresh"] or local is None:
            raise RuntimeError("PX4 position telemetry unavailable before return")
        initial_error = math.hypot(
            local["x"] - home[0], local["y"] - home[1]
        )
        self._brake_before_return(target_down, yaw)
        self._set_state(
            "RETURNING",
            "Measurement complete — centering over takeoff point at "
            f"<= {speed_limit_mps:g} m/s",
        )
        return_timeout = max(
            40.0,
            max(route_length_m, initial_error)
            / max(speed_limit_mps, 0.1)
            * 3.0,
        )
        self._center_over_home_at_limited_speed(
            home,
            target_down,
            yaw,
            speed_limit_mps,
            return_timeout,
        )

        self._set_state("LANDING", "Centered descent above the landing point")
        final_approach_height = min(M100_OFFBOARD_DESCENT_TARGET_M, home[2] - target_down)
        descent_down = home[2] - final_approach_height
        descent_timeout = landing_timeout_limit(
            max(0.0, self._relative_altitude(self._mav.snapshot()))
        ) + 30.0
        deadline = time.monotonic() + descent_timeout
        while time.monotonic() < deadline:
            self._check_abort()
            sample = self._mav.snapshot()
            local = sample.get("local")
            if not sample["fresh"] or local is None:
                raise RuntimeError("PX4 position telemetry became unavailable during descent")
            self._mav.position_setpoint(home[0], home[1], descent_down, yaw)
            error = math.hypot(local["x"] - home[0], local["y"] - home[1])
            horizontal_speed = math.hypot(local["vx"], local["vy"])
            if (
                self._relative_altitude(sample) <= final_approach_height + 0.5
                and error <= M100_HOME_CENTER_TOLERANCE_M
                and horizontal_speed <= M100_HOME_CENTER_SPEED_MPS
            ):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("centered descent did not reach the final landing altitude")

        self._status = f"Final PX4 landing from {final_approach_height:g} m"
        self._mav.land()
        self._wait_until(
            lambda value: value["landed"],
            landing_timeout_limit(final_approach_height),
            "landing",
        )
        self._wait_until(lambda value: not value["armed"], 20.0, "automatic disarm")
        final = self._mav.snapshot().get("local")
        if final is None:
            raise RuntimeError("final landing position is unavailable")
        final_error = math.hypot(final["x"] - home[0], final["y"] - home[1])
        if final_error > M100_FINAL_LANDING_TOLERANCE_M:
            raise RuntimeError(
                f"landing center error {final_error:.3f} m exceeds "
                f"{M100_FINAL_LANDING_TOLERANCE_M:.2f} m"
            )

    def _brake_before_return(self, target_down: float, yaw: float) -> None:
        """Bring measured horizontal motion to rest before commanding homeward motion."""
        sample = self._mav.snapshot()
        local = sample.get("local")
        if not sample["fresh"] or local is None:
            raise RuntimeError("PX4 position telemetry unavailable before return braking")
        command = (local["vx"], local["vy"])
        if not all(math.isfinite(component) for component in command):
            raise RuntimeError("PX4 horizontal velocity invalid before return braking")
        self._set_state("RETURN_BRAKING", "Braking before the homeward leg")
        if sample["mode"] != "OFFBOARD" or not sample["heartbeat_fresh"]:
            self._enter_offboard_velocity(*command, target_down, yaw)

        last_setpoint = time.monotonic()
        # Reserve a second command-ramp duration for measured PX4 settling.
        # A 20 m/s run still had 0.24 m/s residual motion when the previous
        # one-ramp + 10 s supervisory limit expired. The measured-speed and
        # dwell gates below remain the criteria for starting homeward motion.
        ramp_duration_s = math.hypot(*command) / M100_RETURN_HORIZONTAL_ACCEL_MPS2
        deadline = last_setpoint + 2.0 * ramp_duration_s + 10.0
        stopped_since: float | None = None
        while time.monotonic() < deadline:
            self._check_abort()
            sample = self._mav.snapshot()
            local = sample.get("local")
            if not sample["fresh"] or local is None:
                raise RuntimeError("PX4 position telemetry unavailable while braking for return")
            now = time.monotonic()
            command = _ramp_return_velocity_ned(command, (0.0, 0.0), now - last_setpoint)
            self._mav.velocity_setpoint(*command, target_down, yaw)
            last_setpoint = now
            if command == (0.0, 0.0) and math.hypot(local["vx"], local["vy"]) <= M100_HOME_CENTER_SPEED_MPS:
                if stopped_since is None:
                    stopped_since = now
                elif now - stopped_since >= M100_RETURN_STOP_SETTLE_S:
                    return
            else:
                stopped_since = None
            time.sleep(M100_RETURN_SETPOINT_PERIOD_S)
        raise RuntimeError("horizontal motion did not settle before the homeward leg")

    def _center_over_home_at_limited_speed(
        self,
        home: tuple[float, float, float],
        target_down: float,
        yaw: float,
        speed_limit_mps: float,
        timeout_s: float,
    ) -> None:
        """Return over home with bounded XY velocity, then capture position."""
        command = (0.0, 0.0)
        last_setpoint = time.monotonic()
        deadline = last_setpoint + timeout_s
        # Reserve margin for closed-loop speed tracking under the safety cap.
        command_limit_mps = speed_limit_mps * M100_RETURN_SPEED_HEADROOM
        settled_since: float | None = None
        while time.monotonic() < deadline:
            self._check_abort()
            sample = self._mav.snapshot()
            local = sample.get("local")
            if not sample["fresh"] or local is None:
                raise RuntimeError("PX4 position telemetry became unavailable")
            error = math.hypot(local["x"] - home[0], local["y"] - home[1])
            horizontal_speed = math.hypot(local["vx"], local["vy"])
            now = time.monotonic()
            target = _limited_return_velocity_ned(
                local["x"], local["y"], home[0], home[1], command_limit_mps
            )
            command = _ramp_return_velocity_ned(command, target, now - last_setpoint)
            last_setpoint = now
            if target == (0.0, 0.0) and command == (0.0, 0.0):
                self._mav.position_setpoint(home[0], home[1], target_down, yaw)
            else:
                self._mav.velocity_setpoint(*command, target_down, yaw)
            if (
                error <= M100_HOME_CENTER_TOLERANCE_M
                and horizontal_speed <= M100_HOME_CENTER_SPEED_MPS
                and command == (0.0, 0.0)
            ):
                if settled_since is None:
                    settled_since = now
                elif now - settled_since >= M100_HOME_CENTER_SETTLE_S:
                    break
            else:
                settled_since = None
            time.sleep(M100_RETURN_SETPOINT_PERIOD_S)
        else:
            raise RuntimeError("precision return-to-home did not settle over the landing center")

    def _safe_land_after_abort(self, error: bool = False) -> None:
        self._auto_running = False
        self._manual_hold = None
        try:
            sample = self._mav.snapshot()
            if sample["fresh"] and (sample["armed"] or not sample["landed"]):
                landing_timeout_s = landing_timeout_limit(
                    max(0.0, self._relative_altitude(sample))
                )
                self._mav.land()
                deadline = time.monotonic() + landing_timeout_s
                while time.monotonic() < deadline:
                    if self._mav.snapshot()["landed"]:
                        break
                    time.sleep(0.2)
        except Exception as landing_error:
            self.get_logger().error(f"safe landing failed: {landing_error}")
        if self._recording:
            self._stop_recording(outcome="ERROR" if error else "ABORTED")
        if not error:
            self._set_state("ABORTED", "ABORTED — safe landing command completed")

    def _manual_takeoff(self, altitude_m: float) -> None:
        if self._state not in {"READY", "COMPLETE", "IDLE"}:
            raise RuntimeError("manual takeoff requires an idle/ready M100")
        self._abort.clear()
        self._arm_and_takeoff(max(1.0, min(altitude_m, 100.0)))
        self._hold_current_position()
        self._set_state("READY", "Manual takeoff complete — holding position")

    def _manual_land(self) -> None:
        self._set_state("LANDING", "Manual landing")
        self._manual_hold = None
        landing_timeout_s = landing_timeout_limit(
            max(0.0, self._relative_altitude(self._mav.snapshot()))
        )
        self._mav.land()
        self._wait_until(
            lambda value: value["landed"], landing_timeout_s, "manual landing"
        )
        self._set_state("READY" if self._config else "IDLE", "Manual landing complete")

    def _hold_current_position(self) -> None:
        sample = self._mav.snapshot()
        local = sample["local"]
        if not sample["fresh"] or local is None or not sample["armed"]:
            self._status = "REJECTED — M100 must be connected and airborne to hold"
            return
        yaw = (sample["attitude"] or {}).get("yaw", 0.0)
        self._manual_hold = (local["x"], local["y"], local["z"], yaw)
        self._enter_offboard_hold(*self._manual_hold)
        self._status = "M100 holding current position"

    def _manual_home_and_land(self) -> None:
        if self._home_ned is None:
            raise RuntimeError("home position is unavailable")
        sample = self._mav.snapshot()
        local = sample.get("local")
        if not sample["fresh"] or local is None:
            raise RuntimeError("PX4 position telemetry unavailable before manual return")
        altitude = max(2.0, self._relative_altitude(sample))
        yaw = (sample["attitude"] or {}).get("yaw", 0.0)
        target_down = self._home_ned[2] - altitude
        self._manual_hold = None
        speed_limit_mps = _safe_return_speed_limit_mps(self._manual_speed_mps)
        initial_error = math.hypot(
            local["x"] - self._home_ned[0],
            local["y"] - self._home_ned[1],
        )
        self._brake_before_return(target_down, yaw)
        self._set_state(
            "RETURNING",
            f"Manual return to home at <= {speed_limit_mps:g} m/s",
        )
        self._center_over_home_at_limited_speed(
            self._home_ned,
            target_down,
            yaw,
            speed_limit_mps,
            max(90.0, initial_error / max(speed_limit_mps, 0.1) * 3.0),
        )
        self._manual_land()

    def _manual_goto(self, x_enu: float, y_enu: float, z_up: float, yaw_deg: float) -> None:
        if self._home_ned is None or not self._mav.snapshot()["armed"]:
            raise RuntimeError("GO TO POINT requires an airborne M100")
        north = self._home_ned[0] + y_enu
        east = self._home_ned[1] + x_enu
        down = self._home_ned[2] - z_up
        yaw = math.radians(90.0 - yaw_deg)
        self._enter_offboard_hold(north, east, down, yaw)
        self._manual_hold = (north, east, down, yaw)
        self._status = f"M100 GO TO target: X={x_enu:g}, Y={y_enu:g}, Z={z_up:g} m"

    # ------------------------- telemetry and CSV --------------------------
    def _relative_altitude(self, sample: dict[str, Any]) -> float:
        local = sample.get("local")
        if local is None or self._home_ned is None:
            return 0.0
        return self._home_ned[2] - local["z"]

    def _enu_values(self, sample: dict[str, Any]) -> dict[str, float | None]:
        local = sample.get("local")
        if local is None:
            return {key: None for key in ("x", "y", "z", "vx", "vy", "vz", "speed")}
        home = self._home_ned or (0.0, 0.0, 0.0)
        vx, vy, vz = local["vy"], local["vx"], -local["vz"]
        return {
            "x": local["y"] - home[1],
            "y": local["x"] - home[0],
            "z": home[2] - local["z"],
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "speed": math.sqrt(vx * vx + vy * vy + vz * vz),
        }

    def _tick(self) -> None:
        sample = self._mav.snapshot()
        now = time.monotonic()
        if sample["connected"] and now - self._last_heartbeat_tx >= 1.0:
            try:
                self._mav.send_gcs_heartbeat()
                self._last_heartbeat_tx = now
            except Exception:
                pass
        if self._manual_hold and sample["fresh"] and not self._auto_running:
            try:
                self._mav.position_setpoint(*self._manual_hold)
            except Exception as error:
                self._status = f"Manual hold setpoint failed: {error}"

        enu = self._enu_values(sample)
        with self._lock:
            position = (enu["x"], enu["y"], enu["z"])
            if (self._recording and sample.get("armed")
                    and sample.get("landed") is not None
                    and not sample.get("landed")
                    and all(value is not None for value in position)):
                if self._last_position_enu is not None:
                    self._distance_m += math.dist(position, self._last_position_enu)
                self._last_position_enu = position
            else:
                self._last_position_enu = None

            self._publish_status(sample, enu)
            if self._recording:
                self._write_csv_row(sample, enu)

    def _publish_status(self, sample: dict[str, Any], enu: dict[str, Any]) -> None:
        attitude = sample.get("attitude") or {}
        config = self._config
        payload = self._active_payload
        atmosphere = self._applied_atmosphere
        generated_temperature_model = self._generated_temperature_model
        temperature_physical = bool(
            config is not None
            and generated_temperature_model is not None
            and self._active_runtime_model == generated_temperature_model.model_name
        )
        model_energy = self._energy_model.snapshot()
        runtime_energy = self._runtime_energy.snapshot()
        rotor_fresh = time.monotonic() - self._rotor_telemetry_monotonic < 1.0
        electrical_available = (
            rotor_fresh
            and model_energy.electrical_power_w is not None
            and model_energy.phase == self._energy_phase()
        )
        bms_available = (
            model_energy.battery_estimator_mode is not None
            and model_energy.battery_nameplate_remaining_wh is not None
        )
        final_supported_phase_energy_frozen = bool(
            self._state == "COMPLETE"
            and not self._recording
            and model_energy.bms_state == "FROZEN"
        )
        supported_phase_energy_available = bool(
            electrical_available or final_supported_phase_energy_frozen
        )
        battery_budget_summary = _battery_budget_summary(model_energy)
        calibration = self._energy_assessment
        reported_energy_status = calibration.status
        measurement_energy_wh = (
            runtime_energy.e_operational_wh
            if self._measurement_active else self._measurement_energy_wh
        )
        measurement_status = None
        if config is not None:
            measurement_phase = "HOVER" if config.cruise_speed_mps == 0.0 else "CRUISE"
            measurement_assessment = m100_v2_energy_assessment(
                config, measurement_phase, measurement_active=True
            )
            measurement_status = measurement_assessment.status
            if self._stable_cruise_fit.adjustment_w(config) is not None:
                measurement_status = STABLE_CRUISE_MODEL_STATUS
            if (
                measurement_phase == "HOVER"
                and not measurement_assessment.use_limited_hover_mean
            ):
                measurement_status = TRANSIENT_POWER_STATUS
            if config.wind_speed_mps > 0.0:
                measurement_status = "C_NOT_CALIBRATED"
        wind_diagnostic_active = bool(
            config is not None
            and config.wind_speed_mps > 1e-6
            and model_energy.phase == "CRUISE"
            and self._measurement_active
        )
        thrust_reserve = (
            static_thrust_reserve(
                payload.total_mass_kg,
                self._propulsion.thrust_constant_n_per_rad_s2
                * self._applied_density_ratio,
                self._propulsion.max_rot_velocity_rad_s,
                status=(
                    STATIC_THRUST_RESERVE_DENSITY_STATUS
                    if temperature_physical else STATIC_THRUST_RESERVE_STATUS
                ),
                basis=(
                    STATIC_THRUST_RESERVE_DENSITY_BASIS
                    if temperature_physical else STATIC_THRUST_RESERVE_BASIS
                ),
            )
            if payload is not None else None
        )
        nav = {
            "schema": "M100_EXPERIMENT_MANAGER_V2",
            "backend": "M100_PX4_GAZEBO",
            "state": self._state,
            "status": self._status,
            "airborne": bool(sample["armed"] and not sample["landed"]),
            "armed": bool(sample["armed"]),
            "landed": bool(sample["landed"]),
            "px4_connected": bool(sample["fresh"]),
            "px4_mode": sample["mode"],
            "pos": [enu["x"], enu["y"], enu["z"]],
            "vel": [enu["vx"], enu["vy"], enu["vz"]],
            "speed": enu["speed"],
            "yaw_deg": math.degrees(attitude["yaw"]) if "yaw" in attitude else None,
            "experiment": {
                "auto_running": self._auto_running,
                "apply_seq": self._apply_seq,
                "reset_seq": self._reset_seq,
                "safety_level": "READY" if self._state == "READY" else self._state,
                # Legacy field retained as unavailable because the new value is
                # reserve as a percentage of Tmax, not the historical force
                # margin definition used by the SJTU backend.
                "force_margin_pct": None,
                "static_thrust_reserve_pct": (
                    thrust_reserve.reserve_percent_of_maximum
                    if thrust_reserve is not None else None
                ),
                "static_thrust_capacity_n": (
                    thrust_reserve.maximum_static_thrust_n
                    if thrust_reserve is not None else None
                ),
                "thrust_to_weight_ratio": (
                    thrust_reserve.thrust_to_weight_ratio
                    if thrust_reserve is not None else None
                ),
                "force_margin_status": (
                    thrust_reserve.status if thrust_reserve is not None else None
                ),
                "force_margin_basis": (
                    thrust_reserve.basis if thrust_reserve is not None else None
                ),
                "static_thrust_reserve_definition": (
                    "100 * (Tmax - weight) / Tmax"
                    if thrust_reserve is not None else None
                ),
                "force_margin_is_operational_limit": False,
                "payload_kg": payload.payload_kg if payload else None,
                "total_mass_kg": payload.total_mass_kg if payload else None,
                "physical_total_mass_kg": (
                    payload.total_mass_kg if payload else None
                ),
                "published_regression_mass_kg": (
                    model_energy.published_regression_mass_kg
                ),
                "published_regression_mass_status": (
                    model_energy.mass_basis_status
                ),
                "temperature_c": config.temperature_c if config else None,
                "wind_speed_command_mps": (
                    config.wind_speed_mps if config else None
                ),
                "wind_direction_command_deg": (
                    config.wind_direction_deg if config else None
                ),
                "commanded_speed_mps": (
                    config.cruise_speed_mps if config else None
                ),
                "commanded_altitude_m": config.altitude_m if config else None,
                "route_length_command_m": (
                    config.route_length_m if config else None
                ),
                "hover_duration_command_s": (
                    config.hover_duration_s if config else None
                ),
                "measurement_energy_status": measurement_status,
                "route_definition": "ONE_WAY_PLUS_X_ENU_MEASUREMENT_DISTANCE",
                "temperature_physical": temperature_physical,
                "temperature_effect_scope": (
                    TEMPERATURE_EFFECT_SCOPE if temperature_physical else None
                ),
                "temperature_reference_identity": bool(
                    config is not None
                    and atmosphere is not None
                    and atmosphere.density_ratio == 1.0
                    and generated_temperature_model is None
                ),
                "battery_temperature_physical": False,
                "atmosphere_model": (
                    atmosphere.model if atmosphere is not None else None
                ),
                "atmosphere_world_datum_assumption": (
                    atmosphere.world_datum_assumption
                    if atmosphere is not None else None
                ),
                "atmosphere_density_ratio_rho_over_rho15": (
                    atmosphere.density_ratio if atmosphere is not None else None
                ),
                "atmosphere_case_density_kg_m3": (
                    atmosphere.case.density_kg_m3
                    if atmosphere is not None else None
                ),
                "atmosphere_reference_density_kg_m3": (
                    atmosphere.reference.density_kg_m3
                    if atmosphere is not None else None
                ),
                "atmosphere_case_pressure_pa": (
                    atmosphere.case.pressure_pa if atmosphere is not None else None
                ),
                "atmosphere_reference_pressure_pa": (
                    atmosphere.reference.pressure_pa
                    if atmosphere is not None else None
                ),
                "atmosphere_case_temperature_k_at_altitude": (
                    atmosphere.case.temperature_k_at_altitude
                    if atmosphere is not None else None
                ),
                "atmosphere_reference_temperature_k_at_altitude": (
                    atmosphere.reference.temperature_k_at_altitude
                    if atmosphere is not None else None
                ),
                "temperature_model_name": (
                    generated_temperature_model.model_name
                    if temperature_physical else None
                ),
                "temperature_model_case_sha256": (
                    generated_temperature_model.case_sha256
                    if temperature_physical else None
                ),
                "temperature_model_sdf_sha256": (
                    generated_temperature_model.sdf_sha256
                    if temperature_physical else None
                ),
                "temperature_model_scaled_base_sdf_sha256": (
                    generated_temperature_model.scaled_base_sdf_sha256
                    if temperature_physical else None
                ),
            },
            "capabilities": {
                "pid_active": False,
                "predefined_paths": False,
                "manual_takeoff": True,
                "manual_land": True,
                "manual_hover": True,
                "manual_home": True,
                "manual_hold": True,
                "manual_goto": True,
                "manual_speed": True,
                "reset_experiment": True,
                "rpm_available": rotor_fresh and all(
                    value is not None for value in model_energy.rotor_rpm
                ),
                "mechanical_power_available": False,
                "energy_available": electrical_available,
                "supported_phase_energy_available": (
                    supported_phase_energy_available
                ),
            },
        }
        if final_supported_phase_energy_frozen:
            energy_note = (
                "The run is complete. Airborne energy and its time coverage are "
                "frozen simulation estimates. The run-relative battery budget "
                "uses the integrated operational power. Source models and "
                "extrapolation warnings are recorded in the run metadata."
            )
        elif not electrical_available:
            energy_note = (
                "Published aircraft-level electrical power is unavailable for "
                "the current phase or while the simulation time source is stale. "
                "Battery bookkeeping integrates supported simulation-model phases "
                "only and excludes unsupported phases."
            )
        else:
            energy_note = (
                f"{calibration.note} "
                + (
                    "The Gong airspeed-minus-ground-speed delta is research "
                    "only and is not applied to electrical power; the M100 "
                    "field-data identifiability gate rejected a causal "
                    "directional wind transfer. "
                    if wind_diagnostic_active else ""
                )
                + "Rotor mechanical power is intentionally unavailable until an "
                "exact M100 thrust/torque map exists. BMS energy, SOC, and "
                "remaining budget integrate supported phases only and are not "
                "full-flight estimates. BMS alarms are advisory. The "
                "terminal-voltage transfer proxy has a held-out P95 absolute "
                "error of about 1.57 V; TB47D OCV-SOC, RC dynamics, temperature, "
                "SOH, and pack identity remain unresolved."
            )
        energy = {
            "schema": "M100_TELEMETRY_V3",
            "recording": self._recording,
            "recorded_energy_coverage": model_energy.recorded_energy_coverage,
            "airborne_energy_coverage": model_energy.airborne_energy_coverage,
            "sim_time_s": self._gazebo_sim_time_s,
            "roll_deg": math.degrees(attitude["roll"]) if "roll" in attitude else None,
            "pitch_deg": math.degrees(attitude["pitch"]) if "pitch" in attitude else None,
            "distance_m": self._distance_m,
            "physical_total_mass_kg": model_energy.physical_total_mass_kg,
            "payload_mass_kg": model_energy.payload_mass_kg,
            "published_regression_mass_kg": (
                model_energy.published_regression_mass_kg
            ),
            "published_regression_mass_baseline_kg": (
                self._energy_config.published_regression_baseline_mass_kg
            ),
            "published_regression_mass_status": model_energy.mass_basis_status,
            "published_regression_mass_source_doi": (
                model_energy.mass_basis_source_doi
            ),
            "published_regression_mass_source_archive_sha256": (
                model_energy.mass_basis_source_archive_sha256
            ),
            "published_regression_mass_source_archive_root": (
                model_energy.mass_basis_source_archive_root
            ),
            "published_regression_mass_source_path": (
                model_energy.mass_basis_source_path
            ),
            "rotor_rpm": model_energy.rotor_rpm if rotor_fresh else [None] * 4,
            "mechanical_power_w": None,
            "power_w": model_energy.electrical_power_w if electrical_available else None,
            "power_input_density_kg_m3": model_energy.air_density_kg_m3 if electrical_available else None,
            "power_input_altitude_msl_m": (self._power_air_state.altitude_m_msl
                if electrical_available and self._power_air_state is not None else None),
            "wind_power_adjustment_w": model_energy.wind_power_adjustment_w,
            "wind_power_runtime_applied": model_energy.wind_power_adjustment_w is not None,
            "wind_power_empirically_calibrated": False,
            "wind_simulation_model": WIND_SIMULATION_MODEL_ID,
            "wind_force_model": self._wind_simulation.metadata["horizontal_body_drag"],
            "wind_power_diagnostic_delta_w": (
                runtime_energy.wind_diagnostic_delta_w
                if rotor_fresh and wind_diagnostic_active else None
            ),
            "wind_power_diagnostic_status": (
                WIND_DELTA_STATUS if wind_diagnostic_active else None
            ),
            "air_relative_speed_mps": model_energy.air_relative_speed_mps,
            "ground_horizontal_speed_mps": (
                runtime_energy.ground_speed_input_mps if rotor_fresh else None
            ),
            "p_mechanical_w": None,
            "p_operational_w": (
                runtime_energy.p_operational_w if electrical_available else None
            ),
            "e_operational_wh": runtime_energy.e_operational_wh,
            "p_gong_w": runtime_energy.p_gong_w if rotor_fresh else None,
            "e_gong_wh": runtime_energy.e_gong_wh,
            "energy_timestamp_s": runtime_energy.energy_timestamp_s,
            "energy_model_operational": model_energy.model_id or runtime_energy.energy_model_operational,
            "stable_cruise_speed_adjustment_w": (
                model_energy.stable_cruise_speed_adjustment_w if electrical_available else None
            ),
            "energy_model_research": runtime_energy.energy_model_research,
            "gong_model_classification": runtime_energy.gong_model_classification,
            "gong_applicable": runtime_energy.gong_applicable,
            "measurement_distance_m": self._measurement_distance_m,
            "measurement_energy_wh": (
                measurement_energy_wh if supported_phase_energy_available else None
            ),
            "energy_per_distance_wh_per_km": (
                self._live_measurement_energy_per_distance()
                if supported_phase_energy_available else None
            ),
            "energy_wh": (
                model_energy.cumulative_energy_wh
                if supported_phase_energy_available else None
            ),
            "supported_phase_energy_available": supported_phase_energy_available,
            "supported_phase_energy_frozen": final_supported_phase_energy_frozen,
            "battery_pct": (
                model_energy.soc_percent if bms_available else None
            ),
            "endurance_min": (
                model_energy.endurance_min if bms_available else None
            ),
            "battery_voltage_v": (
                model_energy.battery_voltage_v if bms_available else None
            ),
            "battery_voltage_uncertainty_v": (
                model_energy.battery_voltage_uncertainty_v
                if bms_available else None
            ),
            "battery_current_uncertainty_a": (
                model_energy.battery_current_uncertainty_a
                if bms_available else None
            ),
            "battery_voltage_validation_mae_v": (
                model_energy.battery_voltage_validation_mae_v
                if bms_available else None
            ),
            "battery_voltage_validation_rmse_v": (
                model_energy.battery_voltage_validation_rmse_v
                if bms_available else None
            ),
            "battery_terminal_voltage_model_id": (
                model_energy.battery_terminal_voltage_model_id
            ),
            "battery_terminal_voltage_in_envelope": (
                model_energy.battery_terminal_voltage_in_envelope
            ),
            "battery_pack_identity_status": (
                model_energy.battery_pack_identity_status
            ),
            "battery_current_a": (
                model_energy.battery_current_a if bms_available else None
            ),
            "battery_nameplate_remaining_wh": (
                model_energy.battery_nameplate_remaining_wh
                if bms_available else None
            ),
            "battery_remaining_capacity_ah": (
                model_energy.battery_remaining_capacity_ah
                if bms_available else None
            ),
            "battery_consumed_energy_wh": (
                model_energy.battery_consumed_energy_wh
                if bms_available else None
            ),
            "battery_endurance_reference_power_w": (
                model_energy.battery_endurance_reference_power_w
                if bms_available else None
            ),
            "battery_estimator_mode": self._energy_config.bms_config.mode,
            "battery_estimator_available": bms_available,
            "battery_dynamic_model_enabled": (
                model_energy.battery_dynamic_model_enabled
            ),
            "battery_budget_state": model_energy.battery_budget_state,
            "bms_state": model_energy.bms_state,
            "bms_safety_level": model_energy.bms_safety_level,
            "bms_data_quality": model_energy.bms_data_quality,
            "bms_power_source": model_energy.bms_power_source,
            "bms_soc_source": model_energy.bms_soc_source,
            "bms_cutoff_imminent": model_energy.bms_cutoff_imminent,
            "bms_advisory_only": model_energy.bms_advisory_only,
            "battery_energy_scope": BMS_ENERGY_SCOPE,
            "battery_energy_is_full_flight": False,
            "battery_supported_phases": list(BMS_SUPPORTED_PHASES),
            "battery_limited_hover_policy": BMS_LIMITED_HOVER_POLICY,
            "battery_unsupported_phase_policy": BMS_UNSUPPORTED_PHASE_POLICY,
            "battery_budget_quality": model_energy.bms_data_quality,
            "battery_budget_summary": battery_budget_summary,
            "battery_estimator_assumption": (
                "Run-relative nominal-energy budget initialized from the "
                "configured percentage and integrated only while the published "
                "aircraft model supports the active phase. Unsupported phases "
                "are excluded and break continuity, so SOC and remaining energy "
                "are not full-flight battery estimates. Terminal voltage/current "
                "use a limited published-M100 transfer proxy with unresolved pack "
                "identity unless measured telemetry is explicitly enabled"
            ),
            "phase_energy_wh": model_energy.phase_energy_wh,
            "samples": self._sample_count,
            "label": self._run_id or None,
            "last_csv": str(self._csv_path) if self._csv_path else None,
            "last_provenance": (
                str(self._last_run_provenance_path)
                if self._last_run_provenance_path else None
            ),
            "energy_calibration_status": reported_energy_status,
            "rpm_status": model_energy.rpm_status,
            "mechanical_power_status": model_energy.mechanical_power_status,
            "electrical_power_status": (
                reported_energy_status if electrical_available else "UNAVAILABLE"
            ),
            "battery_model_status": model_energy.battery_status,
            "energy_model_status": (
                reported_energy_status if electrical_available else "UNAVAILABLE"
            ),
            "energy_available": electrical_available,
            "final_supported_phase_energy_available": (
                final_supported_phase_energy_frozen
            ),
            "warnings": list(dict.fromkeys([
                *model_energy.warnings, *runtime_energy.warnings,
                *calibration.warnings,
            ])),
            "note": energy_note,
        }
        if model_energy.battery_research is not None:
            energy["battery_research"] = {
                **model_energy.battery_research,
                "sample_fresh": rotor_fresh and model_energy.bms_state == "ACTIVE",
            }
        self._nav_pub.publish(String(data=json.dumps(nav, allow_nan=False)))
        self._energy_pub.publish(String(data=json.dumps(energy, allow_nan=False)))

    def _start_recording(self, prefix: str) -> str:
        with self._lock:
            if self._recording:
                raise RuntimeError("recording is already active")
            if self._state != "READY":
                raise RuntimeError("recording requires a committed READY configuration")
            if self._measurement_active:
                raise RuntimeError("recording cannot start during an active measurement")
            if self._config is None:
                raise RuntimeError("APPLY INPUTS before recording")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._run_id = f"{prefix}_{stamp}"
            self._csv_path = self._csv_root / f"{self._run_id}.csv"
            provenance = self._capture_run_provenance()
            self._csv_handle = self._csv_path.open("x", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=CSV_FIELDS)
            self._csv_writer.writeheader()
            self._csv_handle.flush()
            self._sample_count = 0
            self._last_csv_energy_timestamp_s = None
            self._cached_energy_csv_rows_skipped = 0
            self._reset_csv_sampling()
            self._distance_m = 0.0
            self._last_position_enu = None
            self._energy_model.reset()
            self._energy_model.begin_bms_session()
            self._runtime_energy.reset()
            self._transient_power.reset()
            self._energy_assessment = m100_v2_energy_assessment(None, "IDLE")
            self._clear_measurement_metrics()
            self._run_provenance_context = provenance
            self._last_run_provenance_path = None
            self._recording = True
            return self._run_id

    def _stop_recording(self, *, outcome: str = "MANUAL_STOP") -> Path | None:
        with self._lock:
            if self._recording:
                self._energy_model.freeze_bms_session()
                self._recording = False
                if self._csv_handle:
                    self._csv_handle.flush()
                    os.fsync(self._csv_handle.fileno())
                    self._csv_handle.close()
                self._csv_handle = None
                self._csv_writer = None
            if (
                self._csv_path is not None
                and self._run_provenance_context is not None
            ):
                if "recording_end" not in self._run_provenance_context:
                    sample = self._mav.snapshot()
                    self._run_provenance_context["recording_end"] = {
                        "outcome": outcome,
                        "flight_state": self._state,
                        "landed": sample["landed"] if sample["fresh"] else None,
                        "armed": sample["armed"] if sample["fresh"] else None,
                    }
                self._last_run_provenance_path = self._finalize_run_provenance(
                    self._csv_path,
                    self._run_provenance_context,
                )
                self._run_provenance_context = None
            return self._csv_path

    @staticmethod
    def _sha256_path(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _capture_run_provenance(self) -> dict[str, Any]:
        config = self._config
        atmosphere = self._applied_atmosphere
        runtime_model = self._active_runtime_model
        if config is None or atmosphere is None or runtime_model is None:
            raise RuntimeError("committed model/atmosphere provenance is unavailable")
        runtime_sdf = PX4_MODELS_ROOT / runtime_model / "model.sdf"
        if not runtime_sdf.is_file():
            raise RuntimeError(f"runtime model SDF is unavailable: {runtime_sdf}")
        generated = self._generated_temperature_model
        reference_identity = is_reference_temperature_c(config.temperature_c)
        if reference_identity != (generated is None):
            raise RuntimeError(
                "temperature reference identity and runtime model disagree"
            )
        sources = {
            name: {
                "path": str(path),
                "sha256": self._sha256_path(path),
            }
            for name, path in {
                "m100_experiment_manager": Path(__file__).resolve(),
                "m100_contract": Path(__file__).with_name("m100_contract.py").resolve(),
                "m100_atmosphere": Path(__file__).with_name("m100_atmosphere.py").resolve(),
                "m100_temperature_sdf": Path(__file__).with_name(
                    "m100_temperature_sdf.py"
                ).resolve(),
                "m100_published_energy_v2": Path(__file__).with_name(
                    "m100_published_energy_v2.py"
                ).resolve(),
                "m100_runtime_energy_v2": Path(__file__).with_name(
                    "m100_runtime_energy_v2.py"
                ).resolve(),
                "m100_stable_cruise_speed": Path(__file__).with_name(
                    "m100_stable_cruise_speed.py"
                ).resolve(),
                "m100_stable_cruise_speed_config": self._stable_cruise_config_path,
                "m100_transient_power": Path(__file__).with_name("m100_transient_power.py").resolve(),
                "m100_transient_power_config": self._transient_config_path,
                "m100_wind_simulation": Path(__file__).with_name("m100_wind_simulation.py").resolve(),
                "m100_wind_simulation_config": self._wind_config_path,
                "gazebo_server_config": PX4_ROOT / "src/modules/simulation/gz_bridge/server.config",
                "gazebo_drag_plugin": Path(get_package_share_directory("m100_gazebo_gui")).parents[1] / "lib/m100_gazebo_gui/libM100AerodynamicDrag.so",
                "gazebo_drag_plugin_source": PROJECT_ROOT / "ros2_ws/src/m100_gazebo_gui/aerodynamics/M100AerodynamicDrag.cc",
                "m100_propulsion_v2": Path(__file__).with_name(
                    "m100_propulsion_v2.py"
                ).resolve(),
                "m100_propulsion_inventory": self._propulsion.inventory_path,
                "m100_energy_v2_config": self._energy_config_path,
            }.items()
        }
        speed_adjustment = self._stable_cruise_fit.adjustment_w(config)
        return {
            "schema": "M100_RUN_PROVENANCE_V2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self._run_id,
            "config": config.as_dict(),
            "atmosphere": asdict(atmosphere),
            "energy_density_source": {
                "model": "DRY_HYDROSTATIC_LINEAR_LAPSE_AT_FRESH_PX4_LOCAL_HEIGHT",
                "world_ground_msl_m": 0.0,
                "minimum_supported_height_m": -0.5,
                "subground_handling": "EKF excursions down to -0.5 m use surface density",
                "max_local_position_hold_sim_and_wall_s": self._energy_config.maximum_valid_gap_s,
                "missing_height": "unavailable power breaks integration continuity",
                "gazebo_force_density": "fixed at configured reference altitude",
            },
            "stable_cruise_speed_model": (
                self._stable_cruise_fit.provenance(speed_adjustment)
                if speed_adjustment is not None else None
            ),
            "transient_power_model": self._transient_power.fit.provenance(),
            "wind_simulation_model": self._wind_simulation.metadata,
            "wind_environment_application": getattr(self, "_wind_environment_application", None),
            "published_energy_model": {
                "model_id": self._energy_config.model_id,
                "paper_doi": self._energy_config.source_doi,
                "mass_basis": self._energy_config.mass_basis_provenance(),
                "local_reproduction": (
                    self._energy_config.reproduction_provenance()
                ),
            },
            "temperature_application": {
                "reference_identity": reference_identity,
                "effect_scope": (
                    None if generated is None else TEMPERATURE_EFFECT_SCOPE
                ),
                "battery_temperature_physical": False,
                "runtime_model": runtime_model,
                "runtime_model_sdf_path": str(runtime_sdf.resolve()),
                "runtime_model_sdf_sha256": self._sha256_path(runtime_sdf),
                "generated_case_sha256": (
                    generated.case_sha256 if generated is not None else None
                ),
                "generated_sdf_sha256": (
                    generated.sdf_sha256 if generated is not None else None
                ),
                "generated_scaled_base_sdf_path": (
                    str(generated.scaled_base_sdf_path.resolve())
                    if generated is not None else None
                ),
                "generated_scaled_base_sdf_sha256": (
                    generated.scaled_base_sdf_sha256
                    if generated is not None else None
                ),
                "generated_manifest_path": (
                    str(generated.manifest_path) if generated is not None else None
                ),
                "generated_manifest_sha256": (
                    self._sha256_path(generated.manifest_path)
                    if generated is not None else None
                ),
            },
            "source_files": sources,
        }

    def _finalize_run_provenance(
        self,
        csv_path: Path,
        context: dict[str, Any],
    ) -> Path:
        if not csv_path.is_file():
            raise RuntimeError(f"cannot finalize provenance without CSV: {csv_path}")
        destination = csv_path.with_suffix(".provenance.json")
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite run provenance: {destination}")
        final_energy = self._energy_model.snapshot()
        final_runtime_energy = self._runtime_energy.snapshot()
        cruise_speed_model = context.get("stable_cruise_speed_model")
        cruise_speed_applied = bool(
            cruise_speed_model
            and final_energy.phase_energy_by_phase_wh.get("CRUISE", 0.0) > 0.0
        )
        summary_warnings = list(dict.fromkeys([
            *final_energy.warnings,
            *final_runtime_energy.warnings,
        ]))
        if cruise_speed_applied:
            summary_warnings = [
                "SPEED_AND_WIND_NOT_EXPLICIT_BASE_PHASE_MODEL_INPUTS"
                if warning == "SPEED_AND_WIND_NOT_EXPLICIT_MODEL_INPUTS" else warning
                for warning in summary_warnings
            ]
        measurement_model_id = None
        if self._measurement_energy_wh is not None:
            measurement_model_id = (
                cruise_speed_model["model_id"] if cruise_speed_applied
                else self._energy_config.model_id
            )
            measurement_phase = "HOVER" if context.get("config", {}).get("cruise_speed_mps") == 0.0 else "CRUISE"
            measurement_models = final_energy.phase_power_model_ids.get(measurement_phase, [])
            if len(measurement_models) == 1:
                measurement_model_id = measurement_models[0]
        payload = {
            **context,
            "finalized_utc": datetime.now(timezone.utc).isoformat(),
            "csv": {
                "path": str(csv_path.resolve()),
                "sha256": self._sha256_path(csv_path),
                "size_bytes": csv_path.stat().st_size,
                "schema_field_count": len(CSV_FIELDS),
                "schema": CSV_SCHEMA,
                "fields": list(CSV_FIELDS),
                "value_source": "SIMULATION_MODEL_ESTIMATE",
                "time_basis": "energy_timestamp_s is the strictly increasing simulation energy clock",
                "distance_definition": "distance_m is recorded trajectory length; measurement_distance_m is the outbound +X ENU displacement",
                "cached_energy_rows_skipped": (
                    self._cached_energy_csv_rows_skipped
                ),
                "sampling": {
                    "target_hz": 1.0 / CSV_SAMPLE_INTERVAL_S,
                    "interval_s": CSV_SAMPLE_INTERVAL_S,
                    "clock": "energy_timestamp_s",
                    "selection": "FIRST_OBSERVED_SAMPLE_PER_FIXED_SIMULATION_TIME_WINDOW",
                    "origin_energy_timestamp_s": self._csv_sample_origin_s,
                    "last_recorded_energy_timestamp_s": self._last_csv_energy_timestamp_s,
                    "recorded_rows": self._sample_count,
                    "rate_limited_rows_skipped": self._rate_limited_csv_rows_skipped,
                    "missed_windows": self._csv_missed_sample_windows,
                    "interpolated_values": False,
                    "energy_accumulation": "UNCHANGED_HIGH_RATE_ROTOR_CALLBACK",
                },
            },
            "energy_summary": {
                "scope": BMS_ENERGY_SCOPE,
                "is_full_flight_energy": False,
                "model_id": self._energy_config.model_id,
                "stable_cruise_speed_applied": cruise_speed_applied,
                "wind_power_runtime_applied": any(
                    WIND_SIMULATION_MODEL_ID in models
                    for models in final_energy.phase_power_model_ids.values()
                ),
                "wind_power_empirically_calibrated": False,
                "physical_total_mass_kg": final_energy.physical_total_mass_kg,
                "payload_mass_kg": final_energy.payload_mass_kg,
                "published_regression_mass_kg": (
                    final_energy.published_regression_mass_kg
                ),
                "published_regression_mass_basis": (
                    self._energy_config.mass_basis_provenance()
                ),
                "integrated_supported_phase_energy_wh": (
                    final_energy.cumulative_energy_wh
                ),
                "recorded_energy_coverage": final_energy.recorded_energy_coverage,
                "airborne_energy_coverage": final_energy.airborne_energy_coverage,
                "phase_energy_by_phase_wh": (
                    final_energy.phase_energy_by_phase_wh
                ),
                "phase_power_model_ids": final_energy.phase_power_model_ids,
                "measurement_window_model_id": measurement_model_id,
                "measurement_window_energy_wh": self._measurement_energy_wh,
                "measurement_window_distance_m": self._measurement_distance_m,
                "measurement_energy_per_distance_wh_per_km": (
                    self._measurement_energy_per_distance_wh_per_km
                ),
                "measurement_frozen": final_runtime_energy.measurement_frozen,
                "battery_budget_frozen": final_energy.bms_state == "FROZEN",
                "warnings": summary_warnings,
            },
            "battery_budget": _battery_budget_summary(final_energy),
        }
        battery_research = getattr(final_energy, "battery_research", None)
        if battery_research is not None:
            payload["battery_research"] = battery_research
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return value

    @staticmethod
    def _is_strictly_new_csv_energy_timestamp(
        timestamp_s: float | None,
        previous_timestamp_s: float | None,
    ) -> bool:
        """Reject cached or regressing asynchronous energy snapshots."""

        if timestamp_s is None or not math.isfinite(timestamp_s):
            return False
        return (
            previous_timestamp_s is None
            or timestamp_s > previous_timestamp_s
        )

    def _reset_csv_sampling(self) -> None:
        self._last_csv_candidate_timestamp_s: float | None = None
        self._csv_sample_origin_s: float | None = None
        self._last_csv_sample_window: int | None = None
        self._rate_limited_csv_rows_skipped = 0
        self._csv_missed_sample_windows = 0

    @staticmethod
    def _csv_sample_window_index(timestamp_s: float, origin_s: float) -> int:
        # Fixed windows avoid accumulating timer jitter. Retain actual times;
        # never interpolate timestamps or observations onto an invented grid.
        return math.floor((timestamp_s - origin_s) / CSV_SAMPLE_INTERVAL_S + 1e-9)

    def _write_csv_row(self, sample: dict[str, Any], enu: dict[str, Any]) -> None:
        config = self._config
        writer = self._csv_writer
        if config is None or writer is None:
            return
        model_energy = self._energy_model.snapshot()
        runtime_energy = self._runtime_energy.snapshot()
        if not self._is_strictly_new_csv_energy_timestamp(
            runtime_energy.energy_timestamp_s,
            self._last_csv_candidate_timestamp_s,
        ):
            # The 10 Hz CSV timer and the rotor JointState callback are
            # asynchronous. Do not duplicate a cached energy snapshot
            # under a newer MAVLink timestamp; preserve strict row lineage.
            self._cached_energy_csv_rows_skipped += 1
            return
        timestamp_s = runtime_energy.energy_timestamp_s
        self._last_csv_candidate_timestamp_s = timestamp_s
        if self._csv_sample_origin_s is None:
            self._csv_sample_origin_s = timestamp_s
        window = self._csv_sample_window_index(timestamp_s, self._csv_sample_origin_s)
        if self._last_csv_sample_window is not None and window <= self._last_csv_sample_window:
            self._rate_limited_csv_rows_skipped += 1
            return
        rotor_fresh = time.monotonic() - self._rotor_telemetry_monotonic < 1.0
        electrical_available = (
            rotor_fresh and model_energy.electrical_power_w is not None
        )
        measurement_energy_wh = self._measurement_energy_wh
        if measurement_energy_wh is None and self._measurement_active:
            measurement_energy_wh = runtime_energy.e_operational_wh
        row = {
            "energy_timestamp_s": runtime_energy.energy_timestamp_s,
            "flight_phase": model_energy.phase,
            "temperature_c": config.temperature_c,
            "payload_kg": config.payload.payload_kg,
            "commanded_speed_mps": config.cruise_speed_mps,
            "actual_horizontal_speed_mps": runtime_energy.ground_speed_input_mps,
            "commanded_altitude_m": config.altitude_m,
            "actual_altitude_m": enu["z"],
            "wind_speed_command_mps": config.wind_speed_mps,
            "wind_direction_command_deg": config.wind_direction_deg,
            "distance_m": self._distance_m,
            "measurement_distance_m": self._measurement_distance_m,
            "measurement_energy_Wh": measurement_energy_wh,
            "cumulative_energy_Wh": (
                model_energy.cumulative_energy_wh if electrical_available else None
            ),
            "battery_voltage_V": (
                model_energy.battery_voltage_v if electrical_available else None
            ),
            "battery_current_A": (
                model_energy.battery_current_a if electrical_available else None
            ),
            "electrical_power_W": (
                model_energy.electrical_power_w if electrical_available else None
            ),
            "energy_per_distance_Wh_per_km": (
                self._live_measurement_energy_per_distance()
                if electrical_available else None
            ),
        }
        writer.writerow({key: self._csv_value(row.get(key)) for key in CSV_FIELDS})
        if self._last_csv_sample_window is not None:
            self._csv_missed_sample_windows += window - self._last_csv_sample_window - 1
        self._last_csv_sample_window = window
        self._last_csv_energy_timestamp_s = runtime_energy.energy_timestamp_s
        self._sample_count += 1
        if self._csv_handle:
            self._csv_handle.flush()

    def destroy_node(self):
        self._abort.set()
        try:
            sample = self._mav.snapshot()
            if sample["armed"] or not sample["landed"]:
                self.get_logger().error("Manager closed while airborne; PX4 stack left running")
            else:
                self._stop_owned_stack(require_landed=True)
        except Exception as error:
            self.get_logger().error(f"shutdown cleanup failed: {error}")
        if self._recording:
            self._stop_recording(outcome="SHUTDOWN")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = M100ExperimentManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
