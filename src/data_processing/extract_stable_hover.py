"""Extract a reproducible, validated steady-hover window from a PX4 ULog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from pyulog import ULog


class DataQualityError(ValueError):
    """Raised when an input log cannot support the v1 calculation."""


@dataclass(frozen=True)
class HoverWindowData:
    """Signals aligned to an exact trimmed hover window."""

    timestamps_us: np.ndarray
    omega_command_rad_s: np.ndarray
    air_density_kg_m3: np.ndarray
    posctl_start_us: int
    auto_land_start_us: int
    analysis_start_us: int
    analysis_end_us: int
    log_start_us: int
    quality: dict[str, Any]

    @property
    def duration_s(self) -> float:
        return (self.analysis_end_us - self.analysis_start_us) / 1e6


def _dataset(ulog: ULog, name: str) -> Mapping[str, np.ndarray]:
    matches = [entry.data for entry in ulog.data_list if entry.name == name and entry.multi_id == 0]
    if not matches:
        raise DataQualityError(f"Required ULog dataset is missing: {name}")
    return matches[0]


def validate_timestamps(
    timestamps_us: np.ndarray,
    signal_name: str,
    max_gap_multiplier: float,
) -> None:
    """Require finite, strictly increasing timestamps without large gaps."""

    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    if timestamps.size < 3:
        raise DataQualityError(f"{signal_name} has fewer than three samples")
    gaps = np.diff(timestamps)
    if np.any(gaps <= 0):
        raise DataQualityError(f"{signal_name} has duplicate or non-monotonic timestamps")
    median_gap = float(np.median(gaps))
    if median_gap <= 0 or float(np.max(gaps)) > max_gap_multiplier * median_gap:
        raise DataQualityError(f"{signal_name} has a timestamp gap above the allowed multiplier")


def _find_control_window(status: Mapping[str, np.ndarray], cfg: Mapping[str, Any]) -> tuple[int, int]:
    timestamps = np.asarray(status["timestamp"], dtype=np.int64)
    nav_state = np.asarray(status["nav_state"], dtype=np.int64)
    change_indices = np.flatnonzero(np.r_[True, nav_state[1:] != nav_state[:-1]])

    takeoff = int(cfg["auto_takeoff_nav_state"])
    posctl = int(cfg["position_control_nav_state"])
    auto_land = int(cfg["auto_land_nav_state"])

    for transition_index in range(1, len(change_indices)):
        previous = change_indices[transition_index - 1]
        current = change_indices[transition_index]
        if nav_state[previous] != takeoff or nav_state[current] != posctl:
            continue
        for later in change_indices[transition_index + 1 :]:
            if nav_state[later] == auto_land:
                return int(timestamps[current]), int(timestamps[later])
        break

    raise DataQualityError("No AUTO_TAKEOFF -> POSCTL -> AUTO_LAND sequence was found")


def _interpolation_grid(source_timestamps_us: np.ndarray, start_us: int, end_us: int) -> np.ndarray:
    source = np.asarray(source_timestamps_us, dtype=np.int64)
    if source[0] > start_us or source[-1] < end_us:
        raise DataQualityError("The motor-command signal does not span the complete analysis window")
    interior = source[(source > start_us) & (source < end_us)]
    return np.r_[np.int64(start_us), interior, np.int64(end_us)]


def _interpolate(
    source_timestamps_us: np.ndarray,
    values: np.ndarray,
    target_timestamps_us: np.ndarray,
) -> np.ndarray:
    source_t = np.asarray(source_timestamps_us, dtype=np.float64)
    target_t = np.asarray(target_timestamps_us, dtype=np.float64)
    source_values = np.asarray(values, dtype=np.float64)
    if source_t[0] > target_t[0] or source_t[-1] < target_t[-1]:
        raise DataQualityError("A required signal does not span the complete analysis window")
    return np.interp(target_t, source_t, source_values)


def _max_true_duration_s(timestamps_us: np.ndarray, condition: np.ndarray) -> float:
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    mask = np.asarray(condition, dtype=bool)
    maximum = 0.0
    run_start: int | None = None
    for index, active in enumerate(mask):
        if active and run_start is None:
            run_start = index
        if run_start is not None and (not active or index == len(mask) - 1):
            run_end = index if active and index == len(mask) - 1 else index - 1
            maximum = max(maximum, (timestamps[run_end] - timestamps[run_start]) / 1e6)
            run_start = None
    return maximum


def extract_stable_hover(ulog_path: Path | str, config: Mapping[str, Any]) -> HoverWindowData:
    """Load a ULog and return the accepted v1 hover signals."""

    path = Path(ulog_path)
    if not path.is_file():
        raise DataQualityError(f"ULog does not exist: {path}")

    ulog = ULog(str(path))
    if ulog.dropouts:
        raise DataQualityError("ULog contains dropout records; v1 rejects the log conservatively")

    window_cfg = config["hover_window"]
    quality_cfg = config["quality"]
    max_gap_multiplier = float(quality_cfg["max_timestamp_gap_multiplier"])

    status = _dataset(ulog, "vehicle_status")
    posctl_start_us, auto_land_start_us = _find_control_window(status, window_cfg)
    analysis_start_us = posctl_start_us + int(float(window_cfg["trim_start_s"]) * 1e6)
    analysis_end_us = auto_land_start_us - int(float(window_cfg["trim_end_s"]) * 1e6)
    duration_s = (analysis_end_us - analysis_start_us) / 1e6
    if duration_s < float(window_cfg["minimum_accepted_duration_s"]):
        raise DataQualityError("The trimmed hover window is shorter than the configured minimum")

    actuator = _dataset(ulog, "actuator_outputs")
    actuator_t = np.asarray(actuator["timestamp"], dtype=np.int64)
    validate_timestamps(actuator_t, "actuator_outputs", max_gap_multiplier)
    grid_us = _interpolation_grid(actuator_t, analysis_start_us, analysis_end_us)

    command_fields = config["propulsion"]["motor_command_fields"]
    if config["propulsion"]["motor_command_message"] != "actuator_outputs":
        raise DataQualityError("v1 permits only float actuator_outputs as the motor-command source")
    if len(command_fields) != int(config["vehicle"]["rotor_count"]):
        raise DataQualityError("Motor-command field count does not equal rotor_count")

    commands = np.column_stack(
        [_interpolate(actuator_t, actuator[field], grid_us) for field in command_fields]
    )
    if not np.all(np.isfinite(commands)) or np.any(commands < 0):
        raise DataQualityError("Motor commands contain invalid values")

    max_velocity = float(config["propulsion"]["max_rot_velocity_rad_s"])
    if np.any(commands > max_velocity * 1.01):
        raise DataQualityError("Motor command exceeds maxRotVelocity beyond numerical tolerance")
    high_command = np.any(
        commands > float(quality_cfg["max_motor_command_fraction"]) * max_velocity,
        axis=1,
    )
    high_command_duration = _max_true_duration_s(grid_us, high_command)
    if high_command_duration > float(quality_cfg["max_motor_command_duration_s"]):
        raise DataQualityError("Motor command remains above the configured fraction for too long")

    air = _dataset(ulog, "vehicle_air_data")
    air_t = np.asarray(air["timestamp"], dtype=np.int64)
    validate_timestamps(air_t, "vehicle_air_data", max_gap_multiplier)
    rho = _interpolate(air_t, air["rho"], grid_us)
    if not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        raise DataQualityError("Air density is missing, non-finite, or non-positive")

    ground_truth = _dataset(ulog, "vehicle_local_position_groundtruth")
    ground_t = np.asarray(ground_truth["timestamp"], dtype=np.int64)
    validate_timestamps(ground_t, "vehicle_local_position_groundtruth", max_gap_multiplier)
    ground_mask = (ground_t >= analysis_start_us) & (ground_t < analysis_end_us)
    if np.count_nonzero(ground_mask) < 3:
        raise DataQualityError("Ground-truth motion is unavailable in the analysis window")
    vertical_speed = np.abs(np.asarray(ground_truth["vz"], dtype=float)[ground_mask])
    horizontal_speed = np.hypot(
        np.asarray(ground_truth["vx"], dtype=float)[ground_mask],
        np.asarray(ground_truth["vy"], dtype=float)[ground_mask],
    )
    altitude = np.asarray(ground_truth["z"], dtype=float)[ground_mask]
    altitude_deviation = np.abs(altitude - np.median(altitude))
    violation_fractions = {
        "vertical_speed": float(np.mean(vertical_speed > float(quality_cfg["max_vertical_speed_m_s"]))),
        "horizontal_speed": float(
            np.mean(horizontal_speed > float(quality_cfg["max_horizontal_speed_m_s"]))
        ),
        "altitude_deviation": float(
            np.mean(altitude_deviation > float(quality_cfg["max_altitude_deviation_m"]))
        ),
    }
    allowed_fraction = float(quality_cfg["max_motion_violation_fraction"])
    if any(value > allowed_fraction for value in violation_fractions.values()):
        raise DataQualityError(f"Hover motion thresholds were violated: {violation_fractions}")

    land = _dataset(ulog, "vehicle_land_detected")
    land_t = np.asarray(land["timestamp"], dtype=np.int64)
    land_mask = (land_t >= analysis_start_us) & (land_t < analysis_end_us)
    if not np.any(land_mask) or np.any(np.asarray(land["landed"], dtype=bool)[land_mask]):
        raise DataQualityError("The vehicle is landed or land state is unavailable in the hover window")

    quality = {
        "accepted": True,
        "ulog_dropouts": 0,
        "motion_violation_fractions": violation_fractions,
        "maximum_high_motor_command_duration_s": high_command_duration,
        "rtf_validation": "NOT AVAILABLE IN ULOG; previously validated separately",
    }
    return HoverWindowData(
        timestamps_us=grid_us,
        omega_command_rad_s=commands,
        air_density_kg_m3=rho,
        posctl_start_us=posctl_start_us,
        auto_land_start_us=auto_land_start_us,
        analysis_start_us=analysis_start_us,
        analysis_end_us=analysis_end_us,
        log_start_us=int(ulog.start_timestamp),
        quality=quality,
    )

