"""Synchronize PX4 ULog and Gazebo energy records into the research CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from pyulog import ULog

from data_processing.attitude import attitude_csv_columns
from simulation.scenario import load_scenario_config, scenario_csv_constant_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FlightDataError(ValueError):
    """Raised when flight sources cannot support a defensible unified CSV."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(ulog: ULog, name: str) -> Mapping[str, np.ndarray]:
    matches = [entry.data for entry in ulog.data_list if entry.name == name and entry.multi_id == 0]
    if not matches:
        raise FlightDataError(f"Required ULog dataset is missing: {name}")
    return matches[0]


def _strict_time(timestamps: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(timestamps, dtype=float)
    if values.size < 2 or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise FlightDataError(f"{label} timestamps must be finite and strictly increasing")
    return values


def _deduplicate_last(timestamps: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep the last logged sample at each timestamp and report the raw duplicates."""

    values = np.asarray(timestamps, dtype=float)
    if values.size < 2 or not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0.0):
        raise FlightDataError(f"{label} timestamps must be finite and nondecreasing")
    keep = np.r_[values[1:] != values[:-1], True]
    duplicate_count = int(values.size - np.count_nonzero(keep))
    unique = values[keep]
    _strict_time(unique, label)
    return unique, keep, duplicate_count


def _interp(timestamps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, timestamps, np.asarray(values, dtype=float))


def _interp_quaternion(timestamps: np.ndarray, quaternion: np.ndarray, grid: np.ndarray) -> np.ndarray:
    source = np.asarray(quaternion, dtype=float).copy()
    norms = np.linalg.norm(source, axis=1)
    if not np.all(np.isfinite(source)) or np.any(norms <= np.finfo(float).eps):
        raise FlightDataError("vehicle_attitude contains an invalid quaternion")
    source /= norms[:, None]
    for index in range(1, len(source)):
        if float(np.dot(source[index - 1], source[index])) < 0.0:
            source[index] *= -1.0
    result = np.column_stack([_interp(timestamps, source[:, index], grid) for index in range(4)])
    result /= np.linalg.norm(result, axis=1)[:, None]
    return result


def _load_gazebo_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FlightDataError(f"Invalid Gazebo JSONL at line {line_number}") from exc
            sources[str(record["source"])].append(record)
    required = {
        "rotor_speed_rad_s",
        "shaft_power_w",
        "electrical_load_w",
        "cumulative_energy_j",
        "battery_state",
    }
    if missing := required.difference(sources):
        raise FlightDataError(f"Gazebo recording is missing sources: {sorted(missing)}")
    for source, records in sources.items():
        _strict_time(np.array([record["sim_time_s"] for record in records]), source)
    return sources


def _gazebo_scalar(
    sources: Mapping[str, list[dict[str, Any]]], source: str, grid: np.ndarray
) -> np.ndarray:
    records = sources[source]
    timestamps = np.array([record["sim_time_s"] for record in records], dtype=float)
    values = np.array([record["value"] for record in records], dtype=float)
    return _interp(timestamps, values, grid)


def _phase_labels(markers: list[dict[str, Any]], grid: np.ndarray) -> np.ndarray:
    marker_times = _strict_time(np.array([marker["px4_time_s"] for marker in markers]), "phase")
    labels = np.array([str(marker["phase"]) for marker in markers], dtype=object)
    indices = np.searchsorted(marker_times, grid, side="right") - 1
    if np.any(indices < 0):
        raise FlightDataError("CSV grid begins before the first phase marker")
    return labels[indices]


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "NOT_AVAILABLE"


def _source_tree_revision() -> dict[str, str | None]:
    """Describe source revision availability without inventing a commit."""

    git_metadata = PROJECT_ROOT / ".git"
    if git_metadata.is_dir() and not any(git_metadata.iterdir()):
        return {
            "status": "unavailable_empty_git_directory",
            "git_metadata_path": str(git_metadata),
            "commit": None,
        }
    commit = _command_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"])
    if commit == "NOT_AVAILABLE":
        return {
            "status": "unavailable_no_git_revision",
            "git_metadata_path": str(git_metadata),
            "commit": None,
        }
    return {
        "status": "available",
        "git_metadata_path": str(git_metadata),
        "commit": commit,
    }


def _artifact_record(path: Path) -> dict[str, str]:
    """Return an absolute path and content hash for one immutable artifact."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FlightDataError(f"Required provenance artifact is missing: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _scenario_provenance(
    scenario_manifest_path: Path,
    *,
    config_path: Path,
    expected_run_id: str,
) -> dict[str, Any]:
    """Validate and record the exact scenario assets consumed by a run."""

    try:
        document = json.loads(scenario_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FlightDataError("Scenario manifest is missing or invalid JSON") from exc

    if document.get("run_id") != expected_run_id:
        raise FlightDataError("Scenario manifest run_id does not match the flight configuration")
    if document.get("config_sha256") != _sha256(config_path):
        raise FlightDataError("Scenario manifest was not generated from the supplied configuration")

    generated_files = document.get("generated_files")
    generated_hashes = document.get("generated_file_sha256")
    required = {"model_config", "model_sdf", "world_sdf"}
    if not isinstance(generated_files, Mapping) or not isinstance(generated_hashes, Mapping):
        raise FlightDataError("Scenario manifest does not contain generated-file hashes")
    if required.difference(generated_files) or required.difference(generated_hashes):
        raise FlightDataError("Scenario manifest does not cover every required generated asset")

    assets: dict[str, dict[str, str]] = {}
    for label in sorted(required):
        record = _artifact_record(Path(str(generated_files[label])))
        if record["sha256"] != generated_hashes[label]:
            raise FlightDataError(f"Generated scenario asset hash mismatch: {label}")
        assets[label] = record

    return {
        "scenario_manifest": _artifact_record(scenario_manifest_path),
        "scenario_generated_files": assets,
    }


def build_unified_csv(
    *,
    config_path: Path,
    ulog_path: Path,
    gazebo_jsonl_path: Path,
    phase_markers_path: Path,
    csv_path: Path,
    validation_path: Path,
    manifest_path: Path,
    scenario_manifest_path: Path,
) -> dict[str, Any]:
    inputs = load_scenario_config(config_path)
    scenario_provenance = _scenario_provenance(
        scenario_manifest_path,
        config_path=config_path,
        expected_run_id=inputs.run_id,
    )
    phase_document = json.loads(phase_markers_path.read_text(encoding="utf-8"))
    if phase_document.get("result") != "pass":
        raise FlightDataError("Mission phase document does not report success")
    markers = list(phase_document["phases"])
    required_phases = {
        "prearm",
        "takeoff",
        "hover",
        "move",
        "post_move_hover",
        "landing",
        "landed",
    }
    if {marker["phase"] for marker in markers} != required_phases:
        raise FlightDataError("Mission phase sequence is incomplete")

    ulog = ULog(str(ulog_path))
    if ulog.dropouts:
        raise FlightDataError("ULog contains dropout records")
    position = _dataset(ulog, "vehicle_local_position_groundtruth")
    attitude = _dataset(ulog, "vehicle_attitude")
    angular_velocity = _dataset(ulog, "vehicle_angular_velocity")
    actuator = _dataset(ulog, "actuator_outputs")
    position_t = _strict_time(np.asarray(position["timestamp"], dtype=float) / 1e6, "position")
    attitude_t, attitude_keep, attitude_duplicate_count = _deduplicate_last(
        np.asarray(attitude["timestamp"], dtype=float) / 1e6, "attitude"
    )
    angular_t = _strict_time(
        np.asarray(angular_velocity["timestamp"], dtype=float) / 1e6, "angular velocity"
    )
    actuator_t = _strict_time(np.asarray(actuator["timestamp"], dtype=float) / 1e6, "actuator")
    sources = _load_gazebo_records(gazebo_jsonl_path)

    start_s = float(markers[0]["px4_time_s"])
    desired_end_s = float(markers[-1]["px4_time_s"]) + 2.0
    all_ends = [position_t[-1], attitude_t[-1], angular_t[-1], actuator_t[-1]]
    for records in sources.values():
        all_ends.append(float(records[-1]["sim_time_s"]))
    end_s = min(desired_end_s, *all_ends)
    sample_rate_hz = inputs.logging.sample_rate_hz
    start_tick = int(np.ceil(start_s * sample_rate_hz))
    end_tick = int(np.floor(end_s * sample_rate_hz))
    if end_tick <= start_tick:
        raise FlightDataError("No common time interval exists across flight sources")
    grid = np.arange(start_tick, end_tick + 1, dtype=np.int64) / sample_rate_hz

    columns: dict[str, Any] = {
        "simulation_time_s": grid,
        "mission_elapsed_s": grid - grid[0],
        "position_north_m": _interp(position_t, position["x"], grid),
        "position_east_m": _interp(position_t, position["y"], grid),
        "position_down_m": _interp(position_t, position["z"], grid),
        "altitude_relative_m": -_interp(position_t, position["z"], grid),
        "velocity_north_m_s": _interp(position_t, position["vx"], grid),
        "velocity_east_m_s": _interp(position_t, position["vy"], grid),
        "velocity_down_m_s": _interp(position_t, position["vz"], grid),
        "acceleration_north_m_s2": _interp(position_t, position["ax"], grid),
        "acceleration_east_m_s2": _interp(position_t, position["ay"], grid),
        "acceleration_down_m_s2": _interp(position_t, position["az"], grid),
        "heading_rad": _interp(position_t, position["heading"], grid),
    }
    quaternion = _interp_quaternion(
        attitude_t,
        np.column_stack([attitude[f"q[{index}]"] for index in range(4)])[attitude_keep],
        grid,
    )
    columns.update(attitude_csv_columns(quaternion))
    for index, label in enumerate(("roll_rate_rad_s", "pitch_rate_rad_s", "yaw_rate_rad_s")):
        columns[label] = _interp(angular_t, angular_velocity[f"xyz[{index}]"], grid)
    for index in range(4):
        columns[f"motor_command_{index}_rad_s"] = _interp(
            actuator_t, actuator[f"output[{index}]"], grid
        )

    rotor_records = sources["rotor_speed_rad_s"]
    rotor_t = np.array([record["sim_time_s"] for record in rotor_records], dtype=float)
    rotor_values = np.array([record["values"] for record in rotor_records], dtype=float)
    if rotor_values.shape[1] != 4:
        raise FlightDataError("Gazebo rotor vector does not contain four motors")
    for index in range(4):
        columns[f"actual_rotor_{index}_rad_s"] = _interp(rotor_t, rotor_values[:, index], grid)

    columns["estimated_shaft_power_w"] = _gazebo_scalar(sources, "shaft_power_w", grid)
    columns["estimated_electrical_load_w"] = _gazebo_scalar(
        sources, "electrical_load_w", grid
    )
    plugin_energy = _gazebo_scalar(sources, "cumulative_energy_j", grid)
    columns["plugin_cumulative_energy_j"] = plugin_energy
    columns["cumulative_energy_j"] = plugin_energy - plugin_energy[0]
    columns["cumulative_energy_wh"] = columns["cumulative_energy_j"] / 3600.0

    battery_records = sources["battery_state"]
    battery_t = np.array([record["sim_time_s"] for record in battery_records], dtype=float)
    for source_field, column_name in (
        ("voltage_v", "battery_voltage_v"),
        ("current_a", "battery_current_a"),
        ("charge_ah", "battery_charge_ah"),
        ("capacity_ah", "battery_capacity_ah"),
    ):
        columns[column_name] = _interp(
            battery_t,
            np.array([record[source_field] for record in battery_records], dtype=float),
            grid,
        )
    if "soc_fraction" in battery_records[0]:
        soc = np.array([record["soc_fraction"] for record in battery_records], dtype=float)
    else:
        soc = np.array([record["soc"] / 100.0 for record in battery_records], dtype=float)
    columns["battery_soc_fraction"] = _interp(battery_t, soc, grid)
    columns["flight_phase"] = _phase_labels(markers, grid)
    columns.update(scenario_csv_constant_columns(inputs))

    row_count = len(grid)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for row_index in range(row_count):
            writer.writerow(
                {
                    name: value[row_index] if isinstance(value, np.ndarray) else value
                    for name, value in columns.items()
                }
            )

    numeric_arrays = [value for value in columns.values() if isinstance(value, np.ndarray) and value.dtype != object]
    phase_power = {
        phase: float(np.mean(columns["estimated_electrical_load_w"][columns["flight_phase"] == phase]))
        for phase in required_phases
    }
    checks = {
        "csv_non_empty": row_count > 0 and csv_path.stat().st_size > 0,
        "timestamps_strictly_increasing": bool(np.all(np.diff(grid) > 0.0)),
        "no_duplicate_timestamps": len(np.unique(grid)) == row_count,
        "all_numeric_values_finite": all(bool(np.all(np.isfinite(value))) for value in numeric_arrays),
        "cumulative_energy_nondecreasing": bool(
            np.all(np.diff(columns["cumulative_energy_j"]) >= -1e-8)
        ),
        "takeoff_reached_altitude": float(np.max(columns["altitude_relative_m"])) >= 2.5,
        "landed_near_ground": abs(float(columns["altitude_relative_m"][-1])) <= 0.25,
        "actual_rotors_responded": float(
            np.max(np.abs(np.column_stack([columns[f"actual_rotor_{i}_rad_s"] for i in range(4)])))
        )
        > 100.0,
        "power_rises_during_takeoff": phase_power["takeoff"] > 2.0 * phase_power["landed"],
        "power_rises_during_move": phase_power["move"] > 2.0 * phase_power["landed"],
        "power_drops_after_landing": phase_power["landed"]
        < 0.5 * min(phase_power["takeoff"], phase_power["move"]),
        "fidelity_label_preserved": inputs.fidelity == "C_NOT_CALIBRATED",
        "ulog_has_no_dropouts": len(ulog.dropouts) == 0,
        "final_csv_has_no_source_timestamp_duplicates": bool(np.all(np.diff(grid) > 0.0)),
    }
    validation = {
        "result": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "row_count": row_count,
        "column_count": len(columns),
        "time_range_s": [float(grid[0]), float(grid[-1])],
        "duration_s": float(grid[-1] - grid[0]),
        "phase_mean_estimated_electrical_load_w": phase_power,
        "peak_estimated_electrical_load_w": float(np.max(columns["estimated_electrical_load_w"])),
        "final_cumulative_energy_j": float(columns["cumulative_energy_j"][-1]),
        "maximum_altitude_relative_m": float(np.max(columns["altitude_relative_m"])),
        "source_diagnostics": {
            "vehicle_attitude_duplicate_timestamps_resolved_by_last_sample": attitude_duplicate_count
        },
    }
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if validation["result"] != "pass":
        raise FlightDataError(f"Unified CSV validation failed: {checks}")

    plugin_path = PROJECT_ROOT / "build/gazebo_energy_plugin/libMotorBatteryLoadSystem.so"
    manifest = {
        "schema_version": 2,
        "result": "pass",
        "scientific_claim": "PX4/Gazebo X500 simulation with model-estimated class-C electrical energy",
        "fidelity": inputs.fidelity,
        "scenario": scenario_csv_constant_columns(inputs),
        "versions": {
            "source_tree_revision": _source_tree_revision(),
            "px4_git_commit": _command_output(
                ["git", "-C", inputs.provenance["px4_root"], "rev-parse", "HEAD"]
            ),
            "px4_git_describe": _command_output(
                ["git", "-C", inputs.provenance["px4_root"], "describe", "--always", "--tags"]
            ),
            "gazebo_sim": _command_output(["gz", "sim", "--versions"]),
        },
        "sources": {
            "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
            "ulog": {"path": str(ulog_path.resolve()), "sha256": _sha256(ulog_path)},
            "gazebo_jsonl": {
                "path": str(gazebo_jsonl_path.resolve()),
                "sha256": _sha256(gazebo_jsonl_path),
            },
            "phase_markers": {
                "path": str(phase_markers_path.resolve()),
                "sha256": _sha256(phase_markers_path),
            },
            "energy_plugin": {
                "path": str(plugin_path),
                "sha256": _sha256(plugin_path),
            },
            **scenario_provenance,
        },
        "outputs": {
            "csv": {"path": str(csv_path.resolve()), "sha256": _sha256(csv_path)},
            "validation": {
                "path": str(validation_path.resolve()),
                "sha256": _sha256(validation_path),
            },
        },
        "columns": list(columns),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ulog", required=True, type=Path)
    parser.add_argument("--gazebo-jsonl", required=True, type=Path)
    parser.add_argument("--phase-markers", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scenario-manifest", required=True, type=Path)
    args = parser.parse_args()
    validation = build_unified_csv(
        config_path=args.config,
        ulog_path=args.ulog,
        gazebo_jsonl_path=args.gazebo_jsonl,
        phase_markers_path=args.phase_markers,
        csv_path=args.csv,
        validation_path=args.validation,
        manifest_path=args.manifest,
        scenario_manifest_path=args.scenario_manifest,
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
