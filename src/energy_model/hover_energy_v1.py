"""Minimal offline Hover Energy Model v1.

The model intentionally reports only an ideal induced power/energy lower bound.
It does not estimate electrical power, battery energy, current, or true SOC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from data_processing.extract_stable_hover import DataQualityError, extract_stable_hover


FIDELITY = "ideal_induced_power_energy_lower_bound"


def nominal_vehicle_mass_kg(base_mass_kg: float, rotor_count: int, rotor_mass_kg_each: float) -> float:
    values = (float(base_mass_kg), int(rotor_count), float(rotor_mass_kg_each))
    if values[0] <= 0 or values[1] <= 0 or values[2] < 0:
        raise ValueError("Vehicle mass inputs must be physically valid")
    return values[0] + values[1] * values[2]


def total_mass_kg(nominal_mass_kg: float, payload_mass_kg: float) -> float:
    if nominal_mass_kg <= 0 or payload_mass_kg < 0:
        raise ValueError("Nominal mass must be positive and payload must be non-negative")
    return float(nominal_mass_kg + payload_mass_kg)


def total_disk_area_m2(rotor_count: int, rotor_diameter_m: float) -> float:
    if rotor_count <= 0 or rotor_diameter_m <= 0:
        raise ValueError("Rotor count and diameter must be positive")
    radius_m = float(rotor_diameter_m) / 2.0
    return float(rotor_count * np.pi * radius_m**2)


def required_hover_thrust_N(mass_kg: float, gravity_m_s2: float) -> float:
    if mass_kg <= 0 or gravity_m_s2 <= 0:
        raise ValueError("Mass and gravity must be positive")
    return float(mass_kg * gravity_m_s2)


def command_thrust_N(omega_rad_s: np.ndarray, motor_constant: float) -> np.ndarray:
    omega = np.asarray(omega_rad_s, dtype=float)
    if omega.ndim != 2 or omega.shape[1] == 0:
        raise ValueError("omega_rad_s must be a two-dimensional samples-by-rotors array")
    if not np.all(np.isfinite(omega)) or np.any(omega < 0) or motor_constant <= 0:
        raise ValueError("Motor command and motor constant must be finite and valid")
    return motor_constant * np.sum(omega**2, axis=1)


def ideal_induced_power_lower_bound_W(
    thrust_N: float | np.ndarray,
    air_density_kg_m3: float | np.ndarray,
    total_disk_area_m2_value: float,
) -> np.ndarray:
    thrust = np.asarray(thrust_N, dtype=float)
    density = np.asarray(air_density_kg_m3, dtype=float)
    if np.any(thrust <= 0) or np.any(density <= 0) or total_disk_area_m2_value <= 0:
        raise ValueError("Thrust, density, and disk area must be positive")
    return thrust**1.5 / np.sqrt(2.0 * density * total_disk_area_m2_value)


def integrate_energy_J(timestamps_us: np.ndarray, power_W: np.ndarray) -> float:
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    power = np.asarray(power_W, dtype=float)
    if timestamps.ndim != 1 or power.ndim != 1 or timestamps.size != power.size or timestamps.size < 2:
        raise ValueError("Timestamps and power must be same-length one-dimensional arrays")
    if np.any(np.diff(timestamps) <= 0) or not np.all(np.isfinite(power)):
        raise ValueError("Timestamps must increase strictly and power must be finite")
    return float(np.trapezoid(power, timestamps.astype(float) / 1e6))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["model"]["output_fidelity"] != FIDELITY:
        raise DataQualityError("Configuration attempts to use a non-v1 output fidelity")
    if config["propulsion"]["motor_command_message"] != "actuator_outputs":
        raise DataQualityError("esc_status or another motor source is forbidden in v1")
    labels = [case["label"] for case in config["propulsion"]["rotor_diameter_sensitivity_cases"]]
    if len(labels) != len(set(labels)) or any("sensitivity" not in label for label in labels):
        raise DataQualityError("Every rotor-diameter case needs a unique sensitivity label")
    return config


def analyze_hover(ulog_path: Path | str, config_path: Path | str) -> dict[str, Any]:
    config = load_config(config_path)
    hover = extract_stable_hover(ulog_path, config)

    vehicle = config["vehicle"]
    nominal_mass = nominal_vehicle_mass_kg(
        vehicle["base_mass_kg"], vehicle["rotor_count"], vehicle["rotor_mass_kg_each"]
    )
    total_mass = total_mass_kg(nominal_mass, float(vehicle["payload_mass_kg"]))
    required_thrust = required_hover_thrust_N(total_mass, float(vehicle["gravity_m_s2"]))

    motor_thrust = command_thrust_N(
        hover.omega_command_rad_s,
        float(config["propulsion"]["motor_constant_N_per_rad_s_sq"]),
    )
    mean_motor_thrust = float(np.mean(motor_thrust))
    thrust_balance_error_pct = 100.0 * (mean_motor_thrust - required_thrust) / required_thrust

    sensitivity_results: list[dict[str, Any]] = []
    for case in config["propulsion"]["rotor_diameter_sensitivity_cases"]:
        area = total_disk_area_m2(int(vehicle["rotor_count"]), float(case["diameter_m"]))
        power = ideal_induced_power_lower_bound_W(
            required_thrust,
            hover.air_density_kg_m3,
            area,
        )
        energy_j = integrate_energy_J(hover.timestamps_us, power)
        sensitivity_results.append(
            {
                "label": case["label"],
                "rotor_diameter_m": float(case["diameter_m"]),
                "total_disk_area_m2": area,
                "provenance": case["provenance"],
                "mean_ideal_induced_power_lower_bound_W": energy_j / hover.duration_s,
                "ideal_induced_energy_lower_bound_J": energy_j,
                "ideal_induced_energy_lower_bound_Wh": energy_j / 3600.0,
                "fidelity": FIDELITY,
            }
        )

    ulog = Path(ulog_path).resolve()
    relative = lambda timestamp: (timestamp - hover.log_start_us) / 1e6
    return {
        "model": "Hover Energy Model v1",
        "fidelity": FIDELITY,
        "scientific_claim": "Ideal induced power/energy lower bound only",
        "input": {
            "ulog_path": str(ulog),
            "ulog_sha256": _sha256(ulog),
            "config_path": str(Path(config_path).resolve()),
        },
        "window": {
            "posctl_start_timestamp_us": hover.posctl_start_us,
            "auto_land_start_timestamp_us": hover.auto_land_start_us,
            "analysis_start_timestamp_us": hover.analysis_start_us,
            "analysis_end_timestamp_us": hover.analysis_end_us,
            "posctl_start_relative_s": relative(hover.posctl_start_us),
            "auto_land_start_relative_s": relative(hover.auto_land_start_us),
            "analysis_start_relative_s": relative(hover.analysis_start_us),
            "analysis_end_relative_s": relative(hover.analysis_end_us),
            "analysis_duration_s": hover.duration_s,
            "sample_count_including_interpolated_boundaries": int(hover.timestamps_us.size),
        },
        "vehicle": {
            "nominal_mass_kg": nominal_mass,
            "payload_mass_kg": float(vehicle["payload_mass_kg"]),
            "total_mass_kg": total_mass,
            "gravity_m_s2": float(vehicle["gravity_m_s2"]),
        },
        "thrust": {
            "required_hover_thrust_N": required_thrust,
            "mean_command_based_thrust_N": mean_motor_thrust,
            "thrust_balance_error_pct": thrust_balance_error_pct,
            "motor_command_source": "actuator_outputs.output[0..3] float values in rad/s",
            "interpretation": "Internal PX4-Gazebo consistency check; not external thrust validation",
        },
        "air_density": {
            "mean_kg_m3": float(np.mean(hover.air_density_kg_m3)),
            "minimum_kg_m3": float(np.min(hover.air_density_kg_m3)),
            "maximum_kg_m3": float(np.max(hover.air_density_kg_m3)),
            "usage": "Post-processing only; Gazebo motorConstant is not density-coupled",
        },
        "sensitivity_results": sensitivity_results,
        "quality": hover.quality,
        "unavailable_quantities": [
            "battery energy",
            "measured electrical energy",
            "electrical power",
            "battery current",
            "true state of charge",
        ],
        "limitations": [
            "Momentum theory omits profile, interference, motor, ESC, and avionics losses.",
            "Rotor diameters are temporary sensitivity values, not validated aerodynamic specifications.",
            "Gazebo thrust uses a constant motorConstant and is not coupled to air density.",
            "The result must not be used as an ML target representing battery energy.",
            "A scientific source for momentum theory and figure of merit is required before thesis use.",
        ],
        "provenance": config["provenance"],
    }


def write_report(report: Mapping[str, Any], output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ulog", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_hover(args.ulog, args.config)
    write_report(report, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

