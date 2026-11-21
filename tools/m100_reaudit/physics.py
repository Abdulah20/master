"""Audit the research airframe and call the ordinary runtime's published power model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/m100_reaudit/physics.json"
DEFAULT_RUNTIME = Path.home() / "Master/Drone/project"


def runtime_path(root=None):
    return Path(root or os.environ.get("M100_RUNTIME_ROOT") or DEFAULT_RUNTIME).expanduser().resolve()


def _finite(value, label):
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _enu3(vector):
    x, y, z = vector
    return tuple(_finite(v, "ENU component") for v in (x, y, z))


def wind_enu(speed_mps, toward_deg):
    """Convert horizontal, toward-direction wind to (east,north,up) m/s."""
    speed = _finite(speed_mps, "wind speed")
    if speed < 0:
        raise ValueError("negative wind speed")
    direction = math.radians(_finite(toward_deg, "wind direction") % 360)
    return speed * math.cos(direction), speed * math.sin(direction), 0.0


def air_velocity_enu(ground_enu, wind_enu):
    """Subtract the wind vector in world axes; body yaw is irrelevant."""
    return tuple(v - w for v, w in zip(_enu3(ground_enu), _enu3(wind_enu), strict=True))


def relative_wind(course_deg, speed_mps, toward_deg):
    """Signed along-track and left-of-track wind in m/s, using ground course."""
    angle = math.radians(_finite(course_deg, "ground course") % 360)
    east, north, _ = wind_enu(speed_mps, toward_deg)
    return east * math.cos(angle) + north * math.sin(angle), -east * math.sin(angle) + north * math.cos(angle)


@dataclass(frozen=True)
class Assessment:
    physical_mass_kg: float
    regression_mass_kg: float
    stock_reasons: tuple[str, ...]
    evidence_flags: tuple[str, ...]
    stock_speed_conditions: str
    runtime_eligible: bool
    runtime_rejection: str | None


@dataclass(frozen=True)
class EnergyBudget:
    nominal_remaining_wh: float
    planned_reserve_wh: float
    remaining_above_reserve_wh: float
    status: str


@dataclass
class Model:
    settings: dict
    energy: object
    atmosphere_module: object
    contract: object
    runtime_root: Path

    @classmethod
    def load(cls, runtime_root=None, config_path=CONFIG):
        root = runtime_path(runtime_root)
        package = root / "ros2_ws/src/drone_nav"
        if not (package / "drone_nav/m100_published_energy_v2.py").is_file():
            raise FileNotFoundError(f"M100 runtime missing from {package}; set M100_RUNTIME_ROOT")
        # An installed, older copy in the same Python process must not silently
        # supply the model for a different explicitly selected checkout.
        loaded = sys.modules.get("drone_nav")
        if loaded is not None and Path(loaded.__file__).resolve().parent != (package / "drone_nav").resolve():
            raise ValueError("drone_nav is already loaded from another M100 runtime; restart Python")
        sys.path.insert(0, str(package))
        try:
            atmosphere = import_module("drone_nav.m100_atmosphere")
            contract = import_module("drone_nav.m100_contract")
            published = import_module("drone_nav.m100_published_energy_v2")
        finally:
            sys.path.remove(str(package))
        settings = json.loads(Path(config_path).read_text())
        energy = published.PublishedEnergyConfig.load(package / "config" / settings["energy"]["published_phase_config"])
        physical = math.fsum(settings["mass"]["as_flown_components_kg"].values())
        regression = settings["mass"]["regression_airframe_feature_kg"] + settings["mass"]["regression_battery_feature_kg"]
        if not (math.isclose(physical, energy.physical_runtime_baseline_mass_kg, abs_tol=1e-9, rel_tol=0.0)
                and math.isclose(regression, energy.published_regression_baseline_mass_kg, abs_tol=1e-9, rel_tol=0.0)):
            raise ValueError("component inventory differs from the selected runtime")
        battery = settings["battery"]
        nameplate = energy.bms_config
        if (battery["nominal_voltage_v"], battery["nominal_capacity_ah"], battery["nominal_energy_wh"]) != (
                nameplate.nominal_voltage_v, nameplate.capacity_ah, nameplate.nominal_energy_wh):
            raise ValueError("TB47D nameplate differs from the selected runtime")
        return cls(settings, energy, atmosphere, contract, root)

    def atmosphere(self, temperature_c, altitude_agl_m):
        """Sea-level datum temperature drives hydrostatic pressure at actual MSL height."""
        height_msl = _finite(altitude_agl_m, "height") + self.settings["wind"]["world_ground_msl_m"]
        return self.atmosphere_module.dry_air_state(temperature_c, height_msl)

    def phase_power(self, phase, payload_kg, density_kg_m3):
        """Published phase-average electrical W; unsupported phases return None."""
        return self.energy.estimate_power(phase, payload_kg, density_kg_m3).power_w

    def assess(self, payload_kg, speed_mps, wind_mps, altitude_agl_m, temperature_c,
               route_m=500.0, hover_s=60.0, wind_toward_deg=0.0):
        """Keep spec eligibility, simulator restrictions and research evidence independent."""
        for value in (payload_kg, speed_mps, wind_mps, altitude_agl_m, temperature_c, route_m, hover_s, wind_toward_deg):
            _finite(value, "scenario input")
        if min(payload_kg, speed_mps, wind_mps, altitude_agl_m) < 0:
            raise ValueError("negative mass, speed, wind or altitude")
        physical = math.fsum(self.settings["mass"]["as_flown_components_kg"].values()) + payload_kg
        regression = self.energy.published_regression_mass_kg(payload_kg)
        spec = self.settings["manufacturer"]
        reasons = []
        if physical > spec["maximum_takeoff_kg"]:
            reasons.append("ABOVE_DJI_MAXIMUM_TAKEOFF_MASS")
        if wind_mps > spec["maximum_wind_mps"]:
            reasons.append("ABOVE_DJI_MAXIMUM_WIND")
        if speed_mps > spec["maximum_speed_atti_unloaded_calm_mps"]:
            reasons.append("ABOVE_HIGHEST_DJI_PUBLISHED_SPEED")
        if altitude_agl_m + self.settings["wind"]["world_ground_msl_m"] > spec["maximum_service_ceiling_msl_m"]:
            reasons.append("ABOVE_DJI_SERVICE_CEILING")
        minimum, maximum = spec["ambient_temperature_range_c"]
        if not minimum <= temperature_c <= maximum:
            reasons.append("OUTSIDE_DJI_AMBIENT_TEMPERATURE")
        flags = []
        if wind_mps > 0:
            flags.append("C_NOT_CALIBRATED")
        if temperature_c != 15.0:
            flags.append("ATMOSPHERIC_SENSITIVITY_NO_BATTERY_THERMAL_CALIBRATION")
        if speed_mps > spec["maximum_speed_gps_unloaded_calm_mps"]:
            flags.append("ABOVE_DJI_GPS_UNLOADED_CALM_REFERENCE")
        try:
            args = self.contract.build_config(temperature_c, payload_kg, wind_mps, wind_toward_deg,
                                              speed_mps, altitude_agl_m, route_m, hover_s)
        except ValueError as err:
            return Assessment(physical, regression, tuple(reasons), tuple(flags),
                              "PX4_OFFBOARD_NO_DJI_MODE_EQUIVALENCE", False, str(err))
        assessment = self.contract.m100_v2_energy_assessment(
            args, "HOVER" if speed_mps == 0 else "CRUISE", measurement_active=True)
        if speed_mps == 0:
            flags.append("LIMITED_EMPIRICAL_PROXY")
        else:
            flags.append(assessment.status)
            if assessment.inside_reported_envelope and speed_mps not in self.settings["energy"]["empirical_cruise_speeds_mps"]:
                flags.append("INTERPOLATED_SPEED_BETWEEN_PUBLISHED_LEVELS")
        return Assessment(physical, regression, tuple(reasons), tuple(flags),
                          "PX4_OFFBOARD_NO_DJI_MODE_EQUIVALENCE", True, None)

    def budget(self, consumed_wh, *, complete_coverage=False):
        """Nameplate accounting. Partial energy can prove exceedance only."""
        used = _finite(consumed_wh, "consumed energy")
        if used < 0:
            raise ValueError("negative consumed energy")
        battery = self.settings["battery"]
        nominal = battery["nominal_energy_wh"]
        reserve = nominal * battery["planning_reserve_fraction"]
        remaining = nominal - used
        if remaining <= 0:
            status = "NAMEPLATE_EXHAUSTED"
        elif remaining < reserve:
            status = "BELOW_PLANNED_RESERVE"
        elif complete_coverage:
            status = "WITHIN_NOMINAL_BUDGET"
        else:
            status = "PARTIAL_ENERGY_UPPER_BOUND_ON_REMAINING"
        return EnergyBudget(remaining, reserve, remaining - reserve, status)


PATCH = Path(__file__).with_name("runtime.patch")
PATCH_HASHES = {
    "ros2_ws/src/drone_nav/drone_nav/m100_experiment_manager.py": (
        "72a223d93c0eaa29b2553a0e7d2c8fc1ac5589bfec7ffb71247dc28909dd0c9d",
        "b5de6ec30656a9e9013c5546943f19161878c8ce4b13c60c2f357000083dfcb6"),
    "ros2_ws/src/drone_nav/drone_nav/m100_published_energy_v2.py": (
        "1a332ed4b6115735adec044dec1ab193c57f21091340cb62543d17a8ece93fca",
        "e9dd6e7c2caeee7c721b4fc4a457774279a0dfe19140e8c437c6bdb668ed222a"),
    "ros2_ws/src/drone_nav/test/test_m100_applied_config_telemetry_contract.py": (
        "1fd1aa9e897483909dd59d42a87023f1bd392e4d49f07bd34767ee5feddafafb",
        "e6c6a8b5bc3aad6b5a6488a10fdcecb7553fe4755087fea177a83ef4bad0c627"),
}


def ensure_runtime_patch(runtime_root=None):
    """Apply once to the identified runtime sources or recognize an applied patch."""
    root = runtime_path(runtime_root)
    hashes = [hashlib.sha256((root / name).read_bytes()).hexdigest() for name in PATCH_HASHES]
    before, after = zip(*PATCH_HASHES.values(), strict=True)
    if tuple(hashes) == after:
        return "ALREADY_APPLIED"
    if tuple(hashes) != before:
        raise RuntimeError(f"unexpected runtime source hashes in {root}; see {PATCH.name} for manual review")
    command = ["git", "-C", str(root), "apply"]
    subprocess.run([*command, "--check", str(PATCH)], check=True)
    subprocess.run([*command, str(PATCH)], check=True)
    for name, (_, expected) in PATCH_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise OSError(f"runtime verification failed for {name}")
    return "APPLIED"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply the version-guarded M100 density correction")
    parser.add_argument("--apply-runtime", action="store_true", required=True)
    parser.add_argument("--runtime-root", type=Path)
    options = parser.parse_args()
    print(ensure_runtime_patch(options.runtime_root))
