"""Pure experiment-contract helpers for the PX4 / Gazebo M100 backend.

This module deliberately has no ROS or Gazebo imports so that the physical
configuration rules can be unit-tested without starting the simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from .m100_atmosphere import (
    canonical_temperature_c,
    is_reference_temperature_c,
)


PX4_ROOT = Path("/home/abdulah/PX4-Autopilot")
WORLD_NAME = "m100_experiment_park"
TEMPERATURE_DENSITY_LIMITATION_WARNING = (
    "TEMPERATURE_ROTOR_DENSITY_THEORY_NO_M100_HOLDOUT"
)
STATIC_THRUST_RESERVE_STATUS = "C_M100_SAME_FAMILY_SINGLE_POINT_STATIC_PROXY"
STATIC_THRUST_RESERVE_BASIS = "M100_E800_FAMILY_STATIC_25V_ENDPOINT_PROXY"
STATIC_THRUST_RESERVE_DENSITY_STATUS = (
    "C_DENSITY_SCALED_STATIC_MODEL_PROXY"
)
STATIC_THRUST_RESERVE_DENSITY_BASIS = (
    "M100_E800_FAMILY_FIXED_ALTITUDE_DRY_AIR_DENSITY_SCALED_PROXY"
)
M100_V2_PUBLISHED_STATUS = "B_M100_SYSTEM_LEVEL_EMPIRICAL_PUBLISHED_MODEL"
M100_V2_EXTRAPOLATION_STATUS = "LIMITED_PUBLISHED_M100_PHASE_MODEL_EXTRAPOLATION"
M100_V2_HOVER_STATUS = "LIMITED_REAL_M100_HOVER_MEAN_N2_POST_HOC"
M100_V2_UNAVAILABLE_STATUS = "UNAVAILABLE_UNSUPPORTED_FLIGHT_PHASE"
M100_V2_ROUNDED_COEFFICIENT_WARNING = (
    "PUBLISHED_ONE_DECIMAL_COEFFICIENTS_68_FLIGHT_HOLDOUT_MARE_2P066_PERCENT"
)
M100_V2_FULL_PRECISION_LIMITATION_WARNING = (
    "ORIGINAL_2022_FULL_PRECISION_COEFFICIENTS_NOT_PUBLISHED"
)
M100_V2_PHASE_AVERAGE_WARNING = "PHASE_AVERAGE_NOT_INSTANTANEOUS_OR_COMPONENT_MODEL"
M100_V2_OUTSIDE_ENVELOPE_WARNING = "M100_V2_OUTSIDE_REPORTED_TRANSFER_ENVELOPE"
M100_V2_HOVER_WARNING = "HOVER_MEAN_POST_HOC_NON_PRISTINE_N2"
STANDARD_GRAVITY_MPS2 = 9.80665

# These values size only the experiment manager's wait window. They do not
# command PX4 descent speed or change controller tuning. The current verified
# M100 PX4 runtime uses MPC_Z_V_AUTO_DN=1.5 m/s; a separate 30 s margin covers
# the slower terminal landing segment and automatic-disarm detection.
M100_VERIFIED_AUTO_DESCENT_RATE_MPS = 1.5
M100_LANDING_SETTLE_MARGIN_S = 30.0
M100_MINIMUM_LANDING_TIMEOUT_S = 60.0

# Mission-landing acceptance values.  These are supervisory position / speed
# gates only; they do not change any PX4 controller gain or vehicle parameter.
M100_HOME_CENTER_TOLERANCE_M = 0.15
M100_HOME_CENTER_SPEED_MPS = 0.15
M100_HOME_CENTER_SETTLE_S = 2.0
M100_OFFBOARD_DESCENT_TARGET_M = 4.0
M100_FINAL_LANDING_TOLERANCE_M = 0.35


@dataclass(frozen=True)
class PayloadModel:
    payload_kg: float
    target: str
    model: str
    autostart: int
    total_mass_kg: float


PAYLOAD_MODELS = {
    0.00: PayloadModel(0.00, "gz_m100_cmu_v2", "m100_cmu_v2", 4030, 3.680),
    0.25: PayloadModel(
        0.25,
        "gz_m100_cmu_v2_payload_025",
        "m100_cmu_v2_payload_025",
        4031,
        3.930,
    ),
    0.50: PayloadModel(
        0.50,
        "gz_m100_cmu_v2_payload_050",
        "m100_cmu_v2_payload_050",
        4032,
        4.180,
    ),
    0.75: PayloadModel(
        0.75,
        "gz_m100_cmu_v2_payload_075",
        "m100_cmu_v2_payload_075",
        4033,
        4.430,
    ),
    1.00: PayloadModel(
        1.00,
        "gz_m100_cmu_v2_payload_100",
        "m100_cmu_v2_payload_100",
        4034,
        4.680,
    ),
    1.25: PayloadModel(
        1.25,
        "gz_m100_cmu_v2_payload_125",
        "m100_cmu_v2_payload_125",
        4035,
        4.930,
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    temperature_c: float
    payload: PayloadModel
    wind_speed_mps: float
    wind_direction_deg: float
    wind_vector_enu_mps: tuple[float, float, float]
    cruise_speed_mps: float
    altitude_m: float
    route_length_m: float
    hover_duration_s: float

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["payload_kg"] = self.payload.payload_kg
        result["total_mass_kg"] = self.payload.total_mass_kg
        result["px4_target"] = self.payload.target
        result["gazebo_model"] = self.payload.model
        result.pop("payload")
        measurement_phase = "HOVER" if self.cruise_speed_mps == 0.0 else "CRUISE"
        assessment = m100_v2_energy_assessment(
            self,
            measurement_phase,
            measurement_active=True,
        )
        result["energy_calibration_status"] = assessment.status
        result["energy_calibration_warnings"] = list(assessment.warnings)
        result["route_definition"] = "ONE_WAY_PLUS_X_ENU_MEASUREMENT_DISTANCE"
        return result


@dataclass(frozen=True)
class StaticThrustReserve:
    """Informational thrust reserve at the frozen 25 V model endpoint.

    This single-point proxy is not a wind-, speed-, temperature-, inflow-, or
    battery-aware flight limit and must not be used to accept or reject a run.
    """

    maximum_static_thrust_n: float
    weight_n: float
    thrust_to_weight_ratio: float
    reserve_percent_of_maximum: float
    status: str = STATIC_THRUST_RESERVE_STATUS
    basis: str = STATIC_THRUST_RESERVE_BASIS


@dataclass(frozen=True)
class M100V2EnergyAssessment:
    """Applicability label for the active published whole-aircraft model."""

    status: str
    warnings: tuple[str, ...]
    note: str
    inside_reported_envelope: bool
    use_limited_hover_mean: bool = False


def m100_v2_energy_assessment(
    config: ExperimentConfig | None,
    phase: str,
    *,
    measurement_active: bool = False,
) -> M100V2EnergyAssessment:
    """Classify the active M100 model without claiming component fidelity.

    The Rodrigues / Patterns model is a phase-average, aircraft-level model.
    It does not identify propeller torque, motor or ESC efficiency, speed drag,
    or a causal wind-energy transfer.  The narrow hover value is a separate
    two-flight post-hoc diagnostic and is enabled only in its declared window.
    """
    if config is None:
        return M100V2EnergyAssessment(
            M100_V2_UNAVAILABLE_STATUS,
            (),
            "No applied experiment configuration is available.",
            False,
        )
    phase_name = str(phase).upper()
    zero_wind = math.isclose(config.wind_speed_mps, 0.0, abs_tol=1e-6)
    reference_temperature = is_reference_temperature_c(config.temperature_c)
    if phase_name == "HOVER":
        eligible = (
            measurement_active
            and math.isclose(config.payload.payload_kg, 0.0, abs_tol=1e-6)
            and math.isclose(config.cruise_speed_mps, 0.0, abs_tol=1e-6)
            and zero_wind
            and reference_temperature
            and 3.0 - 1e-6 <= config.altitude_m <= 5.0 + 1e-6
            and 10.0 - 1e-6 <= config.hover_duration_s <= 20.0 + 1e-6
        )
        if eligible:
            return M100V2EnergyAssessment(
                M100_V2_HOVER_STATUS,
                (M100_V2_HOVER_WARNING,),
                "Direct mean real-M100 hover power from two development flights; "
                "post-hoc and not an independent validation.",
                False,
                True,
            )
        return M100V2EnergyAssessment(
            M100_V2_UNAVAILABLE_STATUS,
            (M100_V2_OUTSIDE_ENVELOPE_WARNING,),
            "No general hover model is enabled outside the narrow real-M100 mean.",
            False,
        )
    if phase_name not in {"TAKEOFF", "CLIMB", "CRUISE", "RETURNING", "LANDING"}:
        return M100V2EnergyAssessment(
            M100_V2_UNAVAILABLE_STATUS,
            (),
            f"The published M100 phase model does not support phase {phase_name}.",
            False,
        )
    cruise_envelope = (
        phase_name in {"CRUISE", "RETURNING"}
        and zero_wind
        and reference_temperature
        and _matches_one(config.payload.payload_kg, (0.0, 0.25, 0.5))
        and 4.0 - 1e-6 <= config.cruise_speed_mps <= 12.0 + 1e-6
        and 25.0 - 1e-6 <= config.altitude_m <= 100.0 + 1e-6
    )
    warnings = [
        M100_V2_ROUNDED_COEFFICIENT_WARNING,
        M100_V2_FULL_PRECISION_LIMITATION_WARNING,
        M100_V2_PHASE_AVERAGE_WARNING,
    ]
    if cruise_envelope:
        return M100V2EnergyAssessment(
            M100_V2_PUBLISHED_STATUS,
            tuple(warnings),
            "Within the declared zero-commanded-wind M100 cruise transfer envelope; "
            "the output is phase-average whole-aircraft electrical power.",
            True,
        )
    warnings.append(M100_V2_OUTSIDE_ENVELOPE_WARNING)
    if not reference_temperature:
        warnings.append(TEMPERATURE_DENSITY_LIMITATION_WARNING)
    return M100V2EnergyAssessment(
        M100_V2_EXTRAPOLATION_STATUS,
        tuple(warnings),
        "Published M100 phase-average power is being extrapolated outside the "
        "declared local transfer envelope.",
        False,
    )


def _matches_one(value: float, choices: tuple[float, ...]) -> bool:
    return any(math.isclose(float(value), choice, abs_tol=1e-6) for choice in choices)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def static_thrust_reserve(
    total_mass_kg: float,
    motor_constant: float,
    max_rot_velocity_rad_s: float,
    *,
    rotor_count: int = 4,
    status: str = STATIC_THRUST_RESERVE_STATUS,
    basis: str = STATIC_THRUST_RESERVE_BASIS,
) -> StaticThrustReserve:
    """Calculate a non-operational static-thrust reserve proxy.

    The caller must label any density-scaled motor constant with the matching
    non-calibrated status and basis.  No controller, motor, battery, or Gazebo
    parameter is changed by this calculation.
    """
    mass = _finite(total_mass_kg, "total_mass_kg")
    thrust_constant = _finite(motor_constant, "motor_constant")
    maximum_omega = _finite(max_rot_velocity_rad_s, "max_rot_velocity_rad_s")
    if mass <= 0.0 or thrust_constant <= 0.0 or maximum_omega <= 0.0:
        raise ValueError("static thrust inputs must be positive")
    if isinstance(rotor_count, bool) or int(rotor_count) != rotor_count or rotor_count <= 0:
        raise ValueError("rotor_count must be a positive integer")

    maximum_thrust = int(rotor_count) * thrust_constant * maximum_omega**2
    weight = mass * STANDARD_GRAVITY_MPS2
    return StaticThrustReserve(
        maximum_static_thrust_n=maximum_thrust,
        weight_n=weight,
        thrust_to_weight_ratio=maximum_thrust / weight,
        reserve_percent_of_maximum=100.0 * (maximum_thrust - weight) / maximum_thrust,
        status=str(status),
        basis=str(basis),
    )


def landing_timeout_limit(altitude_m: float) -> float:
    """Return an altitude-aware supervisory wait for PX4 LAND completion."""
    altitude = _finite(altitude_m, "landing_altitude_m")
    if altitude < 0.0:
        raise ValueError("landing_altitude_m must be non-negative")
    return max(
        M100_MINIMUM_LANDING_TIMEOUT_S,
        altitude / M100_VERIFIED_AUTO_DESCENT_RATE_MPS
        + M100_LANDING_SETTLE_MARGIN_S,
    )


def measurement_energy_per_distance(
    measurement_energy_wh: float | None,
    measurement_distance_m: float | None,
) -> float | None:
    """Return Wh/km for the declared measurement segment only.

    Total mission distance must never be used here: it includes climb, return,
    landing and post-touchdown position jitter, while the supervisor's metric
    is the energy of the one-way outbound measurement route.
    """
    if measurement_energy_wh is None or measurement_distance_m is None:
        return None
    energy = float(measurement_energy_wh)
    distance = float(measurement_distance_m)
    if not math.isfinite(energy) or not math.isfinite(distance) or distance <= 0.0:
        return None
    return energy / (distance / 1000.0)


def select_payload(value: float) -> PayloadModel:
    value = _finite(value, "payload_kg")
    for approved, payload in PAYLOAD_MODELS.items():
        if math.isclose(value, approved, abs_tol=1e-6):
            return payload
    allowed = ", ".join(f"{item:.2f}" for item in PAYLOAD_MODELS)
    raise ValueError(f"payload_kg must be one of: {allowed}")


def wind_vector_enu(speed_mps: float, direction_deg: float) -> tuple[float, float, float]:
    """Return the Gazebo vector moving *towards* the entered ENU direction.

    With the experiment outbound route along world +X: 0 degrees is tailwind,
    90 degrees is +Y crosswind and 180 degrees is headwind.
    """
    speed = _finite(speed_mps, "wind_speed_mps")
    direction = _finite(direction_deg, "wind_direction_deg")
    if not 0.0 <= speed <= 15.0:
        raise ValueError("wind_speed_mps must be in [0, 15]")
    if not 0.0 <= direction <= 360.0:
        raise ValueError("wind_direction_deg must be in [0, 360]")
    angle = math.radians(direction % 360.0)
    values = (speed * math.cos(angle), speed * math.sin(angle), 0.0)
    return tuple(0.0 if abs(value) < 1e-12 else round(value, 12) for value in values)


def build_config(
    temperature_c: float,
    payload_kg: float,
    wind_speed_mps: float,
    wind_direction_deg: float,
    cruise_speed_mps: float,
    altitude_m: float,
    route_length_m: float,
    hover_duration_s: float,
) -> ExperimentConfig:
    temperature = canonical_temperature_c(_finite(temperature_c, "temperature_c"))
    speed = _finite(cruise_speed_mps, "cruise_speed_mps")
    altitude = _finite(altitude_m, "altitude_m")
    route = _finite(route_length_m, "route_length_m")
    hover = _finite(hover_duration_s, "hover_duration_s")
    if not -20.0 <= temperature <= 60.0:
        raise ValueError("temperature_c must be in [-20, 60]")
    if not 0.0 <= speed <= 20.0:
        raise ValueError("cruise_speed_mps must be in [0, 20]")
    # 200 m is an explicitly requested simulation scenario, not an extension
    # of the empirical energy envelope (which remains capped at 100 m).
    if not 1.0 <= altitude <= 200.0:
        raise ValueError("altitude_m must be in [1, 200]")
    if speed > 0.0 and route <= 0.0:
        raise ValueError("route_length_m must be positive for cruise")
    if route < 0.0 or route > 550.0:
        raise ValueError("route_length_m must be in [0, 550]")
    if not 5.0 <= hover <= 600.0:
        raise ValueError("hover_duration_s must be in [5, 600]")
    payload = select_payload(payload_kg)
    wind = wind_vector_enu(wind_speed_mps, wind_direction_deg)
    return ExperimentConfig(
        temperature,
        payload,
        float(wind_speed_mps),
        float(wind_direction_deg) % 360.0,
        wind,
        speed,
        altitude,
        route,
        hover,
    )


def parse_apply_command(command: str) -> ExperimentConfig:
    fields = command.split()
    if len(fields) != 15 or fields[0].lower() != "apply":
        raise ValueError("expected APPLY followed by 14 numeric fields")
    values = [float(value) for value in fields[1:]]
    # The last six values are the preserved legacy SJTU PID fields. They are
    # checked for numeric validity by the GUI but intentionally ignored here.
    return build_config(
        temperature_c=values[0],
        wind_speed_mps=values[1],
        wind_direction_deg=values[2],
        payload_kg=values[3],
        cruise_speed_mps=values[4],
        altitude_m=values[5],
        route_length_m=values[6],
        hover_duration_s=values[7],
    )


CSV_SCHEMA = "M100_CURRENT_ENERGY_CSV_V1"
CSV_SAMPLE_INTERVAL_S = 1.0

# The public recording/export contains only current experiment variables.
# Run identity, model provenance and applicability belong in the JSON sidecar.
CSV_FIELDS = (
    "energy_timestamp_s",
    "flight_phase",
    "temperature_c",
    "wind_speed_command_mps",
    "wind_direction_command_deg",
    "payload_kg",
    "commanded_speed_mps",
    "actual_horizontal_speed_mps",
    "commanded_altitude_m",
    "actual_altitude_m",
    "distance_m",
    "measurement_distance_m",
    "battery_voltage_V",
    "battery_current_A",
    "electrical_power_W",
    "cumulative_energy_Wh",
    "measurement_energy_Wh",
    "energy_per_distance_Wh_per_km",
)
