"""Gong et al. multirotor forward-flight electrical-power development model.

Source
------
H. Gong, B. Huang, B. Jia, and H. Dai, "Modeling Power Consumptions
for Multirotor UAVs," IEEE Transactions on Aerospace and Electronic
Systems, 59(6), 7409-7422, 2023. DOI: 10.1109/TAES.2023.3288846.

The published horizontal-flight equation is retained as four positive,
aircraft-level effective parameter groups.  These parameters must not be
interpreted as independently measured raw rotor, drag, motor, or ESC
coefficients.  The module is deliberately independent of PX4 and Gazebo: it
consumes vehicle state produced by the existing simulation; it does not alter
the flight dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


GONG_DOI = "10.1109/TAES.2023.3288846"
ISA_SOURCE = "U.S. Standard Atmosphere 1976; NASA-TM-X-74335"

ISA_T0_K = 288.15
ISA_P0_PA = 101325.0
ISA_LAPSE_K_PER_M = 0.0065
ISA_G_MPS2 = 9.80665
ISA_R_AIR_J_PER_KG_K = 287.05287


def _finite_positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return number


def _finite_nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True)
class GongGeometry:
    """Known M100 geometry and atmosphere constants."""

    rotor_count: int = 4
    rotor_radius_m: float = 0.1651
    gravity_mps2: float = ISA_G_MPS2

    def __post_init__(self) -> None:
        if int(self.rotor_count) != self.rotor_count or self.rotor_count <= 0:
            raise ValueError("rotor_count must be a positive integer")
        _finite_positive(self.rotor_radius_m, "rotor_radius_m")
        _finite_positive(self.gravity_mps2, "gravity_mps2")

    @property
    def rotor_disc_area_m2(self) -> float:
        """Disc area of one rotor, not the sum over all rotors."""
        return math.pi * self.rotor_radius_m**2


@dataclass(frozen=True)
class GongEffectiveParameters:
    """Positive aircraft-level groups used by the M100 development model."""

    theta_p: float
    theta_T: float
    theta_i: float
    theta_d: float

    def __post_init__(self) -> None:
        for name in ("theta_p", "theta_T", "theta_i", "theta_d"):
            _finite_positive(getattr(self, name), name)

    @property
    def linear_groups(self) -> tuple[float, float, float, float]:
        """Return [theta_p, theta_p*theta_T, theta_i, theta_d]."""
        return (
            self.theta_p,
            self.theta_p * self.theta_T,
            self.theta_i,
            self.theta_d,
        )


@dataclass(frozen=True)
class GongPowerComponents:
    """Auditable positive contributions of the implemented equation."""

    profile_hover_w: float
    profile_forward_w: float
    induced_w: float
    parasite_w: float
    induced_velocity_hover_mps: float
    air_density_kg_m3: float
    air_relative_speed_mps: float
    total_mass_kg: float

    @property
    def profile_total_w(self) -> float:
        return self.profile_hover_w + self.profile_forward_w

    @property
    def total_power_w(self) -> float:
        return (
            self.profile_hover_w
            + self.profile_forward_w
            + self.induced_w
            + self.parasite_w
        )


def isa_density_kg_m3(altitude_asl_m: float) -> float:
    """Lower-troposphere U.S. Standard Atmosphere density.

    The altitude is geometric height above mean sea level.  No flight-specific
    density is fitted.
    """
    altitude = float(altitude_asl_m)
    if not math.isfinite(altitude) or altitude < -500.0 or altitude >= 11000.0:
        raise ValueError("altitude_asl_m must be finite and inside the ISA troposphere")
    temperature = ISA_T0_K - ISA_LAPSE_K_PER_M * altitude
    exponent = ISA_G_MPS2 / (ISA_R_AIR_J_PER_KG_K * ISA_LAPSE_K_PER_M)
    pressure = ISA_P0_PA * (temperature / ISA_T0_K) ** exponent
    return pressure / (ISA_R_AIR_J_PER_KG_K * temperature)


def gong_linear_basis(
    total_mass_kg: float,
    air_relative_speed_mps: float,
    altitude_asl_m: float,
    geometry: GongGeometry = GongGeometry(),
) -> tuple[float, float, float, float]:
    """Return the four linear basis values of the effective Gong equation.

    If ``b = gong_linear_basis(...)`` and ``q`` is
    ``parameters.linear_groups``, then ``P = sum(b_i*q_i)``.  The second
    fitted linear group is exactly ``theta_p*theta_T``; it is not a fifth
    independent parameter.
    """
    mass = _finite_positive(total_mass_kg, "total_mass_kg")
    speed = _finite_nonnegative(air_relative_speed_mps, "air_relative_speed_mps")
    density = isa_density_kg_m3(altitude_asl_m)
    rotor_count = float(geometry.rotor_count)
    area = geometry.rotor_disc_area_m2
    weight = mass * geometry.gravity_mps2
    induced_velocity = math.sqrt(weight / (2.0 * rotor_count * density * area))

    speed_ratio = speed**2 / (2.0 * induced_velocity**2)
    # Published term: sqrt(sqrt(1 + V^4/(4*v0^4)) - V^2/(2*v0^2)).
    # The max guards only round-off at the analytical zero lower limit.
    induced_inner = max(0.0, math.sqrt(1.0 + speed_ratio**2) - speed_ratio)
    induced_factor = math.sqrt(induced_inner)

    return (
        weight**1.5 / (8.0 * math.sqrt(rotor_count * density * area)),
        (3.0 / 8.0)
        * math.sqrt(weight * rotor_count * density * area)
        * speed**2,
        weight**1.5
        / math.sqrt(2.0 * rotor_count * density * area)
        * induced_factor,
        (rotor_count / 2.0) * density * speed**3,
    )


def gong_power_components(
    total_mass_kg: float,
    air_relative_speed_mps: float,
    altitude_asl_m: float,
    parameters: GongEffectiveParameters,
    geometry: GongGeometry = GongGeometry(),
) -> GongPowerComponents:
    """Evaluate the effective form of Gong et al. horizontal-flight power.

    Implemented equation::

        P = theta_p/8 * W^(3/2)/sqrt(n*rho*A)
          + 3*theta_p*theta_T/8 * sqrt(W*n*rho*A) * V_a^2
          + theta_i * W^(3/2)/sqrt(2*n*rho*A) * F(V_a, v0)
          + n*theta_d/2 * rho * V_a^3

        F = sqrt(sqrt(1 + V_a^4/(4*v0^4)) - V_a^2/(2*v0^2))
        v0 = sqrt(W/(2*n*rho*A))

    The output estimates whole-aircraft electrical power because conversion
    losses are absorbed into the fitted effective groups.  It is not a shaft-
    power, motor-efficiency, ESC-efficiency, or dynamic-battery model.
    """
    basis = gong_linear_basis(
        total_mass_kg, air_relative_speed_mps, altitude_asl_m, geometry
    )
    groups = parameters.linear_groups
    density = isa_density_kg_m3(altitude_asl_m)
    weight = float(total_mass_kg) * geometry.gravity_mps2
    induced_velocity = math.sqrt(
        weight
        / (2.0 * float(geometry.rotor_count) * density * geometry.rotor_disc_area_m2)
    )
    return GongPowerComponents(
        profile_hover_w=basis[0] * groups[0],
        profile_forward_w=basis[1] * groups[1],
        induced_w=basis[2] * groups[2],
        parasite_w=basis[3] * groups[3],
        induced_velocity_hover_mps=induced_velocity,
        air_density_kg_m3=density,
        air_relative_speed_mps=float(air_relative_speed_mps),
        total_mass_kg=float(total_mass_kg),
    )


def gong_power_w(
    total_mass_kg: float,
    air_relative_speed_mps: float,
    altitude_asl_m: float,
    parameters: GongEffectiveParameters,
    geometry: GongGeometry = GongGeometry(),
) -> float:
    return gong_power_components(
        total_mass_kg,
        air_relative_speed_mps,
        altitude_asl_m,
        parameters,
        geometry,
    ).total_power_w


class GongEnergyIntegrator:
    """Trapezoidal Wh integrator with the established maximum-gap guard."""

    def __init__(self, maximum_valid_gap_s: float = 0.5):
        self.maximum_valid_gap_s = _finite_positive(
            maximum_valid_gap_s, "maximum_valid_gap_s"
        )
        self.reset()

    def reset(self) -> None:
        self.energy_wh = 0.0
        self.last_timestamp_s: float | None = None
        self.last_power_w: float | None = None

    def update(self, timestamp_s: float, power_w: float) -> float:
        timestamp = float(timestamp_s)
        power = _finite_nonnegative(power_w, "power_w")
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if self.last_timestamp_s is not None and self.last_power_w is not None:
            dt = timestamp - self.last_timestamp_s
            if 0.0 < dt <= self.maximum_valid_gap_s:
                self.energy_wh += 0.5 * (self.last_power_w + power) * dt / 3600.0
        self.last_timestamp_s = timestamp
        self.last_power_w = power
        return self.energy_wh


def load_gong_configuration(
    path: str | Path,
) -> tuple[GongGeometry, GongEffectiveParameters, Mapping[str, Any]]:
    """Load the dedicated, frozen Gong configuration."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise ValueError("Gong configuration root must be a mapping")
    if data.get("schema") != "M100_GONG_POWER_MODEL_CONFIG_V1":
        raise ValueError("unsupported Gong configuration schema")
    geometry_data = data.get("geometry", {})
    parameter_data = data.get("parameters", {})
    geometry = GongGeometry(
        rotor_count=int(geometry_data["rotor_count"]),
        rotor_radius_m=float(geometry_data["rotor_radius_m"]),
        gravity_mps2=float(geometry_data["gravity_mps2"]),
    )
    parameters = GongEffectiveParameters(
        theta_p=float(parameter_data["theta_p"]),
        theta_T=float(parameter_data["theta_T"]),
        theta_i=float(parameter_data["theta_i"]),
        theta_d=float(parameter_data["theta_d"]),
    )
    return geometry, parameters, data
