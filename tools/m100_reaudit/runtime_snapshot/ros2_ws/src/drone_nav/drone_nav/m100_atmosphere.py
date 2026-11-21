"""Pure dry-air density model for controlled M100 simulation studies.

The world datum is explicitly assumed to be 0 m mean sea level (MSL), so a
Gazebo world altitude in metres is used directly as geometric altitude above
the pressure datum.  Surface pressure is held at ``P0_PA`` for both the case
temperature and the 15 °C reference.  Humidity, weather-pressure variation,
and non-standard lapse rates are outside this deliberately limited model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


MODEL = "DRY_HYDROSTATIC_LINEAR_LAPSE"
WORLD_DATUM_MSL_M = 0.0
WORLD_DATUM_ASSUMPTION = "GAZEBO_WORLD_DATUM_IS_0_M_MSL"

G_M_S2 = 9.80665
R_DRY_AIR_J_KG_K = 287.05287
LAPSE_RATE_K_PER_M = 0.0065
P0_PA = 101325.0
REFERENCE_TEMPERATURE_C = 15.0
REFERENCE_TEMPERATURE_ABS_TOLERANCE_C = 1e-6
KELVIN_OFFSET = 273.15

# These conservative project validation limits cover the planned thesis
# matrix while remaining inside the lower-troposphere lapse-rate assumption.
MIN_TEMPERATURE_C_AT_DATUM = -20.0
MAX_TEMPERATURE_C_AT_DATUM = 60.0
MIN_ALTITUDE_M_MSL = 0.0
MAX_ALTITUDE_M_MSL = 5000.0


@dataclass(frozen=True)
class DryAirState:
    """Pressure and density at one altitude for one datum temperature."""

    temperature_c_at_datum: float
    altitude_m_msl: float
    temperature_k_at_altitude: float
    pressure_pa: float
    density_kg_m3: float


@dataclass(frozen=True)
class AtmosphereDensityComparison:
    """Case density and its fixed 15 °C reference at the same altitude."""

    model: str
    world_datum_msl_m: float
    world_datum_assumption: str
    case: DryAirState
    reference: DryAirState
    density_ratio: float


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validated_inputs(
    temperature_c_at_datum: Any, altitude_m_msl: Any
) -> tuple[float, float]:
    temperature = _finite_number(
        temperature_c_at_datum, "temperature_c_at_datum"
    )
    if math.isclose(
        temperature,
        REFERENCE_TEMPERATURE_C,
        rel_tol=0.0,
        abs_tol=REFERENCE_TEMPERATURE_ABS_TOLERANCE_C,
    ):
        temperature = REFERENCE_TEMPERATURE_C
    altitude = _finite_number(altitude_m_msl, "altitude_m_msl")
    if not MIN_TEMPERATURE_C_AT_DATUM <= temperature <= MAX_TEMPERATURE_C_AT_DATUM:
        raise ValueError(
            "temperature_c_at_datum must be in "
            f"[{MIN_TEMPERATURE_C_AT_DATUM}, {MAX_TEMPERATURE_C_AT_DATUM}] °C"
        )
    if not MIN_ALTITUDE_M_MSL <= altitude <= MAX_ALTITUDE_M_MSL:
        raise ValueError(
            "altitude_m_msl must be in "
            f"[{MIN_ALTITUDE_M_MSL}, {MAX_ALTITUDE_M_MSL}] m"
        )
    return temperature, altitude


def canonical_temperature_c(value: Any) -> float:
    """Return the single canonical representation of the 15 C reference.

    APPLY inputs within one micro-degree of the reference are stored as exact
    ``15.0``.  Every atmosphere, SDF, and calibration branch therefore makes
    the same decision and cannot label a derived model as reference data.
    """

    temperature = _finite_number(value, "temperature_c_at_datum")
    if math.isclose(
        temperature,
        REFERENCE_TEMPERATURE_C,
        rel_tol=0.0,
        abs_tol=REFERENCE_TEMPERATURE_ABS_TOLERANCE_C,
    ):
        return REFERENCE_TEMPERATURE_C
    return temperature


def is_reference_temperature_c(value: Any) -> bool:
    """Return whether ``value`` canonicalizes to the 15 C reference."""

    return canonical_temperature_c(value) == REFERENCE_TEMPERATURE_C


def dry_air_state(
    temperature_c_at_datum: Any, altitude_m_msl: Any
) -> DryAirState:
    r"""Return the dry hydrostatic state under a linear temperature lapse.

    The implemented equations are

    .. math::

       T(h)=T_0-Lh,
       \quad p(h)=p_0\left(\frac{T(h)}{T_0}\right)^{g/(R L)},
       \quad \rho(h)=\frac{p(h)}{R T(h)}.
    """

    temperature_c, altitude_m = _validated_inputs(
        temperature_c_at_datum, altitude_m_msl
    )
    temperature_k_at_datum = temperature_c + KELVIN_OFFSET
    temperature_k_at_altitude = (
        temperature_k_at_datum - LAPSE_RATE_K_PER_M * altitude_m
    )
    if temperature_k_at_altitude <= 0.0:  # defensive beyond validated limits
        raise ValueError("linear lapse produced a non-physical temperature")
    exponent = G_M_S2 / (R_DRY_AIR_J_KG_K * LAPSE_RATE_K_PER_M)
    pressure_pa = P0_PA * (
        temperature_k_at_altitude / temperature_k_at_datum
    ) ** exponent
    density_kg_m3 = pressure_pa / (
        R_DRY_AIR_J_KG_K * temperature_k_at_altitude
    )
    return DryAirState(
        temperature_c_at_datum=temperature_c,
        altitude_m_msl=altitude_m,
        temperature_k_at_altitude=temperature_k_at_altitude,
        pressure_pa=pressure_pa,
        density_kg_m3=density_kg_m3,
    )


def compare_density_to_15c_reference(
    temperature_c_at_datum: Any, altitude_m_msl: Any
) -> AtmosphereDensityComparison:
    """Return case/reference densities and ``rho_case / rho_reference``.

    The 15 °C branch deliberately reuses the same immutable state so the
    reference condition has exact, rather than merely approximate, identity.
    """

    temperature_c, altitude_m = _validated_inputs(
        temperature_c_at_datum, altitude_m_msl
    )
    reference = dry_air_state(REFERENCE_TEMPERATURE_C, altitude_m)
    if temperature_c == REFERENCE_TEMPERATURE_C:
        case = reference
        density_ratio = 1.0
    else:
        case = dry_air_state(temperature_c, altitude_m)
        density_ratio = case.density_kg_m3 / reference.density_kg_m3
    return AtmosphereDensityComparison(
        model=MODEL,
        world_datum_msl_m=WORLD_DATUM_MSL_M,
        world_datum_assumption=WORLD_DATUM_ASSUMPTION,
        case=case,
        reference=reference,
        density_ratio=density_ratio,
    )


__all__ = [
    "AtmosphereDensityComparison",
    "DryAirState",
    "G_M_S2",
    "LAPSE_RATE_K_PER_M",
    "MAX_ALTITUDE_M_MSL",
    "MAX_TEMPERATURE_C_AT_DATUM",
    "MIN_ALTITUDE_M_MSL",
    "MIN_TEMPERATURE_C_AT_DATUM",
    "MODEL",
    "P0_PA",
    "REFERENCE_TEMPERATURE_ABS_TOLERANCE_C",
    "REFERENCE_TEMPERATURE_C",
    "R_DRY_AIR_J_KG_K",
    "WORLD_DATUM_ASSUMPTION",
    "WORLD_DATUM_MSL_M",
    "canonical_temperature_c",
    "compare_density_to_15c_reference",
    "dry_air_state",
    "is_reference_temperature_c",
]
