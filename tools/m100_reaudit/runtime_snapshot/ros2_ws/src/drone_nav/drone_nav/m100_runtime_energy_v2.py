"""Measurement-window energy bookkeeping for the active M100 V2 model.

The operational input is whole-aircraft electrical power from the published
M100 phase model.  The module deliberately has no propulsion-torque or rotor
mechanical-power dependency.  Gong et al. remains a parallel research-only
diagnostic and never changes the operational power or battery budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
from typing import Any, Mapping

from .gong_power_model import (
    GongEffectiveParameters,
    GongGeometry,
    gong_power_w,
    load_gong_configuration,
)


OPERATIONAL_MODEL = "RODRIGUES_PATTERNS_M100_PHASE_POWER_V1"
RESEARCH_MODEL = "GONG_2023"
RESEARCH_CLASSIFICATION = "C_NOT_CALIBRATED"
WIND_DELTA_STATUS = "RESEARCH_ONLY_WIND_AIRSPEED_DELTA_RUNTIME_REJECTED"
WIND_DELTA_WARNING = (
    "WIND_TRANSFER_REJECTED_CAUSAL_DIRECTIONAL_EFFECT_NOT_IDENTIFIED"
)


class GuardedTrapezoidalIntegrator:
    """Integrate non-negative power without bridging invalid time intervals."""

    def __init__(self, maximum_valid_gap_s: float):
        maximum_gap = float(maximum_valid_gap_s)
        if not math.isfinite(maximum_gap) or maximum_gap <= 0.0:
            raise ValueError("maximum_valid_gap_s must be finite and positive")
        self.maximum_valid_gap_s = maximum_gap
        self.reset()

    def reset(self) -> None:
        self.energy_wh = 0.0
        self.last_timestamp_s: float | None = None
        self.last_power_w: float | None = None
        self.warnings: list[str] = []

    def _warn(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)

    def update(self, timestamp_s: float, power_w: float | None) -> float:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            self._warn("NON_FINITE_ENERGY_TIMESTAMP")
            self.last_timestamp_s = None
            self.last_power_w = None
            return self.energy_wh
        power = None
        if power_w is not None:
            candidate = float(power_w)
            if not math.isfinite(candidate):
                self._warn("NON_FINITE_ENERGY_POWER")
            elif candidate < 0.0:
                self._warn("NEGATIVE_ENERGY_POWER")
            else:
                power = candidate
        if self.last_timestamp_s is not None:
            dt = timestamp - self.last_timestamp_s
            if dt <= 0.0:
                self._warn("ENERGY_TIMESTAMP_RESET_OR_DUPLICATE")
            elif dt > self.maximum_valid_gap_s:
                self._warn("ENERGY_TELEMETRY_GAP_NOT_INTEGRATED")
            elif power is not None and self.last_power_w is not None:
                increment = 0.5 * (self.last_power_w + power) * dt / 3600.0
                if math.isfinite(increment) and increment >= 0.0:
                    self.energy_wh += increment
                else:
                    self._warn("INVALID_ENERGY_INCREMENT")
        self.last_timestamp_s = timestamp
        self.last_power_w = power
        return self.energy_wh


@dataclass(frozen=True)
class WindPowerDelta:
    ground_speed_mps: float
    air_relative_speed_mps: float
    gong_ground_power_w: float
    gong_air_power_w: float
    delta_power_w: float
    status: str = WIND_DELTA_STATUS
    warning: str = WIND_DELTA_WARNING


@dataclass(frozen=True)
class RuntimeEnergyV2Snapshot:
    energy_timestamp_s: float | None = None
    p_operational_w: float | None = None
    e_operational_wh: float | None = None
    p_gong_w: float | None = None
    p_gong_ground_w: float | None = None
    wind_diagnostic_delta_w: float | None = None
    e_gong_wh: float | None = None
    gong_speed_input_mps: float | None = None
    ground_speed_input_mps: float | None = None
    gong_altitude_asl_m: float | None = None
    gong_applicable: bool = False
    measurement_active: bool = False
    measurement_frozen: bool = False
    energy_model_operational: str = OPERATIONAL_MODEL
    energy_model_research: str = RESEARCH_MODEL
    gong_model_classification: str = RESEARCH_CLASSIFICATION
    warnings: tuple[str, ...] = field(default_factory=tuple)


class M100RuntimeEnergyV2:
    """Integrate active M100 power and an isolated research comparison."""

    def __init__(
        self,
        gong_geometry: GongGeometry,
        gong_parameters: GongEffectiveParameters,
        maximum_valid_gap_s: float,
        gong_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.gong_geometry = gong_geometry
        self.gong_parameters = gong_parameters
        self.gong_metadata = dict(gong_metadata or {})
        self._operational_integrator = GuardedTrapezoidalIntegrator(
            maximum_valid_gap_s
        )
        self._gong_integrator = GuardedTrapezoidalIntegrator(maximum_valid_gap_s)
        self._lock = threading.RLock()
        self.reset()

    @classmethod
    def load(
        cls, path: str | Path, maximum_valid_gap_s: float
    ) -> "M100RuntimeEnergyV2":
        geometry, parameters, metadata = load_gong_configuration(path)
        state = metadata.get("scientific_state", {})
        if state.get("runtime_enabled") is not False:
            raise ValueError("Gong must remain disabled as an operational model")
        if state.get("parallel_research_reporting_enabled") is not True:
            raise ValueError("Gong research reporting is not explicitly enabled")
        return cls(geometry, parameters, maximum_valid_gap_s, metadata)

    def reset(self) -> None:
        with self._lock:
            self._operational_integrator.reset()
            self._gong_integrator.reset()
            self._measurement_active = False
            self._measurement_frozen = False
            self._snapshot = RuntimeEnergyV2Snapshot()

    def begin_measurement(self) -> None:
        with self._lock:
            self._operational_integrator.reset()
            self._gong_integrator.reset()
            self._measurement_active = True
            self._measurement_frozen = False
            self._snapshot = RuntimeEnergyV2Snapshot(
                **{
                    **self._snapshot.__dict__,
                    "e_operational_wh": 0.0,
                    "e_gong_wh": 0.0,
                    "measurement_active": True,
                    "measurement_frozen": False,
                }
            )

    def freeze_measurement(self) -> RuntimeEnergyV2Snapshot:
        with self._lock:
            self._measurement_active = False
            self._measurement_frozen = True
            self._snapshot = RuntimeEnergyV2Snapshot(
                **{
                    **self._snapshot.__dict__,
                    "measurement_active": False,
                    "measurement_frozen": True,
                }
            )
            return self._snapshot

    def wind_power_delta(
        self,
        *,
        total_mass_kg: float,
        ground_velocity_enu_mps: tuple[float, float],
        wind_velocity_enu_mps: tuple[float, float],
        altitude_asl_m: float,
    ) -> WindPowerDelta:
        mass = self._positive(total_mass_kg)
        altitude = self._finite(altitude_asl_m)
        ground_x, ground_y = map(float, ground_velocity_enu_mps)
        wind_x, wind_y = map(float, wind_velocity_enu_mps)
        if mass is None or altitude is None or not all(
            math.isfinite(value)
            for value in (ground_x, ground_y, wind_x, wind_y)
        ):
            raise ValueError("wind diagnostic inputs must be finite")
        ground_speed = math.hypot(ground_x, ground_y)
        air_speed = math.hypot(ground_x - wind_x, ground_y - wind_y)
        ground_power = gong_power_w(
            mass, ground_speed, altitude, self.gong_parameters, self.gong_geometry
        )
        air_power = gong_power_w(
            mass, air_speed, altitude, self.gong_parameters, self.gong_geometry
        )
        return WindPowerDelta(
            ground_speed,
            air_speed,
            ground_power,
            air_power,
            air_power - ground_power,
        )

    def optional_wind_power_delta(self, **inputs) -> WindPowerDelta | None:
        """Unavailable research inputs must not interrupt flight telemetry.

        Preserve the strict diagnostic API and its valid numerical results.
        Do not clamp or replace an invalid altitude, speed, or power value.
        """
        try:
            return self.wind_power_delta(**inputs)
        except (ValueError, OverflowError):
            return None

    def update(
        self,
        *,
        timestamp_s: float,
        operational_power_w: float | None,
        total_mass_kg: float | None,
        actual_horizontal_speed_mps: float | None,
        altitude_asl_m: float | None,
        air_relative_speed_mps: float | None = None,
        gong_applicable: bool = True,
    ) -> RuntimeEnergyV2Snapshot:
        with self._lock:
            timestamp = float(timestamp_s)
            operational = self._nonnegative(operational_power_w)
            speed = self._nonnegative(actual_horizontal_speed_mps)
            air_speed = self._nonnegative(air_relative_speed_mps)
            gong_speed = air_speed if air_speed is not None else speed
            altitude = self._finite(altitude_asl_m)
            mass = self._positive(total_mass_kg)
            warnings: list[str] = []
            gong = None
            gong_ground = None
            wind_delta = None
            if (
                gong_applicable
                and mass is not None
                and gong_speed is not None
                and altitude is not None
            ):
                try:
                    gong = gong_power_w(
                        mass,
                        gong_speed,
                        altitude,
                        self.gong_parameters,
                        self.gong_geometry,
                    )
                    if air_speed is not None and speed is not None:
                        gong_ground = gong_power_w(
                            mass,
                            speed,
                            altitude,
                            self.gong_parameters,
                            self.gong_geometry,
                        )
                        wind_delta = gong - gong_ground
                        if not math.isclose(
                            air_speed, speed, rel_tol=0.0, abs_tol=1e-9
                        ):
                            warnings.append(WIND_DELTA_WARNING)
                except (ValueError, OverflowError):
                    gong = None
                    gong_ground = None
                    wind_delta = None
                    warnings.append("INVALID_GONG_INPUT")
            if self._measurement_active:
                operational_energy = self._operational_integrator.update(
                    timestamp, operational
                )
                gong_energy = self._gong_integrator.update(timestamp, gong)
            elif self._measurement_frozen:
                operational_energy = self._operational_integrator.energy_wh
                gong_energy = self._gong_integrator.energy_wh
            else:
                operational_energy = None
                gong_energy = None
            warnings.extend(self._operational_integrator.warnings)
            warnings.extend(self._gong_integrator.warnings)
            self._snapshot = RuntimeEnergyV2Snapshot(
                energy_timestamp_s=(timestamp if math.isfinite(timestamp) else None),
                p_operational_w=operational,
                e_operational_wh=operational_energy,
                p_gong_w=gong,
                p_gong_ground_w=gong_ground,
                wind_diagnostic_delta_w=wind_delta,
                e_gong_wh=gong_energy,
                gong_speed_input_mps=gong_speed,
                ground_speed_input_mps=speed,
                gong_altitude_asl_m=altitude,
                gong_applicable=bool(gong_applicable),
                measurement_active=self._measurement_active,
                measurement_frozen=self._measurement_frozen,
                warnings=tuple(dict.fromkeys(warnings)),
            )
            return self._snapshot

    def snapshot(self) -> RuntimeEnergyV2Snapshot:
        with self._lock:
            return RuntimeEnergyV2Snapshot(**self._snapshot.__dict__)

    @staticmethod
    def _finite(value: float | None) -> float | None:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    @classmethod
    def _nonnegative(cls, value: float | None) -> float | None:
        number = cls._finite(value)
        return number if number is not None and number >= 0.0 else None

    @classmethod
    def _positive(cls, value: float | None) -> float | None:
        number = cls._finite(value)
        return number if number is not None and number > 0.0 else None
