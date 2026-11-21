"""Run-scoped battery-management estimator for the M100 TB47D profile.

The component deliberately separates battery bookkeeping from the aircraft
power model.  In the current configuration its voltage and current are model
estimates derived from the verified TB47D nameplate and the selected
whole-aircraft operational power model.  A
limited terminal-voltage transfer model is fitted on the CALIBRATION flights
of the published M100 dataset and checked on its locked VALIDATION flights.
The public dataset does not resolve the physical pack identity, so this is not
an OCV/SOC or electrochemical TB47D model.  The same interface can accept
measured voltage/current later, but measured input is disabled until a
traceable sensor path exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
from typing import Any


TB47D_BMS_MODE = "TB47D_BMS_ENERGY_BUDGET_V3"
TB47D_BMS_STATUS = "LIMITED_M100_TERMINAL_VOLTAGE_TRANSFER_PACK_ID_UNRESOLVED"
TB47D_BMS_WARNING = "M100_BMS_TRANSFER_ESTIMATE_NOT_PHYSICAL_TB47D_MEASUREMENT"
TB47D_BMS_MODEL_SOURCE = (
    "OPERATIONAL_AIRCRAFT_POWER_AND_M100_PUBLISHED_PACK_TERMINAL_PROXY"
)
TB47D_BMS_MEASURED_SOURCE = "MEASURED_TERMINAL_VOLTAGE_CURRENT"
TB47D_BMS_PROXY_OUTSIDE_ENVELOPE = "BMS_TERMINAL_PROXY_OUTSIDE_VALIDATED_ENVELOPE"
TB47D_BMS_FROZEN_LAST_VALID = "BMS_TERMINAL_PROXY_FROZEN_LAST_VALID_SAMPLE"

BMS_STATES = {"INACTIVE", "ACTIVE", "FROZEN"}
BMS_SAFETY_LEVELS = {"UNAVAILABLE", "NORMAL", "WARNING", "CRITICAL", "DEPLETED"}
BMS_DATA_QUALITIES = {"UNAVAILABLE", "MODEL_ESTIMATE", "MEASURED", "DEGRADED", "FROZEN"}


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class BMSConfig:
    """Validated configuration for the run-scoped BMS estimator."""

    enabled: bool
    mode: str
    nominal_voltage_v: float
    capacity_ah: float
    nominal_energy_wh: float
    initial_soc_percent: float
    minimum_endurance_power_w: float
    maximum_valid_gap_s: float
    endurance_power_time_constant_s: float
    low_soc_warning_percent: float
    critical_soc_percent: float
    measured_telemetry_enabled: bool = False
    measured_power_tolerance_fraction: float = 0.10
    terminal_voltage_proxy_enabled: bool = False
    terminal_voltage_model_id: str | None = None
    terminal_voltage_intercept_v: float | None = None
    terminal_voltage_resistance_ohm: float | None = None
    terminal_voltage_depletion_slope_v_per_ah: float | None = None
    terminal_voltage_minimum_current_a: float = 0.0
    terminal_voltage_maximum_current_a: float | None = None
    terminal_voltage_minimum_power_w: float | None = None
    terminal_voltage_maximum_power_w: float | None = None
    terminal_voltage_maximum_throughput_ah: float | None = None
    terminal_voltage_validation_mae_v: float | None = None
    terminal_voltage_validation_rmse_v: float | None = None
    terminal_voltage_validation_p95_error_v: float | None = None
    terminal_current_validation_p95_error_a: float | None = None

    def __post_init__(self) -> None:
        if self.mode != TB47D_BMS_MODE:
            raise ValueError("unsupported TB47D BMS mode")
        for name in (
            "nominal_voltage_v",
            "capacity_ah",
            "nominal_energy_wh",
            "minimum_endurance_power_w",
            "maximum_valid_gap_s",
            "endurance_power_time_constant_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.initial_soc_percent <= 100.0:
            raise ValueError("initial_soc_percent must be in [0, 100]")
        if not 0.0 < self.critical_soc_percent < self.low_soc_warning_percent < 100.0:
            raise ValueError(
                "SOC thresholds must satisfy 0 < critical < low warning < 100"
            )
        if not 0.0 < self.measured_power_tolerance_fraction <= 1.0:
            raise ValueError("measured power tolerance must be in (0, 1]")
        expected_energy = self.nominal_voltage_v * self.capacity_ah
        if not math.isclose(expected_energy, self.nominal_energy_wh, rel_tol=1e-9):
            raise ValueError(
                "nominal_energy_wh must equal nominal_voltage_v * capacity_ah"
            )
        if self.terminal_voltage_proxy_enabled:
            if not str(self.terminal_voltage_model_id or "").strip():
                raise ValueError("terminal-voltage proxy requires a model id")
            for name in (
                "terminal_voltage_intercept_v",
                "terminal_voltage_resistance_ohm",
                "terminal_voltage_depletion_slope_v_per_ah",
                "terminal_voltage_maximum_current_a",
                "terminal_voltage_minimum_power_w",
                "terminal_voltage_maximum_power_w",
                "terminal_voltage_maximum_throughput_ah",
                "terminal_voltage_validation_mae_v",
                "terminal_voltage_validation_rmse_v",
                "terminal_voltage_validation_p95_error_v",
                "terminal_current_validation_p95_error_a",
            ):
                value = getattr(self, name)
                if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
            if (
                not math.isfinite(self.terminal_voltage_minimum_current_a)
                or self.terminal_voltage_minimum_current_a < 0.0
                or self.terminal_voltage_minimum_current_a
                >= float(self.terminal_voltage_maximum_current_a)
            ):
                raise ValueError("terminal-voltage current envelope is invalid")
            if float(self.terminal_voltage_minimum_power_w) >= float(
                self.terminal_voltage_maximum_power_w
            ):
                raise ValueError("terminal-voltage power envelope is invalid")


@dataclass
class BMSSnapshot:
    """Atomic battery state published to the experiment manager."""

    timestamp_s: float | None = None
    state: str = "INACTIVE"
    safety_level: str = "UNAVAILABLE"
    data_quality: str = "UNAVAILABLE"
    estimator_mode: str | None = None
    power_source: str | None = None
    soc_source: str | None = None
    voltage_v: float | None = None
    voltage_uncertainty_v: float | None = None
    current_uncertainty_a: float | None = None
    voltage_validation_mae_v: float | None = None
    voltage_validation_rmse_v: float | None = None
    terminal_voltage_model_id: str | None = None
    terminal_voltage_in_envelope: bool = False
    pack_identity_status: str = "UNRESOLVED_PUBLISHED_M100_CAMPAIGN"
    current_a: float | None = None
    load_power_w: float | None = None
    consumed_energy_wh: float | None = None
    consumed_capacity_ah: float | None = None
    remaining_energy_wh: float | None = None
    remaining_capacity_ah: float | None = None
    soc_percent: float | None = None
    endurance_min: float | None = None
    endurance_reference_power_w: float | None = None
    cutoff_imminent: bool = False
    advisory_only: bool = True
    dynamic_model_enabled: bool = False
    status: str = TB47D_BMS_STATUS
    warnings: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return (
            self.state != "INACTIVE"
            and self.estimator_mode is not None
            and self.remaining_energy_wh is not None
            and self.soc_percent is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "state": self.state,
            "safety_level": self.safety_level,
            "data_quality": self.data_quality,
            "estimator_mode": self.estimator_mode,
            "power_source": self.power_source,
            "soc_source": self.soc_source,
            "voltage_v": self.voltage_v,
            "voltage_uncertainty_v": self.voltage_uncertainty_v,
            "current_uncertainty_a": self.current_uncertainty_a,
            "voltage_validation_mae_v": self.voltage_validation_mae_v,
            "voltage_validation_rmse_v": self.voltage_validation_rmse_v,
            "terminal_voltage_model_id": self.terminal_voltage_model_id,
            "terminal_voltage_in_envelope": self.terminal_voltage_in_envelope,
            "pack_identity_status": self.pack_identity_status,
            "current_a": self.current_a,
            "load_power_w": self.load_power_w,
            "consumed_energy_wh": self.consumed_energy_wh,
            "consumed_capacity_ah": self.consumed_capacity_ah,
            "remaining_energy_wh": self.remaining_energy_wh,
            "remaining_capacity_ah": self.remaining_capacity_ah,
            "soc_percent": self.soc_percent,
            "endurance_min": self.endurance_min,
            "endurance_reference_power_w": self.endurance_reference_power_w,
            "cutoff_imminent": self.cutoff_imminent,
            "advisory_only": self.advisory_only,
            "dynamic_model_enabled": self.dynamic_model_enabled,
            "status": self.status,
            "warnings": list(self.warnings),
        }


class M100BMS:
    """Thread-safe TB47D energy budget with explicit estimate quality."""

    def __init__(self, config: BMSConfig):
        self.config = config
        self._lock = threading.RLock()
        self.reset()

    def reset(self, initial_soc_percent: float | None = None) -> None:
        """Reset all run state and leave the estimator inactive."""
        with self._lock:
            initial_soc = self._validate_initial_soc(initial_soc_percent)
            self._initial_soc_fraction = initial_soc / 100.0
            self._consumed_energy_wh = 0.0
            self._consumed_capacity_ah = 0.0
            self._filtered_power_w: float | None = None
            self._last_timestamp_s: float | None = None
            self._last_power_w: float | None = None
            self._last_current_a: float | None = None
            self._last_valid_voltage_v: float | None = None
            self._state = "INACTIVE"
            self._warnings: list[str] = []
            self._snapshot = BMSSnapshot()

    def begin(self, initial_soc_percent: float | None = None) -> None:
        """Begin a new run-relative battery budget."""
        with self._lock:
            initial_soc = self._validate_initial_soc(initial_soc_percent)
            self._initial_soc_fraction = initial_soc / 100.0
            self._consumed_energy_wh = 0.0
            self._consumed_capacity_ah = 0.0
            self._filtered_power_w = None
            self._last_timestamp_s = None
            self._last_power_w = None
            self._last_current_a = None
            self._last_valid_voltage_v = None
            self._state = "ACTIVE"
            self._warnings = [TB47D_BMS_WARNING]
            self._snapshot = self._build_snapshot(
                timestamp_s=None,
                voltage_v=None,
                current_a=None,
                load_power_w=None,
                data_quality="UNAVAILABLE",
                power_source=TB47D_BMS_MODEL_SOURCE,
                soc_source="RUN_RELATIVE_ENERGY_COUNT_FROM_OPERATIONAL_POWER",
            )

    def freeze(self) -> None:
        """Freeze final budget values without integrating post-run samples."""
        with self._lock:
            if self._state == "ACTIVE":
                self._state = "FROZEN"
            previous = self._snapshot
            frozen_voltage = (
                previous.voltage_v
                if previous.voltage_v is not None
                else self._last_valid_voltage_v
            )
            if frozen_voltage is not None and previous.voltage_v is None:
                self._clear_warning(TB47D_BMS_PROXY_OUTSIDE_ENVELOPE)
                self._warn(TB47D_BMS_FROZEN_LAST_VALID)
            self._snapshot = BMSSnapshot(
                timestamp_s=previous.timestamp_s,
                state=self._state,
                safety_level=previous.safety_level,
                data_quality=(
                    "FROZEN" if self._state == "FROZEN" else previous.data_quality
                ),
                estimator_mode=(
                    self.config.mode if self._state != "INACTIVE" else None
                ),
                power_source=previous.power_source,
                soc_source=previous.soc_source,
                voltage_v=frozen_voltage,
                voltage_uncertainty_v=(
                    self.config.terminal_voltage_validation_p95_error_v
                    if frozen_voltage is not None
                    and previous.power_source == TB47D_BMS_MODEL_SOURCE
                    else previous.voltage_uncertainty_v
                ),
                current_uncertainty_a=None,
                voltage_validation_mae_v=previous.voltage_validation_mae_v,
                voltage_validation_rmse_v=previous.voltage_validation_rmse_v,
                terminal_voltage_model_id=previous.terminal_voltage_model_id,
                terminal_voltage_in_envelope=(frozen_voltage is not None),
                pack_identity_status=previous.pack_identity_status,
                consumed_energy_wh=previous.consumed_energy_wh,
                consumed_capacity_ah=previous.consumed_capacity_ah,
                remaining_energy_wh=previous.remaining_energy_wh,
                remaining_capacity_ah=previous.remaining_capacity_ah,
                soc_percent=previous.soc_percent,
                endurance_reference_power_w=(
                    previous.endurance_reference_power_w
                ),
                cutoff_imminent=previous.cutoff_imminent,
                advisory_only=True,
                dynamic_model_enabled=False,
                status=TB47D_BMS_STATUS,
                warnings=list(self._warnings),
            )
            self._last_timestamp_s = None
            self._last_power_w = None
            self._last_current_a = None

    def update(
        self,
        timestamp_s: float,
        load_power_w: float | None,
        *,
        measured_voltage_v: float | None = None,
        measured_current_a: float | None = None,
    ) -> BMSSnapshot:
        """Update from the selected aircraft power or an enabled measured path."""
        with self._lock:
            if self._state == "INACTIVE" or not self.config.enabled:
                return self.snapshot()
            if self._state == "FROZEN":
                return self.snapshot()

            timestamp = _finite(timestamp_s)
            power = _finite(load_power_w)
            if timestamp is None:
                self._warn("BMS_NON_FINITE_TIMESTAMP")
                return self.mark_unavailable(None)
            if power is None or power < 0.0:
                self._warn("BMS_INVALID_LOAD_POWER")
                return self.mark_unavailable(timestamp)

            measured_voltage = _finite(measured_voltage_v)
            measured_current = _finite(measured_current_a)
            use_measured = (
                self.config.measured_telemetry_enabled
                and measured_voltage is not None
                and measured_voltage > 0.0
                and measured_current is not None
                and measured_current >= 0.0
            )
            if use_measured:
                self._clear_warning(TB47D_BMS_PROXY_OUTSIDE_ENVELOPE)
                voltage = measured_voltage
                current = measured_current
                data_quality = "MEASURED"
                power_source = TB47D_BMS_MEASURED_SOURCE
                soc_source = "COULOMB_COUNT_FROM_MEASURED_CURRENT"
                measured_power = voltage * current
                tolerance = self.config.measured_power_tolerance_fraction
                scale = max(power, measured_power, 1.0)
                if abs(measured_power - power) / scale > tolerance:
                    self._warn("BMS_MEASURED_POWER_MISMATCH")
            else:
                estimated = self._terminal_voltage_estimate(timestamp, power)
                if estimated is None:
                    voltage = None
                    current = None
                    data_quality = "DEGRADED"
                    self._warn(TB47D_BMS_PROXY_OUTSIDE_ENVELOPE)
                else:
                    voltage, current = estimated
                    self._last_valid_voltage_v = voltage
                    self._clear_warning(TB47D_BMS_PROXY_OUTSIDE_ENVELOPE)
                    data_quality = "MODEL_ESTIMATE"
                power_source = TB47D_BMS_MODEL_SOURCE
                soc_source = "RUN_RELATIVE_ENERGY_COUNT_FROM_OPERATIONAL_POWER"

            valid_dt = self._integrate(timestamp, power, current)
            self._update_filtered_power(power, valid_dt)
            self._snapshot = self._build_snapshot(
                timestamp_s=timestamp,
                voltage_v=voltage,
                current_a=current,
                load_power_w=power,
                data_quality=data_quality,
                power_source=power_source,
                soc_source=soc_source,
            )
            return self.snapshot()

    def mark_unavailable(self, timestamp_s: float | None) -> BMSSnapshot:
        """Break integration continuity while preserving the accumulated budget."""
        with self._lock:
            if self._state == "INACTIVE":
                return self.snapshot()
            if self._state == "FROZEN":
                return self.snapshot()
            self._warn("BMS_INPUT_UNAVAILABLE")
            timestamp = _finite(timestamp_s)
            self._last_timestamp_s = timestamp
            self._last_power_w = None
            self._last_current_a = None
            previous = self._snapshot
            self._snapshot = self._build_snapshot(
                timestamp_s=timestamp,
                voltage_v=None,
                current_a=None,
                load_power_w=None,
                data_quality="DEGRADED",
                power_source=previous.power_source or TB47D_BMS_MODEL_SOURCE,
                soc_source=(
                    previous.soc_source
                    or "RUN_RELATIVE_ENERGY_COUNT_FROM_OPERATIONAL_POWER"
                ),
            )
            return self.snapshot()

    def snapshot(self) -> BMSSnapshot:
        with self._lock:
            return BMSSnapshot(**self._snapshot.as_dict())

    def _validate_initial_soc(self, value: float | None) -> float:
        initial_soc = (
            self.config.initial_soc_percent if value is None else float(value)
        )
        if not math.isfinite(initial_soc) or not 0.0 <= initial_soc <= 100.0:
            raise ValueError("initial SOC must be in [0, 100]")
        return initial_soc

    def _warn(self, code: str) -> None:
        if code not in self._warnings:
            self._warnings.append(code)

    def _clear_warning(self, code: str) -> None:
        """Clear a condition warning once the current sample is valid again."""
        self._warnings = [warning for warning in self._warnings if warning != code]

    def _integrate(
        self, timestamp_s: float, power_w: float, current_a: float | None
    ) -> float | None:
        if self._last_timestamp_s is None:
            self._last_timestamp_s = timestamp_s
            self._last_power_w = power_w
            self._last_current_a = current_a
            return None
        dt = timestamp_s - self._last_timestamp_s
        valid_dt = None
        if dt <= 0.0:
            self._warn("BMS_TIMESTAMP_RESET_OR_DUPLICATE")
        elif dt > self.config.maximum_valid_gap_s:
            self._warn("BMS_TELEMETRY_GAP_NOT_INTEGRATED")
        elif self._last_power_w is not None:
            delta_wh = 0.5 * (self._last_power_w + power_w) * dt / 3600.0
            self._consumed_energy_wh += max(0.0, delta_wh)
            if self._last_current_a is not None and current_a is not None:
                delta_ah = (
                    0.5 * (self._last_current_a + current_a) * dt / 3600.0
                )
                self._consumed_capacity_ah += max(0.0, delta_ah)
            valid_dt = dt
        self._last_timestamp_s = timestamp_s
        self._last_power_w = power_w
        self._last_current_a = current_a
        return valid_dt

    def _update_filtered_power(self, power_w: float, dt: float | None) -> None:
        if self._filtered_power_w is None:
            self._filtered_power_w = power_w
            return
        if dt is None:
            return
        alpha = 1.0 - math.exp(
            -dt / self.config.endurance_power_time_constant_s
        )
        self._filtered_power_w += alpha * (power_w - self._filtered_power_w)

    def _build_snapshot(
        self,
        *,
        timestamp_s: float | None,
        voltage_v: float | None,
        current_a: float | None,
        load_power_w: float | None,
        data_quality: str,
        power_source: str,
        soc_source: str,
    ) -> BMSSnapshot:
        initial_capacity = self._initial_soc_fraction * self.config.capacity_ah
        initial_energy = self._initial_soc_fraction * self.config.nominal_energy_wh
        remaining_capacity = max(
            0.0, initial_capacity - self._consumed_capacity_ah
        )
        remaining_energy = max(0.0, initial_energy - self._consumed_energy_wh)
        soc_percent = 100.0 * remaining_energy / self.config.nominal_energy_wh
        safety_level = self._safety_level(soc_percent)
        if safety_level == "WARNING":
            self._warn("BMS_LOW_SOC")
        elif safety_level == "CRITICAL":
            self._warn("BMS_CRITICAL_SOC")
        elif safety_level == "DEPLETED":
            self._warn("BMS_ENERGY_DEPLETED")
        reference_power = self._filtered_power_w
        endurance = (
            60.0 * remaining_energy / reference_power
            if self._state == "ACTIVE"
            and reference_power is not None
            and reference_power >= self.config.minimum_endurance_power_w
            and remaining_energy > 0.0
            else None
        )
        return BMSSnapshot(
            timestamp_s=timestamp_s,
            state=self._state,
            safety_level=safety_level,
            data_quality=data_quality,
            estimator_mode=self.config.mode,
            power_source=power_source,
            soc_source=soc_source,
            voltage_v=voltage_v,
            voltage_uncertainty_v=(
                self.config.terminal_voltage_validation_p95_error_v
                if voltage_v is not None
                and power_source == TB47D_BMS_MODEL_SOURCE
                and self.config.terminal_voltage_proxy_enabled
                else None
            ),
            current_uncertainty_a=(
                self.config.terminal_current_validation_p95_error_a
                if current_a is not None
                and power_source == TB47D_BMS_MODEL_SOURCE
                and self.config.terminal_voltage_proxy_enabled
                else None
            ),
            voltage_validation_mae_v=(
                self.config.terminal_voltage_validation_mae_v
                if self.config.terminal_voltage_proxy_enabled else None
            ),
            voltage_validation_rmse_v=(
                self.config.terminal_voltage_validation_rmse_v
                if self.config.terminal_voltage_proxy_enabled else None
            ),
            terminal_voltage_model_id=(
                self.config.terminal_voltage_model_id
                if self.config.terminal_voltage_proxy_enabled else None
            ),
            terminal_voltage_in_envelope=(
                voltage_v is not None
                and current_a is not None
                and power_source == TB47D_BMS_MODEL_SOURCE
                and self.config.terminal_voltage_proxy_enabled
            ),
            current_a=current_a,
            load_power_w=load_power_w,
            consumed_energy_wh=self._consumed_energy_wh,
            consumed_capacity_ah=self._consumed_capacity_ah,
            remaining_energy_wh=remaining_energy,
            remaining_capacity_ah=remaining_capacity,
            soc_percent=soc_percent,
            endurance_min=endurance,
            endurance_reference_power_w=reference_power,
            cutoff_imminent=safety_level in {"CRITICAL", "DEPLETED"},
            advisory_only=True,
            dynamic_model_enabled=False,
            status=TB47D_BMS_STATUS,
            warnings=list(self._warnings),
        )

    def _terminal_voltage_estimate(
        self, timestamp_s: float, power_w: float
    ) -> tuple[float, float] | None:
        """Solve the frozen causal transfer model while enforcing its envelope."""
        if not self.config.terminal_voltage_proxy_enabled:
            return self.config.nominal_voltage_v, power_w / self.config.nominal_voltage_v
        if (
            power_w < float(self.config.terminal_voltage_minimum_power_w)
            or power_w > float(self.config.terminal_voltage_maximum_power_w)
        ):
            return None
        throughput = self._consumed_capacity_ah
        maximum_throughput = float(
            self.config.terminal_voltage_maximum_throughput_ah
        )
        if throughput < 0.0 or throughput > maximum_throughput:
            return None
        coefficient = 0.0
        if self._last_timestamp_s is not None and self._last_current_a is not None:
            dt = timestamp_s - self._last_timestamp_s
            if 0.0 < dt <= self.config.maximum_valid_gap_s:
                coefficient = dt / (2.0 * 3600.0)
                throughput += coefficient * self._last_current_a
        depletion_slope = float(
            self.config.terminal_voltage_depletion_slope_v_per_ah
        )
        open_load_voltage = (
            float(self.config.terminal_voltage_intercept_v)
            - depletion_slope * throughput
        )
        resistance = (
            float(self.config.terminal_voltage_resistance_ohm)
            + depletion_slope * coefficient
        )
        discriminant = open_load_voltage * open_load_voltage - 4.0 * resistance * power_w
        if discriminant < 0.0:
            return None
        root = math.sqrt(discriminant)
        current = 0.0 if power_w == 0.0 else 2.0 * power_w / (open_load_voltage + root)
        voltage = open_load_voltage - resistance * current
        resulting_throughput = throughput + coefficient * current
        minimum_current = self.config.terminal_voltage_minimum_current_a
        maximum_current = float(self.config.terminal_voltage_maximum_current_a)
        if (
            not math.isfinite(voltage)
            or not math.isfinite(current)
            or voltage <= 0.0
            or current < minimum_current
            or current > maximum_current
            or resulting_throughput > maximum_throughput
        ):
            return None
        return voltage, current

    def _safety_level(self, soc_percent: float) -> str:
        if soc_percent <= 0.0:
            return "DEPLETED"
        if soc_percent <= self.config.critical_soc_percent:
            return "CRITICAL"
        if soc_percent <= self.config.low_soc_warning_percent:
            return "WARNING"
        return "NORMAL"
