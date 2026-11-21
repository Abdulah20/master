"""Two-RC battery circuit and averaged motor/ESC conduction model for simulation."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, fields
import hashlib
import math
from pathlib import Path
from typing import Callable, Sequence

import yaml


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{name} is outside its allowed range")
    return number


def _positive(value: object, name: str) -> float:
    number = _number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _label(data: dict, name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be provided")
    return value.strip()


@dataclass(frozen=True)
class Circuit:
    ocv_v: float
    r0_ohm: float
    r1_ohm: float
    c1_f: float
    r2_ohm: float
    c2_f: float


@dataclass(frozen=True)
class DriveProfile:
    """Averaged two-phase PWM at a fixed winding and switch temperature."""

    source: str
    torque_constant_nm_per_a: float
    winding_line_to_line_ohm: float
    esc_on_path_ohm: float
    esc_freewheel_path_ohm: float
    auxiliary_power_w: float

    @classmethod
    def from_mapping(cls, data: dict) -> DriveProfile:
        if not isinstance(data, dict):
            raise ValueError("drive must be a mapping")
        return cls(
            source=_label(data, "source"),
            torque_constant_nm_per_a=_positive(
                data.get("torque_constant_nm_per_a"), "torque constant"
            ),
            winding_line_to_line_ohm=_positive(
                data.get("winding_line_to_line_ohm"), "line-to-line resistance"
            ),
            esc_on_path_ohm=_positive(data.get("esc_on_path_ohm"), "ESC on-path resistance"),
            esc_freewheel_path_ohm=_positive(
                data.get("esc_freewheel_path_ohm"), "ESC freewheel-path resistance"
            ),
            auxiliary_power_w=_number(
                data.get("auxiliary_power_w"), "auxiliary load", minimum=0.0
            ),
        )


@dataclass(frozen=True)
class TheveninProfile:
    profile_id: str
    pack_model: str
    source: str
    sha256: str
    temperature_c: float
    capacity_ah: float
    initial_soc_percent: float
    minimum_terminal_voltage_v: float
    maximum_discharge_current_a: float
    soc_percent: tuple[float, ...]
    circuits: tuple[Circuit, ...]
    drive: DriveProfile | None

    @classmethod
    def load(cls, path: str | Path) -> TheveninProfile:
        contents = Path(path).read_bytes()
        data = yaml.safe_load(contents)
        if not isinstance(data, dict) or data.get("schema") != "M100_THEVENIN_2RC_V1":
            raise ValueError("unexpected battery research profile schema")
        knots = data.get("soc_table")
        circuit_fields = fields(Circuit)
        if not isinstance(knots, list) or len(knots) < 2:
            raise ValueError("soc_table requires at least two rows")
        soc: list[float] = []
        circuits: list[Circuit] = []
        for row in knots:
            if not isinstance(row, list) or len(row) != 1 + len(circuit_fields):
                raise ValueError("soc_table rows require SOC, OCV, R0, R1, C1, R2, C2")
            charge = _number(row[0], "table SOC percent", minimum=0.0)
            if charge > 100.0 or (soc and charge <= soc[-1]):
                raise ValueError("table SOC percentages must increase within [0, 100]")
            soc.append(charge)
            circuits.append(Circuit(*(
                _positive(value, item.name) for item, value in zip(circuit_fields, row[1:])
            )))
        initial_soc = _number(data.get("initial_soc_percent"), "initial SOC percent")
        if not soc[0] <= initial_soc <= soc[-1]:
            raise ValueError("initial SOC is outside the documented table range")
        return cls(
            profile_id=_label(data, "profile_id"),
            pack_model=_label(data, "pack_model"),
            source=_label(data, "source"),
            sha256=hashlib.sha256(contents).hexdigest(),
            temperature_c=_number(data.get("temperature_c"), "battery temperature"),
            capacity_ah=_positive(data.get("capacity_ah"), "pack capacity"),
            initial_soc_percent=initial_soc,
            minimum_terminal_voltage_v=_positive(
                data.get("minimum_terminal_voltage_v"), "terminal voltage limit"
            ),
            maximum_discharge_current_a=_positive(
                data.get("maximum_discharge_current_a"), "discharge current limit"
            ),
            soc_percent=tuple(soc),
            circuits=tuple(circuits),
            drive=DriveProfile.from_mapping(data["drive"]) if "drive" in data else None,
        )

    def at_soc(self, soc_percent: float) -> Circuit:
        if not self.soc_percent[0] <= soc_percent <= self.soc_percent[-1]:
            raise ValueError("SOC is outside the documented table range")
        upper = bisect_right(self.soc_percent, soc_percent)
        if upper == len(self.soc_percent):
            return self.circuits[-1]
        low, high = self.circuits[upper - 1], self.circuits[upper]
        z_low, z_high = self.soc_percent[upper - 1:upper + 1]
        fraction = (soc_percent - z_low) / (z_high - z_low)
        return Circuit(*(
            getattr(low, item.name)
            + fraction * (getattr(high, item.name) - getattr(low, item.name))
            for item in fields(Circuit)
        ))


@dataclass(frozen=True)
class ShaftDemand:
    speed_rad_s: float
    torque_nm: float


@dataclass(frozen=True)
class ElectricalPoint:
    terminal_voltage_v: float
    battery_current_a: float
    terminal_power_w: float
    shaft_power_w: float | None = None
    motor_copper_loss_w: float | None = None
    esc_conduction_loss_w: float | None = None
    auxiliary_power_w: float | None = None
    max_pwm_duty: float | None = None


@dataclass(frozen=True)
class _PwmMotor:
    line_current_a: float
    shaft_power_w: float
    voltage_numerator_v: float
    voltage_offset_v: float

    def duty(self, bus_voltage_v: float) -> float:
        if self.line_current_a == 0.0:
            return 0.0
        return self.voltage_numerator_v / (bus_voltage_v - self.voltage_offset_v)


def _ascending_zero(function: Callable[[float], float], lo: float, hi: float) -> float:
    """Bisect an increasing zero crossing using a bounded number of steps."""
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if function(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return hi


def _solve_drive(
    drive: DriveProfile,
    shafts: Sequence[ShaftDemand],
    available_voltage_v: float,
    r0_ohm: float,
    voltage_limit_v: float,
    current_limit_a: float,
) -> ElectricalPoint | None:
    motors = []
    # The same effective K in SI applies to torque and back EMF in this
    # averaged, energy-consistent model. Current is the conducting line current.
    resistance_delta = drive.esc_on_path_ohm - drive.esc_freewheel_path_ohm
    for shaft in shafts:
        line_current = shaft.torque_nm / drive.torque_constant_nm_per_a
        back_emf = drive.torque_constant_nm_per_a * shaft.speed_rad_s
        motors.append(_PwmMotor(
            line_current_a=line_current,
            shaft_power_w=shaft.torque_nm * shaft.speed_rad_s,
            voltage_numerator_v=(back_emf + line_current * (
                drive.winding_line_to_line_ohm + drive.esc_freewheel_path_ohm
            )),
            voltage_offset_v=line_current * resistance_delta,
        ))
    enabled = [motor for motor in motors if motor.line_current_a > 0.0]
    voltage_floor = max(
        voltage_limit_v,
        available_voltage_v - r0_ohm * current_limit_a,
        max(
            (motor.voltage_numerator_v + motor.voltage_offset_v for motor in enabled),
            default=0.0,
        ),
    )
    if voltage_floor > available_voltage_v:
        return None

    def current(voltage: float) -> float:
        return drive.auxiliary_power_w / voltage + sum(
            motor.line_current_a * motor.duty(voltage) for motor in enabled
        )

    def balance(voltage: float) -> float:
        return (voltage - available_voltage_v) + r0_ohm * current(voltage)

    def slope(voltage: float) -> float:
        return 1.0 - r0_ohm * (
            drive.auxiliary_power_w / voltage**2
            + sum(motor.line_current_a * motor.duty(voltage)
                  / (voltage - motor.voltage_offset_v) for motor in enabled)
        )

    # The balance is convex. Search the increasing (high-voltage) branch and
    # require it to satisfy pack and duty limits for every rotor.
    if slope(voltage_floor) < 0.0:
        if slope(available_voltage_v) <= 0.0:
            return None
        voltage_floor = _ascending_zero(slope, voltage_floor, available_voltage_v)
    if balance(voltage_floor) > 0.0:
        return None
    voltage = _ascending_zero(balance, voltage_floor, available_voltage_v)
    amps = current(voltage)
    duties = [motor.duty(voltage) for motor in motors]
    return ElectricalPoint(
        terminal_voltage_v=voltage,
        battery_current_a=amps,
        terminal_power_w=voltage * amps,
        shaft_power_w=sum(motor.shaft_power_w for motor in motors),
        motor_copper_loss_w=drive.winding_line_to_line_ohm * sum(
            motor.line_current_a**2 for motor in enabled
        ),
        esc_conduction_loss_w=sum(
            motor.line_current_a**2 * (
                drive.esc_freewheel_path_ohm + duty * resistance_delta
            )
            for motor, duty in zip(motors, duties)
        ),
        auxiliary_power_w=drive.auxiliary_power_w,
        max_pwm_duty=max(duties, default=0.0),
    )


@dataclass(frozen=True)
class BatteryObservation:
    status: str
    timestamp_s: float | None
    load_history_complete: bool
    integrated_duration_s: float
    soc_percent: float
    ocv_v: float
    polarization_1_v: float
    polarization_2_v: float
    terminal_energy_wh: float
    motor_copper_energy_wh: float | None
    esc_conduction_energy_wh: float | None
    battery_ohmic_loss_w: float | None
    electrical: ElectricalPoint | None


class TheveninStudy:
    """Discharge simulation using held current and actual timestamp intervals."""

    def __init__(self, profile: TheveninProfile, maximum_gap_s: float):
        self.profile = profile
        self.maximum_gap_s = _positive(maximum_gap_s, "maximum sample gap")
        self.reset()

    def reset(self) -> None:
        self._soc = self.profile.initial_soc_percent
        self._polarization = (0.0, 0.0)
        self._terminal_energy_wh = 0.0
        self._motor_copper_energy_wh = 0.0
        self._esc_conduction_energy_wh = 0.0
        self._integrated_duration_s = 0.0
        self._load_seen = False
        self._load_history_complete = True
        self._last_timestamp_s: float | None = None
        self.observation = self._observe("AWAITING_LOAD", None)

    def as_dict(self) -> dict:
        return {
            "model_id": "M100_THEVENIN_2RC_V1",
            "classification": "C_NOT_CALIBRATED",
            "profile_id": self.profile.profile_id,
            "profile_sha256": self.profile.sha256,
            "pack_model": self.profile.pack_model,
            "source": self.profile.source,
            "battery_temperature_c": self.profile.temperature_c,
            "initial_soc_percent": self.profile.initial_soc_percent,
            "drive_source": self.profile.drive.source if self.profile.drive else None,
            "observation": asdict(self.observation),
        }

    def _observe(
        self, status: str, timestamp_s: float | None, electrical: ElectricalPoint | None = None
    ) -> BatteryObservation:
        circuit = self.profile.at_soc(self._soc)
        return BatteryObservation(
            status=status,
            timestamp_s=timestamp_s,
            load_history_complete=self._load_history_complete,
            integrated_duration_s=self._integrated_duration_s,
            soc_percent=self._soc,
            ocv_v=circuit.ocv_v,
            polarization_1_v=self._polarization[0],
            polarization_2_v=self._polarization[1],
            terminal_energy_wh=self._terminal_energy_wh,
            motor_copper_energy_wh=(
                self._motor_copper_energy_wh if self.profile.drive else None
            ),
            esc_conduction_energy_wh=(
                self._esc_conduction_energy_wh if self.profile.drive else None
            ),
            battery_ohmic_loss_w=(
                electrical.battery_current_a**2 * circuit.r0_ohm if electrical else None
            ),
            electrical=electrical,
        )

    def _advance(self, dt: float, electrical: ElectricalPoint) -> None:
        circuit = self.profile.at_soc(self._soc)
        amps = electrical.battery_current_a
        capacity_as = self.profile.capacity_ah * 3600.0
        minimum_soc = self.profile.soc_percent[0]
        if amps > 0.0:
            available_as = (self._soc - minimum_soc) / 100.0 * capacity_as
            time_to_minimum = available_as / amps
            if dt >= time_to_minimum:
                dt = time_to_minimum
                self._soc = minimum_soc
            else:
                self._soc = max(minimum_soc, self._soc - 100.0 * amps * dt / capacity_as)
        self._polarization = tuple(
            previous + (-math.expm1(-dt / (resistance * capacitance)))
            * (amps * resistance - previous)
            for previous, resistance, capacitance in (
                (self._polarization[0], circuit.r1_ohm, circuit.c1_f),
                (self._polarization[1], circuit.r2_ohm, circuit.c2_f),
            )
        )
        self._integrated_duration_s += dt
        self._terminal_energy_wh += electrical.terminal_power_w * dt / 3600.0
        if electrical.motor_copper_loss_w is not None:
            self._motor_copper_energy_wh += electrical.motor_copper_loss_w * dt / 3600.0
        if electrical.esc_conduction_loss_w is not None:
            self._esc_conduction_energy_wh += electrical.esc_conduction_loss_w * dt / 3600.0

    def step(
        self,
        timestamp_s: float | None,
        *,
        terminal_power_w: float | None = None,
        discharge_current_a: float | None = None,
        shafts: Sequence[ShaftDemand] | None = None,
    ) -> BatteryObservation:
        """Update from one source of demand. An explicit zero means rest."""
        inputs = sum(value is not None for value in (terminal_power_w, discharge_current_a, shafts))
        if inputs > 1:
            raise ValueError("provide only one battery load source")
        if self.profile.drive is not None and (
            terminal_power_w is not None or discharge_current_a is not None
        ):
            raise ValueError("a drive profile requires shaft demands")
        if shafts is not None and (self.profile.drive is None or len(shafts) != 4):
            raise ValueError("four shaft demands and a drive profile are required")

        if timestamp_s is None or not math.isfinite(timestamp_s):
            self._load_history_complete &= not self._load_seen
            self.observation = self._observe("INVALID_TIMESTAMP", None)
            return self.observation
        if self._last_timestamp_s is not None:
            dt = timestamp_s - self._last_timestamp_s
            if dt == 0.0:
                return self.observation
            if dt < 0.0:
                self._load_history_complete &= not self._load_seen
                self.observation = self._observe("NON_MONOTONIC_TIMESTAMP", None)
                return self.observation
            if dt <= self.maximum_gap_s and self.observation.electrical is not None:
                self._advance(dt, self.observation.electrical)
            elif self._load_seen:
                self._load_history_complete = False
        self._last_timestamp_s = timestamp_s
        self._load_seen |= bool(inputs)
        status, electrical = self._solve(terminal_power_w, discharge_current_a, shafts)
        if electrical is None and self._load_seen:
            self._load_history_complete = False
        self.observation = self._observe(status, timestamp_s, electrical)
        return self.observation

    def _solve(
        self,
        power: float | None,
        amps: float | None,
        shafts: Sequence[ShaftDemand] | None,
    ) -> tuple[str, ElectricalPoint | None]:
        if power is None and amps is None and shafts is None:
            return "MISSING_LOAD", None
        circuit = self.profile.at_soc(self._soc)
        source_v = circuit.ocv_v - sum(self._polarization)
        limit_v = self.profile.minimum_terminal_voltage_v
        limit_a = self.profile.maximum_discharge_current_a
        if source_v <= 0.0:
            return "VOLTAGE_OUTSIDE_PROFILE", None
        if shafts is not None:
            if not all(
                math.isfinite(x) and x >= 0.0
                for shaft in shafts for x in (shaft.speed_rad_s, shaft.torque_nm)
            ):
                return "INVALID_LOAD", None
            point = _solve_drive(
                self.profile.drive, shafts, source_v, circuit.r0_ohm, limit_v, limit_a
            )
            if point is None:
                return "NO_FEASIBLE_POWERTRAIN_POINT", None
        else:
            if power is not None:
                if not math.isfinite(power) or power < 0.0:
                    return "INVALID_LOAD", None
                # Even at open circuit this demand would exceed the current domain.
                if power > source_v * limit_a:
                    return "CURRENT_OUTSIDE_PROFILE", None
                discriminant = source_v**2 - 4.0 * circuit.r0_ohm * power
                if discriminant < 0.0:
                    return "NO_FEASIBLE_POWER_POINT", None
                amps = 2.0 * power / (source_v + math.sqrt(discriminant))
            if not math.isfinite(amps) or amps < 0.0:
                return "INVALID_LOAD", None
            if amps > limit_a:
                return "CURRENT_OUTSIDE_PROFILE", None
            voltage = source_v - amps * circuit.r0_ohm
            if amps > 0.0 and voltage < limit_v:
                return "VOLTAGE_OUTSIDE_PROFILE", None
            point = ElectricalPoint(voltage, amps, voltage * amps)
        if point.battery_current_a > 0.0 and self._soc <= self.profile.soc_percent[0]:
            return "SOC_LIMIT_REACHED", None
        return "OK", point
