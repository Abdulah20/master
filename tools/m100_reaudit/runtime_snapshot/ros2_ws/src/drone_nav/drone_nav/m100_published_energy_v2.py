"""Published aircraft-level energy model for the DJI Matrice 100.

This module deliberately estimates aircraft electrical power directly from a
published phase model.  Rotor speed remains a diagnostic only: no motor,
propeller, ESC, torque, or mechanical-power value is inferred without a
traceable component map.

The model equation is::

    P = b1 * m**1.5 / sqrt(rho) + b0

where ``P`` is electrical power in watts, ``rho`` is air density in kg/m^3,
and ``m`` is the published regression feature mass, not the simulator's
physical mass.  The official v1.0.0 source constructs that feature as
``payload_kg + 3.08 kg + 0.635 kg`` (3.715 kg plus payload).  Runtime APIs keep
that quantity separate from the 3.680 kg plus payload physical flight mass.
Coefficients and phase applicability are loaded from the versioned YAML
configuration next to this package.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping

import yaml

from .m100_bms import BMSConfig, BMSSnapshot, M100BMS, TB47D_BMS_MODE
from .m100_thevenin import ShaftDemand, TheveninProfile, TheveninStudy
from .m100_stable_cruise_speed import (
    MODEL_ID as STABLE_CRUISE_MODEL_ID,
    MODEL_STATUS as STABLE_CRUISE_MODEL_STATUS,
    MODEL_WARNING as STABLE_CRUISE_MODEL_WARNING,
)


CONFIG_SCHEMA = "M100_PUBLISHED_ENERGY_CONFIG_V2"
MODEL_ID = "RODRIGUES_PATTERNS_M100_PHASE_POWER_V1"
MODEL_STATUS = "PUBLISHED_EMPIRICAL_M100_AIRCRAFT_PHASE_MODEL"
HOVER_STATUS = "LIMITED_EMPIRICAL_PROXY"
UNAVAILABLE_STATUS = "UNAVAILABLE_OUTSIDE_MODEL_APPLICABILITY"
RPM_STATUS = "NUMERICAL_ROTOR_SPEED_DIAGNOSTIC"
MECHANICAL_STATUS = "UNAVAILABLE_NO_TRACEABLE_M100_PROPULSION_TORQUE_MAP"
BATTERY_UNAVAILABLE_STATUS = "UNAVAILABLE_BMS_SESSION_NOT_ACTIVE"
MASS_BASIS_STATUS = "PUBLISHED_REGRESSION_MASS_3P715_KG_PLUS_PAYLOAD"
MASS_BASIS_WARNING = (
    "PUBLISHED_REGRESSION_MASS_BASIS_DIFFERS_FROM_PHYSICAL_RUNTIME_MASS"
)
PUBLISHED_MASS_SOURCE_DOI = "10.5281/zenodo.6726991"
PUBLISHED_MASS_SOURCE_ARCHIVE_SHA256 = (
    "28ef18d85d58d0a14d8ca49c676d559a6b0f50fffe95744a6ecf82d7ec0675ff"
)
PUBLISHED_MASS_SOURCE_ARCHIVE_ROOT = "3d27ac92ba5db36a2fe3fc1449bbb5c422e10ca6"
PUBLISHED_MASS_SOURCE_PATH = "inflightcomponents.py"
PUBLISHED_REGRESSION_AIRFRAME_MASS_KG = 3.08
PUBLISHED_REGRESSION_BATTERY_MASS_KG = 0.635
PUBLISHED_REGRESSION_BASELINE_MASS_KG = 3.715
PHYSICAL_RUNTIME_BASELINE_MASS_KG = 3.680
LOCAL_REPRODUCTION_STATUS = "PASS_NEAR_EXACT_PUBLISHED_ROUNDING"
LOCAL_REPRODUCTION_FULL_PRECISION_HOLDOUT_MARE_PERCENT = 2.0698539594489147
RUNTIME_ROUNDED_HOLDOUT_MARE_PERCENT = 2.0662458588801145
LOCAL_REPRODUCTION_VERIFIER_PATH = (
    "analysis/m100_rodrigues_official_reproduction_v1/"
    "verify_official_reproduction_v1.py"
)
LOCAL_REPRODUCTION_VERIFIER_SHA256 = (
    "56af0fd22e6ffc6a4485802feaf8e4a680749f111e82390576167ae24b6b0773"
)
LOCAL_REPRODUCTION_REPORT_PATH = (
    "analysis/m100_rodrigues_official_reproduction_v1/"
    "reproduction_report_v1.json"
)
LOCAL_REPRODUCTION_REPORT_SHA256 = (
    "5d7548f8f26bef0a390529096aa35b4fe00d44a6c037833065769ed929c6d611"
)
ROTOR_JOINT_NAMES = tuple(f"rotor_{index}_joint" for index in range(4))

DEFAULT_MODEL_WARNINGS = (
    "AIRCRAFT_PHASE_MODEL_NOT_COMPONENT_PROPULSION_PHYSICS",
    "PHASE_AVERAGE_POWER_NOT_INSTANTANEOUS_PROPULSION_DYNAMICS",
    "SPEED_AND_WIND_NOT_EXPLICIT_MODEL_INPUTS",
    "ROTOR_SPEED_NOT_USED_TO_ESTIMATE_POWER",
    "UNSUPPORTED_PHASES_ARE_NOT_INTEGRATED_INTO_ENERGY_OR_BMS",
    MASS_BASIS_WARNING,
)


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _optional_positive(value: Any, name: str) -> float | None:
    return None if value is None else _positive(value, name)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True)
class PhaseCoefficients:
    """Published coefficients for one aircraft-level flight phase."""

    b1: float
    b0: float

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], name: str
    ) -> "PhaseCoefficients":
        return cls(
            b1=_positive(value.get("b1"), f"{name}.b1"),
            b0=_finite(value.get("b0"), f"{name}.b0"),
        )


@dataclass(frozen=True)
class BMSInput:
    """Traceable load input passed to the run-scoped battery estimator."""

    timestamp_s: float | None
    load_power_w: float | None
    source_model_id: str
    fidelity: str
    eligible: bool


@dataclass(frozen=True)
class PowerEstimate:
    """Pure power estimate before time integration or battery bookkeeping."""

    phase: str
    coefficient_phase: str | None
    power_w: float | None
    status: str
    warnings: tuple[str, ...]
    model_id: str = MODEL_ID
    stable_cruise_speed_adjustment_w: float | None = None
    wind_power_adjustment_w: float | None = None

    @property
    def available(self) -> bool:
        return self.power_w is not None


@dataclass(frozen=True)
class PublishedEnergyConfig:
    """Validated, immutable configuration for the published phase model."""

    model_id: str
    source_doi: str
    source_title: str
    source_fidelity: str
    mass_basis_status: str
    published_regression_airframe_mass_kg: float
    published_regression_battery_mass_kg: float
    published_regression_baseline_mass_kg: float
    physical_runtime_baseline_mass_kg: float
    mass_basis_source_doi: str
    mass_basis_source_archive_sha256: str
    mass_basis_source_archive_root: str
    mass_basis_source_path: str
    mass_basis_source_code_expression: str
    coefficients: Mapping[str, PhaseCoefficients]
    phase_mapping: Mapping[str, str]
    hover_enabled: bool
    hover_power_w: float
    hover_status: str
    maximum_valid_gap_s: float
    rotor_joint_velocity_scale: float
    bms_config: BMSConfig
    warnings: tuple[str, ...]
    metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PublishedEnergyConfig":
        if data.get("schema") != CONFIG_SCHEMA:
            raise ValueError("unsupported published M100 energy configuration schema")

        model = _require_mapping(data.get("model"), "model")
        source = _require_mapping(model.get("source"), "model.source")
        reproduction = _require_mapping(
            source.get("local_reproduction"), "model.source.local_reproduction"
        )
        if reproduction.get("status") != LOCAL_REPRODUCTION_STATUS:
            raise ValueError(
                "model.source.local_reproduction.status does not match the "
                "frozen verification result"
            )
        if reproduction.get("training_flights") != 120:
            raise ValueError("local reproduction must use 120 training flights")
        if reproduction.get("holdout_flights") != 68:
            raise ValueError("local reproduction must use 68 holdout flights")
        if reproduction.get("holdout_used_for_fit") is not False:
            raise ValueError("local reproduction holdout must not be used for fit")
        reproduction_metrics = {
            "full_precision_refit_holdout_mean_absolute_relative_error_percent": (
                LOCAL_REPRODUCTION_FULL_PRECISION_HOLDOUT_MARE_PERCENT
            ),
            "runtime_published_rounded_coefficients_holdout_mean_absolute_relative_error_percent": (
                RUNTIME_ROUNDED_HOLDOUT_MARE_PERCENT
            ),
        }
        for name, expected in reproduction_metrics.items():
            observed = _positive(
                reproduction.get(name), f"model.source.local_reproduction.{name}"
            )
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"model.source.local_reproduction.{name} does not match "
                    "the frozen verification result"
                )
        reproduction_sources = {
            "verifier": LOCAL_REPRODUCTION_VERIFIER_PATH,
            "verifier_sha256": LOCAL_REPRODUCTION_VERIFIER_SHA256,
            "report": LOCAL_REPRODUCTION_REPORT_PATH,
            "report_sha256": LOCAL_REPRODUCTION_REPORT_SHA256,
        }
        for name, expected in reproduction_sources.items():
            if str(reproduction.get(name, "")).strip() != expected:
                raise ValueError(
                    f"model.source.local_reproduction.{name} does not match "
                    "the frozen verification artifact"
                )
        mass_basis = _require_mapping(model.get("mass_basis"), "model.mass_basis")
        mass_basis_source = _require_mapping(
            mass_basis.get("source"), "model.mass_basis.source"
        )

        mass_basis_status = str(mass_basis.get("status", "")).strip()
        if mass_basis_status != MASS_BASIS_STATUS:
            raise ValueError(f"model.mass_basis.status must be {MASS_BASIS_STATUS}")

        regression_airframe_mass = _positive(
            mass_basis.get("airframe_component_kg"),
            "model.mass_basis.airframe_component_kg",
        )
        regression_battery_mass = _positive(
            mass_basis.get("battery_component_kg"),
            "model.mass_basis.battery_component_kg",
        )
        regression_baseline_mass = _positive(
            mass_basis.get("published_regression_baseline_kg"),
            "model.mass_basis.published_regression_baseline_kg",
        )
        physical_runtime_baseline_mass = _positive(
            mass_basis.get("physical_runtime_baseline_kg"),
            "model.mass_basis.physical_runtime_baseline_kg",
        )
        if not math.isclose(
            regression_airframe_mass,
            PUBLISHED_REGRESSION_AIRFRAME_MASS_KG,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("published regression airframe component must be 3.08 kg")
        if not math.isclose(
            regression_battery_mass,
            PUBLISHED_REGRESSION_BATTERY_MASS_KG,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("published regression battery component must be 0.635 kg")
        if not math.isclose(
            regression_baseline_mass,
            regression_airframe_mass + regression_battery_mass,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "published regression baseline must equal airframe plus battery"
            )
        if not math.isclose(
            regression_baseline_mass,
            PUBLISHED_REGRESSION_BASELINE_MASS_KG,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("published regression baseline must be 3.715 kg")
        if not math.isclose(
            physical_runtime_baseline_mass,
            PHYSICAL_RUNTIME_BASELINE_MASS_KG,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("physical runtime baseline must be 3.680 kg")

        mass_source_doi = str(mass_basis_source.get("doi", "")).strip()
        mass_source_archive_sha256 = str(
            mass_basis_source.get("archive_sha256", "")
        ).strip()
        mass_source_archive_root = str(
            mass_basis_source.get("archive_root_commit", "")
        ).strip()
        mass_source_path = str(mass_basis_source.get("path", "")).strip()
        mass_source_code_expression = str(
            mass_basis_source.get("code_expression", "")
        ).strip()
        expected_source_values = {
            "doi": (mass_source_doi, PUBLISHED_MASS_SOURCE_DOI),
            "archive_sha256": (
                mass_source_archive_sha256,
                PUBLISHED_MASS_SOURCE_ARCHIVE_SHA256,
            ),
            "archive_root_commit": (
                mass_source_archive_root,
                PUBLISHED_MASS_SOURCE_ARCHIVE_ROOT,
            ),
            "path": (mass_source_path, PUBLISHED_MASS_SOURCE_PATH),
        }
        for name, (actual, expected) in expected_source_values.items():
            if actual != expected:
                raise ValueError(
                    f"model.mass_basis.source.{name} must be {expected}"
                )
        if mass_source_code_expression != "payload / 1000 + 3.08 + 0.635":
            raise ValueError("unexpected published mass-feature code expression")

        raw_coefficients = _require_mapping(
            model.get("coefficients"), "model.coefficients"
        )
        required_coefficient_phases = {"takeoff", "cruise", "landing"}
        if set(raw_coefficients) != required_coefficient_phases:
            raise ValueError(
                "model.coefficients must contain exactly takeoff, cruise, and landing"
            )
        coefficients = {
            name: PhaseCoefficients.from_mapping(
                _require_mapping(value, f"model.coefficients.{name}"),
                f"model.coefficients.{name}",
            )
            for name, value in raw_coefficients.items()
        }

        raw_phase_mapping = _require_mapping(
            model.get("phase_mapping"), "model.phase_mapping"
        )
        phase_mapping: dict[str, str] = {}
        for runtime_phase, coefficient_phase in raw_phase_mapping.items():
            normalized_runtime = str(runtime_phase).strip().upper()
            normalized_coefficient = str(coefficient_phase).strip().lower()
            if not normalized_runtime:
                raise ValueError("runtime phase names must not be empty")
            if normalized_coefficient not in coefficients:
                raise ValueError(
                    f"phase {runtime_phase!r} refers to an unknown coefficient set"
                )
            phase_mapping[normalized_runtime] = normalized_coefficient

        expected_mapping = {
            "TAKEOFF": "takeoff",
            "CLIMB": "takeoff",
            "CRUISE": "cruise",
            "RETURNING": "cruise",
            "LANDING": "landing",
        }
        if phase_mapping != expected_mapping:
            raise ValueError("phase mapping differs from the published V2 contract")

        hover = _require_mapping(model.get("limited_hover"), "model.limited_hover")
        hover_status = str(hover.get("status", "")).strip()
        if hover_status != HOVER_STATUS:
            raise ValueError(f"limited hover status must be {HOVER_STATUS}")

        integration = _require_mapping(data.get("integration"), "integration")
        diagnostics = _require_mapping(data.get("diagnostics"), "diagnostics")
        battery = _require_mapping(data.get("battery"), "battery")
        runtime_bms = _require_mapping(data.get("runtime_bms"), "runtime_bms")
        voltage_proxy = _require_mapping(
            runtime_bms.get("terminal_voltage_proxy"),
            "runtime_bms.terminal_voltage_proxy",
        )

        nominal_voltage = _positive(
            battery.get("nominal_voltage_V"), "battery.nominal_voltage_V"
        )
        capacity = _positive(battery.get("capacity_Ah"), "battery.capacity_Ah")
        nominal_energy = _positive(
            battery.get("nominal_energy_Wh"), "battery.nominal_energy_Wh"
        )
        if not math.isclose(
            nominal_voltage * capacity, nominal_energy, rel_tol=1e-12
        ):
            raise ValueError(
                "battery.nominal_energy_Wh must equal nominal_voltage_V * capacity_Ah"
            )

        bms_mode = str(runtime_bms.get("mode", ""))
        if bms_mode != TB47D_BMS_MODE:
            raise ValueError("unsupported run-scoped BMS mode")

        bms_config = BMSConfig(
            enabled=bool(runtime_bms.get("enabled", False)),
            mode=bms_mode,
            nominal_voltage_v=nominal_voltage,
            capacity_ah=capacity,
            nominal_energy_wh=nominal_energy,
            initial_soc_percent=_nonnegative(
                runtime_bms.get("initial_relative_soc_percent"),
                "runtime_bms.initial_relative_soc_percent",
            ),
            minimum_endurance_power_w=_positive(
                runtime_bms.get("minimum_endurance_power_W"),
                "runtime_bms.minimum_endurance_power_W",
            ),
            maximum_valid_gap_s=_positive(
                integration.get("maximum_valid_gap_s"),
                "integration.maximum_valid_gap_s",
            ),
            endurance_power_time_constant_s=_positive(
                runtime_bms.get("endurance_power_time_constant_s"),
                "runtime_bms.endurance_power_time_constant_s",
            ),
            low_soc_warning_percent=_positive(
                runtime_bms.get("low_soc_warning_percent"),
                "runtime_bms.low_soc_warning_percent",
            ),
            critical_soc_percent=_positive(
                runtime_bms.get("critical_soc_percent"),
                "runtime_bms.critical_soc_percent",
            ),
            measured_telemetry_enabled=bool(
                runtime_bms.get("measured_telemetry_enabled", False)
            ),
            measured_power_tolerance_fraction=_positive(
                runtime_bms.get("measured_power_tolerance_fraction"),
                "runtime_bms.measured_power_tolerance_fraction",
            ),
            terminal_voltage_proxy_enabled=bool(voltage_proxy.get("enabled", False)),
            terminal_voltage_model_id=(
                str(voltage_proxy.get("model_id"))
                if voltage_proxy.get("model_id") is not None
                else None
            ),
            terminal_voltage_intercept_v=_optional_positive(
                voltage_proxy.get("intercept_V"),
                "runtime_bms.terminal_voltage_proxy.intercept_V",
            ),
            terminal_voltage_resistance_ohm=_optional_positive(
                voltage_proxy.get("resistance_ohm"),
                "runtime_bms.terminal_voltage_proxy.resistance_ohm",
            ),
            terminal_voltage_depletion_slope_v_per_ah=_optional_positive(
                voltage_proxy.get("depletion_slope_V_per_Ah"),
                "runtime_bms.terminal_voltage_proxy.depletion_slope_V_per_Ah",
            ),
            terminal_voltage_minimum_current_a=_nonnegative(
                voltage_proxy.get("minimum_current_A", 0.0),
                "runtime_bms.terminal_voltage_proxy.minimum_current_A",
            ),
            terminal_voltage_maximum_current_a=_optional_positive(
                voltage_proxy.get("maximum_current_A"),
                "runtime_bms.terminal_voltage_proxy.maximum_current_A",
            ),
            terminal_voltage_minimum_power_w=_optional_positive(
                voltage_proxy.get("minimum_power_W"),
                "runtime_bms.terminal_voltage_proxy.minimum_power_W",
            ),
            terminal_voltage_maximum_power_w=_optional_positive(
                voltage_proxy.get("maximum_power_W"),
                "runtime_bms.terminal_voltage_proxy.maximum_power_W",
            ),
            terminal_voltage_maximum_throughput_ah=_optional_positive(
                voltage_proxy.get("maximum_throughput_Ah"),
                "runtime_bms.terminal_voltage_proxy.maximum_throughput_Ah",
            ),
            terminal_voltage_validation_mae_v=_optional_positive(
                voltage_proxy.get("validation_MAE_V"),
                "runtime_bms.terminal_voltage_proxy.validation_MAE_V",
            ),
            terminal_voltage_validation_rmse_v=_optional_positive(
                voltage_proxy.get("validation_RMSE_V"),
                "runtime_bms.terminal_voltage_proxy.validation_RMSE_V",
            ),
            terminal_voltage_validation_p95_error_v=_optional_positive(
                voltage_proxy.get("validation_P95_abs_error_V"),
                "runtime_bms.terminal_voltage_proxy.validation_P95_abs_error_V",
            ),
            terminal_current_validation_p95_error_a=_optional_positive(
                voltage_proxy.get("validation_P95_abs_error_A"),
                "runtime_bms.terminal_voltage_proxy.validation_P95_abs_error_A",
            ),
        )

        initial_soc = bms_config.initial_soc_percent
        if initial_soc > 100.0:
            raise ValueError("initial_relative_soc_percent must be in [0, 100]")

        configured_warnings = model.get("warnings", DEFAULT_MODEL_WARNINGS)
        if not isinstance(configured_warnings, (list, tuple)):
            raise ValueError("model.warnings must be a sequence")
        warnings = tuple(str(item).strip() for item in configured_warnings)
        if any(not item for item in warnings):
            raise ValueError("model warnings must not be empty")
        if MASS_BASIS_WARNING not in warnings:
            raise ValueError(f"model warnings must include {MASS_BASIS_WARNING}")

        model_id = str(model.get("id", "")).strip()
        source_doi = str(source.get("doi", "")).strip()
        source_title = str(source.get("title", "")).strip()
        source_fidelity = str(model.get("fidelity", "")).strip()
        if not all((model_id, source_doi, source_title, source_fidelity)):
            raise ValueError("model id, source, and fidelity metadata are required")
        if model_id != MODEL_ID:
            raise ValueError(f"model.id must be {MODEL_ID}")

        return cls(
            model_id=model_id,
            source_doi=source_doi,
            source_title=source_title,
            source_fidelity=source_fidelity,
            mass_basis_status=mass_basis_status,
            published_regression_airframe_mass_kg=regression_airframe_mass,
            published_regression_battery_mass_kg=regression_battery_mass,
            published_regression_baseline_mass_kg=regression_baseline_mass,
            physical_runtime_baseline_mass_kg=physical_runtime_baseline_mass,
            mass_basis_source_doi=mass_source_doi,
            mass_basis_source_archive_sha256=mass_source_archive_sha256,
            mass_basis_source_archive_root=mass_source_archive_root,
            mass_basis_source_path=mass_source_path,
            mass_basis_source_code_expression=mass_source_code_expression,
            coefficients=coefficients,
            phase_mapping=phase_mapping,
            hover_enabled=bool(hover.get("enabled", False)),
            hover_power_w=_positive(
                hover.get("direct_power_W"), "model.limited_hover.direct_power_W"
            ),
            hover_status=hover_status,
            maximum_valid_gap_s=bms_config.maximum_valid_gap_s,
            rotor_joint_velocity_scale=_positive(
                diagnostics.get("rotor_joint_velocity_scale"),
                "diagnostics.rotor_joint_velocity_scale",
            ),
            bms_config=bms_config,
            warnings=warnings,
            metadata=deepcopy(dict(data)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PublishedEnergyConfig":
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        return cls.from_mapping(_require_mapping(data, "configuration root"))

    @property
    def nominal_voltage_v(self) -> float:
        return self.bms_config.nominal_voltage_v

    @property
    def capacity_ah(self) -> float:
        return self.bms_config.capacity_ah

    @property
    def nominal_energy_wh(self) -> float:
        return self.bms_config.nominal_energy_wh

    @property
    def bms_mode(self) -> str:
        return self.bms_config.mode

    @property
    def bms_initial_soc_percent(self) -> float:
        return self.bms_config.initial_soc_percent

    def published_regression_mass_kg(self, payload_mass_kg: float) -> float:
        """Return the exact mass feature used by the published v1.0.0 code."""
        payload = _nonnegative(payload_mass_kg, "payload_mass_kg")
        return self.published_regression_baseline_mass_kg + payload

    def validate_runtime_masses(
        self,
        physical_total_mass_kg: float,
        payload_mass_kg: float,
    ) -> tuple[float, float, float]:
        """Validate physical provenance and return physical, payload, feature mass."""
        physical = _positive(physical_total_mass_kg, "physical_total_mass_kg")
        payload = _nonnegative(payload_mass_kg, "payload_mass_kg")
        expected_physical = self.physical_runtime_baseline_mass_kg + payload
        if not math.isclose(
            physical,
            expected_physical,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "physical_total_mass_kg must equal the configured 3.680 kg "
                "runtime baseline plus payload_mass_kg"
            )
        return physical, payload, self.published_regression_mass_kg(payload)

    def mass_basis_provenance(self) -> dict[str, Any]:
        """Return JSON-safe immutable provenance for the regression mass basis."""
        return {
            "status": self.mass_basis_status,
            "published_regression_equation": (
                "published_regression_mass_kg = 3.715 + payload_mass_kg"
            ),
            "published_regression_airframe_mass_kg": (
                self.published_regression_airframe_mass_kg
            ),
            "published_regression_battery_mass_kg": (
                self.published_regression_battery_mass_kg
            ),
            "published_regression_baseline_mass_kg": (
                self.published_regression_baseline_mass_kg
            ),
            "physical_runtime_equation": (
                "physical_total_mass_kg = 3.680 + payload_mass_kg"
            ),
            "physical_runtime_baseline_mass_kg": (
                self.physical_runtime_baseline_mass_kg
            ),
            "source_doi": self.mass_basis_source_doi,
            "source_archive_sha256": self.mass_basis_source_archive_sha256,
            "source_archive_root_commit": self.mass_basis_source_archive_root,
            "source_path": self.mass_basis_source_path,
            "source_code_expression": self.mass_basis_source_code_expression,
            "warning": MASS_BASIS_WARNING,
        }

    def reproduction_provenance(self) -> dict[str, Any]:
        """Return the validated 120/68 reproduction record for run sidecars."""
        model = _require_mapping(self.metadata.get("model"), "model")
        source = _require_mapping(model.get("source"), "model.source")
        reproduction = _require_mapping(
            source.get("local_reproduction"), "model.source.local_reproduction"
        )
        return {
            "paper_doi": self.source_doi,
            **deepcopy(dict(reproduction)),
        }

    def estimate_power(
        self,
        phase: str,
        payload_mass_kg: float,
        air_density_kg_m3: float,
        *,
        limited_hover_eligible: bool = False,
        stable_cruise_speed_adjustment_w: float | None = None,
    ) -> PowerEstimate:
        """Estimate phase power from payload using the published mass feature."""
        phase_name = str(phase).strip().upper()
        mass = self.published_regression_mass_kg(payload_mass_kg)
        density = _positive(air_density_kg_m3, "air_density_kg_m3")
        if stable_cruise_speed_adjustment_w is not None and (
            phase_name != "CRUISE" or not math.isfinite(stable_cruise_speed_adjustment_w)
        ):
            raise ValueError("stable-cruise adjustment requires finite outbound CRUISE power")

        if phase_name == "HOVER":
            if limited_hover_eligible and self.hover_enabled:
                return PowerEstimate(
                    phase=phase_name,
                    coefficient_phase="limited_hover",
                    power_w=self.hover_power_w,
                    status=self.hover_status,
                    warnings=(
                        "LIMITED_HOVER_VALUE_CALLER_ASSERTED_ENVELOPE",
                        "LIMITED_HOVER_VALUE_NOT_INDEPENDENT_VALIDATION",
                    ),
                )
            return PowerEstimate(
                phase=phase_name,
                coefficient_phase=None,
                power_w=None,
                status=UNAVAILABLE_STATUS,
                warnings=("HOVER_POWER_REQUIRES_EXPLICIT_LIMITED_ELIGIBILITY",),
            )

        if limited_hover_eligible:
            raise ValueError("limited_hover_eligible may only be true in HOVER")

        coefficient_phase = self.phase_mapping.get(phase_name)
        if coefficient_phase is None:
            return PowerEstimate(
                phase=phase_name,
                coefficient_phase=None,
                power_w=None,
                status=UNAVAILABLE_STATUS,
                warnings=(f"POWER_UNAVAILABLE_FOR_PHASE_{phase_name or 'EMPTY'}",),
            )

        coefficients = self.coefficients[coefficient_phase]
        power = coefficients.b1 * mass**1.5 / math.sqrt(density) + coefficients.b0
        corrected = stable_cruise_speed_adjustment_w is not None
        if corrected:
            power += stable_cruise_speed_adjustment_w
        if not math.isfinite(power) or power <= 0.0:
            return PowerEstimate(
                phase=phase_name,
                coefficient_phase=coefficient_phase,
                power_w=None,
                status=UNAVAILABLE_STATUS,
                warnings=("PUBLISHED_EQUATION_PRODUCED_NONPHYSICAL_POWER",),
            )
        return PowerEstimate(
            phase=phase_name,
            coefficient_phase=coefficient_phase,
            power_w=power,
            status=STABLE_CRUISE_MODEL_STATUS if corrected else MODEL_STATUS,
            warnings=(STABLE_CRUISE_MODEL_WARNING,) if corrected else (),
            model_id=STABLE_CRUISE_MODEL_ID if corrected else self.model_id,
            stable_cruise_speed_adjustment_w=stable_cruise_speed_adjustment_w,
        )


@dataclass
class PublishedEnergySnapshot:
    """Atomic runtime view with physical and regression masses kept distinct.

    ``total_mass_kg`` is retained as a compatibility alias for the physical
    simulator mass.  Power uses only ``published_regression_mass_kg``.
    """

    timestamp_s: float | None = None
    phase: str = "IDLE"
    coefficient_phase: str | None = None
    total_mass_kg: float | None = None
    physical_total_mass_kg: float | None = None
    payload_mass_kg: float | None = None
    published_regression_mass_kg: float | None = None
    mass_basis_status: str = MASS_BASIS_STATUS
    mass_basis_source_doi: str = PUBLISHED_MASS_SOURCE_DOI
    mass_basis_source_archive_sha256: str = PUBLISHED_MASS_SOURCE_ARCHIVE_SHA256
    mass_basis_source_archive_root: str = PUBLISHED_MASS_SOURCE_ARCHIVE_ROOT
    mass_basis_source_path: str = PUBLISHED_MASS_SOURCE_PATH
    air_density_kg_m3: float | None = None
    joint_velocity_rad_s: list[float | None] = field(
        default_factory=lambda: [None] * 4
    )
    rotor_omega_rad_s: list[float | None] = field(
        default_factory=lambda: [None] * 4
    )
    rotor_rpm: list[float | None] = field(default_factory=lambda: [None] * 4)
    rotor_torque_nm: list[float | None] = field(default_factory=lambda: [None] * 4)
    rotor_mechanical_power_w: list[float | None] = field(
        default_factory=lambda: [None] * 4
    )
    mechanical_power_total_w: float | None = None
    operational_power_w: float | None = None
    electrical_power_w: float | None = None
    stable_cruise_speed_adjustment_w: float | None = None
    cumulative_energy_wh: float = 0.0
    phase_energy_wh: float | None = None
    phase_energy_by_phase_wh: dict[str, float] = field(default_factory=dict)
    phase_power_model_ids: dict[str, list[str]] = field(default_factory=dict)
    recorded_energy_coverage: dict[str, float | bool | None] | None = None
    airborne_energy_coverage: dict[str, float | bool | None] | None = None
    energy_per_distance_wh_per_km: float | None = None
    wind_power_adjustment_w: float | None = None
    air_relative_speed_mps: float | None = None
    bms_input: BMSInput | None = None
    battery_voltage_v: float | None = None
    battery_voltage_uncertainty_v: float | None = None
    battery_current_uncertainty_a: float | None = None
    battery_voltage_validation_mae_v: float | None = None
    battery_voltage_validation_rmse_v: float | None = None
    battery_terminal_voltage_model_id: str | None = None
    battery_terminal_voltage_in_envelope: bool = False
    battery_pack_identity_status: str = "UNRESOLVED_PUBLISHED_M100_CAMPAIGN"
    battery_current_a: float | None = None
    consumed_capacity_ah: float | None = None
    soc_percent: float | None = None
    endurance_min: float | None = None
    battery_nameplate_remaining_wh: float | None = None
    battery_remaining_capacity_ah: float | None = None
    battery_consumed_energy_wh: float | None = None
    battery_endurance_reference_power_w: float | None = None
    battery_estimator_mode: str | None = None
    battery_dynamic_model_enabled: bool = False
    battery_budget_state: str = "INACTIVE"
    bms_state: str = "INACTIVE"
    bms_safety_level: str = "UNAVAILABLE"
    bms_data_quality: str = "UNAVAILABLE"
    bms_power_source: str | None = None
    bms_soc_source: str | None = None
    bms_cutoff_imminent: bool = False
    bms_advisory_only: bool = True
    battery_research: dict[str, Any] | None = None
    model_id: str | None = None
    source_doi: str | None = None
    source_fidelity: str | None = None
    rpm_status: str = "UNAVAILABLE"
    mechanical_power_status: str = MECHANICAL_STATUS
    electrical_status: str = UNAVAILABLE_STATUS
    battery_status: str = BATTERY_UNAVAILABLE_STATUS
    energy_model_status: str = UNAVAILABLE_STATUS
    warnings: list[str] = field(default_factory=list)

    @property
    def power_available(self) -> bool:
        return self.operational_power_w is not None

    def as_dict(self) -> dict[str, Any]:
        result = deepcopy(self.__dict__)
        if self.bms_input is not None:
            result["bms_input"] = deepcopy(self.bms_input.__dict__)
        return result


class M100PublishedEnergyModel:
    """Thread-safe phase power, energy integration, and BMS input layer."""

    def __init__(
        self, config: PublishedEnergyConfig,
        battery_research_profile: TheveninProfile | None = None,
    ):
        self.config = config
        self._lock = threading.RLock()
        self._bms = M100BMS(config.bms_config)
        self._battery_study = (
            TheveninStudy(battery_research_profile, config.maximum_valid_gap_s)
            if battery_research_profile is not None else None
        )
        self.reset()

    def reset(self, initial_soc_percent: float | None = None) -> None:
        with self._lock:
            self._energy_wh = 0.0
            self._phase_energy_wh: dict[str, float] = {}
            self._phase_models: dict[str, set[str]] = {}
            self._last_timestamp_s: float | None = None
            self._last_power_w: float | None = None
            self._last_phase: str | None = None
            self._warnings: list[str] = []
            self._track_recorded_time = False
            self._reset_recorded_time()
            self._bms.reset(initial_soc_percent)
            if self._battery_study is not None:
                self._battery_study.reset()
            self._snapshot = PublishedEnergySnapshot(
                model_id=self.config.model_id,
                source_doi=self.config.source_doi,
                source_fidelity=self.config.source_fidelity,
                mass_basis_status=self.config.mass_basis_status,
                mass_basis_source_doi=self.config.mass_basis_source_doi,
                mass_basis_source_archive_sha256=(
                    self.config.mass_basis_source_archive_sha256
                ),
                mass_basis_source_archive_root=(
                    self.config.mass_basis_source_archive_root
                ),
                mass_basis_source_path=self.config.mass_basis_source_path,
                battery_research=self._battery_research_payload("INACTIVE"),
                warnings=list(self.config.warnings),
            )

    def begin_bms_session(self, initial_soc_percent: float | None = None) -> None:
        with self._lock:
            self._bms.begin(initial_soc_percent)
            if self._battery_study is not None:
                self._battery_study.reset()
            self._reset_recorded_time()
            self._track_recorded_time = True
            self._snapshot.recorded_energy_coverage = None
            self._apply_bms_snapshot(self._bms.snapshot())

    def freeze_bms_session(self) -> None:
        with self._lock:
            self._bms.freeze()
            self._track_recorded_time = False
            self._apply_bms_snapshot(self._bms.snapshot())

    def begin_nameplate_budget(self, initial_soc_percent: float | None = None) -> None:
        self.begin_bms_session(initial_soc_percent)

    def freeze_nameplate_budget(self) -> None:
        self.freeze_bms_session()

    @property
    def phase_energy_wh(self) -> dict[str, float]:
        with self._lock:
            return dict(self._phase_energy_wh)

    def snapshot(self) -> PublishedEnergySnapshot:
        with self._lock:
            return deepcopy(self._snapshot)

    def update(
        self,
        timestamp_s: float,
        joint_velocity_rad_s: Iterable[float | None],
        phase: str,
        distance_m: float,
        *,
        physical_total_mass_kg: float,
        payload_mass_kg: float,
        air_density_kg_m3: float | None,
        use_limited_hover_mean: bool = False,
        air_relative_speed_mps: float | None = None,
        stable_cruise_speed_adjustment_w: float | None = None,
        transient_power_estimate: PowerEstimate | None = None,
        wind_power_adjustment_w: float | None = None,
        airborne: bool = False,
        research_rotor_torque_coefficient: float | None = None,
    ) -> PublishedEnergySnapshot:
        """Update power and energy without deriving power from rotor telemetry."""
        with self._lock:
            return self._update_unlocked(
                timestamp_s=timestamp_s,
                joint_velocity_rad_s=joint_velocity_rad_s,
                phase=phase,
                distance_m=distance_m,
                physical_total_mass_kg=physical_total_mass_kg,
                payload_mass_kg=payload_mass_kg,
                air_density_kg_m3=air_density_kg_m3,
                use_limited_hover_mean=use_limited_hover_mean,
                air_relative_speed_mps=air_relative_speed_mps,
                stable_cruise_speed_adjustment_w=stable_cruise_speed_adjustment_w,
                transient_power_estimate=transient_power_estimate,
                wind_power_adjustment_w=wind_power_adjustment_w,
                airborne=airborne,
                research_rotor_torque_coefficient=research_rotor_torque_coefficient,
            )

    def _update_unlocked(
        self,
        *,
        timestamp_s: float,
        joint_velocity_rad_s: Iterable[float | None],
        phase: str,
        distance_m: float,
        physical_total_mass_kg: float,
        payload_mass_kg: float,
        air_density_kg_m3: float | None,
        use_limited_hover_mean: bool,
        air_relative_speed_mps: float | None,
        stable_cruise_speed_adjustment_w: float | None,
        transient_power_estimate: PowerEstimate | None,
        wind_power_adjustment_w: float | None,
        airborne: bool,
        research_rotor_torque_coefficient: float | None,
    ) -> PublishedEnergySnapshot:
        timestamp = float(timestamp_s)
        phase_name = str(phase).strip().upper()
        physical_mass, payload_mass, regression_mass = (
            self.config.validate_runtime_masses(
                physical_total_mass_kg,
                payload_mass_kg,
            )
        )
        density = _optional_positive(air_density_kg_m3, "air_density_kg_m3")
        _nonnegative(distance_m, "distance_m")
        clean_air_speed = (
            None
            if air_relative_speed_mps is None
            else _nonnegative(air_relative_speed_mps, "air_relative_speed_mps")
        )
        velocities, omega, rpm, rpm_status = self._rotor_diagnostics(
            joint_velocity_rad_s
        )

        if not math.isfinite(timestamp):
            self._warn("NON_FINITE_TIMESTAMP")
            if self._track_recorded_time:
                self._recorded_clock_valid = False
            self._break_continuity(None)
            bms = self._bms.mark_unavailable(None)
            self._update_battery_study(None, None, omega, None)
            return self._set_snapshot(
                timestamp=None,
                phase=phase_name,
                coefficient_phase=None,
                physical_mass=physical_mass,
                payload_mass=payload_mass,
                regression_mass=regression_mass,
                density=density,
                velocities=velocities,
                omega=omega,
                rpm=rpm,
                rpm_status=rpm_status,
                air_relative_speed_mps=clean_air_speed,
                estimate=PowerEstimate(
                    phase=phase_name,
                    coefficient_phase=None,
                    power_w=None,
                    status=UNAVAILABLE_STATUS,
                    warnings=("NON_FINITE_TIMESTAMP",),
                ),
                bms=bms,
            )

        if density is None:
            estimate = PowerEstimate(
                phase=phase_name, coefficient_phase=None, power_w=None,
                status=UNAVAILABLE_STATUS, warnings=("POWER_DENSITY_POSITION_UNAVAILABLE",),
            )
        else:
            estimate = self.config.estimate_power(
                phase_name, payload_mass, density,
                limited_hover_eligible=use_limited_hover_mean,
                stable_cruise_speed_adjustment_w=stable_cruise_speed_adjustment_w,
            )
        if density is not None and transient_power_estimate is not None:
            if estimate.available or transient_power_estimate.phase != phase_name:
                raise ValueError("transient power only supplements the matching unsupported phase")
            if transient_power_estimate.power_w is None:
                raise ValueError("transient estimate must supply finite positive power")
            _positive(transient_power_estimate.power_w, "transient power")
            estimate = transient_power_estimate
        if wind_power_adjustment_w is not None and estimate.available:
            correction = _finite(wind_power_adjustment_w, "wind power correction")
            power = _positive(estimate.power_w + correction, "wind-corrected power")
            estimate = replace(
                estimate, power_w=power, status="C_NOT_CALIBRATED",
                model_id="M100_WIND_SIMULATION_V1", wind_power_adjustment_w=correction,
                warnings=(*estimate.warnings, "THEORETICAL_COMMANDED_WIND_TRANSFER_NOT_EMPIRICALLY_CALIBRATED"),
            )
        if self._track_recorded_time and estimate.available:
            self._phase_models.setdefault(phase_name, set()).add(estimate.model_id)
        for warning in estimate.warnings:
            self._warn(warning)

        self._integrate(timestamp, estimate.power_w, phase_name, airborne=airborne)
        bms = (
            self._bms.update(timestamp, estimate.power_w)
            if estimate.power_w is not None
            else self._bms.mark_unavailable(timestamp)
        )
        self._update_battery_study(
            timestamp, estimate.power_w, omega, research_rotor_torque_coefficient
        )
        return self._set_snapshot(
            timestamp=timestamp,
            phase=phase_name,
            coefficient_phase=estimate.coefficient_phase,
            physical_mass=physical_mass,
            payload_mass=payload_mass,
            regression_mass=regression_mass,
            density=density,
            velocities=velocities,
            omega=omega,
            rpm=rpm,
            rpm_status=rpm_status,
            air_relative_speed_mps=clean_air_speed,
            estimate=estimate,
            bms=bms,
        )

    def _update_battery_study(
        self, timestamp: float | None, operational_power_w: float | None,
        omega: list[float | None], torque_coefficient: float | None,
    ) -> None:
        study = self._battery_study
        if study is None or not self._track_recorded_time:
            return
        if study.profile.drive is None:
            study.step(timestamp, terminal_power_w=operational_power_w)
            return
        shafts = None
        if (
            operational_power_w is not None
            and torque_coefficient is not None
            and all(speed is not None for speed in omega)
        ):
            shafts = [
                ShaftDemand(speed, torque_coefficient * speed**2) for speed in omega
            ]
        study.step(timestamp, shafts=shafts)

    def _battery_research_payload(self, state: str) -> dict[str, Any] | None:
        if self._battery_study is None:
            return None
        has_drive = self._battery_study.profile.drive is not None
        return {
            **self._battery_study.as_dict(),
            "state": state,
            "load_source": (
                "GAZEBO_IDEAL_HOVER_TORQUE_PROXY" if has_drive
                else "OPERATIONAL_TERMINAL_POWER_REPLAY"
            ),
            "scope": "SUPPORTED_PHASES_ONLY",
        }

    def _rotor_diagnostics(
        self, values: Iterable[float | None] | None
    ) -> tuple[
        list[float | None],
        list[float | None],
        list[float | None],
        str,
    ]:
        if values is None:
            return [None] * 4, [None] * 4, [None] * 4, "UNAVAILABLE"
        raw = list(values)
        if len(raw) != 4:
            raise ValueError("exactly four rotor joint velocities are required")

        clean: list[float | None] = []
        for value in raw:
            if value is None:
                clean.append(None)
                continue
            number = float(value)
            if not math.isfinite(number):
                clean.append(None)
                self._warn("NON_FINITE_ROTOR_TELEMETRY")
            else:
                clean.append(number)

        omega = [
            None
            if value is None
            else abs(value) * self.config.rotor_joint_velocity_scale
            for value in clean
        ]
        rpm = [
            None if value is None else value * 60.0 / (2.0 * math.pi)
            for value in omega
        ]
        status = RPM_STATUS if all(value is not None for value in rpm) else "UNAVAILABLE"
        if status == "UNAVAILABLE":
            self._warn("INCOMPLETE_ROTOR_SPEED_DIAGNOSTIC")
        return clean, omega, rpm, status

    def _integrate(
        self, timestamp_s: float, power_w: float | None, phase: str,
        *, airborne: bool = False,
    ) -> None:
        recorded = self._track_recorded_time and self._previous_sample_recorded
        self._previous_sample_recorded = self._track_recorded_time
        previous_airborne = self._previous_airborne
        self._previous_airborne = airborne
        if self._last_timestamp_s is None:
            self._last_timestamp_s = timestamp_s
            self._last_power_w = power_w
            self._last_phase = phase if power_w is not None else None
            return

        dt = timestamp_s - self._last_timestamp_s
        airborne_duration = dt * (float(previous_airborne) + float(airborne)) / 2.0
        if recorded:
            if dt < 0.0 or not math.isfinite(dt):
                self._recorded_clock_valid = False
            elif dt > 0.0:
                self._recorded_span_s += dt
                self._airborne_span_s += airborne_duration
        if dt <= 0.0:
            self._warn("TIMESTAMP_RESET_OR_DUPLICATE")
        elif dt > self.config.maximum_valid_gap_s:
            self._warn("TELEMETRY_GAP_NOT_INTEGRATED")
            if recorded:
                self._recorded_gap_s += dt
        elif power_w is not None and self._last_power_w is not None:
            if recorded:
                self._integrated_recorded_s += dt
                self._integrated_airborne_s += airborne_duration
                self._airborne_energy_wh += dt / 7200.0 * (
                    self._last_power_w * previous_airborne + power_w * airborne
                )
            previous_delta_wh = 0.5 * self._last_power_w * dt / 3600.0
            current_delta_wh = 0.5 * power_w * dt / 3600.0
            self._energy_wh += previous_delta_wh + current_delta_wh
            previous_phase = self._last_phase or phase
            self._phase_energy_wh[previous_phase] = (
                self._phase_energy_wh.get(previous_phase, 0.0) + previous_delta_wh
            )
            self._phase_energy_wh[phase] = (
                self._phase_energy_wh.get(phase, 0.0) + current_delta_wh
            )

        self._last_timestamp_s = timestamp_s
        self._last_power_w = power_w
        self._last_phase = phase if power_w is not None else None

    def _reset_recorded_time(self) -> None:
        self._previous_sample_recorded = False
        self._previous_airborne = False
        self._recorded_clock_valid = True
        self._recorded_span_s = 0.0
        self._recorded_gap_s = 0.0
        self._integrated_recorded_s = 0.0
        self._airborne_span_s = 0.0
        self._integrated_airborne_s = 0.0
        self._airborne_energy_wh = 0.0

    def _airborne_energy_coverage(self) -> dict[str, float | bool | None] | None:
        if self._airborne_span_s == 0.0:
            return None
        valid = self._recorded_clock_valid
        duration = self._airborne_span_s
        integrated = self._integrated_airborne_s
        return {
            "airborne_duration_s": duration if valid else None,
            "integrated_power_duration_s": integrated if valid else None,
            "power_time_coverage_percent": 100.0 * integrated / duration if valid else None,
            "estimated_airborne_energy_wh": self._airborne_energy_wh if valid else None,
            "complete": valid and math.isclose(duration, integrated, rel_tol=0.0, abs_tol=1e-9),
        }

    def _recorded_energy_coverage(self) -> dict[str, float | bool | None] | None:
        """Report time coverage for the run's energy-clock samples, including gaps."""
        if self._recorded_clock_valid and self._recorded_span_s == 0.0:
            return None
        valid = self._recorded_clock_valid
        span = self._recorded_span_s
        integrated = self._integrated_recorded_s
        gap = self._recorded_gap_s
        return {
            "sample_clock_continuous": valid,
            "recorded_span_s": span if valid else None,
            "integrated_power_duration_s": integrated if valid else None,
            "duration_without_power_s": max(0.0, span - integrated - gap) if valid else None,
            "telemetry_gap_duration_s": gap if valid else None,
            "power_time_coverage_percent": 100.0 * integrated / span if valid else None,
        }

    def _break_continuity(self, timestamp_s: float | None) -> None:
        self._last_timestamp_s = timestamp_s
        self._last_power_w = None
        self._last_phase = None

    def _warn(self, code: str) -> None:
        if code not in self._warnings:
            self._warnings.append(code)

    def _combined_warnings(self, bms: BMSSnapshot) -> list[str]:
        return list(
            dict.fromkeys([*self.config.warnings, *self._warnings, *bms.warnings])
        )

    def _set_snapshot(
        self,
        *,
        timestamp: float | None,
        phase: str,
        coefficient_phase: str | None,
        physical_mass: float,
        payload_mass: float,
        regression_mass: float,
        density: float | None,
        velocities: list[float | None],
        omega: list[float | None],
        rpm: list[float | None],
        rpm_status: str,
        air_relative_speed_mps: float | None,
        estimate: PowerEstimate,
        bms: BMSSnapshot,
    ) -> PublishedEnergySnapshot:
        power = estimate.power_w
        warnings = self._combined_warnings(bms)
        if estimate.stable_cruise_speed_adjustment_w is not None:
            warnings = [
                "WIND_NOT_EXPLICIT_POWER_MODEL_INPUT"
                if warning == "SPEED_AND_WIND_NOT_EXPLICIT_MODEL_INPUTS" else warning
                for warning in warnings
            ]
        if estimate.coefficient_phase is None and estimate.model_id != self.config.model_id:
            warnings = [
                "EMPIRICAL_KINEMATIC_POWER_SIMULATION_TRANSFER"
                if warning == "PHASE_AVERAGE_POWER_NOT_INSTANTANEOUS_PROPULSION_DYNAMICS" else warning
                for warning in warnings
            ]
        bms_input = BMSInput(
            timestamp_s=timestamp,
            load_power_w=power,
            source_model_id=estimate.model_id,
            fidelity=estimate.status,
            eligible=power is not None,
        )
        self._snapshot = PublishedEnergySnapshot(
            timestamp_s=timestamp,
            phase=phase,
            coefficient_phase=coefficient_phase,
            total_mass_kg=physical_mass,
            physical_total_mass_kg=physical_mass,
            payload_mass_kg=payload_mass,
            published_regression_mass_kg=regression_mass,
            mass_basis_status=self.config.mass_basis_status,
            mass_basis_source_doi=self.config.mass_basis_source_doi,
            mass_basis_source_archive_sha256=(
                self.config.mass_basis_source_archive_sha256
            ),
            mass_basis_source_archive_root=(
                self.config.mass_basis_source_archive_root
            ),
            mass_basis_source_path=self.config.mass_basis_source_path,
            air_density_kg_m3=density,
            joint_velocity_rad_s=velocities,
            rotor_omega_rad_s=omega,
            rotor_rpm=rpm,
            rotor_torque_nm=[None] * 4,
            rotor_mechanical_power_w=[None] * 4,
            mechanical_power_total_w=None,
            operational_power_w=power,
            electrical_power_w=power,
            stable_cruise_speed_adjustment_w=estimate.stable_cruise_speed_adjustment_w,
            cumulative_energy_wh=self._energy_wh,
            phase_energy_wh=(
                self._phase_energy_wh.get(phase, 0.0)
                if power is not None
                else None
            ),
            phase_energy_by_phase_wh=dict(self._phase_energy_wh),
            phase_power_model_ids={phase: sorted(models) for phase, models in self._phase_models.items()},
            recorded_energy_coverage=self._recorded_energy_coverage(),
            airborne_energy_coverage=self._airborne_energy_coverage(),
            air_relative_speed_mps=air_relative_speed_mps,
            wind_power_adjustment_w=estimate.wind_power_adjustment_w,
            bms_input=bms_input,
            battery_voltage_v=bms.voltage_v,
            battery_voltage_uncertainty_v=bms.voltage_uncertainty_v,
            battery_current_uncertainty_a=bms.current_uncertainty_a,
            battery_voltage_validation_mae_v=bms.voltage_validation_mae_v,
            battery_voltage_validation_rmse_v=bms.voltage_validation_rmse_v,
            battery_terminal_voltage_model_id=bms.terminal_voltage_model_id,
            battery_terminal_voltage_in_envelope=bms.terminal_voltage_in_envelope,
            battery_pack_identity_status=bms.pack_identity_status,
            battery_current_a=bms.current_a,
            consumed_capacity_ah=bms.consumed_capacity_ah,
            soc_percent=bms.soc_percent,
            endurance_min=bms.endurance_min,
            battery_nameplate_remaining_wh=bms.remaining_energy_wh,
            battery_remaining_capacity_ah=bms.remaining_capacity_ah,
            battery_consumed_energy_wh=bms.consumed_energy_wh,
            battery_endurance_reference_power_w=bms.endurance_reference_power_w,
            battery_estimator_mode=bms.estimator_mode,
            battery_dynamic_model_enabled=bms.dynamic_model_enabled,
            battery_budget_state=bms.state,
            bms_state=bms.state,
            bms_safety_level=bms.safety_level,
            bms_data_quality=bms.data_quality,
            bms_power_source=bms.power_source,
            bms_soc_source=bms.soc_source,
            bms_cutoff_imminent=bms.cutoff_imminent,
            bms_advisory_only=bms.advisory_only,
            battery_research=self._battery_research_payload(bms.state),
            model_id=estimate.model_id,
            source_doi=self.config.source_doi,
            source_fidelity=estimate.status,
            rpm_status=rpm_status,
            mechanical_power_status=MECHANICAL_STATUS,
            electrical_status=estimate.status,
            battery_status=(bms.status if bms.available else BATTERY_UNAVAILABLE_STATUS),
            energy_model_status=estimate.status,
            warnings=warnings,
        )
        return self.snapshot()

    def _apply_bms_snapshot(self, bms: BMSSnapshot) -> None:
        self._snapshot.battery_voltage_v = bms.voltage_v
        self._snapshot.battery_voltage_uncertainty_v = bms.voltage_uncertainty_v
        self._snapshot.battery_current_uncertainty_a = bms.current_uncertainty_a
        self._snapshot.battery_voltage_validation_mae_v = bms.voltage_validation_mae_v
        self._snapshot.battery_voltage_validation_rmse_v = bms.voltage_validation_rmse_v
        self._snapshot.battery_terminal_voltage_model_id = bms.terminal_voltage_model_id
        self._snapshot.battery_terminal_voltage_in_envelope = (
            bms.terminal_voltage_in_envelope
        )
        self._snapshot.battery_pack_identity_status = bms.pack_identity_status
        self._snapshot.battery_current_a = bms.current_a
        self._snapshot.consumed_capacity_ah = bms.consumed_capacity_ah
        self._snapshot.soc_percent = bms.soc_percent
        self._snapshot.endurance_min = bms.endurance_min
        self._snapshot.battery_nameplate_remaining_wh = bms.remaining_energy_wh
        self._snapshot.battery_remaining_capacity_ah = bms.remaining_capacity_ah
        self._snapshot.battery_consumed_energy_wh = bms.consumed_energy_wh
        self._snapshot.battery_endurance_reference_power_w = (
            bms.endurance_reference_power_w
        )
        self._snapshot.battery_estimator_mode = bms.estimator_mode
        self._snapshot.battery_dynamic_model_enabled = bms.dynamic_model_enabled
        self._snapshot.battery_budget_state = bms.state
        self._snapshot.bms_state = bms.state
        self._snapshot.bms_safety_level = bms.safety_level
        self._snapshot.bms_data_quality = bms.data_quality
        self._snapshot.bms_power_source = bms.power_source
        self._snapshot.bms_soc_source = bms.soc_source
        self._snapshot.bms_cutoff_imminent = bms.cutoff_imminent
        self._snapshot.bms_advisory_only = bms.advisory_only
        self._snapshot.battery_research = self._battery_research_payload(bms.state)
        self._snapshot.battery_status = (
            bms.status if bms.available else BATTERY_UNAVAILABLE_STATUS
        )
        self._snapshot.warnings = self._combined_warnings(bms)


# Short aliases keep the integration import concise while the module filename
# makes the active model provenance explicit.
EnergyModelConfig = PublishedEnergyConfig
EnergySnapshot = PublishedEnergySnapshot
M100EnergyModel = M100PublishedEnergyModel
