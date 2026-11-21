"""Continuous CMU kinematic regression transferred to simulator transients."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import yaml

from .m100_published_energy_v2 import PowerEstimate


MODEL_ID = "M100_CMU_TRANSIENT_POWER_V1"
STATUS = "LIMITED_EMPIRICAL_PROXY"
PHASES = {"STABILIZING", "RETURN_BRAKING", "HOVER"}
WARNING = "CMU_KINEMATIC_POWER_TRANSFER_TO_LONG_TRANSIENTS_IS_SIMULATION_EXTRAPOLATION"
FEATURES = (
    "offset", "mass_over_density_feature", "ascent_work_W", "descent_work_W",
    "mass_feature_times_horizontal_speed_squared", "horizontal_speed_cubed",
    "horizontal_acceleration_work_W", "horizontal_braking_work_W",
)


@dataclass(frozen=True)
class TransientPowerFit:
    coefficients: tuple[float, ...]
    evidence_artifact: str
    evidence_sha256: str
    dataset_doi: str

    @classmethod
    def load(cls, path: Path) -> TransientPowerFit:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data["schema"] != MODEL_ID or data["model_id"] != MODEL_ID:
            raise ValueError("unsupported M100 transient-power configuration")
        coefficients = tuple(float(data["coefficients"][name]) for name in FEATURES)
        if not all(math.isfinite(value) for value in coefficients):
            raise ValueError("non-finite M100 transient-power coefficient")
        return cls(coefficients, str(data["evidence_artifact"]),
                   str(data["evidence_sha256"]), str(data["dataset_doi"]))

    def power_w(self, *, payload_kg: float, density_kg_m3: float,
                horizontal_speed_mps: float, vertical_speed_up_mps: float,
                horizontal_speed_rate_mps2: float) -> float:
        values = (payload_kg, density_kg_m3, horizontal_speed_mps,
                  vertical_speed_up_mps, horizontal_speed_rate_mps2)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("transient-power inputs must be finite")
        if payload_kg < 0.0 or density_kg_m3 <= 0.0 or horizontal_speed_mps < 0.0:
            raise ValueError("invalid mass, density, or speed for transient power")
        mass = 3.715 + payload_kg  # The published regression feature mass.
        mass_feature = mass**1.5 / math.sqrt(density_kg_m3)
        vertical_work = mass * 9.81 * vertical_speed_up_mps
        kinetic_work = mass * horizontal_speed_mps * horizontal_speed_rate_mps2
        features = (
            1.0, mass_feature, max(vertical_work, 0.0), min(vertical_work, 0.0),
            mass_feature * horizontal_speed_mps**2, horizontal_speed_mps**3,
            max(kinetic_work, 0.0), min(kinetic_work, 0.0),
        )
        return math.fsum(coefficient * value for coefficient, value in
                         zip(self.coefficients, features, strict=True))

    def provenance(self) -> dict[str, object]:
        return {"model_id": MODEL_ID, "status": STATUS,
                "coefficients": dict(zip(FEATURES, self.coefficients, strict=True)),
                "evidence_artifact": self.evidence_artifact,
                "evidence_sha256": self.evidence_sha256, "dataset_doi": self.dataset_doi,
                "phases": sorted(PHASES), "simulation_transfer_warning": WARNING}


class TransientPowerEstimator:
    def __init__(self, fit: TransientPowerFit, maximum_gap_s: float):
        self.fit = fit
        self.maximum_gap_s = maximum_gap_s
        self.reset()

    def reset(self) -> None:
        self._previous_time: float | None = None
        self._previous_speed: float | None = None
        self._speed_rate: float = 0.0

    def estimate(self, *, phase: str, timestamp_s: float | None,
                 velocity_enu_mps: tuple[float, float, float],
                 payload_kg: float, density_kg_m3: float) -> PowerEstimate | None:
        if timestamp_s is None or not math.isfinite(timestamp_s):
            self.reset()
            return None
        if not all(math.isfinite(value) for value in velocity_enu_mps):
            self.reset()
            return None
        speed = math.hypot(*velocity_enu_mps[:2])
        if self._previous_time != timestamp_s:
            delta = timestamp_s - self._previous_time if self._previous_time is not None else None
            self._speed_rate = (
                (speed - self._previous_speed) / delta
                if delta is not None and 0.0 < delta <= self.maximum_gap_s else 0.0
            )
            self._previous_time = timestamp_s
            self._previous_speed = speed
        if phase not in PHASES:
            return None
        power = self.fit.power_w(
            payload_kg=payload_kg, density_kg_m3=density_kg_m3,
            horizontal_speed_mps=speed, vertical_speed_up_mps=velocity_enu_mps[2],
            horizontal_speed_rate_mps2=self._speed_rate,
        )
        if not math.isfinite(power) or power <= 0.0:
            return None
        return PowerEstimate(phase=phase, coefficient_phase=None, power_w=power,
                             status=STATUS, warnings=(WARNING,), model_id=MODEL_ID)
