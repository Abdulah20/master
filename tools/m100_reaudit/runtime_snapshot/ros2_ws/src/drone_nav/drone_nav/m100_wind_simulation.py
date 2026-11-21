"""Theoretical ambient-wind extension of the M100 simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import yaml

from .m100_transient_power import TransientPowerFit


MODEL_ID = "M100_WIND_SIMULATION_V1"
STATUS = "C_NOT_CALIBRATED"
WARNING = "THEORETICAL_COMMANDED_WIND_TRANSFER_NOT_EMPIRICALLY_CALIBRATED"


@dataclass(frozen=True)
class WindSimulation:
    drag_coefficient: float
    reference_area_m2: float
    metadata: Mapping[str, object]

    @classmethod
    def load(cls, path: Path) -> WindSimulation:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data["schema"] != MODEL_ID or data["classification"] != STATUS:
            raise ValueError("unsupported M100 wind simulation configuration")
        values = data["horizontal_body_drag"]
        coefficient, area = float(values["drag_coefficient"]), float(values["reference_area_m2"])
        if not all(math.isfinite(value) and value > 0.0 for value in (coefficient, area)):
            raise ValueError("invalid published M100 drag coefficient or area")
        return cls(coefficient, area, data)

    def horizontal_force_n(self, velocity_enu_mps, wind_enu_mps, density_kg_m3):
        relative = tuple(v - w for v, w in zip(velocity_enu_mps[:2], wind_enu_mps[:2], strict=True))
        factor = -0.5 * density_kg_m3 * self.drag_coefficient * self.reference_area_m2 * math.hypot(*relative)
        return (factor * relative[0], factor * relative[1], 0.0)

    @staticmethod
    def power_delta_w(fit: TransientPowerFit, *, payload_kg: float, density_kg_m3: float,
                      velocity_enu_mps, wind_enu_mps) -> float:
        ground_speed = math.hypot(*velocity_enu_mps[:2])
        relative_speed = math.hypot(velocity_enu_mps[0] - wind_enu_mps[0],
                                    velocity_enu_mps[1] - wind_enu_mps[1])
        inputs = {"payload_kg": payload_kg, "density_kg_m3": density_kg_m3,
                  "vertical_speed_up_mps": velocity_enu_mps[2], "horizontal_speed_rate_mps2": 0.0}
        return fit.power_w(horizontal_speed_mps=relative_speed, **inputs) - fit.power_w(
            horizontal_speed_mps=ground_speed, **inputs)
