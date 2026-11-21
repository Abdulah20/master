"""Read-only propulsion inventory for the active M100 CMU V2 airframe.

The values exposed here are used only for simulator configuration checks,
rotor-RPM decoding, and the explicitly labelled static-thrust reserve proxy.
They are not used to predict electrical power; operational power is provided
by the independent whole-aircraft M100 phase model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


DEFAULT_INVENTORY = Path(
    "/home/abdulah/PX4-Autopilot/Tools/simulation/gz/models/"
    "m100_cmu_v2/m100_physical_parameters.yaml"
)


def _entry(section: dict[str, Any], name: str) -> dict[str, Any]:
    value = section.get(name)
    if not isinstance(value, dict) or "value" not in value:
        raise ValueError(f"missing inventory entry {name}")
    return value


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True)
class M100PropulsionV2:
    inventory_path: Path
    max_rot_velocity_rad_s: float
    thrust_constant_n_per_rad_s2: float
    yaw_moment_ratio_m: float
    rotor_velocity_slowdown_sim: float
    rotor_drag_coefficient: float
    rolling_moment_coefficient: float
    body_drag_enabled: bool
    axial_inflow_enabled: bool

    @classmethod
    def load(cls, path: str | Path = DEFAULT_INVENTORY) -> "M100PropulsionV2":
        inventory_path = Path(path).resolve()
        data = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
        if data.get("schema") != "M100_CMU_PHYSICAL_PARAMETERS_V2":
            raise ValueError("unexpected M100 propulsion inventory schema")
        propulsion = data.get("propulsion")
        rotor = data.get("rotor_aerodynamics")
        motor = data.get("motor_dynamics")
        body = data.get("body_aerodynamics")
        if not all(isinstance(item, dict) for item in (propulsion, rotor, motor, body)):
            raise ValueError("M100 propulsion inventory sections are incomplete")
        result = cls(
            inventory_path=inventory_path,
            max_rot_velocity_rad_s=_positive(
                _entry(propulsion, "max_rot_velocity")["value"],
                "max_rot_velocity",
            ),
            thrust_constant_n_per_rad_s2=_positive(
                _entry(propulsion, "motor_constant")["value"],
                "motor_constant",
            ),
            yaw_moment_ratio_m=_positive(
                _entry(propulsion, "moment_constant")["value"],
                "moment_constant",
            ),
            rotor_velocity_slowdown_sim=_positive(
                _entry(motor, "rotor_velocity_slowdown_sim")["value"],
                "rotor_velocity_slowdown_sim",
            ),
            rotor_drag_coefficient=float(
                _entry(rotor, "rotor_drag_coefficient")["value"]
            ),
            rolling_moment_coefficient=float(
                _entry(rotor, "rolling_moment_coefficient")["value"]
            ),
            body_drag_enabled=bool(_entry(body, "body_drag_enabled")["value"]),
            axial_inflow_enabled=bool(_entry(rotor, "axial_inflow_enabled")["value"]),
        )
        if result.rotor_drag_coefficient != 0.0:
            raise ValueError("unvalidated rotor drag must remain disabled")
        if result.rolling_moment_coefficient != 0.0:
            raise ValueError("unvalidated rolling moment must remain disabled")
        if result.body_drag_enabled:
            raise ValueError("unvalidated body drag must remain disabled")
        if result.axial_inflow_enabled:
            raise ValueError("unvalidated axial inflow must remain disabled")
        return result
