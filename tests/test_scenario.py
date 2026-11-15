from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

from simulation.scenario import (
    ScenarioConfigError,
    build_gazebo_scenario,
    load_scenario_config,
    scenario_csv_constant_columns,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "x500_first_flight_v1.yaml"


def test_baseline_inputs_are_explicit_and_safe() -> None:
    inputs = load_scenario_config(CONFIG)
    assert inputs.payload.mass_kg == 0.0
    assert inputs.wind.velocity_enu_m_s == (0.0, 0.0, 0.0)
    assert inputs.mission.target_relative_altitude_m == 3.0
    assert inputs.fidelity == "C_NOT_CALIBRATED"
    assert inputs.logging.required_attitude_columns[-3:] == (
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
    )
    csv_inputs = scenario_csv_constant_columns(inputs)
    assert csv_inputs["payload_mass_kg"] == 0.0
    assert csv_inputs["target_relative_altitude_m"] == 3.0
    assert csv_inputs["energy_fidelity"] == "C_NOT_CALIBRATED"


def test_baseline_scenario_generation_omits_zero_mass_payload(tmp_path: Path) -> None:
    inputs = load_scenario_config(CONFIG)
    manifest = build_gazebo_scenario(inputs, tmp_path, config_path=CONFIG)
    model_path = tmp_path / "models" / "x500_energy_scenario" / "model.sdf"
    world_path = tmp_path / "worlds" / "x500_energy_first_flight.sdf"
    model_root = ET.parse(model_path).getroot()
    world_root = ET.parse(world_path).getroot()
    assert model_root.find(".//link[@name='payload_link']") is None
    assert world_root.findtext(".//wind/linear_velocity") == "0 0 0"
    assert manifest["total_vehicle_mass_kg"] == pytest.approx(2.0643076923076924)
    assert len(manifest["config_sha256"]) == 64


def test_payload_and_wind_are_written_to_gazebo_sdf(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["scenario"]["run_id"] = "payload_wind_case"
    raw["operating_conditions"]["payload"]["mass_kg"] = 0.25
    raw["operating_conditions"]["wind"]["velocity_enu_m_s"] = [5.0, 2.0, 0.0]
    changed = tmp_path / "case.yaml"
    changed.write_text(yaml.safe_dump(raw), encoding="utf-8")

    inputs = load_scenario_config(changed)
    output = tmp_path / "generated"
    build_gazebo_scenario(inputs, output, config_path=changed)
    model_root = ET.parse(output / "models" / "x500_energy_scenario" / "model.sdf").getroot()
    world_root = ET.parse(output / "worlds" / "x500_energy_first_flight.sdf").getroot()
    assert float(model_root.findtext(".//link[@name='payload_link']/inertial/mass")) == 0.25
    assert model_root.find(".//joint[@name='payload_fixed_joint']") is not None
    assert world_root.findtext(".//wind/linear_velocity") == "5 2 0"
    assert world_root.find(".//plugin[@name='gz::sim::systems::WindEffects']") is not None
    assert world_root.find(".//model[@name='wind_vector_indicator']") is not None
    assert world_root.find(".//model[@name='target_altitude_marker']") is not None
    assert len(world_root.findall(".//model[@name='wind_vector_indicator']//cone")) == 3
    manifest = json.loads((output / "scenario_manifest.json").read_text(encoding="utf-8"))
    assert manifest["wind"]["speed_m_s"] == pytest.approx(5.3851648071)
    assert manifest["total_vehicle_mass_kg"] == pytest.approx(2.3143076923)


def test_zero_wind_omits_only_the_wind_indicator(tmp_path: Path) -> None:
    inputs = load_scenario_config(CONFIG)
    build_gazebo_scenario(inputs, tmp_path)
    root = ET.parse(tmp_path / "worlds" / "x500_energy_first_flight.sdf").getroot()
    assert root.find(".//model[@name='wind_vector_indicator']") is None
    assert root.find(".//model[@name='target_altitude_marker']") is not None
    assert root.find(".//plugin[@name='gz::sim::systems::WindEffects']") is not None


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("operating_conditions", "payload", "mass_kg"), -0.1, "non-negative"),
        (("mission", "target_relative_altitude_m"), 0.0, "positive"),
        (("operating_conditions", "wind", "velocity_enu_m_s"), [1.0, 2.0], "three"),
    ],
)
def test_invalid_operating_inputs_are_rejected(
    tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    changed = tmp_path / "invalid.yaml"
    changed.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ScenarioConfigError, match=message):
        load_scenario_config(changed)
