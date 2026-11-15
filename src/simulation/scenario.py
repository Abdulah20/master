"""Validate X500 operating inputs and generate project-owned Gazebo SDF files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


FIDELITY = "C_NOT_CALIBRATED"
NOMINAL_X500_MASS_KG = 2.0643076923076924
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ScenarioConfigError(ValueError):
    """Raised when a scenario input is ambiguous or physically invalid."""


@dataclass(frozen=True)
class MissionInputs:
    target_relative_altitude_m: float
    hover_duration_s: float
    translation_north_m: float
    translation_east_m: float
    translation_speed_m_s: float


@dataclass(frozen=True)
class PayloadInputs:
    mass_kg: float
    dimensions_m: tuple[float, float, float]
    position_body_m: tuple[float, float, float]


@dataclass(frozen=True)
class WindInputs:
    velocity_enu_m_s: tuple[float, float, float]

    @property
    def speed_m_s(self) -> float:
        return math.sqrt(sum(component * component for component in self.velocity_enu_m_s))


@dataclass(frozen=True)
class LoggingInputs:
    sample_rate_hz: float
    quaternion_order: str
    euler_sequence: str
    angle_unit: str
    required_attitude_columns: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioInputs:
    mission_name: str
    run_id: str
    fidelity: str
    mission: MissionInputs
    payload: PayloadInputs
    wind: WindInputs
    logging: LoggingInputs
    provenance: dict[str, str]

    @property
    def total_vehicle_mass_kg(self) -> float:
        return NOMINAL_X500_MASS_KG + self.payload.mass_kg


def scenario_csv_constant_columns(inputs: ScenarioInputs) -> dict[str, float | str]:
    """Return operating inputs that must be repeated on every unified CSV row."""

    east, north, up = inputs.wind.velocity_enu_m_s
    return {
        "mission_name": inputs.mission_name,
        "run_id": inputs.run_id,
        "energy_fidelity": inputs.fidelity,
        "payload_mass_kg": inputs.payload.mass_kg,
        "total_mass_kg": inputs.total_vehicle_mass_kg,
        "target_relative_altitude_m": inputs.mission.target_relative_altitude_m,
        "wind_east_m_s": east,
        "wind_north_m_s": north,
        "wind_up_m_s": up,
        "wind_speed_m_s": inputs.wind.speed_m_s,
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScenarioConfigError(f"{label} must be a mapping")
    return value


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScenarioConfigError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ScenarioConfigError(f"{label} must be finite")
    return result


def _vector3(value: Any, label: str, *, positive: bool = False) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ScenarioConfigError(f"{label} must contain exactly three values")
    result = tuple(_finite_float(component, label) for component in value)
    if positive and any(component <= 0.0 for component in result):
        raise ScenarioConfigError(f"{label} values must be positive")
    return result


def load_scenario_config(config_path: Path | str) -> ScenarioInputs:
    """Load and strictly validate the first-flight operating inputs."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        raw = _mapping(yaml.safe_load(stream), "configuration")

    if int(raw.get("schema_version", -1)) != 1:
        raise ScenarioConfigError("Only scenario schema_version 1 is supported")

    scenario = _mapping(raw.get("scenario"), "scenario")
    mission_raw = _mapping(raw.get("mission"), "mission")
    conditions = _mapping(raw.get("operating_conditions"), "operating_conditions")
    payload_raw = _mapping(conditions.get("payload"), "operating_conditions.payload")
    wind_raw = _mapping(conditions.get("wind"), "operating_conditions.wind")
    logging_raw = _mapping(raw.get("logging"), "logging")
    provenance_raw = _mapping(raw.get("provenance"), "provenance")

    mission_name = str(scenario.get("mission_name", ""))
    run_id = str(scenario.get("run_id", ""))
    if not _NAME_PATTERN.fullmatch(mission_name) or not _NAME_PATTERN.fullmatch(run_id):
        raise ScenarioConfigError("mission_name and run_id must be filesystem-safe identifiers")
    fidelity = str(scenario.get("fidelity", ""))
    if fidelity != FIDELITY:
        raise ScenarioConfigError(f"X500 electrical fidelity must remain {FIDELITY}")

    mission = MissionInputs(
        target_relative_altitude_m=_finite_float(
            mission_raw.get("target_relative_altitude_m"), "target_relative_altitude_m"
        ),
        hover_duration_s=_finite_float(mission_raw.get("hover_duration_s"), "hover_duration_s"),
        translation_north_m=_finite_float(
            mission_raw.get("translation_north_m"), "translation_north_m"
        ),
        translation_east_m=_finite_float(
            mission_raw.get("translation_east_m"), "translation_east_m"
        ),
        translation_speed_m_s=_finite_float(
            mission_raw.get("translation_speed_m_s"), "translation_speed_m_s"
        ),
    )
    if mission.target_relative_altitude_m <= 0.0:
        raise ScenarioConfigError("target_relative_altitude_m must be positive")
    if mission.hover_duration_s < 0.0 or mission.translation_speed_m_s <= 0.0:
        raise ScenarioConfigError("Mission durations must be non-negative and speed must be positive")

    payload = PayloadInputs(
        mass_kg=_finite_float(payload_raw.get("mass_kg"), "payload.mass_kg"),
        dimensions_m=_vector3(payload_raw.get("dimensions_m"), "payload.dimensions_m", positive=True),
        position_body_m=_vector3(payload_raw.get("position_body_m"), "payload.position_body_m"),
    )
    if payload.mass_kg < 0.0:
        raise ScenarioConfigError("payload.mass_kg must be non-negative")

    wind = WindInputs(
        velocity_enu_m_s=_vector3(wind_raw.get("velocity_enu_m_s"), "wind.velocity_enu_m_s")
    )
    required_columns = tuple(str(value) for value in logging_raw.get("required_attitude_columns", []))
    expected_columns = ("q_w", "q_x", "q_y", "q_z", "roll_rad", "pitch_rad", "yaw_rad")
    logging = LoggingInputs(
        sample_rate_hz=_finite_float(logging_raw.get("sample_rate_hz"), "logging.sample_rate_hz"),
        quaternion_order=str(logging_raw.get("quaternion_order", "")),
        euler_sequence=str(logging_raw.get("euler_sequence", "")),
        angle_unit=str(logging_raw.get("angle_unit", "")),
        required_attitude_columns=required_columns,
    )
    if logging.sample_rate_hz <= 0.0:
        raise ScenarioConfigError("logging.sample_rate_hz must be positive")
    if (logging.quaternion_order, logging.euler_sequence, logging.angle_unit) != (
        "wxyz",
        "roll_pitch_yaw",
        "rad",
    ):
        raise ScenarioConfigError("Attitude convention must be wxyz, roll_pitch_yaw, radians")
    if required_columns != expected_columns:
        raise ScenarioConfigError("The required quaternion and Euler attitude columns may not be omitted")

    provenance = {str(key): str(value) for key, value in provenance_raw.items()}
    return ScenarioInputs(
        mission_name=mission_name,
        run_id=run_id,
        fidelity=fidelity,
        mission=mission,
        payload=payload,
        wind=wind,
        logging=logging,
        provenance=provenance,
    )


def _box_inertia(payload: PayloadInputs) -> tuple[float, float, float]:
    x, y, z = payload.dimensions_m
    mass = payload.mass_kg
    return (
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    )


def _payload_sdf(payload: PayloadInputs) -> str:
    if payload.mass_kg == 0.0:
        return ""
    size = " ".join(f"{value:.12g}" for value in payload.dimensions_m)
    pose = " ".join(f"{value:.12g}" for value in payload.position_body_m)
    ixx, iyy, izz = _box_inertia(payload)
    return f"""
    <link name="payload_link">
      <pose relative_to="base_link">{pose} 0 0 0</pose>
      <inertial>
        <mass>{payload.mass_kg:.12g}</mass>
        <inertia>
          <ixx>{ixx:.12g}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{iyy:.12g}</iyy><iyz>0</iyz><izz>{izz:.12g}</izz>
        </inertia>
      </inertial>
      <collision name="payload_collision">
        <geometry><box><size>{size}</size></box></geometry>
      </collision>
      <visual name="payload_visual">
        <geometry><box><size>{size}</size></box></geometry>
        <material><diffuse>0.15 0.15 0.15 1</diffuse></material>
      </visual>
      <gravity>true</gravity>
      <enable_wind>true</enable_wind>
    </link>
    <joint name="payload_fixed_joint" type="fixed">
      <parent>base_link</parent>
      <child>payload_link</child>
    </joint>"""


def _model_sdf(inputs: ScenarioInputs) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sdf version="1.9">
  <model name="x500_energy_scenario">
    <include merge="true">
      <uri>model://x500_energy</uri>
    </include>{_payload_sdf(inputs.payload)}
  </model>
</sdf>
"""


def _pose_quaternion_from_z_axis(direction: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Return an xyzw quaternion that rotates the local +Z axis to direction."""

    x, y, z = direction
    if z < -1.0 + 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    cross_x = -y
    cross_y = x
    scalar = 1.0 + z
    norm = math.sqrt(cross_x * cross_x + cross_y * cross_y + scalar * scalar)
    return (cross_x / norm, cross_y / norm, 0.0, scalar / norm)


def _wind_indicator_sdf(inputs: ScenarioInputs) -> str:
    """Create non-physical arrows that display the configured ENU wind vector."""

    speed = inputs.wind.speed_m_s
    if speed <= 0.0:
        return ""
    direction = tuple(component / speed for component in inputs.wind.velocity_enu_m_s)
    qx, qy, qz, qw = _pose_quaternion_from_z_axis(direction)
    shaft_length = min(2.4, 0.9 + 0.35 * speed)
    head_length = 0.38
    anchors = ((-2.8, -1.4, 0.8), (-2.8, 0.8, 2.0), (-2.8, 3.0, 3.2))

    visuals: list[str] = []
    for index, anchor in enumerate(anchors):
        shaft_center = tuple(anchor[i] + 0.5 * shaft_length * direction[i] for i in range(3))
        head_center = tuple(
            anchor[i] + (shaft_length + 0.5 * head_length) * direction[i] for i in range(3)
        )
        visuals.append(
            f"""
      <visual name="wind_arrow_{index}_shaft">
        <pose rotation_format="quat_xyzw">{' '.join(f'{v:.12g}' for v in shaft_center)} {qx:.12g} {qy:.12g} {qz:.12g} {qw:.12g}</pose>
        <geometry><cylinder><radius>0.055</radius><length>{shaft_length:.12g}</length></cylinder></geometry>
        <material>
          <ambient>0.02 0.65 1 1</ambient><diffuse>0.02 0.65 1 1</diffuse>
          <emissive>0.01 0.20 0.35 1</emissive>
        </material>
      </visual>
      <visual name="wind_arrow_{index}_head">
        <pose rotation_format="quat_xyzw">{' '.join(f'{v:.12g}' for v in head_center)} {qx:.12g} {qy:.12g} {qz:.12g} {qw:.12g}</pose>
        <geometry><cone><radius>0.18</radius><length>{head_length:.12g}</length></cone></geometry>
        <material>
          <ambient>0.02 0.65 1 1</ambient><diffuse>0.02 0.65 1 1</diffuse>
          <emissive>0.01 0.20 0.35 1</emissive>
        </material>
      </visual>"""
        )
    return f"""
    <!-- Cyan arrows display the configured wind vector only; they apply no force. -->
    <model name="wind_vector_indicator">
      <static>true</static>
      <link name="indicator_link">{''.join(visuals)}
      </link>
    </model>"""


def _target_altitude_marker_sdf(inputs: ScenarioInputs) -> str:
    """Create an orange reference frame at the commanded relative altitude."""

    target_z = 0.25 + inputs.mission.target_relative_altitude_m
    return f"""
    <!-- Orange frame marks the mission target altitude; it has no collision. -->
    <model name="target_altitude_marker">
      <static>true</static>
      <link name="marker_link">
        <visual name="south"><pose>0 -1 {target_z:.12g} 0 0 0</pose><geometry><box><size>4 0.035 0.035</size></box></geometry><material><ambient>1 0.35 0.02 1</ambient><diffuse>1 0.35 0.02 1</diffuse><emissive>0.35 0.08 0 1</emissive></material></visual>
        <visual name="north"><pose>0 3 {target_z:.12g} 0 0 0</pose><geometry><box><size>4 0.035 0.035</size></box></geometry><material><ambient>1 0.35 0.02 1</ambient><diffuse>1 0.35 0.02 1</diffuse><emissive>0.35 0.08 0 1</emissive></material></visual>
        <visual name="west"><pose>-2 1 {target_z:.12g} 0 0 0</pose><geometry><box><size>0.035 4 0.035</size></box></geometry><material><ambient>1 0.35 0.02 1</ambient><diffuse>1 0.35 0.02 1</diffuse><emissive>0.35 0.08 0 1</emissive></material></visual>
        <visual name="east"><pose>2 1 {target_z:.12g} 0 0 0</pose><geometry><box><size>0.035 4 0.035</size></box></geometry><material><ambient>1 0.35 0.02 1</ambient><diffuse>1 0.35 0.02 1</diffuse><emissive>0.35 0.08 0 1</emissive></material></visual>
        <visual name="height_reference"><pose>-2 -1 {0.5 * target_z:.12g} 0 0 0</pose><geometry><box><size>0.035 0.035 {target_z:.12g}</size></box></geometry><material><ambient>1 0.35 0.02 1</ambient><diffuse>1 0.35 0.02 1</diffuse><emissive>0.35 0.08 0 1</emissive></material></visual>
      </link>
    </model>"""


def _world_sdf(inputs: ScenarioInputs) -> str:
    east, north, up = inputs.wind.velocity_enu_m_s
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sdf version="1.9">
  <world name="x500_energy_first_flight">
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-air-speed-system" name="gz::sim::systems::AirSpeed"/>
    <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>
    <plugin filename="gz-sim-wind-effects-system" name="gz::sim::systems::WindEffects"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <wind><linear_velocity>{east:.12g} {north:.12g} {up:.12g}</linear_velocity></wind>
    <scene><grid>true</grid><ambient>0.45 0.45 0.45 1</ambient><background>0.72 0.80 0.90 1</background><shadows>true</shadows></scene>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>500 500</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>500 500</size></plane></geometry>
        </visual>
        <enable_wind>false</enable_wind>
      </link>
    </model>
    <light name="sun" type="directional">
      <pose>0 0 500 0 0 0</pose><cast_shadows>true</cast_shadows>
      <direction>0.001 0.625 -0.78</direction>
    </light>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg>
      <longitude_deg>8.546163739800146</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>{_wind_indicator_sdf(inputs)}{_target_altitude_marker_sdf(inputs)}
    <include>
      <uri>model://x500_energy_scenario</uri>
      <name>x500_energy</name>
      <pose>0 0 0.25 0 0 0</pose>
    </include>
  </world>
</sdf>
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_gazebo_scenario(
    inputs: ScenarioInputs,
    output_directory: Path | str,
    *,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Generate a scenario wrapper without modifying PX4 or the source model."""

    output = Path(output_directory)
    model_directory = output / "models" / "x500_energy_scenario"
    world_directory = output / "worlds"
    model_directory.mkdir(parents=True, exist_ok=True)
    world_directory.mkdir(parents=True, exist_ok=True)

    model_config = model_directory / "model.config"
    model_sdf = model_directory / "model.sdf"
    world_sdf = world_directory / "x500_energy_first_flight.sdf"
    manifest_path = output / "scenario_manifest.json"

    model_config.write_text(
        """<?xml version="1.0"?>
<model>
  <name>x500_energy_scenario</name><version>0.1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>UAV Energy Thesis</name></author>
  <description>Generated X500 energy scenario with explicit payload inputs.</description>
</model>
""",
        encoding="utf-8",
    )
    model_sdf.write_text(_model_sdf(inputs), encoding="utf-8")
    world_sdf.write_text(_world_sdf(inputs), encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mission_name": inputs.mission_name,
        "run_id": inputs.run_id,
        "fidelity": inputs.fidelity,
        "nominal_x500_mass_kg": NOMINAL_X500_MASS_KG,
        "total_vehicle_mass_kg": inputs.total_vehicle_mass_kg,
        "mission": asdict(inputs.mission),
        "payload": asdict(inputs.payload),
        "wind": {**asdict(inputs.wind), "speed_m_s": inputs.wind.speed_m_s, "frame": "ENU"},
        "logging": asdict(inputs.logging),
        "provenance": inputs.provenance,
        "generated_files": {
            "model_config": str(model_config.resolve()),
            "model_sdf": str(model_sdf.resolve()),
            "world_sdf": str(world_sdf.resolve()),
        },
        "generated_file_sha256": {
            "model_config": _sha256(model_config),
            "model_sdf": _sha256(model_sdf),
            "world_sdf": _sha256(world_sdf),
        },
    }
    if config_path is not None:
        config = Path(config_path).resolve()
        manifest["config_path"] = str(config)
        manifest["config_sha256"] = _sha256(config)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = load_scenario_config(args.config)
    manifest = build_gazebo_scenario(inputs, args.output, config_path=args.config)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
