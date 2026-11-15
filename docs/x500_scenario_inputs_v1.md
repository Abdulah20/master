# X500 Scenario Inputs v1

Date: 2026-08-20

## Scope

The first-flight configuration now records payload, wind, and target altitude
as explicit, unit-labelled inputs. The baseline remains zero payload, zero
wind, and 3 m relative target altitude. This does not start a payload or wind
experiment before the first unified CSV succeeds.

Configuration:

```text
config/x500_first_flight_v1.yaml
```

## Input semantics

- `payload.mass_kg` is added as a fixed box-shaped link in a generated Gazebo
  model. Its dimensions and body-frame centre position are also explicit.
- `wind.velocity_enu_m_s` is the Gazebo world wind vector ordered East, North,
  Up. This avoids ambiguity between meteorological "from" direction and a
  force-vector direction.
- `mission.target_relative_altitude_m` is positive above the takeoff point. It
  is consumed by the forthcoming PX4 mission runner rather than by the Gazebo
  world generator.

Generated files are kept below a caller-selected directory inside the thesis
project. The generator does not modify PX4:

```bash
PYTHONPATH=src python3 -m simulation.scenario \
  --config config/x500_first_flight_v1.yaml \
  --output build/scenarios/x500_first_flight_v1_baseline
```

The generated manifest records the nominal and total mass, wind vector and
magnitude, mission inputs, logging convention, source paths, configuration
SHA-256, and the SHA-256 of all three generated assets (`model.config`,
`model.sdf`, and the world SDF). A unified-flight manifest must consume this
scenario manifest and verify those hashes before it records a run.

The unified CSV builder uses `scenario_csv_constant_columns` to put
`payload_mass_kg`, `total_mass_kg`, target altitude, all three ENU wind
components, wind magnitude, mission name, run ID, and energy fidelity on every
row. Measured altitude remains a separate output column.

## Attitude convention

PX4 `vehicle_attitude.q[0..3]` is retained in the unified CSV as `q_w`, `q_x`,
`q_y`, and `q_z`. It is also converted to aerospace roll, pitch, and yaw using
`src/data_processing/attitude.py`. Euler outputs are explicitly named
`roll_rad`, `pitch_rad`, and `yaw_rad`; the quaternion is normalized before
conversion and invalid zero or non-finite quaternions are rejected.

## Scientific limits

Wind and payload are simulation inputs, not measurements. Gazebo's wind-effect
system is a model and the present propulsion and electrical parameters remain
`C_NOT_CALIBRATED`. The first nonzero-payload or nonzero-wind experiment must be
deferred until the baseline automated flight produces an accepted unified CSV.

## Gazebo visual references

Generated worlds now load Gazebo's `WindEffects` system explicitly. When the
configured wind speed is nonzero, three cyan arrows display the ENU vector;
these arrows are visual references only and apply no force. An orange frame
marks the commanded relative-altitude level. The actual wind force remains the
Gazebo `WindEffects` model acting on links whose `WindMode` is enabled.

The first visible wind demonstration is configured in
`config/x500_wind_demo_v1.yaml` with a 3 m/s vector toward East. It does not add
payload mass and retains the `C_NOT_CALIBRATED` energy label. It is not a
single-factor comparison with the baseline because its initial hover is 4 s
rather than 5 s; its baseline comparison is descriptive, not a wind-only causal
estimate.
