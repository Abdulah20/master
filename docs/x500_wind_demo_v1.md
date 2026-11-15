# X500 Visible Wind Demonstration v1

Date: 2026-08-20

## Outcome

A short automated X500 simulation completed with a configured Gazebo wind
vector of `[3, 0, 0] m/s` in the ENU frame (toward East). PX4 armed, took off,
hovered, translated 2 m north, landed, and disarmed without a fatal error.
Gazebo's runtime `wind_info` service reported `enable_wind: true` and an East
component of 3 m/s.

The Gazebo view contains three cyan direction arrows and an orange frame at the
3 m relative target altitude. The arrows and frame are visual references with
no collisions or applied forces. The physical simulated response comes from
Gazebo's `WindEffects` system and `WindMode` on the X500 body and rotor links.

This remains a simulation with model-estimated electrical energy classified as
`C_NOT_CALIBRATED`; it is not an X500 measurement or calibrated wind-tunnel
result.

## Comparison contract

This recorded run is a wind-path demonstration, not a single-factor wind
experiment relative to `x500_first_flight_v1_baseline`. In addition to changing
the ENU wind vector from `[0, 0, 0] m/s` to `[3, 0, 0] m/s`, it uses a 4 s
initial hover while the baseline uses 5 s. The remaining declared mission,
payload, logging, fidelity, and provenance inputs match.

Accordingly, baseline values below are descriptive, phase-normalized context.
They demonstrate that the configured wind path is active, but they do not
identify a wind-only causal energy effect.

## Run evidence

- Configuration: `config/x500_wind_demo_v1.yaml`.
- Generated world: `build/scenarios/x500_wind_demo_v1_3ms_east/worlds/x500_energy_first_flight.sdf`.
- Unified CSV: `results/runs/x500_wind_demo_v1_3ms_east/x500_wind_demo_v1.csv`.
- Validation: `results/runs/x500_wind_demo_v1_3ms_east/validation.json`.
- Provenance and versions: `results/runs/x500_wind_demo_v1_3ms_east/run_manifest.json`.
- Supplemental immutable-artifact hashes:
  `results/runs/x500_wind_demo_v1_3ms_east/integrity_manifest.json`.
- Mission phases: `results/runs/x500_wind_demo_v1_3ms_east/phase_markers.json`.
- Raw Gazebo energy stream: `results/runs/x500_wind_demo_v1_3ms_east/gazebo_energy.jsonl`.
- Gazebo view capture: `results/runs/x500_wind_demo_v1_3ms_east/gazebo_wind_demo.png`.

The supplemental manifest hashes both PNG captures as visual evidence. It also
records that the project has an empty `.git` directory, so no source-tree
revision can honestly be recovered or assigned to this legacy run.

The CSV passed all registered acceptance checks: 554 unique rows, 52 columns,
strictly increasing time, finite numeric values, nondecreasing cumulative
energy, expected power rise in flight, power drop after landing, no ULog
dropouts, and preserved fidelity classification.

The run manifest retains the byte hash of the configuration used in August
2026. The current source configuration later gained documentation comments and
therefore has a different byte hash, while its parsed operating inputs remain
the same. The historical hash is not rewritten or relabelled as the current
file.

## Observed response

At the hover marker, the vehicle had moved approximately 1.18 m East before
PX4 compensated; the zero-wind baseline hover marker was approximately 0.01 m
East. Across hover and post-move-hover samples, the mean horizontal tilt was
approximately 19.49 degrees with wind versus 0.23 degrees in the baseline.
Mean model-estimated electrical load over those phases was 392.31 W versus
366.83 W, and mean simulated battery current was 16.17 A versus 15.07 A.

These differences demonstrate that the configured wind path is active and that
PX4 responds to it. They are not aerodynamic calibration results. The demo also
uses a 4 s initial hover instead of the baseline's 5 s, so phase-normalized
means are shown under the non-isolated comparison contract above.

## Verification

The registered project Python suite and the C++ power-model test pass. Gazebo
8.11.0 loaded the world, the explicit
`gz::sim::systems::WindEffects` plugin, the X500 model, the energy plugin, and
the visual marker entities without a fatal error.
