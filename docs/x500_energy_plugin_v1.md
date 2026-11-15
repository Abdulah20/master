# X500 Dynamic Battery Load Plugin v1

Date: 2026-08-17

## Outcome

The external `x500_energy` model now couples actual Gazebo rotor-joint motion to
the official Gazebo Sim 8 `LinearBatteryPlugin`.  PX4 source files are not
modified.

The implementation is deliberately labelled `C_NOT_CALIBRATED`.  It is a
dynamic, physically structured estimate, not electrical ground truth and not a
validated X500 battery model.

## Implemented path

The model reads the four actual rotor joint velocities, reverses the X500
simulation slowdown, and computes:

\[
P_{shaft}=\sum_i k_q k_m |\omega_i|^3
\]

\[
P_{electrical}=P_{aux}+P_{shaft}/\eta
\]

The initial, uncalibrated parameters are:

| Parameter | Value | Origin/status |
|---|---:|---|
| `motor_constant` | `8.54858e-06` | PX4 X500 motor model |
| `moment_constant` | `0.016` | PX4 X500 motor model |
| `rotor_velocity_slowdown` | `10` | PX4 X500 motor model |
| `propulsion_efficiency` | `0.70` | Placeholder; must be calibrated |
| `auxiliary_power_w` | `10` | Placeholder; must be calibrated |
| `maximum_power_w` | `2000` | Numerical safety bound, not a rating |

The resulting electrical load is written into a Gazebo
`BatteryPowerLoad` component on every simulation update.  The official battery
plugin therefore computes voltage, current, charge, and state of charge from a
load that changes with actual rotor motion.

## Relevant files

```text
/home/abdulah/uav-energy-thesis/src/gazebo_energy_plugin/
  CMakeLists.txt
  include/uav_energy/PowerModel.hpp
  include/uav_energy/MotorBatteryLoadSystem.hpp
  src/MotorBatteryLoadSystem.cpp
  tests/power_model_test.cpp

/home/abdulah/uav-energy-thesis/models/x500_energy/model.sdf
/home/abdulah/uav-energy-thesis/worlds/x500_energy_smoke.sdf
```

The plugin was built against the effective ROS Jazzy Gazebo Sim 8.11.0 vendor
libraries.  The standalone smoke world also loads PX4's existing
`MotorFailurePlugin` from the PX4 build tree; this avoids changing or copying
PX4 code.

## Runtime topics

| Quantity | Gazebo topic |
|---|---|
| Actual physical rotor speeds | `/model/x500_energy/energy/rotor_speed_rad_s` |
| Estimated total shaft power | `/model/x500_energy/energy/shaft_power_w` |
| Estimated electrical load | `/model/x500_energy/energy/electrical_load_w` |
| Cumulative estimated energy | `/model/x500_energy/energy/cumulative_energy_j` |
| Official battery state | `/model/x500_energy/battery/x500_energy_battery/state` |
| Motor command for standalone test | `/x500_energy/command/motor_speed` |

The battery topic uses `gz.msgs.BatteryState`.  The four diagnostic energy
topics use Gazebo scalar/vector messages.  They do not automatically enter PX4
ULog or QGroundControl.

## Smoke-test evidence

The server loaded Gazebo Sim 8.11.0, the official linear battery, all four X500
motor systems, PX4's motor-failure system, and `MotorBatteryLoadSystem`.  The
custom system reported that it connected to `x500_energy_battery`.

Measured steady snapshots were:

| Motor command (rad/s) | Actual joint speeds (rad/s) | Shaft power (W) | Electrical load (W) | Battery current (A) | Battery voltage (V) |
|---:|---|---:|---:|---:|---:|
| 0 | `0, 0, 0, 0` | 0 | 10.0000 | 0.3972 | 25.1789 |
| 200 | `200, 200, -200, -200` | 4.3769 | 16.2527 | 0.6458 | 25.1643 |
| 600 | `600, 600, -600, -600` | 118.1756 | 178.8222 | 7.2031 | 24.8212 |
| 0 after high load | returned toward zero | 0 | 10.0000 | 0.4398 | 25.1428 |

The opposite signs are the expected counter-rotating rotor directions; the
power model uses absolute angular speed.  The current filter has a one-second
time constant, so the post-load current does not jump immediately to its final
value.

This demonstrates the required causal chain:

```text
actual joint speed -> shaft-power estimate -> dynamic battery load
                   -> current and voltage response -> SOC / charge integration
```

The result log is stored at:

```text
/home/abdulah/uav-energy-thesis/results/logs/x500_energy_smoke_v1.json
```

## Verification

- C++ analytical power-model test: passed.
- Standalone Gazebo load-coupling smoke test: passed.
- No plugin-loading error occurred after the PX4 plugin directory was included.
- PX4 remains an external dependency and is not modified by this implementation.

Warnings about `gz_frame_id`, Ogre material scripts, missing `robotNamespace`,
and deprecated `motorNumber` come from the existing PX4 X500 SDF and do not
prevent the test from running.

## Research limits and next step

The current battery coefficients, 70% propulsion efficiency, and 10 W auxiliary
load are placeholders.  The current output is model-derived, not measured.
Therefore this remains fidelity class C.

The next investigation is to identify a target 6S battery parameter set
(OCV-SOC, resistance, capacity, and preferably pulse/rest dynamics) and a
motor-ESC-propeller power map.  Only after calibrating those values and
validating on held-out physical data may the combined model be considered for
class B.  Integrating the topics into the one-flight research CSV follows that
calibration/telemetry definition; it should not relabel the present estimate as
ground truth.

