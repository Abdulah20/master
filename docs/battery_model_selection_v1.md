# Battery Model Selection v1

Date: 2026-08-17

## Decision

There is no maintained, drop-in Gazebo Harmonic plugin that turns the PX4 X500
motor simulation into validated electrical current automatically. The selected
research architecture therefore combines two maintained, permissively licensed
components:

1. the official Gazebo Sim 8 `LinearBatteryPlugin` for real-time Gazebo battery
   state and `BatteryPowerLoad` integration;
2. NREL `thevenin` v0.2.1 for a higher-fidelity equivalent-circuit battery model
   and offline/stepwise validation.

A thesis-owned `MotorBatteryLoadSystem` is still required between the X500
motors and the battery. It will calculate a dynamic load from actual rotor
motion instead of supplying a constant number of watts.

This is a model architecture, not ground truth. Before calibration its fidelity
is class C. It may reach class B after target-platform parameter identification
and independent validation. It cannot be class A without matched physical X500
or propulsion-bench measurements.

## Installed and downloaded components

### Gazebo Sim battery

Gazebo Harmonic is already installed. The effective PX4/ROS environment reports
Gazebo Sim 8.11.0 and provides:

```text
/opt/ros/jazzy/opt/gz_sim_vendor/lib/
  libgz-sim8-linearbatteryplugin-system.so.8.11.0
```

A separate system installation contains 8.14.0. Any thesis plugin must build
against the effective 8.11.0 ROS vendor ABI and must not mix the two versions.

Relevant local interfaces include:

```text
/usr/include/gz/sim8/gz/sim/components/BatteryPowerLoad.hh
/usr/include/gz/sim8/gz/sim/components/BatterySoC.hh
```

The official system supports capacity, charge, open-circuit-voltage terms,
internal resistance, current smoothing, recharge control, and a battery-state
topic. It sums `BatteryPowerLoad` consumers every simulation update.

Primary sources:

- <https://gazebosim.org/api/sim/8/battery.html>
- <https://raw.githubusercontent.com/gazebosim/gz-sim/gz-sim8/src/systems/battery_plugin/LinearBatteryPlugin.cc>
- <https://raw.githubusercontent.com/gazebosim/gz-sim/gz-sim8/LICENSE>

### NREL Thevenin model

The source was downloaded to:

```text
/home/abdulah/uav-energy-thesis/external/third_party/thevenin
```

Pinned provenance:

```text
release: v0.2.1
commit: bc44f88e12a37c3ca6346a4735c6ba9e8ecc4573
license: BSD-3-Clause
DOI: 10.11578/dc.20241125.2
```

It was installed without `sudo` in the isolated environment:

```text
/home/abdulah/uav-energy-thesis/.venv-thevenin
```

Installed versions include `thevenin 0.2.1`, `numpy 2.5.2`, and `scipy 1.18.0`.
All 44 upstream tests pass when unrelated globally registered ROS pytest plugins
are disabled. A constant-current functional simulation also completed and
showed decreasing voltage and state of charge.

Primary sources:

- <https://github.com/NREL/thevenin>
- <https://thevenin.readthedocs.io/stable/>
- <https://doi.org/10.11578/dc.20241125.2>

The package models open-circuit voltage versus SOC, series resistance, any
number of RC polarization pairs, Coulomb counting, hysteresis, and lumped
thermal behavior. It supports power-controlled experiments. Its bundled default
parameters are explicitly tutorial values and must never be labelled as X500 or
Matrice 100 battery parameters.

## Required motor-to-battery coupling

The X500 Gazebo motor system already defines:

```text
motorConstant = 8.54858e-06
momentConstant = 0.016
maxRotVelocity = 1000 rad/s
rotorVelocitySlowdownSim = 10
```

For an initial auditable load model:

\[
T_i = k_m \omega_i^2
\]

\[
Q_i = k_q T_i
\]

\[
P_{shaft} = \sum_i |Q_i\omega_i|
\]

\[
P_{load} = P_{aux} + \frac{P_{shaft}}{\eta_{motor}\eta_{ESC}}
\]

The adapter will read actual Gazebo joint velocity, correct the simulation rotor
slowdown, and update a `BatteryPowerLoad` component every simulation step. This
makes payload, manoeuvres, and rotor demand affect current and SOC. Command echo
or `esc_status.esc_rpm` must not be substituted for independent rotor truth.

The initial cubic power relation follows directly from the X500 motor model's
quadratic thrust and reaction-torque relationship. It does not identify motor
or ESC losses. Efficiency and auxiliary load remain calibration parameters.

## Expected outputs

The combined model should expose, at simulation time:

- actual rotor angular velocity for each motor;
- estimated shaft power for each motor;
- auxiliary and total electrical load;
- battery voltage and current;
- charge and SOC;
- accumulated DC energy;
- calibration/fidelity label and parameter-set identifier.

Gazebo publishes battery state at:

```text
/model/<model>/battery/<battery_name>/state
```

It can be bridged to ROS 2 `sensor_msgs/msg/BatteryState` and synchronized into
the research CSV. It does not enter PX4 `battery_status`, QGroundControl, or ULog
automatically; that adapter is a separate optional phase.

## Alternatives rejected

| Candidate | Decision | Reason |
|---|---|---|
| Gazebo battery with constant `power_load` | Rejected as final model | Time/load decoupling remains |
| RotorS or Gazebo Classic battery plugins | Rejected | Old simulator generation and no maintained Harmonic/X500 coupling |
| `brass_gazebo_battery` | Rejected | Tested against ROS Kinetic/Gazebo 7 and accepts external/static load |
| ArduPilot `SIM_Battery` | Reference only | GPL-3 and simplified caller-side propulsion load |
| MathWorks eVTOL Simscape | Rejected | Requires proprietary MATLAB/Simscape toolboxes and weak PX4/Gazebo integration |
| Unlicensed GitHub motor/battery repositories | Rejected | Cannot be safely reused or reproduced in the thesis |

## Calibration limits

The M100 flight data can test processing and gross electrical behavior but
cannot uniquely identify X500 OCV, series resistance, RC time constants, motor
efficiency, or ESC efficiency. A battery pulse/rest or HPPC-style dataset and a
matched motor-propeller bench map are required for defensible target parameters.

Until those measurements exist:

- Gazebo constant-load battery: C/D;
- dynamic motor load plus uncalibrated battery: C;
- calibrated and independently validated combined model: potentially B;
- X500 electrical ground truth: not available.

## Implementation status and next step

`MotorBatteryLoadSystem` has now been built outside PX4 against the effective
Gazebo Sim 8.11.0 environment and tested in an external `x500_energy` model.
Changing actual rotor speed changed shaft power, electrical load, battery
current, voltage, charge, and SOC while PX4 remained external and unchanged.
See `docs/x500_energy_plugin_v1.md` and
`results/logs/x500_energy_smoke_v1.json`.

The next step is parameter identification and independent validation: replace
the placeholder battery coefficients, propulsion efficiency, and auxiliary
load with a named target parameter set and matched measurements.  Until that is
done the implementation remains fidelity class C, not electrical ground truth.
