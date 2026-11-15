# X500 First Automated Energy Flight v1

Date: 2026-08-20

## Outcome

The first short automated PX4/Gazebo flight using the external
`x500_energy` model completed without a fatal error. The vehicle armed, took
off, reached approximately 3 m, hovered, moved approximately 2 m north, held,
landed, and disarmed automatically.

The result is a simulation experiment with model-estimated electrical energy.
Its fidelity remains `C_NOT_CALIBRATED`; it is not physical X500 measurement.

## Scenario

- Payload: 0 kg.
- Wind ENU: `[0, 0, 0] m/s`.
- Relative target altitude: 3 m.
- North translation: 2 m.
- CSV sample rate: 20 Hz.
- Mission duration retained in CSV: 28.25 s.

The configuration is `config/x500_first_flight_v1.yaml`.

## Flight evidence

PX4 phase markers report:

- takeoff command at simulation time 219.172 s;
- hover phase at 226.644 s and approximately 2.75 m;
- movement phase at 231.644 s;
- landing phase at 237.308 s;
- landed phase at 243.408 s.

PX4 reported takeoff detection, landing detection, and automatic disarming by
landing. The ULog was closed normally with no dropout records.

## Unified CSV

The accepted CSV is:

```text
results/runs/x500_first_flight_v1_baseline/x500_first_flight_v1.csv
```

It contains 566 unique, strictly increasing rows and 52 columns. It includes
PX4/Gazebo position, velocity, acceleration, heading, quaternion, roll, pitch,
yaw, angular rate, motor commands, actual rotor speed, battery state, estimated
shaft power, estimated electrical load, cumulative energy, flight phase, and
scenario identifiers and operating inputs.

The raw ULog contained 118 duplicate `vehicle_attitude` timestamps. The merger
kept the last quaternion at each duplicate raw timestamp and documented the
count. The final 20 Hz CSV itself has no duplicated timestamp.

## Acceptance result

All registered checks passed:

- CSV non-empty;
- timestamps strictly increasing and unique;
- all numeric values finite;
- cumulative energy nondecreasing;
- target altitude reached and landing near ground;
- actual Gazebo rotors responded;
- power rose during takeoff and movement;
- power dropped after landing;
- fidelity label preserved;
- ULog contains no dropout records.

Key model-derived results:

- maximum altitude: 3.001 m;
- peak estimated electrical load: 560.504 W;
- final estimated flight-window energy: 7628.901 J (2.119 Wh);
- mean estimated load: takeoff 261.288 W, hover 366.948 W, movement
  370.074 W, and landed 12.773 W.

Machine-readable evidence is stored in `validation.json`, `run_manifest.json`,
`phase_markers.json`, `gazebo_energy.jsonl`, and the PX4 ULog under the same run
directory.

This historical run predates generated-asset hashes in `scenario_manifest.json`
and a validation hash in `run_manifest.json`. Those omissions are covered
without rewriting either historical manifest by
`integrity_manifest.json` in the run directory. Future runs record these hashes
directly.

## Current boundary

This run intentionally used no wind and no payload. Nonzero wind and payload
sweeps may now be introduced as controlled follow-on experiments, one factor at
a time, while retaining the baseline mission and CSV acceptance checks. The
recorded `x500_wind_demo_v1` run is a wind-path demonstration rather than such a
controlled sweep: it uses a 4 s initial hover, while this baseline uses 5 s.
