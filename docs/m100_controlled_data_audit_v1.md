# Matrice 100 Controlled-Data Audit v1

Date: 2026-08-17

> Historical artifact: v1 is preserved byte-for-byte. Its ground-energy fields
> for flights 214 and 216 integrated across samples excluded by the current-band
> filter. The corrected, reproducible result is documented in
> `docs/m100_controlled_data_audit_v2.md`. Hover powers, hover energies, aggregate
> statistics, and scientific claim limits are unchanged.

## Outcome

The controlled Matrice 100 data are internally consistent enough to support an
**exploratory, single-condition electrical calibration for the Matrice 100**.
They do not provide X500 battery ground truth and do not yet validate a payload
relationship.

The four independently recorded hover flights produce a mean flight-level DC
battery-side power of **478.659 W**, with a between-flight standard deviation of
**12.661 W** and coefficient of variation of **2.645%**.  A provisional
zero-current offset correction changes the mean to **481.956 W**.  Both values
must be carried until sensor-offset uncertainty is resolved.

All four original strict motion diagnostics fail.  The power result is therefore
labelled exploratory rather than confirmatory even though electrical
repeatability is good.

## Reproducible artifacts

- Protocol: `config/m100_controlled_audit_v1.json`
- Processor: `src/data_processing/audit_m100_controlled.py`
- Machine-readable result: `results/logs/m100_controlled_audit_v1.json`
- Regression tests: `tests/test_m100_controlled_audit.py`

At the time this historical v1 result was published, the then-registered project
suite reported **19 passed**. Current verification is recorded with the v2
correction rather than by rewriting this historical result.

## Source integrity

| File | Bytes | Verified MD5 |
|---|---:|---|
| `README.txt` | 25,101 | `af6afbb550286df945c9ecaf4c5f72a4` |
| `parameters.csv` | 7,867 | `7daa897a1af7847b592ccb7a459477ee` |
| `flights.zip` | 41,274,360 | `cd1d267fd9b025219f9633b95158eea2` |

The ZIP integrity test reports no compressed-data errors.  Flights 211--219 and
221--224 are present and their routes match `parameters.csv`.

## Current-sign finding

The README describes the raw ROS `sensor_msgs/BatteryState` convention as
negative current during discharge.  The processed flight CSV files do not
follow that sign in their motor-active portions:

- A1, motors not moving: approximately -0.152 A pooled mean;
- A2, motors moving without propellers: approximately +0.40 A;
- A3, motors moving with propellers: approximately +1.47 A;
- H, hover: approximately +21--22 A.

The processed CSV is therefore analysed with its observed sign:

\[
P_{DC}(t)=V(t)I(t)
\]

No absolute value is applied.  Applying `abs(I)` to A1 would convert a negative
zero offset into fictitious positive power.

## Ground-test results

Transient/off samples were excluded with route-specific current bands fixed in
the protocol before calculation.

| Route | Flights | Physical state | Mean of flight mean powers | Between-flight SD | Interpretation |
|---|---|---|---:|---:|---|
| A1 | 211--213 | No motor movement | -3.850 W | 0.499 W | Sensor-zero diagnostic; not physical auxiliary power |
| A2 | 214--216 | Minimum motor movement, no propellers | 10.031 W | 0.194 W | Motor/ESC idle diagnostic |
| A3 | 217--219 | Minimum motor movement, propellers installed | 35.986 W | 0.646 W | Propeller-idle diagnostic |

A1 cannot identify auxiliary power.  Its pooled current, -0.15153 A, is retained
as a provisional zero-current offset and uncertainty case.  Subtracting A1 as
an auxiliary-power term is rejected.

## Hover-window selection

Each H flight is documented by the source as hover at nominal 3 m with no
horizontal route.  The files include powered and, in some cases, terminal
unpowered samples.  The processor:

1. requires a continuous powered interval with current above 10 A;
2. evaluates 20 s candidates using kinematics only;
3. chooses the least-motion candidate without looking at voltage, current, or
   power;
4. preserves the original irregular timestamps;
5. integrates signed `V*I` using the trapezoidal rule.

The selection score is the maximum of the three normalized 95th-percentile
motion diagnostics.  A score no greater than 1 would pass every pre-registered
strict threshold.

| Flight | Window (s) | Mean V | Mean I | Mean power | Energy | Motion score | Strict pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 221 | 55.9--75.9 | 21.640 V | 21.277 A | 460.382 W | 2.5568 Wh | 1.311 | No |
| 222 | 54.7--74.7 | 22.129 V | 22.075 A | 488.406 W | 2.7135 Wh | 1.195 | No |
| 223 | 83.5--103.5 | 21.848 V | 22.233 A | 485.706 W | 2.6984 Wh | 1.071 | No |
| 224 | 81.0--101.0 | 21.419 V | 22.420 A | 480.141 W | 2.6656 Wh | 1.348 | No |

Mean-power-times-duration and trapezoidal energy differ by at most 0.069% for
these windows, below the planned 1% limit.

## Acceptance assessment

| Check | Result |
|---|---|
| Source hashes and ZIP integrity | Pass |
| Finite, positive voltage | Pass |
| Finite, strictly increasing timestamps | Pass |
| Four powered 20 s H windows | Pass |
| Hover power coefficient of variation <=10% | Pass: 2.645% |
| Mean-power and trapezoidal-energy agreement <=1% | Pass |
| Original strict motion diagnostics | **Fail: 0 of 4** |
| Payload validation | Not testable from H: all four have zero payload |
| X500 electrical validation | Not provided by this dataset |

The failure is not hidden by changing thresholds.  The selected windows may be
used for an explicitly exploratory calibration and a threshold-sensitivity
analysis, but not described as having passed the original confirmatory motion
protocol.

## Claim boundary

These are measured DC battery-side values for the experiment's Matrice 100.
They include propulsion, electronics, and other battery-fed loads that the
sensor sees.  They are not shaft power, per-motor power, or X500 power.  A fitted
Matrice 100 coefficient may test the processing and model form; transferring it
to X500 is only a surrogate sensitivity scenario.

## Important remaining gaps and next step

The remaining blockers for the first physical coefficient are the effective
DJI 1345s rotor diameter, air density at the experiment, mass uncertainty, and
the unresolved separation of sensor zero offset from auxiliary consumption.

The **only next investigation step** is to establish defensible ranges for
rotor diameter, air density, and mass from primary evidence, then calculate a
single Matrice 100 total electrical multiplier with leave-one-flight-out
validation and sensitivity to raw versus zero-offset-corrected current.  No X500
battery claim or payload ML should begin before that check.
