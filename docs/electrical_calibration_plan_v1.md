# Electrical Calibration and External Validation Plan v1

## 1. Decision

The first defensible electrical-validation route is a **two-platform, two-claim
workflow**:

1. Use the public DJI Matrice 100 measurements to calibrate and validate the
   electrical extension of the model **for the Matrice 100 empirical case**.
2. Keep PX4/Gazebo X500 results labelled as an ideal induced lower bound until
   matched X500 propulsion/electrical measurements become available.

The Matrice 100 data must not be presented as X500 electrical ground truth.
Transferring a fitted coefficient from Matrice 100 to X500 would be a separate,
uncertain surrogate experiment, not a completed validation.

## 2. Why this route was selected

The selected dataset provides measured battery voltage and current, motion,
wind, payload, altitude, and speed. It contains 195 repeated route flights and
14 ancillary hover/idle recordings. The published experiment used a calibrated
Mauch PL-200 voltage/current sensor and synchronized the signals at approximately
5 Hz.

The Matrice 100 and X500 are not the same vehicle:

- Matrice 100 experimental mass without payload: 3.680 kg.
- X500 simulation nominal mass: 2.064307692 kg.
- Matrice 100: DJI 3510 motor, DJI 1345s propeller, E SERIES 620D ESC.
- X500: a lumped Gazebo motor model with no identified electrical motor, ESC,
  or battery parameters.
- The effective X500 aerodynamic rotor diameter remains unknown.

This mismatch prevents direct absolute calibration of X500, but the real data
can validate the model form, units, processing, repeatability, and expected
payload/operating-condition trends.

## 3. Primary sources

1. Rodrigues, T. A. et al., *In-flight positional and energy use data set of a
   DJI Matrice 100 quadcopter for small package delivery*, Scientific Data 8,
   155 (2021). DOI:
   <https://doi.org/10.1038/s41597-021-00930-x>

2. Dataset record: *Data Collected with Package Delivery Quadcopter Drone*.
   DOI: <https://doi.org/10.1184/R1/12683453>

3. DJI Matrice 100 official support/specifications:
   <https://www.dji.com/support/product/matrice100>

4. UIUC Propeller Database, measured small-UAV propeller thrust and torque data:
   <https://m-selig.ae.illinois.edu/props/propDB.html>

The UIUC database is supplementary. No matched DJI 1345s data have yet been
verified there, so it must not silently replace the actual propeller.

## 4. Source-data audit completed for this plan

The Figshare record exposes the following files:

| File | Size | MD5 | Decision |
|---|---:|---|---|
| `parameters.csv` | 7,867 bytes | `7daa897a1af7847b592ccb7a459477ee` | Required |
| `README.txt` | 25,101 bytes | `af6afbb550286df945c9ecaf4c5f72a4` | Required |
| `flights.zip` | 41,274,360 bytes | `cd1d267fd9b025219f9633b95158eea2` | Downloaded and verified |
| `flights.csv` | 107,068,831 bytes | `93630afc334df8a4eafe4fd87e3c253f` | Not needed initially |
| `raw_files.zip` | 106,101,296 bytes | `351de11d5e7db5521606a931ea10ce63` | Deferred |

Only the processed `flights.zip`, metadata, and README are needed for the first
calibration audit. The larger combined and raw files should not be downloaded
until a specific validation need justifies them.

## 5. Selected recordings

The metadata identifies these controlled recordings:

| Route | Flight IDs | Meaning | Planned use |
|---|---|---|---|
| `A1` | 211, 212, 213 | Ground, no propellers, no motor movement | Sensor-zero diagnostic; not identifiable auxiliary power |
| `A2` | 214, 215, 216 | Ground, no propellers, minimum motor movement | Motor/ESC idle diagnostic |
| `A3` | 217, 218, 219 | Ground, propellers installed, minimum motor movement | Propeller-idle diagnostic |
| `H` | 221, 222, 223, 224 | Hover, no horizontal movement, nominal altitude 3 m | Hover electrical calibration/repeatability |

The four hover recordings all have zero added payload. Therefore they cannot by
themselves validate the payload relationship. Payload validation must later use
held-out route flights and must remain a separate phase.

## 6. Measured and modelled quantities

### Measured electrical power

The README documents negative discharge current for the raw ROS BatteryState
topic, but the processed CSV files show positive current during A2, A3, and H
motor operation. A1 has a small negative zero offset. The processed values are
therefore retained with their observed sign:

\[
P_{measured}(t)=V(t)I(t)
\]

where:

- \(V\): battery/system voltage, V.
- \(I\): battery/system current, A.
- \(P_{measured}\): DC electrical input power, W.

Energy is calculated from the original irregular timestamps:

\[
E_{measured,J}=\int P_{measured}(t)\,dt
\]

\[
E_{measured,Wh}=E_{measured,J}/3600
\]

No interpolation is required for integration if the timestamps are valid and
trapezoidal integration is used directly. Applying an absolute value is
forbidden because it would convert A1's negative sensor offset into fictitious
positive power.

### Sensor-zero and idle diagnostics

A1 cannot estimate auxiliary power because all three A1 recordings report a
small negative current while the system is operating. Its pooled current is
used only as a provisional sensor-zero offset and uncertainty case. A2 and A3
quantify minimum motor/ESC and propeller-installed idle states and remain
separate diagnostics. No auxiliary-power term is identified from these data.

### Ideal induced hover lower bound

For the Matrice 100 empirical case:

\[
P_{i,ideal}=\frac{(mg)^{3/2}}{\sqrt{2\rho A_\Sigma}}
\]

The experiment reports \(m=3.680\,kg\) without added payload. The effective DJI
1345s disk diameter and local air density must be sourced or carried as explicit
uncertainty ranges before calculating a final number. Propeller nomenclature
alone is not accepted as a dimensional measurement.

### Preliminary total electrical multiplier

Because auxiliary power is not separately identifiable, v1 may calculate only
a total empirical multiplier:

\[
\kappa_{total}=\frac{P_{measured,hover}}{P_{i,ideal}}
\]

and:

\[
P_{electrical,predicted}
=
\kappa_{total} P_{i,ideal}
\]

Both raw-current and provisional zero-offset-corrected cases must be reported.
For a physically consistent lower-bound model, \(\kappa_{total}\) should be
greater than one. A value below one triggers rejection and investigation of
mass, rotor area, air density, sign convention, window selection, or sensor
processing.

## 7. Identification limitation

Four hover runs at one mass cannot independently identify a complex efficiency
curve. A1 reveals a sensor-zero problem rather than separating an auxiliary
term, and the following remain confounded:

- motor and ESC losses;
- rotor profile and interference power;
- armed electronics overhead;
- effective rotor diameter;
- air density;
- ground effect and hover-window selection.

Consequently, v1 may fit only one total multiplier \(\kappa_{total}\), with raw
and zero-offset-corrected current treated as sensitivity cases. It must not fit
separate auxiliary, motor, ESC, battery, or propeller parameters from these few
recordings.

## 8. Data-quality and window rules

Each selected CSV must pass all checks:

- file hash/source recorded;
- timestamps finite and strictly increasing;
- no duplicated timestamps;
- voltage finite and positive;
- current finite, with discharge sign handled explicitly;
- no silent row interpolation;
- no impossible energy decrease after cumulative integration;
- stable section at least 20 s where available;
- stable hover selected using motion, not electrical power or a fixed row count;
- horizontal speed no more than 0.15 m/s for at least 95% of accepted samples;
- absolute vertical speed no more than 0.10 m/s for at least 95% of samples;
- altitude remains within 0.25 m of its stable-window median for at least 95% of
  samples;
- takeoff, landing, and transient sections excluded;
- all rejection reasons stored, never silently dropped.

The first audit found that no 20 s H window passes all three strict motion
thresholds. Those failures remain recorded. Exploratory v1 selects the
least-motion continuous powered 20 s window without using electrical power and
must carry an explicit non-confirmatory label. A sensitivity check must show how
conclusions change when thresholds or windows are varied.

## 9. Calibration and validation split

Because there are only four hover recordings, ordinary random row splitting is
forbidden. Rows from one flight are correlated.

Use leave-one-flight-out validation:

1. Select three hover flights as calibration flights.
2. Estimate \(\kappa_{total}\) from flight-level mean values.
3. Predict the held-out fourth hover flight.
4. Repeat until every hover flight has been held out once.
5. Bootstrap at the flight level, not the row level.

Use A1 flight-level estimates to report sensor-zero variability. A2 and A3
remain separate idle diagnostics and are not merged with A1.

## 10. Preliminary acceptance criteria

This stage is accepted only if:

- all four H recordings have valid stable windows;
- at least two of three A1 recordings have valid stable windows;
- stable-hover flight-level measured-power coefficient of variation is no more
  than 10%;
- integration of constant/mean power and trapezoidal energy agrees within 1%
  for stable windows;
- every fitted \(\kappa_{total}>1\);
- leave-one-flight-out hover-power MAPE is no more than 10%;
- uncertainty intervals and all excluded recordings are reported;
- no fitted coefficient is labelled as X500 ground truth.

These criteria assess repeatability at one hover condition. They do not prove
generalization across payload, speed, wind, or altitude.

## 11. Uncertainty analysis

At minimum propagate or sweep:

- measured voltage/current uncertainty;
- total-mass uncertainty;
- effective rotor diameter;
- air density;
- current-zero correction (raw versus A1-offset-corrected);
- hover-window thresholds;
- between-flight variability.

Report median, mean, standard deviation, 95% bootstrap interval, and the number
of independent flights. Do not present thousands of time rows as thousands of
independent experiments.

## 12. Relationship to X500

After the Matrice 100 case passes:

- the equations and processing pipeline are externally tested;
- the real-data trend between payload/conditions and measured power can be
  compared with the X500 lower-bound trend;
- the Matrice 100 multiplier may be shown only as a **surrogate sensitivity
  scenario** for X500, with explicit platform-transfer uncertainty;
- X500 absolute electrical power remains `NOT YET AVAILABLE` until matched
  motor-propeller-ESC-battery measurements or a justified physical bench model
  are obtained.

No Random Forest target should combine X500 simulated lower-bound power and
Matrice 100 measured electrical power as if they were the same quantity.

## 13. Rejected alternatives for v1

| Alternative | Decision | Reason |
|---|---|---|
| Fit X500 directly to all M100 flights | Rejected | Different mass, propulsion, geometry, and battery |
| Treat Gazebo battery percentage as calibration data | Rejected | Time-based and not load-coupled |
| Add a custom electrical Gazebo plugin first | Deferred | Would encode unvalidated assumptions, not create ground truth |
| Use UIUC propeller data as DJI 1345s data | Rejected unless matched | No verified exact propeller match yet |
| Fit many motor/ESC/battery parameters | Rejected | Parameters are not identifiable from four same-mass hover runs |
| Begin payload ML now | Rejected | No validated electrical target for X500 |

## 14. Next investigation step only

The controlled-data audit is complete and recorded in
`docs/m100_controlled_data_audit_v1.md`. Establish defensible primary-source
ranges for effective DJI 1345s rotor diameter, experimental air density, and
mass uncertainty. Then calculate one exploratory Matrice 100 total electrical
multiplier using leave-one-flight-out validation and raw-versus-offset-corrected
sensitivity. Do not transfer the coefficient to X500 as ground truth.
