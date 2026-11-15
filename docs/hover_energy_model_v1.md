# Hover Energy Model v1

## Scientific scope

This offline model calculates an **ideal induced power/energy lower bound** for a
validated steady-hover window. It does not calculate battery energy, measured
electrical energy, electrical power, battery current, or true state of charge.

The input is an existing PX4 ULog. PX4 and Gazebo are not modified or run by this
pipeline.

## Equations

For total mass \(m\), gravity \(g\), air density \(\rho\), and the sum of four
non-overlapping rotor disk areas \(A_\Sigma\):

\[
T_{required}=mg
\]

\[
T_{command}=k_f\sum_{i=1}^{4}\omega_{i,command}^2
\]

\[
P_{induced,ideal,LB}=\frac{T_{required}^{3/2}}{\sqrt{2\rho A_\Sigma}}
\]

\[
E_{induced,ideal,LB}=\int P_{induced,ideal,LB}\,dt
\]

The code uses simulation timestamps and trapezoidal integration. Joules are
converted to watt-hours using \(1\,Wh=3600\,J\).

## Signal provenance

- Motor command: float `actuator_outputs.output[0..3]`, interpreted as rad/s for
  the current X500 Gazebo actuator mapping.
- Air density: `vehicle_air_data.rho`, kg/m³.
- Motion acceptance: `vehicle_local_position_groundtruth`.
- Flight-state window: `vehicle_status.nav_state`.
- Land-state check: `vehicle_land_detected.landed`.

`esc_status.esc_rpm` is forbidden as a calculation input. It is an integer command
echo in this simulation, not measured RPM.

## Reproducible window

The control window begins at the `AUTO_TAKEOFF -> POSCTL` transition and ends at
the following `AUTO_LAND` transition. The analysis removes 10 seconds from both
ends and rejects the result if fewer than 10 seconds remain or any configured
data-quality rule fails.

## Rotor-diameter sensitivity

Two local geometry-derived values are evaluated separately. They are temporary
sensitivity cases, not validated aerodynamic specifications. Neither is selected
as the true effective diameter.

## Air-density limitation

Air density affects the offline momentum-theory calculation. The current Gazebo
motor plugin still uses constant `motorConstant` in \(T=k_f\omega^2\); it is not
density-coupled. Therefore this model cannot yet support a realistic altitude
propulsion experiment.

## References and calibration status

A primary scientific source for momentum theory and figure of merit is required
before thesis use. No citation is assigned here to avoid fabricating or selecting
an unreviewed source.

Figure of merit, motor efficiency, ESC efficiency, auxiliary power, real battery
capacity, and real current are **NOT YET AVAILABLE**. Electrical outputs remain
disabled until independent calibration and validation data exist.

## Historical reproducibility status

`results/logs/hover_energy_v1_run.json` is a preserved legacy result. Its exact
input ULog is identified by SHA-256
`430113e447b75526e10ad7742eef3bf586c85852cd0ca2911cf6055479517f19`.
On 2026-11-11 that exact file was restored from the verified member
`PX4-Autopilot/build/px4_sitl_default/rootfs/log/2026-08-17/00_55_32.ulg` in
`FULL_03_PX4_Autopilot_COMPLETE_2026-08-23.tar.zst`. The archive SHA-256 is
`8691708fc06fdf9e157c0cc2ffe75a2898a04e64bcd28531fe693120ddf5b782`.

The historical recorded path is inside PX4 `build/rootfs`, which can be
recreated by a later SITL run. On 2026-11-12 the exact ULog was therefore also
installed at the durable thesis path
`data/restored/hover_energy_v1/00_55_32.ulg`. Integration tests use this durable
copy. The compatibility copy at the original recorded path is restored too,
but is no longer the sole current reproduction input.

The analysis was rebuilt to a temporary file with the restored ULog and current
configuration. The rebuilt output and preserved result both have SHA-256
`d4b7f214f144f2351b7c4dc07516183c570a5ccc6bc586d5e0e72afd9b21d2e5`;
their parsed structures also have zero differences. The preserved result was
not overwritten. A newer or different ULog remains an invalid substitute.

The machine-readable status and preserved-result hash are in
`results/logs/hover_energy_v1_provenance.json`. They can be checked without the
missing ULog:

```bash
PYTHONPATH=src pytest -q tests/test_provenance_integrity.py
```

To restore both copies reproducibly from the verified archive:

```bash
./scripts/restore_hover_energy_v1_ulog.sh
```

The script verifies the 2.5 GB archive hash, extracts only the approved member
to a temporary directory, verifies the ULog hash, and then installs the durable
and compatibility copies. To repeat the analysis, rebuild to a temporary
output and verify the input hash first:

```bash
export HOVER_V1_ULOG="$PWD/data/restored/hover_energy_v1/00_55_32.ulg"
test -f "$HOVER_V1_ULOG" || { echo "missing restored ULog" >&2; exit 1; }
printf '%s  %s\n' \
  430113e447b75526e10ad7742eef3bf586c85852cd0ca2911cf6055479517f19 \
  "$HOVER_V1_ULOG" | sha256sum -c -
PYTHONPATH=src python3 -m energy_model.hover_energy_v1 \
  --config config/hover_energy_v1.yaml \
  --ulog "$HOVER_V1_ULOG" \
  --output /tmp/hover_energy_v1_rebuild.json
```

Compare the temporary report with the legacy artifact; do not silently replace
the baseline. The legacy report did not record a configuration hash, so the
byte-identical rebuild proves numerical reproducibility with the current
configuration but does not prove that its bytes are the original historical
configuration file.
