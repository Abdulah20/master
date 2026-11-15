# Matrice 100 Controlled-Data Audit v2

Date: 2026-11-10

## Outcome

This revision corrects only the ground-energy integration contract. The v1
processor first filtered samples by the declared ground-current band and then
integrated the flattened selection. That could connect two selected samples
across one or more excluded source rows as if the interval were continuously
in-band.

The v2 processor groups selected samples into contiguous source runs and applies
the trapezoidal rule inside each run only. It reports both the selected span
(`duration_s`) and the actual summed integration time
(`integrated_duration_s`), plus the excluded gap time and segment count.

## Corrected values

Only two published v1 ground-energy values change:

| Flight | v1 energy (Wh) | v2 energy (Wh) | Removed bridge (Wh) | Selected span (s) | Integrated duration (s) | Segments |
|---:|---:|---:|---:|---:|---:|---:|
| 214 | 0.348200920551 | 0.343769571621 | 0.004431348930 | 123.21 | 121.32 | 4 |
| 216 | 0.143775909055 | 0.138604242250 | 0.005171666805 | 51.49 | 49.40 | 3 |

The corresponding mean powers remain exactly `10.192421399610 W` and
`10.086061745318 W`. All other ground energies are unchanged.

## Unchanged scientific result

All four hover flights, their selected windows and energies, the mean
battery-side hover power (`478.658727 W`), between-flight coefficient of
variation (`2.645145%`), strict-motion result (`0/4`), and all claim boundaries
are unchanged. The hover result remains exploratory Matrice 100 DC battery-side
evidence, not X500 ground truth or a calibrated payload relationship.

## Provenance and artifacts

- Historical result: `results/logs/m100_controlled_audit_v1.json`, preserved at
  SHA-256 `5c68050e0a365bc1a3b0963cfd25b76fb23fcba2e3a4ab4e9425e5b55a60fa34`.
- Corrected result: `results/logs/m100_controlled_audit_v2.json`, SHA-256
  `94ae05cf5f6c80a428bd69ef9488397c8f0f996faf22f512cdcaa854e06f1783`.
- v2 protocol: `config/m100_controlled_audit_v2.json`.
- Processor: `src/data_processing/audit_m100_controlled.py`.
- Regression tests: `tests/test_m100_controlled_audit.py`.
- Article DOI: `10.1038/s41597-021-00930-x`.
- Dataset DOI: `10.1184/R1/12683453`.

The v2 result carries the source DOI values and verifies the three local source
files against their recorded MD5 values. The legacy artifact hash and correction
scope are also embedded in `report_metadata`.

## Reproduction

From the project root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m data_processing.audit_m100_controlled \
  --project-root /home/abdulah/uav-energy-thesis \
  --config config/m100_controlled_audit_v2.json \
  --output results/logs/m100_controlled_audit_v2.json
```

The regression suite independently regenerates the in-memory report and requires
it to equal the published v2 JSON exactly.
