# NREL Thevenin provenance

- Upstream: <https://github.com/NREL/thevenin>
- Release: `v0.2.1`
- Commit: `bc44f88e12a37c3ca6346a4735c6ba9e8ecc4573`
- Retrieved: 2026-08-17
- Local source: `external/third_party/thevenin`
- License: BSD-3-Clause; retained in the upstream source directory
- Software DOI: <https://doi.org/10.11578/dc.20241125.2>
- Isolated environment: `.venv-thevenin`
- Verification: 44 upstream tests passed with pytest plugin autoload disabled

The upstream default parameter file contains tutorial values that do not
represent a real battery. No default value may be used as an X500 or Matrice 100
parameter. The software is installed as a modelling engine only; vehicle
parameters require separate identification and validation.
