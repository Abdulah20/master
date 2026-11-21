"""Calibrate M100 phase power, airborne energy and loaded voltage from archived CMU flights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml

from . import cmu_flight_calibration, cmu_voltage_calibration, external
from .physics import CONFIG, PORTABLE_RUNTIME
from .runtime_snapshot.ros2_ws.src.drone_nav.drone_nav import m100_atmosphere


ROOT = external.ROOT
DESTINATION = ROOT / "results/m100_reaudit/cmu_calibration"
MANIFEST = external.OUTPUT / "flight_manifest.csv"
MANIFEST_SHA256 = "171d8176bbddab97b25d19ada43c54c67d760a6e7680f0380233d54c08c044f3"
MODEL_CONFIG = PORTABLE_RUNTIME / "ros2_ws/src/drone_nav/config/m100_energy_model_v2.yaml"
REFERENCE_C = m100_atmosphere.REFERENCE_TEMPERATURE_C


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def reference_powers(reference, *, payload_kg, commanded_altitude_agl_m):
    """Evaluate the stored phase reference algebraically using preflight inputs."""
    density = m100_atmosphere.dry_air_state(reference["temperature_at_datum_C"],
                                            reference["ground_datum_msl_m"] + commanded_altitude_agl_m).density_kg_m3
    feature = (reference["regression_baseline_kg"] + payload_kg) ** 1.5 / math.sqrt(density)
    return {phase: coef["b1"] * feature + coef["b0"] for phase, coef in reference["coefficients_by_phase"].items()}


def development_flights():
    # Verify recorded sources and the preassigned split before opening any flight member.
    for name in ("flights.zip", "parameters.csv", "README.txt"):
        external.check_hash(external.INPUT / name, external.SHA256[name])
    external.check_hash(external.CONTEXT, external.SHA256["published_context.csv"])
    external.check_hash(MANIFEST, MANIFEST_SHA256)
    manifest = pd.read_csv(MANIFEST).set_index("flight_id")
    metadata = pd.read_csv(external.INPUT / "parameters.csv").set_index("flight")
    context = pd.read_csv(external.CONTEXT).set_index("flight_id")
    if not all(table.index.is_unique for table in (manifest, metadata, context)):
        raise ValueError("Duplicate flight ID in a source table")
    manifest = manifest.loc[manifest["included"] & manifest["split"].isin(["train", "validation"])]
    assigned = external.assign_splits(metadata, context[["published_role"]])
    if manifest["split"].value_counts().to_dict() != {"train": 75, "validation": 35}:
        raise ValueError("Development membership differs from the frozen 75/35 manifest")

    model = yaml.safe_load(MODEL_CONFIG.read_text())["model"]
    mass_kg = model["mass_basis"]["published_regression_baseline_kg"]
    coefficients = model["coefficients"]
    ground_msl_m = json.loads(CONFIG.read_text())["wind"]["world_ground_msl_m"]
    base = {
        "model": model["id"],
        "equation": "P_W = b1 * (regression_baseline_kg + payload_kg)^1.5 / sqrt(rho_kg_m3) + b0",
        "coefficients_by_phase": coefficients,
        "regression_baseline_kg": mass_kg,
        "original_fit_flights": model["source"]["training_flights"],
        "development_overlap": "The published reference was derived from the original 120 fitting flights, "
                               "including both current development partitions.",
        "reference_atmosphere": "dry hydrostatic air at commanded AGL plus ground MSL; actual flight weather unknown",
        "ground_datum_msl_m": ground_msl_m, "temperature_at_datum_C": REFERENCE_C,
        "wind_power_runtime_applied": False,
    }
    flights = []
    with ZipFile(external.INPUT / "flights.zip") as archive:
        for fid, member in manifest.sort_index().iterrows():
            if assigned[fid] != member.split or context.loc[fid, "published_role"] != "fit":
                raise ValueError(f"Flight {fid} is outside its registered development split")
            params = metadata.loc[fid]
            with archive.open(f"flights/{fid}.csv") as file:
                samples = pd.read_csv(file, usecols=external.COLUMNS)
            flags = external.quality_flags(samples)
            if flags:
                raise ValueError(f"Flight {fid} no longer satisfies data quality rules: {flags}")
            scopes, _, _ = external.measured_scopes(samples, params, context.loc[fid])
            if any(scopes[p]["observed_mean_power_W"] <= 0 for p in external.PHASES):
                raise ValueError(f"Unexpected discharge-current sign in flight {fid}")
            height_m = float(params.altitude)
            payload_kg = float(params.payload) / 1000.0
            flights.append({
                "flight_id": int(fid), "split": member.split, "date": str(params.date), "route": params.route,
                "payload_kg": payload_kg, "commanded_speed_mps": float(params.speed),
                "commanded_altitude_agl_m": height_m, "scopes": scopes, "samples": samples,
                "bounds": external.phase_boundaries(samples["position_z"].to_numpy()),
                "base_phase_power_W": reference_powers(base, payload_kg=payload_kg, commanded_altitude_agl_m=height_m),
            })
    return flights, base


def condition_table(predictions):
    """Flight-equal diagnostics for each observed speed, payload, height and route."""
    factors = ["commanded_speed_mps", "payload_kg", "commanded_altitude_agl_m", "route"]
    rows = []
    for factor in factors:
        for (split, scope, model, value), group in predictions.groupby(["split", "scope", "model", factor], sort=True):
            row = dict(split=split, scope=scope, model=model, condition=factor, value=value, flights=len(group))
            for quantity, unit in [("mean_power_W", "W"), ("energy_Wh", "Wh"), ("duration_s", "s")]:
                error = group[f"predicted_{quantity}"] - group[f"observed_{quantity}"]
                row[f"mae_{unit}"] = error.abs().mean()
                row[f"bias_{unit}"] = error.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def calibrate(destination):
    flights, base = development_flights()
    flight_fit, predictions = cmu_flight_calibration.fit(flights)
    voltage_fit, voltages = cmu_voltage_calibration.fit(flights)
    scores = flight_fit["metrics"]["validation"]["airborne"]
    candidates = [name for name in scores if name != cmu_flight_calibration.MEAN]
    flight_fit["development_choice"] = {
        "criterion": "Lowest development MAE among the coherent phase models sharing the same predicted durations. "
                     "Each chosen model produces its own mutually consistent power, duration and energy.",
        "airborne_energy_Wh": min(candidates, key=lambda name: scores[name]["MAE_Wh"]),
        "airborne_mean_power_W": min(candidates, key=lambda name: scores[name]["MAE_W"]),
    }
    cohorts = {}
    for split in ("train", "validation"):
        members = [f for f in flights if f["split"] == split]
        cohorts[split] = {
            "flight_ids": [f["flight_id"] for f in members],
            "calendar_dates": sorted({f["date"] for f in members}),
            "routes": dict(Counter(f["route"] for f in members)),
        }
    source_files = [MANIFEST, external.CONTEXT, MODEL_CONFIG, CONFIG, Path(external.__file__),
                    Path(m100_atmosphere.__file__), Path(__file__), Path(cmu_flight_calibration.__file__),
                    Path(cmu_voltage_calibration.__file__)]
    source_hashes = {str(path.resolve().relative_to(ROOT)): digest(path) for path in source_files}
    source_hashes.update({str((external.INPUT / name).relative_to(ROOT)): external.SHA256[name]
                          for name in ("flights.zip", "parameters.csv", "README.txt")})
    summary = {
        "source": "Rodrigues et al., Carnegie Mellon University, https://doi.org/10.1184/R1/12683453 (CC BY 4.0)",
        "protocol": "Raw synchronized CSVs, accepted development flight IDs only. New coefficients use the 75 "
                    "train flights. The 35 validation flights come from other development dates in the same campaign. "
                    "The original published phase reference predates this split. No test-flight CSV is opened or scored.",
        "measurement": "Battery output on the instrumented CMU aircraft (Mauch PL-200). Positive discharge in the "
                       "processed CSVs. Observed airborne energy integrates V times I at the actual adjacent times, "
                       "in Wh. The published altitude regime boundaries exclude ground time.",
        "cohorts": cohorts, "input_and_code_sha256": source_hashes, "published_power_reference": base,
        "software_versions": {"Python": platform.python_version(), "NumPy": np.__version__,
                              "pandas": pd.__version__, "PyYAML": yaml.__version__},
        "aircraft": flight_fit, "loaded_voltage": voltage_fit,
        "scope": {"source_of_power_and_energy_targets": "measured pack electrical power and its airborne integral",
                  "battery_voltage_fidelity": "LIMITED_EMPIRICAL_PROXY",
                  "tb47d_soc_ocv_rc_capacity_temperature_aging": "C_NOT_CALIBRATED",
                  "3510_1345s_shaft_thrust_torque_and_620d_losses": "C_NOT_CALIBRATED",
                  "wind_power_runtime_applied": False, "controller_or_runtime_modified": False},
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        bundle = Path(tmp) / "results"
        bundle.mkdir()
        outputs = {"flight_residuals.csv": predictions, "voltage_by_flight.csv": voltages,
                   "by_condition.csv": condition_table(predictions)}
        for name, frame in outputs.items():
            external.write_table(frame, bundle / name)
        summary["table_sha256"] = {name: digest(bundle / name) for name in outputs}
        (bundle / "fit.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if destination.exists():
            names = [*outputs, "fit.json"]
            if not all((destination / name).is_file() and digest(destination / name) == digest(bundle / name)
                       for name in names):
                raise FileExistsError(f"Different results already exist in {destination}; use a new --output directory")
        else:
            bundle.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    print(calibrate(parser.parse_args().output))


if __name__ == "__main__":
    main()
