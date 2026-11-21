"""Compare fixed M100 power models to synchronized CMU electrical measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml

from .physics import Model

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/external/m100"
OUTPUT = ROOT / "results/m100_reaudit/external"
CONTEXT = OUTPUT / "published_context.csv"
SHA256 = {
    "flights.zip": "98f65a1ab54a8b618c169fdf49311a53ff1fefed86fdc3f440d6a0166ce49fb1",
    "parameters.csv": "568dffaed11a4d335ef4f8a932b342c836907af2f4aa54da9b4a0e2ffef2108a",
    "README.txt": "8601dd211d6c297c5411992dc34923bdb98b72cdc287517ed18e479f0e9540fd",
    "published_context.csv": "b2ce6705326edb6174e5191901195e21d0c73cd161d28b9f1c0759d9bcee0ea8",
}
REPRODUCTION_SHA256 = {
    "energy_summary_v1.csv": "df560db3c856ddf3bba92e889de4862c1c5b53de1a100cd7fbfa6d319aeaedb9",
    "official_sample_v1.csv": "a5d3922846cba85047e6ecd31726cc1727bc4941a767975c12a12645415df6ea",
}
PHASES = ("takeoff", "cruise", "landing")
PUBLISHED = {"takeoff": (80.4, 13.8), "cruise": (68.9, 16.8), "landing": (71.5, -24.3)}
BASELINE = "published_phase"
SPEED_EXTENSION = "existing_stable_speed"
STABLE_CONFIG_SHA256 = "a702c1f1a4b2498e29554b10c0e10ff1f9d6541333fe82fc0b1b36736ce0f2c7"
MAX_GAP_S = 0.5  # Existing per-recording QC rule. Do not bridge missing data.
TIME_ATOL_S = 1e-7  # Agreement with archived spans published to ~0.01 s.
ENERGY_ATOL_WH = 1e-8  # Float64 accumulation only.
COLUMNS = ("time", "battery_voltage", "battery_current", "position_z",
           "velocity_x", "velocity_y", "velocity_z", "wind_speed")
FEATURES = ["payload_kg", "commanded_speed_mps", "commanded_altitude_agl_m"]


def check_hash(path, expected):
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path.name} SHA-256 differs from the registered source: {actual}")


def rebuild_context(reproduction_dir: Path, destination: Path):
    """Recreate the compact weather, phase-span and membership inputs without copying targets."""
    for name, digest in REPRODUCTION_SHA256.items():
        check_hash(reproduction_dir / name, digest)
    with (reproduction_dir / "official_sample_v1.csv").open() as file:
        fit_ids = {int(row["flight"]) for row in csv.DictReader(file)}
    with (reproduction_dir / "energy_summary_v1.csv").open() as file:
        source = sorted(csv.DictReader(file), key=lambda row: int(row["flight"]))
    with destination.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["flight_id", "published_role", "reconstructed_density_kg_m3",
                         *(f"published_{phase}_span_s" for phase in PHASES)])
        for row in source:
            fid = int(row["flight"])
            mass_kg = 3.715 + float(row["payload"]) / 1000
            rho = (mass_kg**1.5 / float(row["mass_rho"]))**2
            writer.writerow([fid, "fit" if fid in fit_ids else "holdout", format(rho, ".16g"),
                             *(format(float(row[f"time_{p}"]), ".12g") for p in PHASES)])
    check_hash(destination, SHA256["published_context.csv"])


def assign_splits(metadata, context):
    """Published testing IDs stay locked. Reserve entire fitting days for new-model validation."""
    fit_days = sorted({metadata.loc[fid, "date"] for fid, role in context["published_role"].items()
                       if role == "fit"})
    # Select one fifth of development dates by a fixed hash, before reading any observations.
    by_hash = sorted(fit_days, key=lambda day: hashlib.sha256(f"cmu-v1:{day}".encode()).digest())
    validation_dates = set(by_hash[:max(1, len(fit_days) // 5)])
    return {
        fid: ("test" if row.published_role == "holdout"
              else "validation" if metadata.loc[fid, "date"] in validation_dates else "train")
        for fid, row in context.iterrows()
    }


def phase_boundaries(altitude):
    """Indices from Rodrigues regime.py v1.0.0, with explicit ordering checks."""
    # Equivalent to the published Gaussian (sigma 5 samples, four-sigma radius,
    # half-sample symmetric edges). NumPy keeps it available without a SciPy ABI.
    radius = 20
    positions = np.arange(-radius, radius + 1, dtype=float)
    weights = np.exp(-0.5 * (positions / 5.0)**2)
    padded = np.pad(altitude - np.min(altitude), radius, mode="symmetric")
    smooth = np.convolve(padded, weights / weights.sum(), mode="valid")
    slope = np.diff(smooth)
    climb_peak, descent_peak = np.argmax(slope), np.argmin(slope)
    lift = np.flatnonzero(slope > 0.05)
    level = np.flatnonzero(slope[climb_peak:] <= 0)
    descent = np.flatnonzero(slope[:descent_peak + 1] >= 0)
    stop = np.flatnonzero(slope[descent_peak:] >= -0.025)
    if not (lift.size and level.size and descent.size):
        raise ValueError("cannot resolve the published flight regimes from altitude")
    bounds = [int(lift[0]), int(climb_peak + level[0]), int(descent[-1]),
              int(descent_peak + stop[0]) if stop.size else len(slope) - 1]
    if min(np.diff(bounds)) < 3:
        raise ValueError("phase boundaries unordered or too short")
    return bounds


def quality_flags(samples):
    if len(samples) < 3 or not np.isfinite(samples.to_numpy()).all():
        return ["nonfinite_or_too_few_samples"]
    flags = []
    dt = np.diff(samples["time"])
    if np.any(dt <= 0):
        flags.append("non_increasing_time")
    if np.any(dt > MAX_GAP_S + 1e-10):
        flags.append("gap_over_0.5s")
    if samples["time"].iloc[0] != 0:
        flags.append("time_origin_not_zero")
    if not samples["battery_voltage"].between(0, 33, inclusive="right").all():
        flags.append("voltage_outside_PL200_range")
    if np.any(abs(samples["battery_current"]) > 200):
        flags.append("current_outside_PL200_range")
    return flags


def midpoint(values):
    return (values[:-1] + values[1:]) / 2


def summarize_intervals(dt, mean_power, mean_speed, mean_airflow, mask):
    """Integrate piecewise linear samples only over selected *adjacent* intervals."""
    durations = dt[mask]
    span = float(durations.sum())
    if span <= 0:
        raise ValueError("empty measurement window")
    watt_seconds = np.dot(mean_power[mask], durations)
    return {
        "included_intervals": int(mask.sum()),
        "covered_duration_s": span,
        "observed_mean_power_W": float(watt_seconds / span),
        "observed_energy_Wh": float(watt_seconds / 3600),
        "observed_ground_speed_mps": float(np.dot(mean_speed[mask], durations) / span),
        "onboard_airflow_mps": float(np.dot(mean_airflow[mask], durations) / span),
    }


def measured_scopes(samples, params, context):
    t = samples["time"].to_numpy()
    z = samples["position_z"].to_numpy()
    dt = np.diff(t)
    measured_power = (samples["battery_voltage"] * samples["battery_current"]).to_numpy()
    ground_speed = np.hypot(samples["velocity_x"], samples["velocity_y"]).to_numpy()
    airflow = samples["wind_speed"].to_numpy()
    start, level, descent, end = phase_boundaries(z)
    # Confirm the official open-endpoint masks independently before making a continuous budget.
    for p, a, b in zip(PHASES, (start, level, descent), (level, descent, end), strict=True):
        span = t[b - 1] - t[a + 1]
        if abs(span - context[f"published_{p}_span_s"]) > TIME_ATOL_S:
            raise ValueError(f"{p} source duration mismatch in flight {params.name}")
    # Both ends use authors' airborne window. Every interior time interval belongs to one phase.
    i = np.arange(dt.size)
    airborne = (i >= start + 1) & (i < end - 1)
    phase_masks = {"takeoff": airborne & (i < level), "cruise": (i >= level) & (i < descent),
                   "landing": airborne & (i >= descent), "airborne": airborne}
    speed = params.speed
    agl = float(params.altitude)
    home = np.median(z[t <= 5.0])
    steady = ((abs(ground_speed - speed) <= max(0.5, 0.1 * speed))
              & (abs(z - home - agl) <= max(3.0, 0.1 * agl))
              & (abs(samples["velocity_z"].to_numpy()) <= 1.5))
    stable_mask = steady[1:] & steady[:-1]
    if speed in (4, 6, 8) and dt[stable_mask].sum() >= 20.0:
        phase_masks["stable_selected"] = stable_mask
    scope_rows = {}
    for name, mask in phase_masks.items():
        measure = summarize_intervals(dt, midpoint(measured_power), midpoint(ground_speed),
                                      midpoint(airflow), mask)
        indices = i[mask]
        measure.update(window_first_time_s=t[indices[0]], window_last_time_s=t[indices[-1]+1],
                       contiguous_windows=int(1 + sum(np.diff(indices) > 1)))
        scope_rows[name] = measure
    budget_error = abs(scope_rows["airborne"]["observed_energy_Wh"]
                       - sum(scope_rows[p]["observed_energy_Wh"] for p in PHASES))
    span_error = abs(scope_rows["airborne"]["covered_duration_s"]
                     - sum(scope_rows[p]["covered_duration_s"] for p in PHASES))
    if budget_error > ENERGY_ATOL_WH or span_error > TIME_ATOL_S:
        raise ValueError("inconsistent airborne phase allocation")
    return scope_rows, budget_error, span_error


def compare_models(scopes, base_power, stable_adjustment_w):
    """Matched window, including common measured duration in each energy comparison."""
    for phase, obs in scopes.items():
        if phase == "airborne":
            power = sum(base_power[p] * scopes[p]["covered_duration_s"] for p in PHASES)
            power /= obs["covered_duration_s"]
        else:
            power = base_power["cruise" if phase == "stable_selected" else phase]
        models = {BASELINE: power}
        if phase == "stable_selected":
            models[SPEED_EXTENSION] = power + stable_adjustment_w
        for name, estimate in models.items():
            yield {"scope": phase, "model": name, **obs, "estimated_mean_power_W": estimate,
                   "estimated_energy_Wh": estimate * obs["covered_duration_s"] / 3600}


def grouped_errors(frame):
    df = frame.copy()
    df["airflow_band_mps"] = pd.cut(df["onboard_airflow_mps"], [-np.inf, 4., 8., 12., np.inf],
                                    labels=["<4", "4_to_<8", "8_to_<12", ">=12"], right=False)
    rows = []
    for (split, phase, model), cohort in df.groupby(["split", "scope", "model"], sort=True):
        strata = [("all", "all", cohort)]
        for key in [*FEATURES, "airflow_band_mps"]:
            strata.extend((key, str(value), group)
                          for value, group in cohort.groupby(key, observed=True, sort=True))
        for field, value, subset in strata:
            row = {"split": split, "scope": phase, "model": model, "stratum": field,
                   "value": value, "flights": len(subset)}
            for qty, unit in [("mean_power", "W"), ("energy", "Wh")]:
                residual = subset[f"estimated_{qty}_{unit}"] - subset[f"observed_{qty}_{unit}"]
                row.update({f"MAE_{unit}": abs(residual).mean(),
                            f"RMSE_{unit}": math.sqrt((residual**2).mean()),
                            f"bias_{unit}": residual.mean()})
            rows.append(row)
    return pd.DataFrame(rows)


def preflight_reference(measured):
    """One prespecified diagnostic fit to actual cruise means, using whole training flights only."""
    if measured.flight_id.duplicated().any():
        raise ValueError("repeated flights in measured training data")
    features = np.column_stack([np.ones(len(measured)), measured[FEATURES].to_numpy()])
    target = measured["observed_mean_power_W"].to_numpy()
    train = measured.split.eq("train").to_numpy()
    coefficients, _, rank, _ = np.linalg.lstsq(features[train], target[train], rcond=None)
    if rank != features.shape[1]:
        raise ValueError("training flight settings do not identify the selected features")
    prediction = features @ coefficients
    scores = {}
    for name in ("train", "validation", "test"):
        mask = measured.split.eq(name).to_numpy()
        error = prediction[mask] - target[mask]
        scores[name] = {"flights": int(mask.sum()), "MAE_W": float(np.abs(error).mean()),
                        "RMSE_W": float(np.sqrt(np.mean(error**2))), "bias_W": float(error.mean())}
    return {"target": "measured whole-cruise mean electrical power (W)",
            "model": "ordinary least squares; intercept + payload_kg + commanded_speed_mps + commanded_altitude_agl_m",
            "coefficients": dict(zip(["intercept_W", *FEATURES], coefficients.tolist(), strict=True)),
            "training_scope": "train only; no model selection or retuning on validation or test",
            "scores": scores}


def write_table(frame, path):
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def evaluate(dataset, context_path, destination):
    runtime = Model.load()
    config_dir = runtime.runtime_root / "ros2_ws/src/drone_nav/config"
    stable_path = config_dir / "m100_stable_cruise_speed_v1.yaml"
    check_hash(stable_path, STABLE_CONFIG_SHA256)
    stable = yaml.safe_load(stable_path.read_text())
    published_path = config_dir / runtime.settings["energy"]["published_phase_config"]
    with published_path.open("rb") as file:
        published_sha = hashlib.file_digest(file, "sha256").hexdigest()
    for name in ("flights.zip", "parameters.csv", "README.txt"):
        check_hash(dataset / name, SHA256[name])
    check_hash(context_path, SHA256["published_context.csv"])
    metadata = pd.read_csv(dataset / "parameters.csv").set_index("flight")
    context = pd.read_csv(context_path).set_index("flight_id")
    if not metadata.index.is_unique or not context.index.is_unique:
        raise ValueError("duplicate flight IDs in a provenance input")
    if context["published_role"].value_counts().to_dict() != {"fit": 120, "holdout": 68}:
        raise ValueError("published 120/68 cohort identity mismatch")
    splits = assign_splits(metadata, context)
    manifest, comparisons, energy_closures, time_closures = [], [], [], []
    with ZipFile(dataset / "flights.zip") as archive:
        recorded_ids = {int(Path(p).stem) for p in archive.namelist() if p.startswith("flights/") and p.endswith(".csv")}
        if recorded_ids != set(metadata.index):
            raise ValueError("archive recording IDs do not match published metadata")
        for fid, row in metadata.sort_index().iterrows():
            with archive.open(f"flights/{fid}.csv") as handle:
                samples = pd.read_csv(handle, usecols=COLUMNS)
            flags = quality_flags(samples)
            if fid not in context.index:
                if row.route in {"A1", "A2", "A3", "H"}:
                    flags.append("ancillary_or_hover")
                elif row.payload not in (0, 250, 500):
                    flags.append("outside_study_payloads")
                else:
                    flags.append("route_outside_published_cohort")
            base = {"flight_id": fid, "split": splits.get(fid, "outside"),
                    "published_role": context.loc[fid, "published_role"] if fid in context.index else "outside",
                    "date": row.date, "route": row.route, "payload_kg": row.payload / 1000,
                    "commanded_speed_mps": row.speed}
            manifest.append({**base, "commanded_altitude_original_m": row.altitude,
                             "samples": len(samples), "max_interval_s": np.diff(samples.time).max(),
                             "included": not flags, "reasons": "|".join(flags)})
            if flags:
                continue
            info = context.loc[fid]
            base.update(commanded_altitude_agl_m=float(row.altitude),
                        density_proxy_kg_m3=info.reconstructed_density_kg_m3)
            feature = (3.715 + base["payload_kg"])**1.5 / math.sqrt(info.reconstructed_density_kg_m3)
            base_power = {p: runtime.phase_power(p, base["payload_kg"], info.reconstructed_density_kg_m3)
                          for p in PHASES}
            for phase, (slope, intercept) in PUBLISHED.items():
                if not math.isclose(base_power[phase], slope * feature + intercept, rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError(f"runtime/{phase} differs from the published rounded estimator")
            stable_adjustment = stable["offset_W"] + stable["slope_W_per_mps"] * (row.speed - stable["reference_speed_mps"])

            scopes, de, dt = measured_scopes(samples, row, info)
            energy_closures.append(de)
            time_closures.append(dt)
            comparisons.extend({**base, **measure} for measure in compare_models(scopes, base_power, stable_adjustment))
    manifest = pd.DataFrame(manifest)
    table = pd.DataFrame(comparisons)
    if table.duplicated(["flight_id", "scope", "model"]).any():
        raise ValueError("duplicate measurement/model comparison")
    measured = table[(table.scope == "cruise") & (table.model == BASELINE)][
        ["flight_id", "split", *FEATURES, "observed_mean_power_W"]]
    if not measured.flight_id.is_unique or set(measured.split) != {"train", "validation", "test"}:
        raise ValueError("missing split or repeated cruise target")
    metrics = grouped_errors(table)
    included = manifest[manifest.included]
    summary = {
        "dataset": "https://doi.org/10.1184/R1/12683453",
        "software": "https://doi.org/10.5281/zenodo.6726991",
        "release_commit": "3d27ac92ba5db36a2fe3fc1449bbb5c422e10ca6",
        "source_sha256": SHA256, "context_parent_sha256": REPRODUCTION_SHA256,
        "runtime_config_sha256": {published_path.name: published_sha, stable_path.name: STABLE_CONFIG_SHA256},
        "actual_recordings": len(manifest), "included_complete_flights": len(included),
        "excluded_recordings": len(manifest) - len(included),
        "published_before_qc": context.published_role.value_counts().to_dict(),
        "published_after_qc": included.published_role.value_counts().to_dict(),
        "measured_train_validation_test": measured.split.value_counts().to_dict(),
        "excluded_reason_combinations": manifest.loc[~manifest.included, "reasons"].value_counts().to_dict(),
        "split_unit": "complete flight; new-flight generalization in one campaign",
        "validation_dates_in_development_partition": sorted(included.loc[included.split == "validation", "date"].unique()),
        "train_test_overlap_calendar_dates": sorted(set(included.loc[included.split == "train", "date"])
                                                    & set(included.loc[included.split == "test", "date"])),
        "measured_training_features": FEATURES,
        "measured_target": "time-weighted battery electrical mean power W in cruise including turns",
        "limits": {"maximum_gap_s": MAX_GAP_S, "duration_check_absolute_s": TIME_ATOL_S,
                   "phase_sum_absolute_Wh": ENERGY_ATOL_WH},
        "max_measured_phase_sum_error_Wh": max(energy_closures),
        "max_measured_phase_sum_error_s": max(time_closures),
        "same_campaign_external_to_simulator": True, "newly_fit_simulator_parameters": 0,
        "diagnostic_preflight_regression": preflight_reference(measured),
        "independent_aircraft_or_campaign": False, "ambient_wind_ground_truth": False,
        "density": "per-flight retrospective KAGC METAR proxy from the frozen publication-protocol reconstruction",
        "physical_operating_scope": "CMU research configuration, 3.680 kg plus payload, above DJI published 3.6 kg MTOW",
        "holdout": metrics[(metrics.split == "test") & (metrics.stratum == "all")].to_dict("records"),
    }
    destination.mkdir(parents=True, exist_ok=True)
    context_copy = destination / "published_context.csv"
    if context_path.resolve() != context_copy.resolve():
        shutil.copyfile(context_path, context_copy)
    for filename, frame in [("flight_manifest.csv", manifest), ("phase_comparisons.csv", table),
                            ("grouped_errors.csv", metrics), ("measured_cruise.csv", measured)]:
        write_table(frame, destination / filename)
    (destination / "validation.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=INPUT)
    parser.add_argument("--context", type=Path, default=CONTEXT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rebuild-context-from", type=Path,
                        help="reconstruct the compact reference from pinned local reproduction CSVs")
    args = parser.parse_args()
    if args.rebuild_context_from:
        rebuild_context(args.rebuild_context_from, args.context)
    result = evaluate(args.dataset, args.context, args.output)
    print(json.dumps({k: result[k] for k in ("actual_recordings", "included_complete_flights",
                                            "measured_train_validation_test", "holdout")}, indent=2))


if __name__ == "__main__":
    main()
