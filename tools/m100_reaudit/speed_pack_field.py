"""Reproduce measured CMU pack-power targets for the native cruise-speed check.

Run from the thesis root: python3 -m tools.m100_reaudit.speed_pack_field
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import transient_cmu
from .evaluate_gong_cruise import digest, write_csv
from .pack_drive_cmu_context import gnss_track


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/m100_reaudit"
WINDOWS = ("locked_stable", "straight_level")
SPEEDS = (4, 5, 6, 8, 10, 12, 15, 20)
MAX_GAP_S = 0.5
MIN_ELIGIBLE_S = 20.0


def level_mask(speed, height, vertical, command, target):
    return ((np.abs(speed - command) <= max(0.5, 0.1 * command))
            & (np.abs(height - target) <= max(3.0, 0.1 * target))
            & (np.abs(vertical) <= 1.5))


def straight_level_mask(time_s, ground_speed_mps, altitude_agl_m, vertical_speed_mps,
                        east_m, north_m, command_mps, target_agl_m):
    """Original level/speed gate plus the project's one-second acceleration and track guards."""
    t, speed, height, vertical, east, north = (
        np.asarray(col, dtype=float) for col in
        (time_s, ground_speed_mps, altitude_agl_m, vertical_speed_mps, east_m, north_m))
    if not all(arr.shape == t.shape and np.isfinite(arr).all() for arr in
               (t, speed, height, vertical, east, north)) or np.any(np.diff(t) <= 0):
        raise ValueError("Kinematics must be aligned, finite and in strictly increasing time order")
    good = level_mask(speed, height, vertical, command_mps, target_agl_m)
    policy = transient_cmu.POLICY
    good &= np.abs(vertical) <= policy["max_abs_vertical_up_mps"]
    # Restart interpolation on either side of missing telemetry. GNSS gives two
    # consecutive ground-track segments without relying on body yaw or airspeed.
    for segment in np.split(np.arange(len(t)), np.flatnonzero(np.diff(t) > MAX_GAP_S) + 1):
        if not len(segment):
            continue
        ts, sp = t[segment], speed[segment]
        lag = policy["causal_accel_lag_s"]
        along = (sp - np.interp(ts - lag, ts, sp)) / lag
        resolved, turning = transient_cmu.heading_filter(ts, east[segment], north[segment])
        good[segment] &= ((ts >= ts[0] + lag) & (np.abs(along) < policy["small_accel_mps2"])
                          & resolved & ~turning)
    return good


def integrate(t, speed, power, mask):
    dt = np.diff(t)
    use = mask[:-1] & mask[1:] & (dt > 0) & (dt <= MAX_GAP_S)
    weights = dt[use]
    duration = float(weights.sum())
    if not duration:
        return dict(eligible_s=0.0, intervals=0, mean_W=None, mean_speed_mps=None)

    def average(values):
        return float(np.dot(0.5 * (values[:-1][use] + values[1:][use]), weights) / duration)

    return dict(eligible_s=duration, intervals=int(use.sum()), mean_W=average(power),
                mean_speed_mps=average(speed))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measured_rows(project):
    # The existing cross-check independently rebuilt power from V x I, replayed
    # the locked selector on all 170 original flights, and recorded input hashes.
    archived = RESULTS / "gong_cruise"
    evidence = read_json(archived / "evaluation.json")["inputs"]
    paths = {}
    for key in ("samples", "split", "summary", "selector", "official_sample", "qc"):
        path = project / evidence[key]["path"]
        if digest(path) != evidence[key]["sha256"]:
            raise ValueError(f"Changed CMU reference input: {path}")
        paths[key] = path
    protocol = read_json(paths["split"])
    stable = protocol["stable_cruise"]
    fit, held = (set(protocol[key]) for key in ("fit_flight_ids", "evaluation_flight_ids"))
    locked_fit, locked_held = (set(stable[key]) for key in ("fit_flight_ids", "evaluation_flight_ids"))
    published_fit = set(pd.read_csv(paths["official_sample"]).flight)
    if fit & held or fit != (fit | held) & published_fit or not locked_fit <= fit or not locked_held <= held:
        raise ValueError("Unexpected overlap or drift in the frozen CMU partitions")
    model_fit_sets = {
        "stable_speed": locked_fit,
        "aircraft_power_and_loaded_voltage": set(read_json(RESULTS / "cmu_calibration/fit.json")["cohorts"]["train"]["flight_ids"]),
        "native_thevenin": set(read_json(RESULTS / "mission_150m/battery_fit.json")["flight_ids"]["train"]),
        "manoeuvre": set(read_json(RESULTS / "transient_airborne/model.json")["fit_ids"]),
    }
    if any(held & ids for ids in model_fit_sets.values()):
        raise ValueError("CMU evaluation rows overlap a training set")
    with (archived / "flights.csv").open(newline="") as stream:
        locked_rows = {int(r["flight_id"]): r for r in csv.DictReader(stream)}
    if set(locked_rows) != locked_fit | locked_held:
        raise ValueError("Archived stable flight IDs differ from the original protocol")
    summary = pd.read_csv(paths["summary"]).set_index("flight")
    if not summary.index.is_unique:
        raise ValueError("Duplicate flight IDs in the published phase summary")
    columns = ("flight_id", "time_s", "voltage_v", "current_a", "power_real_signed_w", "longitude_deg",
               "latitude_deg", "altitude_asl_m", "actual_horizontal_speed_mps", "velocity_z_mps",
               "commanded_speed_mps", "target_altitude_m", "payload_g", "date")
    samples = pd.read_csv(paths["samples"], usecols=columns)
    rows = []
    for fid, samples_f in samples[samples.flight_id.isin(fit | held)].groupby("flight_id"):
        fid = int(fid)
        meta = summary.loc[fid]
        command, height, payload = float(meta["speed"]), float(meta["altitude"]), float(meta["payload"] / 1000)
        for key, expected in (("commanded_speed_mps", command), ("target_altitude_m", height), ("payload_g", 1000 * payload)):
            if not samples_f[key].eq(expected).all():
                raise ValueError(f"Field command differs from summary for {fid}: {key}")
        t = samples_f.time_s.to_numpy()
        power = (samples_f.voltage_v * samples_f.current_a).to_numpy()
        if not np.allclose(power, samples_f.power_real_signed_w, rtol=1e-7, atol=1e-7):
            raise ValueError(f"Signed battery power differs from V x I in flight {fid}")
        speed = samples_f.actual_horizontal_speed_mps.to_numpy()
        vertical = samples_f.velocity_z_mps.to_numpy()
        z = samples_f.altitude_asl_m.to_numpy()
        agl = z - np.median(z[t <= 5])
        locked = integrate(t, speed, power, level_mask(speed, agl, vertical, command, height))
        eligible = locked["eligible_s"] >= MIN_ELIGIBLE_S
        if eligible != (fid in locked_rows):
            raise ValueError(f"Published >=20-second cohort changed for flight {fid}")
        if eligible:
            old = locked_rows[fid]
            if not np.allclose([locked["eligible_s"], locked["mean_W"], locked["intervals"], locked["mean_speed_mps"]],
                               [float(old[k]) for k in ("duration_s", "measured_W", "eligible_intervals", "mean_ground_speed_mps")],
                               rtol=1e-11, atol=1e-8):
                raise ValueError(f"Published stable-flight mean/duration differs for {fid}")
        _, east, north = gnss_track(samples_f.rename(columns={"longitude_deg": "position_x", "latitude_deg": "position_y"}))
        straight = integrate(t, speed, power, straight_level_mask(t, speed, agl, vertical, east, north, command, height))
        row = dict(flight_id=fid, split="fit" if fid in fit else "evaluation", date=str(samples_f.date.iloc[0]),
                   payload_kg=payload, altitude_m=height, command_mps=command,
                   whole_cruise_phase_W=meta["Power_cruise"], whole_cruise_phase_s=meta["time_cruise"])
        for window, stats in zip(WINDOWS, (locked, straight), strict=True):
            qualified = stats["eligible_s"] >= MIN_ELIGIBLE_S
            row[f"{window}_eligible"] = qualified
            for key, value in stats.items():
                row[f"{window}_{key}"] = value if qualified or key in ("intervals", "eligible_s") else None
        rows.append(row)
    if {r["flight_id"] for r in rows} != fit | held:
        raise ValueError("Missing CMU source flights")
    return rows, protocol, {key: evidence[key] for key in paths}


def speed_support(rows):
    counts = []
    for speed in SPEEDS:
        row = {"command_mps": speed}
        for split in ("fit", "evaluation"):
            selected = [r for r in rows if r["command_mps"] == speed and r["split"] == split]
            row[f"whole_phase_{split}_flights"] = len(selected)
            for window in WINDOWS:
                row[f"{window}_{split}_flights"] = sum(r[f"{window}_eligible"] for r in selected)
            row[f"max_locked_selected_{split}_s"] = max((r["locked_stable_eligible_s"] for r in selected), default=None)
        counts.append(row)
    return counts


def field_target(rows, speed, payload, altitude, window):
    selected = [r for r in rows if r["split"] == "evaluation" and r["command_mps"] == speed
                and r["payload_kg"] == payload and r["altitude_m"] == altitude and r[f"{window}_eligible"]]
    watts = [r[f"{window}_mean_W"] for r in selected]
    return dict(command_mps=speed, payload_kg=payload, altitude_m=altitude, window=window,
                field_n=len(selected), field_flight_ids=" ".join(str(r["flight_id"]) for r in selected),
                field_mean_W=float(np.mean(watts)) if watts else None,
                field_min_W=min(watts, default=None), field_max_W=max(watts, default=None),
                field_mean_groundspeed_mps=float(np.mean([r[f"{window}_mean_speed_mps"] for r in selected])) if watts else None)


def native_comparison(rows, out):
    """Join audited on/off pairs to the independent field flight IDs for the identical window."""
    protocol = read_json(out / "protocol.json")
    with (out / "native_flights.csv").open(newline="") as stream:
        native_rows = list(csv.DictReader(stream))
    native = {r["case_id"]: r for r in native_rows}
    if len(native) != len(native_rows) or set(native) != {r["case_id"] for r in protocol["cases"]}:
        raise ValueError("Comparison requires exactly one accepted native flight per registered case")
    pairs = {}
    for case in protocol["cases"]:
        r = native[case["case_id"]]
        if any(r[k] != case[k] for k in ("pair_id", "mode")) or any(
            float(r[k]) != case[k] for k in ("seed", "payload_kg", "altitude_m", "cruise_speed_mps", "wind_speed_mps", "temperature_c")
        ):
            raise ValueError(f"Native case differs from registered conditions: {case['case_id']}")
        modes = pairs.setdefault(r["pair_id"], {})
        if r["mode"] in modes:
            raise ValueError(f"Duplicate native treatment in pair {r['pair_id']}")
        modes[r["mode"]] = r
    output, individual = [], []
    for pair, modes in sorted(pairs.items()):
        if modes.keys() != {"on", "off"}:
            raise ValueError(f"Missing comparable physical on/off records for {pair}")
        on, off = modes["on"], modes["off"]
        conditions = ("payload_kg", "altitude_m", "cruise_speed_mps", "seed", "wind_speed_mps", "temperature_c")
        if any(float(on[k]) != float(off[k]) for k in conditions):
            raise ValueError(f"Native pair {pair} changed conditions")
        speed, payload, altitude = (float(on[k]) for k in ("cruise_speed_mps", "payload_kg", "altitude_m"))
        for window, prefix in zip(WINDOWS, ("stable", "strict"), strict=True):
            target = field_target(rows, speed, payload, altitude, window)
            baseline = dict(pair_id=pair, **target, wind_speed_sim_mps=float(on["wind_speed_mps"]),
                            temperature_sim_c=float(on["temperature_c"]), seed=int(on["seed"]))
            enough = True
            for mode, record in modes.items():
                baseline[f"native_{mode}_case"] = record["case_id"]
                duration_s = float(record[f"{prefix}_duration_s"])
                raw_watts = record[f"{prefix}_pack_W"]
                if not np.isfinite(duration_s) or duration_s < 0 or (raw_watts and not np.isfinite(float(raw_watts))):
                    raise ValueError(f"Invalid native measurement in {record['case_id']}, {window}")
                baseline[f"native_{mode}_duration_s"] = duration_s
                baseline[f"native_{mode}_W"] = float(raw_watts) if raw_watts else None
                for quantity in ("mean_groundspeed_mps", "applied_time_fraction"):
                    raw = record[f"{prefix}_{quantity}"]
                    baseline[f"native_{mode}_{quantity}"] = float(raw) if raw else None
                enough &= duration_s >= MIN_ELIGIBLE_S - 1e-8 and bool(raw_watts)
            before, after = baseline["native_off_W"], baseline["native_on_W"]
            baseline["native_on_minus_off_W"] = after - before if None not in (before, after) else None
            baseline["comparison_status"] = ("NO_ELIGIBLE_FIELD_FLIGHT" if not target["field_n"] else
                                             "INSUFFICIENT_NATIVE_DURATION" if not enough else "MATCHED")
            residuals = {"on": [], "off": []}
            if baseline["comparison_status"] == "MATCHED":
                ids = {int(s) for s in target["field_flight_ids"].split()}
                for field in (r for r in rows if r["flight_id"] in ids):
                    watts = field[f"{window}_mean_W"]
                    detail = dict(pair_id=pair, command_mps=speed, payload_kg=payload, altitude_m=altitude,
                                  window=window, field_flight_id=field["flight_id"], date=field["date"],
                                  field_eligible_duration_s=field[f"{window}_eligible_s"], field_mean_W=watts)
                    for mode in ("off", "on"):
                        native_watts = baseline[f"native_{mode}_W"]
                        detail[f"native_{mode}_W"] = native_watts
                        detail[f"{mode}_signed_error_W"] = native_watts - watts
                        absolute = abs(native_watts - watts)
                        detail[f"{mode}_abs_error_W"] = absolute
                        residuals[mode].append(absolute)
                    individual.append(detail)
            for mode in ("off", "on"):
                errors = residuals[mode]
                baseline[f"{mode}_abs_error_to_field_mean_W"] = (
                    abs(baseline[f"native_{mode}_W"] - target["field_mean_W"]) if errors else None)
                baseline[f"{mode}_mean_per_flight_abs_error_W"] = float(np.mean(errors)) if errors else None
            for metric in ("abs_error_to_field_mean_W", "mean_per_flight_abs_error_W"):
                baseline[f"on_minus_off_{metric}"] = (
                    baseline[f"on_{metric}"] - baseline[f"off_{metric}"] if residuals["on"] else None)
            output.append(baseline)
    write_csv(out / "comparison.csv", output)
    detail_fields = ("pair_id", "command_mps", "payload_kg", "altitude_m", "window", "field_flight_id", "date",
                     "field_eligible_duration_s", "field_mean_W", "native_off_W", "off_signed_error_W",
                     "off_abs_error_W", "native_on_W", "on_signed_error_W", "on_abs_error_W")
    with (out / "field_per_flight_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=detail_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(individual)


def run(project, out, payload, altitude, join_native=False):
    rows, protocol, evidence = measured_rows(project)
    out.mkdir(exist_ok=True, parents=True)
    write_csv(out / "field_heldout_flights.csv", [r for r in rows if r["split"] == "evaluation"])
    write_csv(out / "field_speed_support.csv", speed_support(rows))
    write_csv(out / "field_targets.csv", [field_target(rows, speed, payload, altitude, w) for speed in SPEEDS for w in WINDOWS])
    receipt = dict(
        dataset_doi=protocol["dataset_doi"], published_split_and_phases_doi=protocol["phase_processing_and_fixed_flight_ids_doi"],
        field_study="https://pmc.ncbi.nlm.nih.gov/articles/PMC8213793/",
        sources_relative_to="--project", sources=evidence,
        development_flights=len(protocol["fit_flight_ids"]), evaluation_flights=len(protocol["evaluation_flight_ids"]),
        same_field_campaign=True, evaluation_flights_used_for_speed_fit=False,
        overlap_with_cruise_aircraft_voltage_2rc_or_manoeuvre_fit_ids=0, comparison_payload_kg=payload,
        comparison_target_agl_m=altitude,
        locked_stable="Each endpoint requires absolute speed error <= max(0.5 m/s, 10% command), absolute height error <= "
                      "max(3 m, 10% command), abs(vz) <=1.5 m/s. Field AGL uses median ASL in first five seconds. "
                      "All eligible intervals are summed per flight. The 20 s may comprise disjoint intervals.",
        straight_level="The nested sensitivity uses the project transient_cmu movement and ground-track guards. "
                       "Each endpoint requires |vz| <= 0.8 m/s, "
                       "|speed(t)-speed(t-1 s)|/1 s <0.30 m/s2, two resolved 0.75 s GNSS ground-track legs of "
                       "at least 1 m with bend <=20 degrees. Kinematic history resets across gaps >0.5 s. "
                       "Identical >=20 s eligibility and quadrature. Guard uses no power or field-fit outcome.",
        integration="Trapezoidal signed V*I on adjacent original times with both endpoints selected and 0<dt<=0.5 s. "
                    "Divide the sum by the selected duration. Each held-out flight has equal weight in the field cell. "
                    "The comparison measures mean electrical power over the eligible intervals.",
        native_window="Apply the corresponding kinematic mask to native ground truth inside the audited outbound "
                      "measurement phase, plus a fresh acknowledged native pack ledger, bounded synchronized telemetry, "
                      "and adjacent native pack packets <=0.30 s apart. Require >=20 s total in each treatment.",
        conditions="Exact nominal payload, AGL command and speed command. Wind, ambient temperature, route, starting "
                   "SOC and density are unmatched across field and simulator flights. The aircraft-carried wind "
                   "sensor measures relative airflow. This is a same-campaign check with sparse exact cells.",
        whole_phase="The held-out table includes the published whole cruise-phase mean for context. It includes "
                    "changes of heading and speed. The steady comparison uses eligible kinematic intervals.",
        error_comparison="Absolute error to the field cell mean is |Pnative - mean(Pflight)|. "
                         "Per-flight MAE is mean(|Pnative-Pflight|) across the identical field IDs. "
                         "Negative on-minus-off errors mean closer agreement in these observations.",
        reproduction="python3 -m tools.m100_reaudit.speed_pack_field --project <CMU project root>",
        reproduction_with_native="Add --join-native after the native audit has written all accepted flights.",
    )
    if join_native:
        native_comparison(rows, out)
        receipt["join_sha256"] = {f: digest(out / f) for f in (
            "protocol.json", "native_flights.csv", "field_targets.csv", "field_heldout_flights.csv", "comparison.csv",
            "field_per_flight_errors.csv")}
    (out / "field_protocol.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project", type=Path, default=Path.home() / "Master/Drone/project")
    cli.add_argument("--out", type=Path, default=RESULTS / "speed_pack_link")
    cli.add_argument("--payload", type=float, default=0.25)
    cli.add_argument("--altitude", type=float, default=50.0)
    cli.add_argument("--join-native", action="store_true")
    args = cli.parse_args()
    run(args.project, args.out, args.payload, args.altitude, args.join_native)
