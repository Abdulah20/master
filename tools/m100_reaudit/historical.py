"""Audit the original accepted M100 export against its acquisition records.

No simulator, middleware or spreadsheet library is required. The recorded
internal integral and power quadrature are independent code paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

SUMMARY_NAME = "M100_69_experiments_summary.csv"
ORIGINAL_SHA256 = "ea24ff674b74940b2f1ffaf12fcee49164eb5180f895ab0bdcf9bda1b060028c"
INPUTS = ("payload_kg", "cruise_speed_mps", "wind_speed_mps",
          "wind_direction_toward_deg_ENU", "altitude_m", "temperature_c")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_close(value, expected, name, tolerance=1e-8):
    require(math.isclose(value, expected, rel_tol=0, abs_tol=tolerance),
            f"{name}: observed {value}, expected {expected} ± {tolerance}")


def csv_table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trapezoid(timestamps, values):
    require(len(timestamps) == len(values) >= 2, "At least two paired observations required")
    durations = [b - a for a, b in zip(timestamps, timestamps[1:])]
    require(all(dt > 0 for dt in durations), "Non-increasing timestamps")
    return math.fsum(dt * (a + b) / 2 for dt, a, b in zip(durations, values, values[1:]))


def original_study(study=None):
    if study:
        source = Path(study).expanduser()
        return source.resolve().parent if source.is_file() else source.resolve()
    locations = (Path.home() / "Master/Drone/project/outputs", Path.cwd() / "outputs")
    for directory in locations:
        if directory.is_dir():
            matches = [p for p in sorted(directory.glob("*/" + SUMMARY_NAME)) if digest(p) == ORIGINAL_SHA256]
            if matches:
                return matches[0].parent.resolve()
    raise FileNotFoundError(f"Supply --study with the directory containing {SUMMARY_NAME}")


def safe_destination(out, protected):
    out = out.resolve()
    require(all(not out.is_relative_to(p.resolve()) for p in protected), "Output overlaps original study")
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_study(study):
    summary = study / SUMMARY_NAME
    sidecar = load_json(study / (SUMMARY_NAME + ".export.json"))
    check_paths = {summary: ORIGINAL_SHA256, study / "results.json": sidecar["results_sha256"],
                   study / "protocol.json": sidecar["protocol_sha256"]}
    for path, expected in check_paths.items():
        require(digest(path) == expected, f"Original hash mismatch: {path}")
    require(sidecar["csv_sha256"] == ORIGINAL_SHA256, "Export identity mismatch")
    fields, summary_rows = csv_table(summary)
    require(fields == sidecar["columns"], "Export headers changed")
    protocol = load_json(study / "protocol.json")
    records = load_json(study / "results.json")
    require(len(set(row["run_id"] for row in summary_rows)) == len(summary_rows), "Duplicate run IDs")
    for case, record, row in zip(protocol["cases"], sidecar["sources"], summary_rows, strict=True):
        identifier = case["case_id"]
        selected = records[identifier]
        require(selected["accepted"] and record["case_id"] == identifier, f"Unselected case {identifier}")
        require(record["run_id"] == row["run_id"], f"Flight identity mismatch for {identifier}")
        require(record["source_sha256"] == selected["source_sha256"], f"Flight hash mismatch for {identifier}")
        for key, observed in row.items():
            expected = selected["metrics"][key]
            if isinstance(expected, (int, float)):
                # Both are round-trip decimal encodings of binary64 in the exporter.
                require(float(observed) == expected, f"{identifier} export mismatch for {key}")
            else:
                require(observed == (expected or ""), f"{identifier} export mismatch for {key}")
    return protocol, records, sidecar


def flight_sources(study, record, flight_dir=None):
    name = Path(record["source_csv"]).name
    candidate = (flight_dir or study.parent / "m100_csv") / name
    if not candidate.exists() and flight_dir is None:
        candidate = Path(record["source_csv"])
    require(digest(candidate) == record["source_sha256"], f"Flight CSV changed: {name}")
    provenance_path = candidate.with_suffix(".provenance.json")
    meta = load_json(provenance_path)
    require(meta["csv"]["sha256"] == record["source_sha256"], f"Unbound metadata: {name}")
    require(meta["run_id"] == record["metrics"]["run_id"], f"Different run in metadata: {name}")
    return candidate, provenance_path, meta


def sampled_window(rows, phase):
    indices = [i for i, row in enumerate(rows) if row["flight_phase"] == phase]
    require(len(indices) >= 2 and indices == list(range(indices[0], indices[-1] + 1)),
            "Missing or interrupted measurement phase")
    segment = rows[indices[0]:indices[-1] + 1]
    t = [float(row["energy_timestamp_s"]) for row in segment]
    p = [float(row["electrical_power_W"]) for row in segment]
    energy_ws = trapezoid(t, p)
    first, last = segment[0], segment[-1]
    energy_Wh = energy_ws / 3600
    counter_change = float(last["measurement_energy_Wh"]) - float(first["measurement_energy_Wh"])
    disp = None
    if phase == "CRUISE":
        disp = float(last["measurement_distance_m"]) - float(first["measurement_distance_m"])
    return dict(sampled_first_s=t[0], sampled_last_s=t[-1], sampled_duration_s=t[-1] - t[0],
                aligned_mean_power_W=energy_ws / (t[-1] - t[0]),
                aligned_1Hz_energy_Wh=energy_Wh, aligned_high_rate_counter_change_Wh=counter_change,
                aligned_1Hz_minus_high_rate_Wh=energy_Wh - counter_change,
                counter_at_first_sample_Wh=float(first["measurement_energy_Wh"]),
                counter_at_last_sample_Wh=float(last["measurement_energy_Wh"]),
                aligned_outbound_distance_m=disp, samples=len(segment))


def audit_case(study, case, record, flight_dir=None):
    identifier = case["case_id"]
    source, provenance_path, meta = flight_sources(study, record, flight_dir)
    fields, rows = csv_table(source)
    m, energy, sampling = record["metrics"], meta["energy_summary"], meta["csv"]["sampling"]
    require(fields == meta["csv"]["fields"] and len(fields) == 18, "Unexpected flight schema")
    times = [float(row["energy_timestamp_s"]) for row in rows]
    require(all(b > a for a, b in zip(times, times[1:])), "Disordered original recording")
    slots = [math.floor((t - sampling["origin_energy_timestamp_s"]) / sampling["interval_s"] + 1e-9)
             for t in times]
    require(sampling["interval_s"] == 1 and sampling["missed_windows"] == 0 and
            slots == list(range(len(rows))), "Original sampling gaps")
    for field in INPUTS:
        raw_key = "wind_direction_deg" if field == "wind_direction_toward_deg_ENU" else field
        check_close(float(m[field]), float(case[field]), field)
        check_close(float(meta["config"][raw_key]), float(case[field]), field)
    require(energy["airborne_energy_coverage"]["complete"] and meta["recording_end"]["landed"]
            and not meta["recording_end"]["armed"] and meta["recording_end"]["outcome"] == "COMPLETE",
            f"Original outcome for {identifier} violates the archived admission criteria")
    check_close(energy["airborne_energy_coverage"]["power_time_coverage_percent"], 100, "airborne coverage")
    require(energy["wind_power_runtime_applied"] == (case["wind_speed_mps"] > 0), "Wind dispatch mismatch")
    require(energy["wind_power_empirically_calibrated"] is False, "Unexpected wind calibration status")
    require(energy["is_full_flight_energy"] is False, "Unsupported full flight claim")
    phase = "CRUISE" if case["cruise_speed_mps"] else "HOVER"
    observations = [r for r in rows if r["flight_phase"] == phase]
    sample = sampled_window(rows, phase)
    check_close(sample["aligned_mean_power_W"], m["sampled_mean_power_W"], "sampled mean")
    check_close(sample["sampled_duration_s"], m["sampled_measurement_span_s"], "sampled span")
    target_energy = energy["measurement_window_energy_wh"]
    check_close(target_energy, m["measurement_energy_Wh"], "original target")
    mission = energy["integrated_supported_phase_energy_wh"]
    phase_sum = math.fsum(energy["phase_energy_by_phase_wh"].values())
    check_close(phase_sum, mission, "phase energy sum")
    check_close(mission, m["supported_phase_mission_energy_Wh"], "supported energy")
    nominal = record["nominal_energy_Wh"]
    check_close(m["ideal_steady_endurance_min"], 60 * nominal / sample["aligned_mean_power_W"], "endurance")
    check_close(m["nominal_energy_remaining_percent"], 100 * max(0., nominal - mission) / nominal, "budget")
    if phase == "CRUISE":
        check_close(m["energy_per_distance_Wh_per_km"], 1000 * target_energy / m["measurement_distance_m"],
                    "distance efficiency")
        check_close(m["ideal_steady_range_km"], nominal / m["energy_per_distance_Wh_per_km"], "range")
    else:
        require(all(m[k] is None for k in ("ideal_steady_range_km", "measurement_distance_m",
                                           "energy_per_distance_Wh_per_km")), "Undefined hover fill")
    vi_errors, missing = [], 0
    for row in rows:
        if not row["electrical_power_W"]:
            continue
        v, i = row["battery_voltage_V"], row["battery_current_A"]
        require(bool(v) == bool(i), "Broken V/I pair")
        if v:
            vi_errors.append(abs(float(row["electrical_power_W"]) - float(v) * float(i)))
        else:
            missing += 1
    vi_error = max(vi_errors, default=0.)
    check_close(vi_error, 0., "power = VI where available", 1e-7)
    require(m["data_quality"] == ("PASS_ENERGY_VI_PARTIAL" if missing else "PASS"), "VI label mismatch")
    difference = target_energy - sample["aligned_1Hz_energy_Wh"]
    speed_error = max(abs(float(r["actual_horizontal_speed_mps"]) - case["cruise_speed_mps"]) for r in observations)
    height_error = max(abs(float(r["actual_altitude_m"]) - case["altitude_m"]) for r in observations)
    require(speed_error <= max(.3, .05 * case["cruise_speed_mps"]) and height_error <= .5,
            f"Original stabilization criteria violated by {identifier}")
    atmosphere = meta["atmosphere"]["case"]
    calm_base_error = None
    if phase == "CRUISE" and not case["wind_speed_mps"]:
        mass = energy["published_regression_mass_kg"]
        formula = 68.9 * mass**1.5 / math.sqrt(atmosphere["density_kg_m3"]) + 16.8
        calm_base_error = sample["aligned_mean_power_W"] - formula
        check_close(calm_base_error, 0, "published cruise equation")
    result = dict(case_id=identifier, run_id=m["run_id"], **{k: case[k] for k in INPUTS},
                  original_energy_evidence_status=m["energy_evidence_status"], original_data_quality=m["data_quality"],
                  physical_mass_kg=energy["physical_total_mass_kg"], regression_mass_kg=energy["published_regression_mass_kg"],
                  datum_temperature_c=atmosphere["temperature_c_at_datum"], fixed_density_kg_m3=atmosphere["density_kg_m3"],
                  nominal_Wh=nominal, original_measurement_energy_Wh=target_energy,
                  original_measurement_distance_m=m["measurement_distance_m"],
                  original_supported_mission_Wh=mission, original_phase_sum_error_Wh=phase_sum - mission,
                  **sample, after_last_sample_Wh=target_energy - sample["counter_at_last_sample_Wh"],
                  original_minus_aligned_1Hz_Wh=difference,
                  mismatch_percent_original=100 * difference / target_energy,
                  mismatch_equivalent_s=3600 * difference / sample["aligned_mean_power_W"],
                  max_abs_speed_error_mps=speed_error, max_abs_height_error_m=height_error,
                  calm_base_formula_error_W=calm_base_error, source_rows=len(rows),
                  paired_VI_rows=len(vi_errors), missing_VI_rows=missing, max_abs_VI_error_W=vi_error,
                  source_csv_sha256=digest(source), source_provenance_sha256=digest(provenance_path))
    return result


def run(study, out, flight_dir=None):
    protocol, records, manifest = read_study(study)
    out = safe_destination(out, [study, flight_dir or study.parent / "m100_csv"])
    cases = [audit_case(study, case, records[case["case_id"]], flight_dir) for case in protocol["cases"]]
    winds = [r for r in cases if r["wind_speed_mps"] > 0]
    def stats(key):
        values = [r[key] for r in cases]
        return dict(min=min(values), mean=mean(values), max=max(values))
    summary = dict(source_csv_sha256=ORIGINAL_SHA256, source_protocol_sha256=manifest["protocol_sha256"],
                   source_results_sha256=manifest["results_sha256"], cases=len(cases), columns=len(manifest["columns"]),
                   source_cells_checked=len(cases) * len(manifest["columns"]), original_rows=sum(c["source_rows"] for c in cases),
                   logical_rows=sum(map(len, protocol["groups"].values())),
                   excluded_original_attempts=sum(len(r.get("previous_attempts", [])) for r in records.values()),
                   temperature_c_counts=Counter(c["temperature_c"] for c in cases),
                   altitude_m_counts=Counter(c["altitude_m"] for c in cases), wind_cases=len(winds),
                   wind_by_command_speed=Counter(c["cruise_speed_mps"] for c in winds),
                   evidence=Counter(c["original_energy_evidence_status"] for c in cases),
                   quality=Counter(c["original_data_quality"] for c in cases),
                   mismatched_window_percent=stats("mismatch_percent_original"), equivalent_seconds=stats("mismatch_equivalent_s"),
                   max_abs_aligned_1Hz_minus_high_rate_Wh=max(abs(c["aligned_1Hz_minus_high_rate_Wh"]) for c in cases),
                   max_abs_phase_sum_error_Wh=max(abs(c["original_phase_sum_error_Wh"]) for c in cases),
                   nominal_energy_values_Wh=sorted({c["nominal_Wh"] for c in cases}),
                   paired_VI_rows=sum(c["paired_VI_rows"] for c in cases), missing_VI_rows=sum(c["missing_VI_rows"] for c in cases),
                   max_abs_VI_error_W=max(c["max_abs_VI_error_W"] for c in cases),
                   max_abs_speed_error_mps=max(c["max_abs_speed_error_mps"] for c in cases),
                   max_abs_height_error_m=max(c["max_abs_height_error_m"] for c in cases),
                   hover_cases=sum(c["cruise_speed_mps"] == 0 for c in cases),
                   calm_max_abs_formula_error_W=max(abs(c["calm_base_formula_error_W"]) for c in cases if c["calm_base_formula_error_W"] is not None))
    write_csv(out / "per_case.csv", cases)
    (out / "audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, help="Original study directory or original 69-row CSV")
    parser.add_argument("--flight-dir", type=Path, help="Directory of original raw CSV and provenance pairs, if relocated")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "results/m100_reaudit/historical")
    args = parser.parse_args()
    print(json.dumps(run(original_study(args.study), args.out, args.flight_dir), indent=2))
