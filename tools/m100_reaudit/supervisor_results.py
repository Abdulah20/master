"""Audit each closed M100 supervisor acquisition and preserve its original energy labels."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import yaml

from . import direct_airborne_evidence as evidence


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "tmp/m100_reaudit_supervisor_rerun"
OUT = ROOT / "results/m100_reaudit/supervisor_rerun"


def measurements(case, flight_path, provenance):
    """Apply the original 69-case CSV definitions to the new recorder data."""
    energy = provenance["energy_summary"]
    battery = provenance["battery_budget"]
    with flight_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    phase = "CRUISE" if case["cruise_speed_mps"] else "HOVER"
    points = [(float(r["energy_timestamp_s"]), float(r["electrical_power_W"]))
              for r in rows if r["flight_phase"] == phase and r["electrical_power_W"]]
    span = points[-1][0] - points[0][0] if len(points) >= 2 else None
    mean = (math.fsum((a[1] + b[1]) * 0.5 * (b[0] - a[0]) for a, b in zip(points, points[1:])) / span
            if span and span > 0 else None)
    segment = energy["measurement_window_energy_wh"]
    distance = energy["measurement_window_distance_m"] if phase == "CRUISE" else None
    rate = energy["measurement_energy_per_distance_wh_per_km"] if distance else None
    if rate is not None:
        evidence.confirm(math.isclose(rate, 1000 * segment / distance, rel_tol=1e-9), "Segment Wh/km mismatch")
    mission = energy["integrated_supported_phase_energy_wh"]
    evidence.confirm(math.isclose(battery["consumed_supported_phase_energy_wh"], mission, abs_tol=1e-8),
                     "Legacy budget and supported-phase energy disagree")
    available = missing = 0
    for row in rows:
        if not row["electrical_power_W"]:
            continue
        voltage, current = row["battery_voltage_V"], row["battery_current_A"]
        evidence.confirm(bool(voltage) == bool(current), "Unpaired manager terminal voltage/current")
        if voltage:
            evidence.confirm(math.isclose(float(voltage)*float(current), float(row["electrical_power_W"]),
                                          abs_tol=1e-7), "Manager sample P differs from V times I")
            available += 1
        else:
            missing += 1
    config_source = provenance["source_files"]["m100_energy_v2_config"]
    canonical = ROOT / "tools/m100_reaudit/runtime_snapshot/ros2_ws/src/drone_nav/config/m100_energy_model_v2.yaml"
    evidence.confirm(evidence.digest(canonical) == config_source["sha256"], "Nameplate configuration mismatch")
    nominal = float(yaml.safe_load(canonical.read_bytes())["battery"]["nominal_energy_Wh"])
    sampling = provenance["csv"]["sampling"]
    origin, interval = sampling["origin_energy_timestamp_s"], sampling["interval_s"]
    windows = [math.floor((float(r["energy_timestamp_s"]) - origin) / interval + 1e-9) for r in rows]
    coverage = energy["airborne_energy_coverage"]
    end = provenance["recording_end"]
    legacy_complete = (end["outcome"] == "COMPLETE" and end["landed"] and not end["armed"] and coverage["complete"]
                       and math.isclose(coverage["power_time_coverage_percent"], 100) and interval == 1
                       and sampling["missed_windows"] == 0 and windows == list(range(len(windows))))
    quality = ("PASS_ENERGY_VI_PARTIAL" if missing else "PASS") if legacy_complete else "EXCLUDE_GAPS_OR_INCOMPLETE"
    return {
        "measurement_energy_Wh": segment, "sampled_mean_power_W": mean, "energy_per_distance_Wh_per_km": rate,
        "supported_phase_mission_energy_Wh": mission, "ideal_steady_endurance_min": 60*nominal/mean if mean else None,
        "ideal_steady_range_km": nominal/rate if rate else None,
        "nominal_energy_remaining_percent": 100 * max(0, 1 - mission/nominal),
        "measurement_distance_m": distance, "sampled_measurement_span_s": span,
        "measurement_frozen": energy["measurement_frozen"], "nameplate_Wh": nominal,
        "manager_power_rows_with_vi": available, "manager_power_rows_without_vi": missing,
        "mean_segment_power_definition": "TIME_WEIGHTED_ONE_HZ_MEASUREMENT_PHASE",
        "legacy_wind_power_runtime_applied": energy["wind_power_runtime_applied"],
        "data_quality": quality, "energy_evidence_status": provenance["config"]["energy_calibration_status"],
    }


def horizontal_path(ulog, start_us, end_us):
    """Linearly clip simulated ground-truth local positions to the PX4 airborne gate."""
    import numpy as np
    from pyulog import ULog

    topic = "vehicle_local_position_groundtruth"
    log = ULog(str(ulog), [topic])
    candidates = [d for d in log.data_list if d.name == topic and d.multi_id == 0]
    if len(candidates) != 1:
        return None, "GROUND_TRUTH_TOPIC_UNAVAILABLE"
    data = candidates[0].data
    ts = np.asarray(data["timestamp_sample"], dtype=np.int64)
    if not len(ts) or ts[0] > start_us or ts[-1] < end_us or (np.diff(ts) <= 0).any():
        return None, "GROUND_TRUTH_CLOCK_UNAVAILABLE"
    lo = max(0, int(np.searchsorted(ts, start_us, side="right")) - 1)
    hi = int(np.searchsorted(ts, end_us, side="left")) + 1
    samples = ts[lo:hi]
    if np.max(np.diff(samples)) > 500000 or len(set(data["xy_reset_counter"][lo:hi])) != 1:
        return None, "GROUND_TRUTH_GAP_OR_ORIGIN_RESET"
    clipped = np.r_[start_us, samples[(samples > start_us) & (samples < end_us)], end_us]
    coords = [np.interp(clipped, samples, data[k][lo:hi]) for k in ("x", "y")]
    if not all(np.isfinite(v).all() for v in coords):
        return None, "GROUND_TRUTH_POSITION_UNAVAILABLE"
    return float(np.hypot(np.diff(coords[0]), np.diff(coords[1])).sum()), "ULOG_SIMULATED_GROUND_TRUTH_XY"


def hashes_for_case(out, protocol, case):
    hashes = dict(protocol["source_sha256"])
    supplemental = out / "model_identities.json"
    if supplemental.exists():
        models = evidence.json_file(supplemental)
        evidence.confirm(models["protocol_sha256"] == evidence.digest(out / "protocol.json"), "Model register mismatch")
        matches = [m for m in models["models"].values() if m["payload_kg"] == case["payload_kg"]]
        evidence.confirm(len(matches) == 1, "Unregistered payload")
        key = "base_sdf" if not case["payload_kg"] else f"payload{round(100*case['payload_kg']):03d}"
        evidence.confirm(key not in hashes or hashes[key] == matches[0]["model_sdf_sha256"], "Conflicting SDF register")
        hashes[key] = matches[0]["model_sdf_sha256"]
    return hashes


def mission_summary(raw, case, attempt, hashes):
    cid = case["case_id"]
    receipt_path = raw / f"{cid}.json"
    receipt = evidence.json_file(receipt_path) if receipt_path.exists() else {}
    run = receipt.get("run_id")
    outcome = receipt.get("outcome", "ACQUISITION_EXIT_WITHOUT_RECEIPT")
    metrics = dict(case_id=cid, attempt=attempt, run_id=run, mission_outcome=outcome,
                   training_status="NO_VERIFIED_AIRBORNE_TARGET", label_source="unavailable", training_eligible=False,
                   estimated_airborne_energy_Wh=None, uncovered_s=None, native_VI_coverage_s=None,
                   charge_fit_exhaustion_observed=None, first_charge_fit_exhaustion_s=None, last_accounted_discharge_Ah=None)
    payload = {"case_id": cid, "attempt": attempt, "metrics": metrics, "sources": {}, "audit": {}}
    if receipt:
        evidence.confirm(all(receipt[k] == v for k, v in case.items()), "Receipt differs from requested case")
        payload["sources"]["receipt_sha256"] = evidence.digest(receipt_path)
    if outcome != "COMPLETE":
        metrics["rejection_reason"] = receipt.get("reason") or outcome
        return payload
    evidence.confirm(run and receipt["landed"] and not receipt["armed"], "Missing disarmed landing")
    provenance_path = raw / f"{run}.provenance.json"
    doc = evidence.json_file(provenance_path)
    row = doc["airborne_training"]
    wire = raw / f"{run}.pack_drive.jsonl"
    pack_trace = doc["native_pack_drive"]["trace"]
    evidence.confirm(row["relative_path"] == f"training/{run}.csv" and pack_trace["filename"] == wire.name
                     and not pack_trace["io_error"], "Recorder output missing, truncated or unattributed")
    logs = list(raw.glob("state/rootfs/log/*/*.ulg"))
    evidence.confirm(len(logs) == 1, "Expected one ULog in the isolated mission")
    ulog = logs[0]
    ulog_sha = evidence.digest(ulog)
    sources = {"receipt.json": receipt_path, "provenance.json": provenance_path, "flight.csv": raw / f"{run}.csv",
               "training.csv": raw / row["relative_path"]}
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        for name, path in sources.items():
            (directory / name).symlink_to(path.resolve())
        drops = evidence.ulog_extract(ulog, directory / "ulog_states.csv")
        manifest = {"case_id": cid, "attempt": attempt, "run_id": run,
                    "selected_for_training": row["training_eligible"], "source_ulog_sha256": ulog_sha,
                    "ulog_dropout_events": drops, "launcher_sha256": (raw / "launcher_sha256").read_text().strip(),
                    "files_sha256": {}}
        evidence.write_json(directory / "manifest.json", manifest)
        audit = evidence.audit_flight(directory, case, hashes, raw_trace_path=wire)
    distance, motion_source = horizontal_path(ulog, audit["ulog_takeoff_us"], audit["ulog_landing_us"])
    evidence.confirm(ulog_sha == evidence.digest(ulog), "PX4 ULog changed during verification")
    metrics.update(measurements(case, sources["flight.csv"], doc))
    energy = audit[evidence.TARGET]
    duration = audit["duration_s"]
    metrics.update({
        evidence.TARGET: energy, "training_status": audit["training_status"], "training_eligible": audit["training_eligible"],
        "label_source": row["label_source"], "airborne_duration_s": duration, "uncovered_s": audit["native_uncovered_s"],
        "airborne_covered_s": audit["native_covered_s"], "airborne_known_subtotal_Wh": audit["native_known_Wh"],
        "airborne_mean_power_W": energy*3600/duration if energy is not None else None,
        "airborne_manager_fallback_s": audit["source_bins"]["recorded_manager"]["seconds"],
        "airborne_manager_fallback_Wh": audit["source_bins"]["recorded_manager"]["energy_Wh"],
        "native_VI_coverage_s": audit["downsampled_VI_diagnostic"]["eligible_seconds"],
        "nominal_energy_remaining_percent_after_airborne": 100*max(0, 1 - energy/metrics["nameplate_Wh"]) if energy is not None else None,
        "direct_row_sha256": row["sha256"], "airborne_horizontal_distance_m": distance,
        "airborne_distance_source": motion_source, "rejection_reason": audit["budget_reason"] if energy is None else "",
    })
    diag = audit["pack_fit_airborne_diagnostics"]
    for name in ("charge_fit_exhaustion_observed", "first_charge_fit_exhaustion_s", "last_accounted_discharge_Ah"):
        metrics[name] = diag[name]
    payload["sources"].update({name: evidence.digest(path) for name, path in sources.items()})
    payload["sources"].update(original_ulog=str(ulog.relative_to(raw)), ulog_sha256=ulog_sha,
                              raw_pack_drive_jsonl_sha256=audit["original_trace_sha256"])
    payload["audit"] = audit
    return payload


def collect(work, out, case_id=None, attempt=None, verify=False):
    protocol = evidence.json_file(out / "protocol.json")
    planned = {c["case_id"]: c for c in protocol["cases"]}
    if case_id is not None and case_id not in planned:
        raise ValueError("Case is not registered")
    candidates = sorted(work.glob(f"{case_id or 'case_*'}/attempt_{attempt or '*'}/acquisition_exit.json"))
    if case_id and attempt and not candidates:
        raise RuntimeError("Attempt has not closed")
    for marker in candidates:
        raw = marker.parent
        cid = raw.parent.name
        case = planned[cid]
        attempt_number = int(raw.name.split("_")[-1])
        file = out / "audits" / cid / f"{raw.name}.json"
        if file.exists() and not verify:
            continue
        acquisition = raw / "acquisition_protocol.json"
        recorded = evidence.json_file(marker)
        actual_protocol = evidence.json_file(acquisition)
        evidence.confirm(recorded["case_id"] == cid and recorded["attempt"] == attempt_number
                         and recorded["protocol_sha256"] == evidence.digest(acquisition)
                         and case in actual_protocol["cases"] and actual_protocol["attempt"] == attempt_number,
                         "Attempt protocol registration mismatch")
        doc = mission_summary(raw, case, attempt_number, hashes_for_case(out, actual_protocol, case))
        doc["protocol_sha256"] = recorded["protocol_sha256"]
        doc["sources"]["acquisition_exit_sha256"] = evidence.digest(marker)
        if file.exists():
            evidence.confirm(evidence.json_file(file) == doc, "Previous audited attempt differs")
        else:
            file.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=file.parent, delete=False) as tmp:
                tmp.write((json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
                tmp_path = Path(tmp.name)
            tmp_path.replace(file)
        print(cid, raw.name, doc["metrics"]["training_status"], doc["metrics"][evidence.TARGET], flush=True)


def follow(work, out):
    """Audit closed acquisitions independently while the campaign is underway."""
    cases = {c["case_id"] for c in evidence.json_file(out / "protocol.json")["cases"]}
    completed, retry_after = set(), {}
    while completed != cases:
        for marker in sorted(work.glob("case_*/attempt_*/acquisition_exit.json")):
            raw = marker.parent
            cid, attempt = raw.parent.name, int(raw.name.removeprefix("attempt_"))
            if time.monotonic() < retry_after.get(marker, 0):
                continue
            try:
                collect(work, out, cid, attempt)
                file = out / "audits" / cid / f"{raw.name}.json"
                if evidence.json_file(file)["metrics"][evidence.TARGET] is not None:
                    completed.add(cid)
            except (ValueError, OSError, KeyError, RuntimeError) as error:
                print(f"Audit {cid}/{raw.name} could not be published: {error}", file=sys.stderr, flush=True)
                retry_after[marker] = time.monotonic() + 60
        if completed != cases:
            time.sleep(8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--case")
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--verify", action="store_true", help="Re-audit even previously published attempts")
    parser.add_argument("--follow", action="store_true", help="Watch all registered cases for verified airborne labels")
    args = parser.parse_args()
    if args.follow:
        follow(args.work, args.out)
    else:
        collect(args.work, args.out, args.case, args.attempt, args.verify)
