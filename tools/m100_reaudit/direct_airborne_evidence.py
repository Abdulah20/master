"""Archive and audit directly recorded Gazebo airborne energy against original PX4 data.

Run as python3 -m tools.m100_reaudit.direct_airborne_evidence. Use --verify-saved
to reproduce the audit using the released recorder bytes and extracted ULog states.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from . import native_training
from .pack_drive.m100_airborne_training import BOUNDARY_SOURCE, CALIBRATION_ID, POWER_SOURCES


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "results/m100_reaudit/pack_drive_three/protocol.json"
WORK = ROOT / "tmp/m100_direct_airborne_three"
EVIDENCE = ROOT / "results/m100_reaudit/direct_airborne_three/evidence"
TARGET = native_training.TARGET


def digest(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def confirm(ok, message):
    if not ok:
        raise ValueError(message)


def json_file(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def ulog_extract(ulog, destination):
    from pyulog import ULog

    topics = {"vehicle_land_detected": ("landed", "ground_contact", "maybe_landed"),
              "vehicle_status": ("arming_state", "failsafe", "nav_state")}
    log = ULog(str(ulog), list(topics))
    with destination.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(("topic", "timestamp_us", *topics["vehicle_land_detected"], *topics["vehicle_status"]))
        for topic, keys in topics.items():
            data = log.get_dataset(topic).data
            for i, timestamp in enumerate(data["timestamp"]):
                observed = [int(data[key][i]) for key in keys]
                if topic == "vehicle_land_detected":
                    writer.writerow((topic, int(timestamp), *observed, "", "", ""))
                else:
                    writer.writerow((topic, int(timestamp), "", "", "", *observed))
    return len(log.dropouts)


def archive_flight(raw, evidence, case, protocol_path, attempt):
    cid = case["case_id"]
    receipt_file = raw / f"{cid}.json"
    receipt = json_file(receipt_file)
    confirm(receipt["case_id"] == cid, "Acquisition receipt identity differs")
    run = receipt["run_id"]
    provenance_file = raw / f"{run}.provenance.json"
    doc = json_file(provenance_file)
    trace = doc["native_pack_drive"]["trace"]
    row_info = doc["airborne_training"]
    selected = row_info["training_eligible"] is True and receipt["outcome"] == "COMPLETE"
    folder = evidence / cid if selected else evidence / "prior_attempts" / cid / f"attempt_{attempt}"
    confirm(trace["filename"] == f"{run}.pack_drive.jsonl" and not trace["io_error"],
            "Native trace recording incomplete")
    confirm(row_info["relative_path"] == f"training/{run}.csv", "Unattributed recorder row")
    sources = {"receipt.json": receipt_file, "provenance.json": provenance_file, "acquisition_protocol.json": protocol_path,
               "training.csv": raw / row_info["relative_path"], "flight.csv": raw / f"{run}.csv"}
    wire = raw / trace["filename"]
    confirm(digest(wire) == trace["sha256"], "Original pack trace SHA-256 differs from recorder")
    confirm(digest(sources["training.csv"]) == row_info["sha256"], "Original row SHA-256 differs from recorder")
    confirm(Path(doc["csv"]["path"]).name == sources["flight.csv"].name and
            digest(sources["flight.csv"]) == doc["csv"]["sha256"], "Original flight telemetry differs from recorder")
    logs = list(raw.glob("state/rootfs/log/*/*.ulg"))
    confirm(len(logs) == 1, f"{cid}: expected one original ULog, found {len(logs)}")
    ulog = logs[0]
    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as stage:
        dest = Path(stage) / cid
        dest.mkdir()
        for name, path in sources.items():
            shutil.copyfile(path, dest / name)
        with (dest / "pack_drive.jsonl.gz").open("wb") as target, wire.open("rb") as source:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=target, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)
        ulog_sha = digest(ulog)
        log_dropouts = ulog_extract(ulog, dest / "ulog_states.csv")
        confirm(digest(ulog) == ulog_sha, f"{cid}: PX4 ULog is still changing")
        manifest = {"case_id": cid, "run_id": run, "attempt": attempt, "selected_for_training": selected,
                    "files_sha256": {p.name: digest(p) for p in sorted(dest.iterdir())},
                    "source_training_relative_path": row_info["relative_path"],
                    "source_training_mtime_ns": sources["training.csv"].stat().st_mtime_ns,
                    "source_provenance_mtime_ns": provenance_file.stat().st_mtime_ns,
                    "source_acquisition_receipt_mtime_ns": receipt_file.stat().st_mtime_ns,
                    "launcher_sha256": (raw / "launcher_sha256").read_text().strip(),
                    "source_ulog_relative_path": str(ulog.relative_to(raw)), "source_ulog_sha256": ulog_sha,
                    "source_ulog_bytes": ulog.stat().st_size, "ulog_dropout_events": log_dropouts,
                    "ulog_columns": "Every instance-0 observation of selected vehicle_land_detected and vehicle_status fields."}
        write_json(dest / "manifest.json", manifest)
        if folder.exists():
            previous = json_file(folder / "manifest.json")
            frozen_files = previous.pop("files_sha256")
            new_files = manifest["files_sha256"]
            confirm(previous == {k: v for k, v in manifest.items() if k != "files_sha256"}
                    and all(sha == new_files.get(name) == digest(folder / name) for name, sha in frozen_files.items()),
                    f"{cid}: previous evidence refers to different recorder bytes; choose a new output")
            added = new_files.keys() - frozen_files.keys()
            for name in added:
                confirm(not (folder / name).exists() or digest(folder / name) == new_files[name],
                        f"{cid}: archived {name} conflicts with its recorder")
                (dest / name).replace(folder / name)
            if added:
                (dest / "manifest.json").replace(folder / "manifest.json")
        else:
            dest.rename(folder)
    return folder


def gate_from_ulog(path):
    with path.open(newline="", encoding="utf-8") as src:
        data = list(csv.DictReader(src))
    land = [r for r in data if r["topic"] == "vehicle_land_detected"]
    states = [r for r in data if r["topic"] == "vehicle_status"]
    for group in (land, states):
        stamps = [int(r["timestamp_us"]) for r in group]
        confirm(stamps and sorted(stamps) == stamps, "Missing or nonmonotonic PX4 state observations")
    edges = [(int(r["timestamp_us"]), int(r["landed"])) for i, r in enumerate(land)
             if not i or r["landed"] != land[i-1]["landed"]]
    confirm([level for stamp, level in edges] == [1, 0, 1], f"Expected one PX4 airborne episode, found {edges}")
    start, end = edges[1][0], edges[2][0]
    confirm(int(states[0]["timestamp_us"]) <= start < end <= int(states[-1]["timestamp_us"]),
            "PX4 status does not bracket airborne episode")
    armed_us = failsafe_us = 0
    for left, right in zip(states, states[1:]):
        dt = max(0, min(end, int(right["timestamp_us"])) - max(start, int(left["timestamp_us"])))
        if left["arming_state"] == "2":
            armed_us += dt
        if left["failsafe"] != "0":
            failsafe_us += dt
    return start, end, armed_us, failsafe_us


def rejected_gate_observations(samples, ulog_states):
    """Retain the original land-detector and published budget transitions for a rejected label."""
    px4_edges = []
    with ulog_states.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["topic"] == "vehicle_land_detected":
                landed = bool(int(row["landed"]))
                if not px4_edges or landed != px4_edges[-1]["landed"]:
                    px4_edges.append({"timestamp_us": int(row["timestamp_us"]), "landed": landed})
    publications = []
    last = object()
    previous = None
    for native in samples:
        budget = native["airborne_budget"]
        state = (native["px4_land_sequence"], native["px4_landed"], native["px4_land_transition_s"],
                 budget["status"], budget["reason"])
        if state != last:
            publications.append({"native_publication_s": native["time_s"], "previous_publication_s": previous,
                                 "detector_stamp_s": native["px4_land_time_s"], "sequence": state[0],
                                 "landed": state[1], "transition_s": state[2], "budget_status": state[3],
                                 "budget_reason": state[4]})
        last = state
        previous = native["time_s"]
    return {"ulog_land_edges": px4_edges, "native_land_and_budget_transitions": publications}


def flight_telemetry(folder, doc, case, start, end):
    path = folder / "flight.csv"
    csv_info = doc["csv"]
    sampling = csv_info["sampling"]
    confirm(Path(csv_info["path"]).name == doc["run_id"] + ".csv" and digest(path) == csv_info["sha256"]
            and path.stat().st_size == csv_info["size_bytes"], "Flight CSV differs from recorder provenance")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        confirm(reader.fieldnames == csv_info["fields"] and len(reader.fieldnames) == csv_info["schema_field_count"]
                and len(rows) == sampling["recorded_rows"], "Flight CSV header or row count differs from recorder")
    clock = sampling["clock"]
    times = [float(r[clock]) for r in rows]
    confirm(times and all(math.isfinite(t) for t in times) and all(a < b for a, b in zip(times, times[1:])) and
            times[0] == sampling["origin_energy_timestamp_s"] and times[-1] == sampling["last_recorded_energy_timestamp_s"],
            "Flight CSV energy clock differs from recorder")
    command_fields = {"payload_kg": "payload_kg", "commanded_speed_mps": "cruise_speed_mps",
                      "commanded_altitude_m": "altitude_m", "wind_speed_command_mps": "wind_speed_mps",
                      "wind_direction_command_deg": "wind_direction_toward_deg_ENU", "temperature_c": "temperature_c"}
    confirm(all(float(r[col]) == case[key] for r in rows for col, key in command_fields.items()),
            "Flight telemetry commands differ from the frozen acquisition")
    return {"sha256": csv_info["sha256"], "size_bytes": csv_info["size_bytes"], "rows": len(times), "clock": clock,
            "first_timestamp_s": times[0], "last_timestamp_s": times[-1], "target_hz": sampling["target_hz"],
            "missed_windows": sampling["missed_windows"], "brackets_ulog_airborne_window": times[0] <= start < end <= times[-1],
            "rows_within_ulog_airborne_window": sum(start <= t < end for t in times),
            "rows_with_blank_electrical_power": sum(not r["electrical_power_W"] for r in rows)}


def vi_integral(samples, start, end):
    """Independent low-rate V*I trapezoids on native-only, single-source spans.

    Cumulative *time* is used only to select intervals having unmixed support.
    All energy below is computed from recorded voltage/current samples separately
    from the physics ledger. Mixed/boundary and fallback spans are excluded.
    """
    total_s = vi_wh = ledger_wh = 0.
    count = 0
    for left, right in zip(samples, samples[1:]):
        t0, t1 = left["time_s"], right["time_s"]
        dt = t1-t0
        source = left["effective_demand_source"]
        if (not start <= t0 < t1 <= end or dt > 0.5 or source not in POWER_SOURCES[:3]
                or right["effective_demand_source"] != source or left["command_limited"] or right["command_limited"]):
            continue
        values = [s[k] for s in (left, right) for k in ("voltage_v", "current_a")]
        if not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values):
            continue
        prev, nxt = (s["airborne_budget"] for s in (left, right))
        if prev["status"] != "AIRBORNE" or nxt["status"] != "AIRBORNE":
            continue
        before, after = (budget["sources"][source] for budget in (prev, nxt))
        if abs(after["seconds"] - before["seconds"] - dt) > 1e-8:
            continue
        total_s += dt
        vi_wh += 0.5 * (values[0]*values[1] + values[2]*values[3]) * dt / 3600.
        ledger_wh += after["energy_Wh"] - before["energy_Wh"]
        count += 1
    return {"eligible_intervals": count, "eligible_seconds": total_s, "published_VI_trapezoids_Wh": vi_wh,
            "same_intervals_native_Wh": ledger_wh, "published_minus_native_Wh": vi_wh - ledger_wh}


def publication_partitions(samples, run):
    count = 0
    max_time_error = max_energy_error = 0.
    for sample in samples:
        b = sample["airborne_budget"]
        if b["run_id"] != run or b["start_s"] is None or b["duration_s"] is None:
            continue
        count += 1
        bins = b["sources"]
        confirm(set(bins) == set(POWER_SOURCES), "Published source bins changed during the flight")
        for values in bins.values():
            confirm(all(math.isfinite(x) and x >= 0. for x in (values["seconds"], values["energy_Wh"])),
                    "Published source bin has a negative or nonfinite value")
        elapsed = (sample["time_s"] if b["end_s"] is None else b["end_s"]) - b["start_s"]
        source_time = math.fsum(v["seconds"] for v in bins.values())
        source_energy = math.fsum(v["energy_Wh"] for v in bins.values())
        max_time_error = max(max_time_error, abs(source_time - b["covered_s"]),
                             abs(source_time + b["uncovered_s"] - elapsed), abs(b["duration_s"] - elapsed))
        max_energy_error = max(max_energy_error, abs(source_energy - b["known_energy_Wh"]))
    confirm(max_time_error < 1e-8 and max_energy_error < 1e-10,
            "A published airborne clock or energy subtotal differs from its source bins")
    return {"checked_publications": count, "max_time_partition_residual_s": max_time_error,
            "max_energy_subtotal_residual_Wh": max_energy_error}


def uncovered_events(records, run):
    """Publish brackets are observable; an increase can cover only part of its bracket."""
    gaps, previous = [], None
    for record in records:
        native = record["native"]
        budget = native["airborne_budget"]
        if budget["run_id"] != run or budget["uncovered_s"] is None:
            continue
        start = previous["time_s"] if previous else budget["start_s"]
        before = previous["native"]["airborne_budget"]["uncovered_s"] if previous else 0.
        delta = budget["uncovered_s"] - before
        confirm(delta >= -1e-9, "Native uncovered-time counter reversed inside airborne window")
        if delta > 1e-9:
            snapshots = []
            for item in (previous, record):
                if item is None:
                    continue
                state = item["native"]
                demand = item["acknowledged_request"] or {}
                snapshots.append({"time_s": item["time_s"], "phase": demand.get("phase"),
                                  "manager_power_w": demand.get("manager_power_w"),
                                  "phase_source": demand.get("phase_source"),
                                  "raw_fresh": state["raw_command_fresh"], "clipped": state["command_limited"],
                                  "native_status": state["status"], "effective_source": state["effective_demand_source"]})
            gaps.append({"previous_publish_s": start, "next_publish_s": record["time_s"],
                         "newly_uncovered_s": delta, "endpoints": snapshots})
        previous = record
    return gaps


def audit_flight(folder, case, protocol_hashes, *, raw_trace_path=None):
    manifest, receipt, doc = [json_file(folder / name) for name in ("manifest.json", "receipt.json", "provenance.json")]
    for name, sha in manifest["files_sha256"].items():
        confirm(digest(folder / name) == sha, f"Archived digest mismatch: {folder / name}")
    cid, run = case["case_id"], receipt["run_id"]
    confirm(manifest["run_id"] == doc["run_id"] == run and manifest["case_id"] == receipt["case_id"] == cid,
            "Acquisition run identities disagree")
    confirm(all(receipt[k] == v for k, v in case.items()), f"{cid}: protocol differs from previous flight")
    confirm(receipt["observed_seed"] == case["seed"] and abs(receipt["observed_physics_step_s"]-case["physics_step_s"]) < 1e-9,
            "Unconfirmed simulation seed or step")
    source_names = {"manager": "m100_experiment_manager", "training_writer": "m100_airborne_training",
                    "px4_binary": "px4_executable_with_stamped_land_state", "plugin": "gazebo_m100_pack_drive_plugin",
                    "energy_config": "m100_energy_v2_config", "base_sdf": "m100_canonical_base_sdf"}
    for key, source in source_names.items():
        confirm(doc["source_files"][source]["sha256"] == protocol_hashes[key], f"Different acquisition source: {source}")
    sdf = "base_sdf" if not case["payload_kg"] else f"payload{round(case['payload_kg']*100):03d}"
    confirm(doc["temperature_application"]["runtime_model_sdf_sha256"] == protocol_hashes[sdf]
            and manifest["launcher_sha256"] == protocol_hashes["launcher"]
            and doc["native_pack_drive"]["cmu_development_fit_sha256"] == protocol_hashes["fit"],
            "Different runtime model, launcher or voltage fit")
    direct, pack = doc["airborne_training"], doc["native_pack_drive"]
    row_path = folder / "training.csv"
    with row_path.open(newline="", encoding="utf-8") as f:
        row, = list(csv.DictReader(f))
    example = native_training.training_example(row_path.read_bytes(), direct, cid, run, receipt["outcome"])
    confirm(manifest["selected_for_training"] == (example is not None), "Published row eligibility differs from archive")
    config = doc["config"]
    for col, key, config_key in (
            ("payload_kg", "payload_kg", "payload_kg"), ("commanded_speed_mps", "cruise_speed_mps", "cruise_speed_mps"),
            ("commanded_altitude_agl_m", "altitude_m", "altitude_m"),
            ("commanded_wind_mps", "wind_speed_mps", "wind_speed_mps"),
            ("commanded_wind_direction_deg", "wind_direction_toward_deg_ENU", "wind_direction_deg"),
            ("ambient_reference_C", "temperature_c", "temperature_c")):
        confirm(float(row[col]) == case[key] == config[config_key], f"Recorded condition differs for {col}")
    confirm(config["route_length_m"] == case["measurement_distance_m"]
            and config["hover_duration_s"] == case["hover_duration_s"], "Changed mission plan")
    cruise = case["cruise_speed_mps"] != 0
    confirm(float(row["planned_outbound_measurement_m"]) == (case["measurement_distance_m"] if cruise else 0)
            and (row["planned_hover_duration_s"] == "" if cruise else
                 float(row["planned_hover_duration_s"]) == case["hover_duration_s"]), "Changed measurement mode")
    info = direct["native_budget"]
    confirm(info["run_id"] == run and row["calibration_id"] == info["calibration_id"] == pack["airborne_calibration_id"] == CALIBRATION_ID
            and info["boundary_source"] == BOUNDARY_SOURCE, "Different native gate or fit")
    with (raw_trace_path.open("rb") if raw_trace_path is not None else
          gzip.open(folder / "pack_drive.jsonl.gz", "rb")) as stream:
        wire_sha = hashlib.sha256()
        records, epochs = [], set()
        for line in stream:
            wire_sha.update(line)
            item = json.loads(line)
            native = item["native"]
            confirm(item["run_id"] == run and item["time_s"] == native["time_s"]
                    and native["calibration_id"] == CALIBRATION_ID, "Different native run, clock or fit")
            records.append(item)
            epochs.add(item["epoch"])
    samples = [r["native"] for r in records]
    confirm(wire_sha.hexdigest() == pack["trace"]["sha256"] and len(samples) == pack["trace"]["rows"],
            "Raw archive does not reproduce recorder pack trace")
    confirm(len(epochs) == 1 and all(a["time_s"] < b["time_s"] for a, b in zip(samples, samples[1:])),
            "Nonmonotonic or mixed native entity epochs")
    budgets = [s["airborne_budget"] for s in samples if s["airborne_budget"]["run_id"] == run]
    confirm(budgets and budgets[-1] == info, "Final recorder receipt does not match the terminal Gazebo budget")
    start_us, end_us, armed_us, failsafe_us = gate_from_ulog(folder / "ulog_states.csv")
    start, end = start_us / 1e6, end_us / 1e6
    status = info["status"]
    expected_duration = (end_us-start_us)/1e6
    source_s = math.fsum(b["seconds"] for b in info["sources"].values())
    source_wh = math.fsum(b["energy_Wh"] for b in info["sources"].values())
    target = float(row[TARGET]) if row[TARGET] else None
    if receipt["outcome"] == "COMPLETE":
        confirm(receipt["landed"] and not receipt["armed"] and
                doc["recording_end"]["landed"] and not doc["recording_end"]["armed"], "Landing confirmation missing")
        confirm(armed_us == end_us-start_us, "Aircraft was not continuously armed in the PX4 window")
    bracketed = info["start_s"] is not None and info["end_s"] is not None
    if bracketed:
        confirm(round(info["start_s"] * 1e6) == start_us and round(info["end_s"] * 1e6) == end_us,
                f"Native edges differ from PX4 ULog {start_us},{end_us}: {info['start_s']},{info['end_s']}")
        confirm(info["uncovered_s"] == float(row["uncovered_s"]) and
                all(abs(v-expected_duration) < 1e-8 for v in (info["covered_s"] + info["uncovered_s"],
                      info["duration_s"], float(row["airborne_duration_s"]), source_s + info["uncovered_s"])),
                "Native sources plus uncovered time do not partition PX4 airborne time")
        confirm(abs(source_wh-info["known_energy_Wh"]) < 1e-10 and set(info["sources"]) == set(POWER_SOURCES) and
                all(abs(float(row[f"{key}_s"]) - val["seconds"]) < 1e-10 for key, val in info["sources"].items()),
                "Native known subtotal or source coverage in training row differs")
    if status == "COMPLETE":
        confirm(info["start_s"] is not None and info["end_s"] is not None and info["uncovered_s"] == 0 and
                target is not None and abs(source_wh-target) < 1e-10 and abs(info[TARGET]-target) < 1e-10,
                "Native source energies do not add up to a fully covered direct target")
    else:
        confirm(target is None and example is None, "An incomplete native budget was exported as a target")
    model_coverage = {}
    if raw_trace_path is not None:
        if status != "COMPLETE":
            model_coverage["rejected_gate_observations"] = rejected_gate_observations(samples, folder / "ulog_states.csv")
        in_air = [s for s in samples if start <= s["time_s"] <= end]
        exhausted = [s["time_s"] for s in in_air if s["q_domain_exhausted"]]
        model_coverage["pack_fit_airborne_diagnostics"] = {
            "publications": len(in_air),
            "charge_fit_exhaustion_observed": bool(exhausted) if in_air else None,
            "first_charge_fit_exhaustion_s": exhausted[0] if exhausted else None,
            "last_accounted_discharge_Ah": in_air[-1]["discharged_ah"] if in_air else None,
        }
    return {**model_coverage, "case_id": cid, "run_id": run, "attempt": manifest["attempt"],
            "mission_outcome": receipt["outcome"], "native_budget_status": status,
            "training_status": row["training_status"], "budget_reason": info["reason"], "training_eligible": example is not None,
            "direct_row_sha256": direct["sha256"], "original_trace_sha256": pack["trace"]["sha256"],
            "calibration_id": info["calibration_id"], "trace_records": len(samples), "entity_epochs": sorted(epochs),
            "ulog_sha256": manifest["source_ulog_sha256"], "ulog_dropout_events": manifest["ulog_dropout_events"],
            "ulog_takeoff_us": start_us, "ulog_landing_us": end_us, "ulog_airborne_armed_s": armed_us / 1e6,
            "ulog_airborne_failsafe_s": failsafe_us / 1e6, "native_start_s": info["start_s"], "native_end_s": info["end_s"],
            "ulog_and_native_edges_match": bracketed, "native_source_time_partition_verified": bracketed,
            "training_row_matches_recorder_and_terminal_native_budget": True,
            "duration_s": expected_duration, "native_covered_s": info["covered_s"], "native_uncovered_s": info["uncovered_s"],
            "source_bins": info["sources"], "source_sum_s": source_s, "source_sum_Wh": source_wh,
            "native_known_Wh": info["known_energy_Wh"], TARGET: target,
            "flight_csv": flight_telemetry(folder, doc, case, start, end),
            "published_ledger_checks": publication_partitions(samples, run),
            "uncovered_event_brackets": uncovered_events(records, run),
            "downsampled_VI_diagnostic": vi_integral(samples, start, end)}


def reference_cases(protocol):
    original = {c["case_id"]: c for c in json_file(PREVIOUS)["cases"]}
    cases = protocol["cases"]
    confirm(cases and all(c == original.get(c["case_id"]) for c in cases)
            and len({c["case_id"] for c in cases}) == len(cases)
            and protocol["previous_protocol_sha256"] == digest(PREVIOUS),
            "Acquisition protocol does not reference original case commands")
    return cases


def write_or_verify_index(evidence, verify):
    indexed = []
    for path in sorted(evidence.rglob("manifest.json")):
        folder = path.parent
        manifest = json_file(path)
        audit_path = folder / "audit.json"
        audited = json_file(audit_path)
        cid, attempt = manifest["case_id"], manifest["attempt"]
        expected = evidence / cid if audited["training_eligible"] else evidence / "prior_attempts" / cid / f"attempt_{attempt}"
        confirm(folder == expected, "Unexpected evidence location for case and eligibility")
        if attempt > 1:
            plan = json_file(folder / "acquisition_protocol.json")
            original = evidence / "prior_attempts" / cid / "attempt_1"
            earlier = json_file(original / "audit.json")
            frozen = plan["initial_attempts"][cid]
            confirm(attempt == plan["attempt"] and cid in plan["planned_replacement_cases"] and
                    plan["initial_direct_protocol_sha256"] == digest(original / "acquisition_protocol.json") and
                    frozen["run_id"] == earlier["run_id"] and frozen["mission_outcome"] == earlier["mission_outcome"] and
                    frozen["training_status"] == earlier["training_status"] and
                    frozen["original_training_row_sha256"] == earlier["direct_row_sha256"] and
                    frozen["uncovered_s"] == earlier["native_uncovered_s"] and earlier[TARGET] is None,
                    "Replacement differs from the frozen plan or its original training target was populated")
        indexed.append({"case_id": cid, "run_id": manifest["run_id"], "attempt": attempt,
                        "training_eligible": audited["training_eligible"], "training_status": audited["training_status"],
                        "path": str(folder.relative_to(evidence)), "manifest_sha256": digest(path),
                        "acquisition_protocol_sha256": manifest["files_sha256"]["acquisition_protocol.json"],
                        "ulog_sha256": manifest["source_ulog_sha256"], "audit_sha256": digest(audit_path)})
    confirm(indexed and len({(x["case_id"], x["attempt"]) for x in indexed}) == len(indexed)
            and len({x["run_id"] for x in indexed}) == len(indexed), "Missing evidence or duplicate acquisition ID")
    indexed.sort(key=lambda x: (x["case_id"], x["attempt"]))
    index = evidence / "index.json"
    if verify:
        confirm(indexed == json_file(index), "Released evidence inventory differs")
    else:
        write_json(index, indexed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--protocol", type=Path, default=EVIDENCE.parent / "protocol.json")
    parser.add_argument("--case", choices=["load_000", "load_025", "load_050"])
    parser.add_argument("--attempt", type=int, help="Acquisition attempt (default 1 for archiving, all for --verify-saved)")
    parser.add_argument("--verify-saved", action="store_true")
    args = parser.parse_args()
    if args.verify_saved:
        folders = [p.parent for p in sorted(args.evidence.rglob("manifest.json"))
                   if (not args.case or json_file(p)["case_id"] == args.case)
                   and (not args.attempt or json_file(p)["attempt"] == args.attempt)]
    else:
        attempt = args.attempt or 1
        folders = [archive_flight(args.work / c["case_id"] / f"attempt_{attempt}", args.evidence,
                                  c, args.protocol, attempt) for c in reference_cases(json_file(args.protocol))
                   if not args.case or c["case_id"] == args.case]
    confirm(folders, "No matching evidence or protocol cases")
    for folder in folders:
        cid = json_file(folder / "manifest.json")["case_id"]
        protocol = json_file(folder / "acquisition_protocol.json")
        case, = (c for c in reference_cases(protocol) if c["case_id"] == cid)
        audited = audit_flight(folder, case, protocol["source_sha256"])
        audit_path = folder / "audit.json"
        if args.verify_saved:
            confirm(audited == json_file(audit_path), f"Audit differs for {cid}")
        else:
            write_json(audit_path, audited)
        print(cid, audited["training_status"], audited[TARGET], "Wh")
    write_or_verify_index(args.evidence, args.verify_saved)


if __name__ == "__main__":
    main()
