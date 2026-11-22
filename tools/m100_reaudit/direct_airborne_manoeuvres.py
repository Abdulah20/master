"""Audit in-flight transient power in the released Gazebo pack traces.

Each manoeuvre total is the difference of the *native physics-step counters*
between successive in-air trace samples that acknowledge the same phase and
run. Transition or unmatched spans are deliberately left unassigned. Pointwise
power ranges and transfer-flag counts use the recorded states at their stamps.
Neither calculation provides the full-flight training label.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/m100_reaudit/direct_airborne_three"
PHASES = ("STABILIZING", "RETURN_BRAKING")
CALIBRATED = ("cmu_local", "cmu_pooled")
SOURCES = ("native_pack_model", *CALIBRATED, "recorded_manager")


def load_recording(evidence, case_id):
    directory = evidence / case_id
    provenance = json.loads((directory / "provenance.json").read_text())
    trace = provenance["native_pack_drive"]["trace"]
    with gzip.open(directory / "pack_drive.jsonl.gz", "rb") as stream:
        raw = stream.read()
    if trace["io_error"] or hashlib.sha256(raw).hexdigest() != trace["sha256"]:
        raise ValueError(f"{case_id}: native pack trace differs from the recorder")
    records = [json.loads(line) for line in raw.splitlines()]
    run = provenance["run_id"]
    if (len(records) != trace["rows"] or not records or len({r["epoch"] for r in records}) != 1
            or any(r["run_id"] != run or r["time_s"] != r["native"]["time_s"] for r in records)
            or any(b["time_s"] <= a["time_s"] for a, b in zip(records, records[1:]))):
        raise ValueError(f"{case_id}: invalid pack trace identity, count or clock")
    calibration_id = provenance["airborne_training"]["native_budget"]["calibration_id"]
    for record in records:
        n, request = record["native"], record["acknowledged_request"]
        if n["effective_demand_source"] in CALIBRATED:
            if (n["calibration_id"] != calibration_id or not n["raw_command_fresh"]
                    or not n["demand_fresh"] or not math.isfinite(n["effective_demand_power_w"])
                    or n["effective_demand_power_w"] <= 0 or (request and request["phase"] not in PHASES)):
                raise ValueError(f"{run}: calibrated power lacks current native inputs or phase support")
    return provenance, records


def phase_of(record, run):
    request = record["acknowledged_request"] or {}
    native = record["native"]
    budget = native["airborne_budget"]
    if (record["demand_fresh"] and native["demand_fresh"] and request.get("run_id") == run
            and budget["run_id"] == run and budget["status"] == "AIRBORNE"
            and request.get("model") == native["model"] and request.get("phase") in PHASES):
        return request["phase"]
    return None


def product(native, voltage, current):
    volts, amps = native[voltage], native[current]
    return None if volts is None or amps is None else volts * amps


def compact_counts(counts):
    return "|".join(f"{name}={counts[name]}" for name in sorted(counts))


def summarize(case, provenance, records):
    run = provenance["run_id"]
    phases = [phase_of(record, run) for record in records]
    native = [record["native"] for record in records]
    rows = []
    for phase in PHASES:
        points = [n for n, p in zip(native, phases) if p == phase]
        intervals = [(a, b) for i, (a, b) in enumerate(zip(native, native[1:]))
                     if phases[i] == phase == phases[i+1]]
        elapsed = math.fsum(b["time_s"] - a["time_s"] for a, b in intervals)
        row = {
            "case_id": case["case_id"], "run_id": run, "payload_kg": case["payload_kg"], "phase": phase,
            "first_ack_phase_state_s": points[0]["time_s"] if points else None,
            "last_ack_phase_state_s": points[-1]["time_s"] if points else None,
            "bracketed_phase_s": elapsed,
        }
        for source in SOURCES:
            for field, unit in (("seconds", "s"), ("energy_Wh", "Wh")):
                deltas = [b["airborne_budget"]["sources"][source][field]
                          - a["airborne_budget"]["sources"][source][field] for a, b in intervals]
                if any(d < -1e-9 for d in deltas):
                    raise ValueError(f"{run}: {source} counter moved backwards")
                row[f"{source}_{unit}"] = math.fsum(deltas)
        missing = math.fsum(b["airborne_budget"]["uncovered_s"] - a["airborne_budget"]["uncovered_s"]
                            for a, b in intervals)
        if not math.isclose(elapsed, missing + sum(row[s + "_s"] for s in SOURCES), abs_tol=1e-7):
            raise ValueError(f"{run}: non-exhaustive phase interval coverage")
        row["uncovered_s"] = missing
        for source in CALIBRATED:
            duration = row[source + "_s"]
            row[source + "_mean_W"] = row[source + "_Wh"] * 3600 / duration if duration else None
            source_points = [p for p in points if p["effective_demand_source"] == source]
            source_power = [p["effective_demand_power_w"] for p in source_points]
            row[f"{source}_states"] = len(source_points)
            row[f"{source}_min_recorded_demand_W"] = min(source_power, default=None)
            row[f"{source}_max_recorded_demand_W"] = max(source_power, default=None)
        calibrated = [p for p in points if p["effective_demand_source"] in CALIBRATED]
        volts = [p["voltage_v"] for p in calibrated if p["voltage_v"] is not None]
        amps = [p["current_a"] for p in calibrated if p["current_a"] is not None]
        vi_errors, requested_errors = [], []
        flagged, regimes, flags, local_reasons = 0, Counter(), Counter(), Counter()
        for state in calibrated:
            effective = state["effective_demand_power_w"]
            requested = product(state, "requested_load_voltage_v", "requested_pack_current_a")
            vi = product(state, "voltage_v", "current_a")
            if requested is not None:
                requested_errors.append(abs(requested - effective))
            if vi is not None:
                vi_errors.append(abs(vi - effective))
            if state["command_limited"] and vi is not None:
                raise ValueError(f"{run}: clipped commands claim a qualified pack VI value")
            regimes[state["transient_regime"]] += 1
            local_reasons[state["transient_local_reason"]] += 1
            flags.update(state["transient_transfer_flags"])
            flagged += bool(state["transient_transfer_flags"])
        row.update(
            ack_phase_states=len(points),
            cmu_states_with_requested_VI=len(requested_errors), cmu_states_with_qualified_VI=len(vi_errors),
            cmu_min_qualified_voltage_V=min(volts, default=None), cmu_max_qualified_voltage_V=max(volts, default=None),
            cmu_min_qualified_current_A=min(amps, default=None), cmu_max_qualified_current_A=max(amps, default=None),
            cmu_max_abs_requested_VI_minus_demand_W=max(requested_errors, default=None),
            cmu_max_abs_qualified_VI_minus_demand_W=max(vi_errors, default=None),
            command_limited_states=sum(p["command_limited"] for p in points),
            cmu_states_with_transfer_flags=flagged, cmu_motion_state_counts=compact_counts(regimes),
            cmu_local_support_reason_state_counts=compact_counts(local_reasons),
            cmu_transfer_flag_state_counts=compact_counts(flags),
        )
        rows.append(row)
    return rows


def main(protocol, evidence, output):
    cases = json.loads(protocol.read_text())["cases"]
    rows = []
    for case in cases:
        provenance, records = load_recording(evidence, case["case_id"])
        rows.extend(summarize(case, provenance, records))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: format(v, ".12g") if isinstance(v, float) else v for k, v in row.items()}
                         for row in rows)


def export_failed_inputs(evidence, captures, output):
    """Extract original manager ticks around loss documented by the independent native audit."""
    rows = []
    for audit_path in sorted(evidence.glob("prior_attempts/*/attempt_1/audit.json")):
        audit = json.loads(audit_path.read_text())
        if not audit["uncovered_event_brackets"]:
            continue
        case, run = audit["case_id"], audit["run_id"]
        attempt = audit_path.parent.name
        _, records = load_recording(audit_path.parent.parent, attempt)
        native_at_time = {round(r["time_s"], 9): r for r in records}
        trace_path = captures / case / attempt / f"{run}.trace.csv.gz"
        manager_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        with gzip.open(trace_path, "rt", newline="") as stream:
            ticks = list(csv.DictReader(stream))
        for gap in audit["uncovered_event_brackets"]:
            start, end = gap["previous_publish_s"], gap["next_publish_s"]
            # Two real manager samples on each side show whether the signal disappears and returns.
            before = [r for r in ticks if float(r["sim_time_s"]) < start]
            during = [r for r in ticks if start <= float(r["sim_time_s"]) <= end]
            after = [r for r in ticks if float(r["sim_time_s"]) > end]
            for tick in (*before[-2:], *during, *after[:2]):
                record = native_at_time.get(round(float(tick["sim_time_s"]), 9), {})
                n, q = record.get("native", {}), record.get("acknowledged_request") or {}
                rows.append({
                    "case_id": case, "attempt": attempt, "run_id": run,
                    "native_bracket_start_s": start, "native_bracket_end_s": end,
                    "newly_uncovered_s": gap["newly_uncovered_s"],
                    "manager_trace_filename": trace_path.name, "manager_trace_gzip_sha256": manager_sha,
                    "native_trace_uncompressed_sha256": audit["original_trace_sha256"],
                    **{f"manager_{k}": tick[k] for k in (
                        "sim_time_s", "px4_time_s", "phase", "local_position_age_wall_s",
                        "rho_kg_m3", "power_w", "armed", "landed")},
                    **{f"native_{k}": n.get(k) for k in (
                        "time_s", "status", "effective_demand_source", "effective_demand_power_w",
                        "demand_time_s", "demand_fresh", "raw_command_time_s", "raw_command_fresh",
                        "px4_land_time_s", "px4_landed", "armed", "command_limited")},
                    "ack_phase_source": q.get("phase_source"),
                    "joint_observation_s": (record.get("latest_joint_observation") or {}).get("time_s"),
                })
    if rows:
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=OUT / "protocol.json")
    parser.add_argument("--evidence", type=Path, default=OUT / "evidence")
    parser.add_argument("--output", type=Path, default=OUT / "manoeuvres.csv")
    parser.add_argument("--failed-gap-inputs", type=Path, help="write just the original loss-bracket extract")
    parser.add_argument("--capture-root", type=Path, default=ROOT / "tmp/m100_direct_airborne_three")
    args = parser.parse_args()
    if args.failed_gap_inputs:
        export_failed_inputs(args.evidence, args.capture_root, args.failed_gap_inputs)
    else:
        main(args.protocol, args.evidence, args.output)
