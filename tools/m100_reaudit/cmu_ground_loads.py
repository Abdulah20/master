"""Summarize the nine published CMU ground recordings at the battery sensor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from itertools import groupby
import json
from pathlib import Path
from statistics import fmean
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RECORDINGS = {"A1": (211, 212, 213), "A2": (214, 215, 216), "A3": (217, 218, 219)}
CONDITIONS = {"A1": "no propellers, motors stopped", "A2": "no propellers, minimum motor movement",
              "A3": "propellers installed, minimum motor movement"}
SOURCE_HASHES = {
    "flights.zip": "98f65a1ab54a8b618c169fdf49311a53ff1fefed86fdc3f440d6a0166ce49fb1",
    "parameters.csv": "568dffaed11a4d335ef4f8a932b342c836907af2f4aa54da9b4a0e2ffef2108a",
    "README.txt": "8601dd211d6c297c5411992dc34923bdb98b72cdc287517ed18e479f0e9540fd",
}
SIGNALS = ("time", "battery_voltage", "battery_current")
MAX_GAP_S = 0.5  # Same adjacent-interval rule as the CMU flight-level calibration.
RELATIVE_PLATEAU_BAND = 0.15
GUARD_S = 1.0


def read_ground(source: Path):
    for name, expected in SOURCE_HASHES.items():
        with (source / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Unexpected release file: {name}")
    with (source / "parameters.csv").open(newline="") as f:
        metadata = {int(r["flight"]): r for r in csv.DictReader(f) if r["route"] in RECORDINGS}
    expected_routes = {fid: route for route, ids in RECORDINGS.items() for fid in ids}
    if {fid: r["route"] for fid, r in metadata.items()} != expected_routes:
        raise ValueError("Ground flight IDs differ from the published selection")
    traces, member_hashes = {}, {}
    with ZipFile(source / "flights.zip") as archive:
        for fid in expected_routes:
            member = f"flights/{fid}.csv"
            raw = archive.read(member)
            member_hashes[member] = hashlib.sha256(raw).hexdigest()
            samples = np.array([[float(row[key]) for key in SIGNALS]
                                for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))])
            t, voltage, current = samples.T
            if not (np.isfinite(samples).all() and len(t) > 1 and t[0] == 0 and np.all(np.diff(t) > 0)):
                raise ValueError(f"Invalid signal or timestamp in recording {fid}")
            traces[fid] = (t, voltage, current)
    return metadata, traces, member_hashes


def operating_windows(t, current, route, low_limit):
    """Find the long operating plateau and the flanking low-current windows."""
    end = float(t[-1])
    if route == "A1":
        return [("publisher_motor_off", 0.0, end)]
    center = float(np.median(current))
    second = np.floor(t).astype(int)
    states = []
    for index in range(second[-1] + 1):
        samples = current[second == index]
        if not len(samples):
            states.append("gap")
            continue
        value = float(np.median(samples))
        if value <= low_limit:
            states.append("low")
        elif abs(value - center) <= center * RELATIVE_PLATEAU_BAND:
            states.append("plateau")
        else:
            states.append("transition")
    runs = []
    for state, entries in groupby(enumerate(states), key=lambda item: item[1]):
        indices = [entry[0] for entry in entries]
        runs.append((state, float(indices[0]), min(float(indices[-1] + 1), end)))
    plateau = max((r for r in runs if r[0] == "plateau"), key=lambda r: r[2] - r[1])
    selected = [("minimum_movement_plateau", *plateau[1:])]
    if runs[0][0] == "low":
        selected.append(("initial_low_current", *runs[0][1:]))
    if runs[-1][0] == "low":
        selected.append(("final_low_current", *runs[-1][1:]))
    windows = []
    previous = 0.0
    for label, start, stop in sorted(selected, key=lambda item: item[1]):
        start = start + GUARD_S if start > 0 else start
        stop = stop - GUARD_S if stop < end else stop
        if start >= stop:
            continue
        if previous < start:
            windows.append(("transition", previous, start))
        windows.append((label, start, stop))
        previous = stop
    if previous < end:
        windows.append(("transition", previous, end))
    return windows


def summarize_window(trace, start: float, end: float, reference_power_W: float | None = None) -> dict:
    t, voltage, current = trace
    dt = np.diff(t)
    left = np.maximum(t[:-1], start)
    right = np.minimum(t[1:], end)
    keep = (right > left) & (dt <= MAX_GAP_S + 1e-10)
    width = right[keep] - left[keep]
    duration = float(width.sum())
    if duration <= 0:
        raise ValueError("No covered intervals in ground measurement window")
    a = (left[keep] - t[:-1][keep]) / dt[keep]
    b = (right[keep] - t[:-1][keep]) / dt[keep]

    def integral(signal):
        midpoint = signal[:-1][keep] + (a + b) / 2.0 * np.diff(signal)[keep]
        return float(np.dot(midpoint, width))

    samples = (t >= start) & (t <= end)
    peak = np.flatnonzero(samples)[np.argmax(current[samples])]
    power = voltage * current
    watt_seconds = integral(power)
    incremental_ws = integral(power - reference_power_W) if reference_power_W is not None else None
    return {"start_s": start, "end_s": end, "covered_s": duration, "sample_count": int(samples.sum()),
            "covered_intervals": int(keep.sum()), "mean_pack_voltage_V": integral(voltage) / duration,
            "mean_signed_current_A": integral(current) / duration, "mean_signed_pack_power_W": watt_seconds / duration,
            "signed_energy_Wh": watt_seconds / 3600.0, "min_sample_voltage_V": float(voltage[samples].min()),
            "min_sample_current_A": float(current[samples].min()), "peak_sample_current_A": float(current[peak]),
            "peak_sample_time_s": float(t[peak]), "peak_sample_power_W": float(power[samples].max()),
            "initial_low_reference_W": reference_power_W,
            "mean_power_above_initial_low_W": incremental_ws / duration if incremental_ws is not None else None,
            "energy_above_initial_low_Wh": incremental_ws / 3600.0 if incremental_ws is not None else None}


def recording_stats(values):
    return {"recordings": len(values), "mean": fmean(values), "min": min(values), "max": max(values)}


def run(source: Path, output: Path):
    metadata, traces, member_hashes = read_ground(source)
    low_limit = max(float(traces[fid][2].max()) for fid in RECORDINGS["A1"])
    rows, acquisition, primary, paired = [], {}, {}, {}
    for fid, trace in traces.items():
        t, _, current = trace
        meta = metadata[fid]
        route = meta["route"]
        spans = operating_windows(t, current, route, low_limit)
        initial_span = next((s for s in spans if s[0] == "initial_low_current"), None)
        initial_power = summarize_window(trace, *initial_span[1:])["mean_signed_pack_power_W"] if initial_span else None
        windows = [("full_recording", 0.0, float(t[-1])), *spans]
        by_label = {}
        for label, begin, end in windows:
            metrics = summarize_window(trace, begin, end, initial_power)
            rows.append({"flight_id": fid, "route": route, "date": meta["date"], "local_time": meta["local_time"],
                         "window": label, **metrics})
            by_label[label] = metrics
        primary[fid] = by_label["publisher_motor_off" if route == "A1" else "minimum_movement_plateau"]
        if "initial_low_current" in by_label:
            initial = by_label["initial_low_current"]
            paired[fid] = {"delta_current_A": primary[fid]["mean_signed_current_A"] - initial["mean_signed_current_A"],
                           "delta_power_W": primary[fid]["mean_power_above_initial_low_W"],
                           "incremental_energy_during_plateau_Wh": primary[fid]["energy_above_initial_low_Wh"],
                           "initial_low_window_s": [initial["start_s"], initial["end_s"]],
                           "initial_mean_signed_power_W": initial["mean_signed_pack_power_W"]}
        dt = np.diff(t)
        acquisition[fid] = {"samples": len(t), "median_dt_s": float(np.median(dt)), "max_dt_s": float(dt.max()),
                            "gaps_over_0_5s": int((dt > MAX_GAP_S + 1e-10).sum()),
                            "primary_window_s": [primary[fid]["start_s"], primary[fid]["end_s"]]}
    by_route = {}
    for route, ids in RECORDINGS.items():
        primary_values = {key: recording_stats([primary[fid][key] for fid in ids]) for key in (
            "mean_pack_voltage_V", "mean_signed_current_A", "mean_signed_pack_power_W")}
        comparisons = [paired[fid] for fid in ids if fid in paired]
        by_route[route] = {"condition": CONDITIONS[route], "recordings": list(ids), "primary": primary_values,
                           "paired_recordings": [fid for fid in ids if fid in paired],
                           "within_recording_increment": {key: recording_stats([row[key] for row in comparisons])
                                                          for key in ("delta_current_A", "delta_power_W")} if comparisons else None}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_data = buffer.getvalue().encode("utf-8")
    summary = {
        "schema": "CMU_M100_GROUND_LOADS_V1", "dataset_doi": "https://doi.org/10.1184/R1/12683453.v1",
        "paper_doi": "https://doi.org/10.1038/s41597-021-00930-x", "input_sha256": SOURCE_HASHES,
        "opened_measurement_members_sha256": member_hashes, "source_columns": list(SIGNALS),
        "source_definitions": "Publisher README.txt lines 257-262 and parameters.csv records 211-219. All on 2019-07-18, in successive condition blocks.",
        "analysis": "python3 -B -m tools.m100_reaudit.cmu_ground_loads",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(csv_data).hexdigest(),
        "sign_and_units": "Preserve the signed PL-200 pack readings, with positive current under load. The three stopped-motor traces establish a small negative measured baseline. Sensor offset and the true accessory load cannot be separated from these signals.",
        "segmentation": {"hardware_state": "The publisher provides one label per recording. Signal windows describe changes in measured current, with no independent onset, RPM, torque or thrust channel.",
                         "low_current_upper_A": low_limit, "source_of_low_bound": "Highest individual current sample among all three A1 recordings.",
                         "binning": "Bin samples by the integer floor of elapsed seconds and take the median current of each populated bin.",
                         "plateau": "For A2/A3 take the longest contiguous run of one-second bin medians within the relative band around the recording median. The plateau occupies the majority of each such recording.",
                         "relative_band": RELATIVE_PLATEAU_BAND, "internal_edge_guard_s": GUARD_S,
                         "edges": "Exclude one second inside each detected internal boundary of a plateau or a flanking low-current run. Retain original recording endpoints. All remaining time is labelled transition.",
                         "A1": "Use the entire recording as the publisher-designated stopped-motor window."},
        "integration": "Use adjacent native timestamps. Integrate signed V*I, V and I by the trapezoidal rule. Interpolate each sampled quantity linearly at a window boundary. Skip intervals over 0.5 s. Closed endpoint sample counts may share a boundary sample. The full_recording row overlaps its disjoint state windows.",
        "aggregation": "One equally weighted mean per recording, with the range between recordings. Integrate the pointwise difference V(t)*I(t) minus time-weighted mean V*I from that same recording's initial low-current window, where available. The power increment equals this integral divided by the plateau duration. No cross-recording zero correction or matched propeller comparison.",
        "sampling": {"recordings": len(traces), "samples": sum(info["samples"] for info in acquisition.values()),
                     "covered_s": sum(r["covered_s"] for r in rows if r["window"] == "full_recording"),
                     "gaps_over_0_5s": sum(info["gaps_over_0_5s"] for info in acquisition.values())},
        "acquisition": acquisition, "by_route": by_route, "paired_by_recording": paired,
        "interpretation": "These nine ground recordings cover stopped or minimum motor movement. Current and electrical power describe the complete monitored DC path. The sequence changes pack voltage and the records have no independently verified matching rotor speeds, battery identity, or separate ESC, motor-phase or shaft measurements.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ground_recordings.csv").write_bytes(csv_data)
    (output / "ground_loads.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/external/m100")
    parser.add_argument("--output", type=Path, default=ROOT / "results/m100_reaudit/cmu_component_evidence")
    args = parser.parse_args()
    result = run(args.source, args.output)
    print(json.dumps(result["by_route"], indent=2))
