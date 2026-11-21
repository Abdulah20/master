"""Describe terminal-voltage changes during recorded takeoff and landing load changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from . import external
from .cmu_calibration import MANIFEST, MANIFEST_SHA256, digest


OUTPUT = external.ROOT / "results/m100_reaudit/cmu_component_evidence"
WINDOW_S = 2.0
SEARCH_RADIUS_S = 10.0
MIN_CHANGE_A = 5.0


def paired_means(time, signal, centres):
    """Integrate a piecewise linear signal exactly across each pair of adjacent windows."""
    cumulative = np.r_[0.0, np.cumsum(np.diff(time) * (signal[1:] + signal[:-1]) / 2)]
    query = np.vstack([centres - WINDOW_S, centres, centres + WINDOW_S])
    left = np.clip(np.searchsorted(time, query, side="right") - 1, 0, time.size - 2)
    # Add the partial last trapezoid to the integral through the left sample.
    edge_signal = np.interp(query, time, signal)
    integral = cumulative[left] + (query - time[left]) * (signal[left] + edge_signal) / 2
    return np.diff(integral, axis=0) / WINDOW_S


def transition(samples, event_time, direction):
    time, current, voltage = samples[["time", "battery_current", "battery_voltage"]].to_numpy().T
    eligible = (np.abs(time - event_time) <= SEARCH_RADIUS_S) & (time >= time[0] + WINDOW_S)
    centres = time[eligible & (time <= time[-1] - WINDOW_S)]
    if not centres.size:
        return {"status": "no_full_window_pair"}
    i_before, i_after = paired_means(time, current, centres)
    delta = i_after - i_before
    index = int(np.argmax(direction * delta))
    change_a = float(delta[index])
    centre = float(centres[index])
    v_before, v_after = paired_means(time, voltage, np.array([centre]))[:, 0]
    change_v = float(v_after - v_before)
    resolved = direction * change_a >= MIN_CHANGE_A
    return {
        "status": "included" if resolved else "current_change_below_cut",
        "transition_time_s": centre, "transition_minus_event_s": centre - event_time,
        "current_before_A": float(i_before[index]), "current_after_A": float(i_after[index]),
        "voltage_before_V": float(v_before), "voltage_after_V": float(v_after),
        "delta_current_A": change_a, "delta_voltage_V": change_v,
        "apparent_mV_per_A": -1000 * change_v / change_a if resolved else None,
    }


def summarize(frame):
    result = {}
    for name, events in frame.groupby("event", sort=False):
        included = events.loc[events.status == "included"]
        slopes = included["apparent_mV_per_A"]
        result[name] = {
            "flights": len(events), "included_flights": len(included),
            "status_counts": {str(s): int(n) for s, n in events.status.value_counts().items()},
            "opposite_voltage_and_current_changes": int((slopes > 0).sum()),
            "median_delta_current_A": float(included.delta_current_A.median()) if len(included) else None,
            "median_delta_voltage_V": float(included.delta_voltage_V.median()) if len(included) else None,
            "apparent_mV_per_A_percentiles": {f"p{x}": float(slopes.quantile(x / 100))
                                               for x in (10, 25, 50, 75, 90)} if len(slopes) else {},
        }
    return result


def extract(output):
    archive_path = external.INPUT / "flights.zip"
    external.check_hash(archive_path, external.SHA256["flights.zip"])
    external.check_hash(MANIFEST, MANIFEST_SHA256)
    manifest = pd.read_csv(MANIFEST)
    train = manifest.loc[manifest.included & manifest.split.eq("train") & manifest.published_role.eq("fit")]
    rows = []
    with ZipFile(archive_path) as archive:
        for fid in train.flight_id:
            with archive.open(f"flights/{fid}.csv") as member:
                samples = pd.read_csv(member, usecols=["time", "position_z", "battery_current", "battery_voltage"])
            if flags := external.quality_flags(samples):
                raise ValueError(f"Invalid source for flight {fid}: {flags}")
            start, _, _, end = external.phase_boundaries(samples.position_z.to_numpy())
            for name, event, direction in (("takeoff_load_rise", start, 1), ("landing_load_release", end, -1)):
                event_time = float(samples.time.iloc[event])
                rows.append(dict(flight_id=int(fid), split="train", event=name, event_time_s=event_time,
                                 **transition(samples, event_time, direction)))
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table = output / "load_transitions.csv"
    external.write_table(frame, table)
    report = {
        "dataset": "CMU Rodrigues et al., doi:10.1184/R1/12683453 (CC BY 4.0)",
        "scope": "Descriptive natural load transitions in archived train flights only. One candidate per flight per event.",
        "selector": "Within the search radius of the published altitude event, select the sampled centre with the "
                    "largest current rise at takeoff, or largest fall at landing. Average each signal by time in two "
                    "adjacent windows around that centre. The voltage does not enter event selection. Preserve signs.",
        "search_radius_s": SEARCH_RADIUS_S, "window_each_side_s": WINDOW_S, "minimum_signed_current_change_A": MIN_CHANGE_A,
        "maximum_allowed_sampling_gap_s": external.MAX_GAP_S,
        "statistic": "For each qualifying pair: -1000*(V_after-V_before)/(I_after-I_before), mV/A. "
                     "Summaries use one value per flight in each direction. Percentiles describe the sample, "
                     "not uncertainty intervals or a controlled pulse experiment.",
        "interpretation": "Effective pack-output voltage change on these time windows. Includes unresolved dynamics, "
                          "electrical path and sensing. Neither instantaneous TB47D R0 nor a motor or ESC parameter.",
        "events": summarize(frame),
        "sha256": {"flights.zip": external.SHA256["flights.zip"], "flight_manifest.csv": MANIFEST_SHA256,
                   "external.py": digest(Path(external.__file__)), "cmu_load_transitions.py": digest(Path(__file__)),
                   table.name: digest(table)},
    }
    (output / "load_transitions.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    extract(parser.parse_args().output)
