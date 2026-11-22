"""Compare direct M100 training labels to their previous estimates and matched field flights."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from . import external, native_training


RESULTS = external.ROOT / "results/m100_reaudit"
PREVIOUS = RESULTS / "transient_airborne"
DIRECT = RESULTS / "direct_airborne_three"


def table(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def field_measurements(cases):
    """Reintegrate the measured V and I at the original CMU timestamps and published altitude gates."""
    calibration = RESULTS / "cmu_calibration"
    fit = json.loads((calibration / "fit.json").read_text())
    manifest_path = external.OUTPUT / "flight_manifest.csv"
    params_path = external.INPUT / "parameters.csv"
    traces_path = external.INPUT / "flights.zip"
    for path in (params_path, traces_path, manifest_path):
        external.check_hash(path, fit["input_and_code_sha256"][str(path.relative_to(external.ROOT))])
    external.check_hash(calibration / "flight_residuals.csv", fit["table_sha256"]["flight_residuals.csv"])
    members = pd.read_csv(manifest_path)
    members = members[members.included & members.split.isin(("train", "validation"))].copy()
    members["commanded_altitude_original_m"] = pd.to_numeric(members.commanded_altitude_original_m)
    params = pd.read_csv(params_path).set_index("flight")
    archived = pd.read_csv(calibration / "flight_residuals.csv")
    archived = archived[archived.scope.eq("airborne") & archived.model.eq("published_phase_preflight")]
    archived = archived.set_index("flight_id")
    if not params.index.is_unique or not archived.index.is_unique:
        raise ValueError("Duplicate CMU flight identifiers in a reference table")
    previous_matches = {r["case_id"]: r["cmu_flight_ids"] for r in table(PREVIOUS / "matched_field_flights.csv")}
    records = []
    with ZipFile(traces_path) as archive:
        for case in cases:
            cid = case["case_id"]
            peers = members[members.payload_kg.eq(case["payload_kg"])
                            & members.commanded_speed_mps.eq(case["cruise_speed_mps"])
                            & members.commanded_altitude_original_m.eq(case["altitude_m"])].sort_values("flight_id")
            if "|".join(map(str, peers.flight_id)) != previous_matches[cid]:
                raise ValueError(f"{cid}: field cohort differs from the previous matched comparison")
            for peer in peers.itertuples():
                fid = peer.flight_id
                meta = params.loc[fid]
                if (meta.speed, float(meta.altitude), meta.payload / 1000, meta.route) != (
                        case["cruise_speed_mps"], case["altitude_m"], case["payload_kg"], peer.route):
                    raise ValueError(f"CMU flight {fid}: inconsistent command metadata")
                source = archive.read(f"flights/{fid}.csv")
                signals = pd.read_csv(io.BytesIO(source), usecols=external.COLUMNS)
                if external.quality_flags(signals):
                    raise ValueError(f"CMU flight {fid}: unusable observation")
                left, _, _, right = external.phase_boundaries(signals.position_z.to_numpy())
                # Same outer endpoints as external.measured_scopes. All interior samples are contiguous.
                airborne = signals.iloc[left + 1:right]
                t = airborne.time.to_numpy()
                p = (airborne.battery_voltage * airborne.battery_current).to_numpy()
                duration = t[-1] - t[0]
                energy = math.fsum((p[:-1] + p[1:]) * np.diff(t)) / 7200
                watts = energy * 3600 / duration
                saved = archived.loc[fid]
                if not np.allclose((energy, duration, watts), (saved.observed_energy_Wh,
                        saved.observed_duration_s, saved.observed_mean_power_W), rtol=1e-10, atol=1e-10):
                    raise ValueError(f"CMU flight {fid}: measured airborne totals differ from the calibration archive")
                records.append(dict(case_id=cid, flight_id=fid, split=peer.split, route=meta.route,
                                    payload_kg=peer.payload_kg, commanded_speed_mps=meta.speed,
                                    commanded_altitude_agl_m=float(meta.altitude), airborne_first_time_s=t[0],
                                    airborne_last_time_s=t[-1], airborne_samples=len(t), airborne_duration_s=duration,
                                    measured_airborne_energy_Wh=energy, measured_airborne_mean_power_W=watts,
                                    csv_member_sha256=hashlib.sha256(source).hexdigest()))
    return pd.DataFrame(records)


def direct_row(receipt, training, provenance):
    if not training.exists():
        if provenance.exists() and "airborne_training" in json.loads(provenance.read_text()):
            raise ValueError(f"Published airborne row is missing: {training}")
        return {"run_id": receipt.get("run_id"), "training_status": "NO_DIRECT_TRAINING_FILE"}
    recording = json.loads(provenance.read_text())
    proof = recording["airborne_training"]
    contents = training.read_bytes()
    native_training.training_example(contents, proof, receipt["case_id"], receipt["run_id"], receipt["outcome"])
    row = next(csv.DictReader(io.StringIO(contents.decode("utf-8"), newline="")))
    row["runtime_pack_plugin_sha256"] = recording["source_files"]["gazebo_m100_pack_drive_plugin"]["sha256"]
    budget = proof.get("native_budget") or {}
    if (budget.get("run_id") == receipt["run_id"] and budget.get("start_s") is not None
            and budget.get("end_s") is not None and budget.get("uncovered_s") is not None):
        row["known_subtotal_Wh"] = budget["known_energy_Wh"]
        row["covered_s"] = budget["covered_s"]
        row["missing_s"] = budget["uncovered_s"]
    return row


def comparison(case, row, old, peers):
    energy = float(row["estimated_airborne_energy_Wh"]) if row.get("estimated_airborne_energy_Wh") else math.nan
    duration = float(row["airborne_duration_s"]) if row.get("airborne_duration_s") else math.nan
    if math.isfinite(energy) and not (energy > 0 and math.isfinite(duration) and duration > 0):
        raise ValueError(f"{case['case_id']}: nonphysical training label or duration")
    power = energy * 3600 / duration if duration > 0 else math.nan
    old_energy = float(old["practical_total_Wh"])
    old_seconds = float(old["duration_s"])
    old_power = float(old["practical_mean_power_W"])
    energies = peers.measured_airborne_energy_Wh
    powers = peers.measured_airborne_mean_power_W
    durations = peers.airborne_duration_s
    known = row.get("known_subtotal_Wh", math.nan)
    covered = row.get("covered_s", math.nan)
    return dict(case_id=case["case_id"], payload_kg=case["payload_kg"], direct_run_id=row["run_id"],
                previous_run_id=old["run_id"], training_status=row["training_status"], training_eligible=row.get("training_eligible"),
                direct_energy_Wh=energy, previous_energy_Wh=old_energy, cmu_mean_energy_Wh=energies.mean(),
                direct_minus_previous_Wh=energy-old_energy, direct_minus_previous_energy_pct=100*(energy/old_energy-1),
                direct_minus_cmu_Wh=energy-energies.mean(), direct_minus_cmu_energy_pct=100*(energy/energies.mean()-1),
                direct_duration_s=duration, previous_duration_s=old_seconds, cmu_mean_duration_s=durations.mean(),
                direct_mean_power_W=power, previous_mean_power_W=old_power, cmu_mean_power_W=powers.mean(),
                direct_minus_previous_power_pct=100*(power/old_power-1), direct_minus_cmu_power_pct=100*(power/powers.mean()-1),
                known_subtotal_Wh=known, covered_s=covered, missing_s=row.get("missing_s", math.nan),
                known_interval_mean_power_W=3600*known/covered if covered > 0 else math.nan,
                duration_coverage_pct=100*covered/duration if duration > 0 else math.nan,
                known_subtotal_minus_previous_Wh=known-old_energy, known_subtotal_minus_cmu_Wh=known-energies.mean(),
                cmu_flight_ids="|".join(map(str, peers.flight_id)), cmu_flights=len(peers),
                cmu_min_energy_Wh=energies.min(), cmu_max_energy_Wh=energies.max(),
                cmu_min_mean_power_W=powers.min(), cmu_max_mean_power_W=powers.max(),
                cmu_min_duration_s=durations.min(), cmu_max_duration_s=durations.max(),
                runtime_pack_plugin_sha256=row.get("runtime_pack_plugin_sha256"))


def run(output, runs=None):
    cases = json.loads((output / "protocol.json").read_text())["cases"]
    selected = json.loads((output / "selection.json").read_text())["cases"]
    choices = {r["case_id"]: r for r in selected}
    if len(choices) != len(selected) or set(choices) != {case["case_id"] for case in cases}:
        raise ValueError("Final recording selection differs from the planned cases")
    hashes = {name: digest for digest, name in (s.split() for s in (PREVIOUS / "SHA256SUMS").read_text().splitlines())}
    for name in ("matched_field_flights.csv", "practical_flights.csv", "training_estimates.csv"):
        external.check_hash(PREVIOUS / name, hashes[name])
    old = {r["case_id"]: r for r in table(PREVIOUS / "practical_flights.csv")}
    old_training = {r["case_id"]: r for r in table(PREVIOUS / "training_estimates.csv")}
    field = field_measurements(cases)
    rows = []
    for case in cases:
        cid = case["case_id"]
        choice = choices[cid]
        if runs is None:
            folder = output / "evidence" / cid
            receipt = json.loads((folder / "receipt.json").read_text())
            recorded = direct_row(receipt, folder / "training.csv", folder / "provenance.json")
        else:
            folder = runs / cid / choice["attempt"]
            receipt = json.loads((folder / (cid + ".json")).read_text())
            rid = receipt.get("run_id")
            recorded = direct_row(receipt, folder / "training" / f"{rid}.csv", folder / f"{rid}.provenance.json")
        if any(receipt.get(k) != v for k, v in case.items()):
            raise ValueError(f"{cid}: acquired plan differs from protocol")
        if receipt.get("run_id") != choice["run_id"]:
            raise ValueError(f"{cid}: acquired training data differs from the published selection")
        if old[cid]["run_id"] != old_training[cid]["run_id"] or not math.isclose(
                float(old[cid]["practical_total_Wh"]), float(old_training[cid]["estimated_airborne_energy_Wh"]), abs_tol=1e-8):
            raise ValueError(f"{cid}: previous practical budget differs from the training label")
        result = comparison(case, recorded, old[cid], field[field.case_id.eq(cid)])
        result["attempt"] = choice["attempt"]
        rows.append(result)
    external.write_table(field, output / "cmu_matched_measurements.csv")
    external.write_table(pd.DataFrame(rows), output / "comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DIRECT)
    parser.add_argument("--runs", type=Path, help="Original acquisition directory, in place of the archived receipts")
    args = parser.parse_args()
    run(args.output, args.runs)
