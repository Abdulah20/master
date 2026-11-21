"""Record one preflight feature row and the Gazebo pack's airborne energy label."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
from pathlib import Path
import tempfile


CALIBRATION_ID = "CMU_TRANSIENT_AIRBORNE_V1:831973dc40cdc338b7086a1ba824c57c8670bdeff90f7fcdf806a56905ff02a9"
BOUNDARY_SOURCE = "PX4_VEHICLE_LAND_DETECTED"
POWER_SOURCES = ("native_pack_model", "cmu_local", "cmu_pooled", "recorded_manager")


def _number(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def _same(a, b):
    # Sum round-off for native physics intervals (s or Wh).
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)


def _budget_status(budget, run_id, recording_start_s):
    if not isinstance(budget, dict) or budget.get("run_id") != run_id:
        return "NO_MATCHING_NATIVE_BUDGET"
    if budget.get("calibration_id") != CALIBRATION_ID:
        return "CALIBRATION_ID_MISMATCH"
    if budget.get("boundary_source") != BOUNDARY_SOURCE:
        return "NATIVE_BOUNDARIES_UNAVAILABLE"
    if budget.get("status") != "COMPLETE":
        return str(budget.get("status") or "NO_COMPLETED_AIRBORNE_WINDOW")
    start, end, duration = (budget.get(k) for k in ("start_s", "end_s", "duration_s"))
    covered, gap, energy = (budget.get(k) for k in (
        "covered_s", "uncovered_s", "estimated_airborne_energy_Wh"))
    if (not all(_number(x) for x in (start, end, duration, covered, gap, energy))
            or recording_start_s is None or start < recording_start_s
            or end <= start or energy <= 0 or gap != 0
            or not _same(duration, end - start) or not _same(covered, duration)):
        return "INCOMPLETE_AIRBORNE_COVERAGE"
    bins = budget.get("sources")
    if not isinstance(bins, dict) or set(bins) != set(POWER_SOURCES):
        return "INVALID_SOURCE_LEDGER"
    for item in bins.values():
        if (not isinstance(item, dict) or not _number(item.get("seconds"))
                or not _number(item.get("energy_Wh"))
                or (item["seconds"] == 0) != (item["energy_Wh"] == 0)):
            return "INVALID_SOURCE_LEDGER"
    if (not _same(sum(item["seconds"] for item in bins.values()), covered)
            or not _same(sum(item["energy_Wh"] for item in bins.values()), energy)):
        return "INVALID_SOURCE_LEDGER"
    return "COMPLETE"


def training_row(config, run_id, outcome, budget, *, recording_start_s, epoch_changed):
    status = ("ENTITY_OR_CLOCK_RESET_DURING_RECORDING" if epoch_changed else
              _budget_status(budget, run_id, recording_start_s))
    numeric_energy = budget["estimated_airborne_energy_Wh"] if status == "COMPLETE" else None
    auto_complete = outcome == "COMPLETE" and run_id.startswith("m100_")
    if numeric_energy is not None and not auto_complete:
        status = "MISSION_PLAN_NOT_COMPLETED"
    # Speed/altitude are marginal commands of the fitted CMU release. Real
    # aircraft airflow is not a direct reference for a commanded Gazebo wind.
    inside = (config["payload_kg"] in (0.0, 0.25, 0.5)
              and 4 <= config["cruise_speed_mps"] <= 12 and 25 <= config["altitude_m"] <= 100
              and math.isclose(config["temperature_c"], 15, abs_tol=1e-6)
              and math.isclose(config["wind_speed_mps"], 0, abs_tol=1e-6))
    label_source = "CMU_calibrated_simulation" if inside else "simulation_transfer_outside_CMU_reference"
    if numeric_energy is None:
        label_source = "unavailable"
    elif budget["sources"]["recorded_manager"]["seconds"] >= budget["covered_s"] - 1e-8:
        label_source = "simulation_manager_fallback_only"
    eligible = auto_complete and numeric_energy is not None and label_source == "CMU_calibrated_simulation"
    if status == "COMPLETE" and not eligible:
        status = ("COMPLETE_MANAGER_FALLBACK_ONLY" if label_source == "simulation_manager_fallback_only"
                  else "COMPLETE_OUTSIDE_CMU_REFERENCE")
    matched = (budget if isinstance(budget, dict) and budget.get("run_id") == run_id
               and not epoch_changed else {})
    bins = matched.get("sources")
    row = {
        "run_id": run_id,
        "label_source": label_source,
        "training_eligible": eligible,
        "training_status": status,
        "mission_outcome": outcome,
        "payload_kg": config["payload_kg"],
        "commanded_speed_mps": config["cruise_speed_mps"],
        "commanded_altitude_agl_m": config["altitude_m"],
        "planned_outbound_measurement_m": config["route_length_m"] if config["cruise_speed_mps"] else 0,
        "planned_hover_duration_s": config["hover_duration_s"] if config["cruise_speed_mps"] == 0 else None,
        "commanded_wind_mps": config["wind_speed_mps"],
        "commanded_wind_direction_deg": config["wind_direction_deg"],
        "ambient_reference_C": config["temperature_c"],
        "estimated_airborne_energy_Wh": numeric_energy,
        "calibration_id": matched.get("calibration_id"),
        "airborne_duration_s": matched.get("duration_s"),
        "uncovered_s": matched.get("uncovered_s"),
        "budget_reason": matched.get("reason") or (status if numeric_energy is None else None),
    }
    for source in POWER_SOURCES:
        entry = bins.get(source) if isinstance(bins, dict) else None
        row[f"{source}_s"] = entry.get("seconds") if isinstance(entry, dict) else None
    return row


def publish_training_row(csv_path, row, budget):
    """Install a whole CSV atomically. A finalization retry reuses identical bytes."""
    folder = csv_path.parent / "training"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / csv_path.name
    data = io.StringIO(newline="")
    writer = csv.DictWriter(data, fieldnames=list(row), lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    encoded = data.getvalue().encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=folder, prefix=f".{csv_path.stem}.") as temporary:
        temporary.write(encoded)
        temporary.flush()
        os.fsync(temporary.fileno())
        try:
            os.link(temporary.name, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"training row already exists with different contents: {path}")
    return {
        "schema": "M100_NATIVE_AIRBORNE_TRAINING_V1",
        "relative_path": str(path.relative_to(csv_path.parent)),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "status": row["training_status"],
        "training_eligible": row["training_eligible"],
        "label_source": row["label_source"],
        "estimated_airborne_energy_Wh": row["estimated_airborne_energy_Wh"],
        "native_budget": budget,
    }
