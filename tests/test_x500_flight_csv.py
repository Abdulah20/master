from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from data_processing.build_x500_flight_csv import _deduplicate_last


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "runs" / "x500_first_flight_v1_baseline"
CSV_PATH = RUN / "x500_first_flight_v1.csv"
VALIDATION_PATH = RUN / "validation.json"


def test_last_sample_resolves_raw_source_duplicates() -> None:
    timestamps, keep, count = _deduplicate_last(np.array([1.0, 2.0, 2.0, 3.0]), "test")
    assert timestamps == pytest.approx([1.0, 2.0, 3.0])
    assert keep.tolist() == [True, False, True, True]
    assert count == 1


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="Accepted first-flight CSV is unavailable")
def test_accepted_first_flight_csv_regression() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 566
    timestamps = np.array([float(row["simulation_time_s"]) for row in rows])
    energy = np.array([float(row["cumulative_energy_j"]) for row in rows])
    assert np.all(np.diff(timestamps) > 0.0)
    assert len(np.unique(timestamps)) == len(timestamps)
    assert np.all(np.diff(energy) >= -1e-8)
    assert {row["flight_phase"] for row in rows} == {
        "prearm",
        "takeoff",
        "hover",
        "move",
        "post_move_hover",
        "landing",
        "landed",
    }
    required = {
        "q_w",
        "q_x",
        "q_y",
        "q_z",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "payload_mass_kg",
        "target_relative_altitude_m",
        "wind_east_m_s",
        "wind_north_m_s",
        "wind_up_m_s",
    }
    assert required.issubset(rows[0])
    assert {row["energy_fidelity"] for row in rows} == {"C_NOT_CALIBRATED"}


@pytest.mark.skipif(not VALIDATION_PATH.is_file(), reason="First-flight validation is unavailable")
def test_accepted_first_flight_validation_regression() -> None:
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    assert validation["result"] == "pass"
    assert all(validation["checks"].values())
    power = validation["phase_mean_estimated_electrical_load_w"]
    assert power["takeoff"] > power["landed"]
    assert power["move"] > power["landed"]
