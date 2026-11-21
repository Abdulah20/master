"""Map estimated aircraft electrical power to a pack operating point at a recorded discharged charge."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from . import external
from .cmu_calibration import DESTINATION as CALIBRATION, digest


OUTPUT = external.ROOT / "results/m100_reaudit/cmu_component_evidence"


def operating_point(voltage_fit, power_W, discharged_Ah):
    """Static high-voltage solution of P=VI and the measured-current regression, inside its ranges."""
    if voltage_fit["fit_status"] != "FULL_RANK_EXPECTED_SIGNS":
        raise ValueError("The loaded-voltage fit requires identified, positive coefficients")
    if not math.isfinite(power_W) or not math.isfinite(discharged_Ah) or power_W <= 0:
        return {"status": "invalid_loaded_input", "estimated_bus_V": None, "estimated_pack_A": None}
    ranges = voltage_fit["training_predictor_ranges"]
    q_min, q_max = ranges["net_discharged_Ah"]
    if not q_min <= discharged_Ah <= q_max:
        return {"status": "charge_outside_training", "estimated_bus_V": None, "estimated_pack_A": None}
    coeff = voltage_fit["coefficients"]
    source_term = coeff["a_V"] - coeff["k_V_per_Ah"] * discharged_Ah
    discriminant = source_term**2 - 4 * coeff["r_V_per_A"] * power_W
    if source_term <= 0 or discriminant < 0:
        return {"status": "no_positive_load_solution", "estimated_bus_V": None, "estimated_pack_A": None}
    volts = (source_term + math.sqrt(discriminant)) / 2
    amps = power_W / volts
    i_min, i_max = ranges["current_A"]
    if not i_min <= amps <= i_max:
        return {"status": "current_outside_training", "estimated_bus_V": None, "estimated_pack_A": None}
    return {"status": "within_marginal_training_ranges", "estimated_bus_V": volts, "estimated_pack_A": amps}


def extract(output):
    fit_path = CALIBRATION / "fit.json"
    saved = json.loads(fit_path.read_text())
    table = CALIBRATION / "flight_residuals.csv"
    external.check_hash(table, saved["table_sha256"][table.name])
    external.check_hash(external.INPUT / "flights.zip", external.SHA256["flights.zip"])
    model = saved["aircraft"]["development_choice"]["airborne_energy_Wh"]
    frame = pd.read_csv(table)
    forecasts = frame.loc[frame.split.eq("train") & frame.model.eq(model) & frame.scope.isin(external.PHASES)]
    forecasts = forecasts.set_index(["flight_id", "scope"])
    if not forecasts.index.is_unique:
        raise ValueError("Duplicate phase in the frozen training forecasts")

    rows = []
    with ZipFile(external.INPUT / "flights.zip") as archive:
        for fid in saved["cohorts"]["train"]["flight_ids"]:
            with archive.open(f"flights/{fid}.csv") as member:
                samples = pd.read_csv(member, usecols=["time", "position_z", "battery_current", "battery_voltage"])
            if flags := external.quality_flags(samples):
                raise ValueError(f"Invalid measurement stream in flight {fid}: {flags}")
            time = samples.time.to_numpy()
            current = samples.battery_current.to_numpy()
            q = np.r_[0.0, np.cumsum(np.diff(time) * (current[1:] + current[:-1]) / 7200)]
            start, level, descent, end = external.phase_boundaries(samples.position_z.to_numpy())
            boundaries = (start + 1, level, descent, end - 1)
            for phase, left, right in zip(external.PHASES, boundaries[:-1], boundaries[1:], strict=True):
                midpoint = (time[left] + time[right]) / 2
                index = left + int(np.argmin(np.abs(time[left:right + 1] - midpoint)))
                power = float(forecasts.loc[(fid, phase), "predicted_mean_power_W"])
                rows.append({
                    "flight_id": fid, "phase": phase, "recorded_time_s": float(time[index]),
                    "recorded_net_discharged_Ah": float(q[index]), "estimated_phase_mean_W": power,
                    **operating_point(saved["loaded_voltage"], power, float(q[index])),
                })
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "pack_operating_points.csv"
    external.write_table(pd.DataFrame(rows), destination)
    report = {
        "source": "CMU Rodrigues et al., doi:10.1184/R1/12683453 (CC BY 4.0)",
        "scope": "One static, model-derived pack operating point per recorded phase of each training flight.",
        "model_for_total_pack_power": model,
        "power_input": "Frozen preflight estimate of phase-average electrical power for the entire aircraft.",
        "charge_input": "Integral of signed measured current since recording start, sampled nearest the actual "
                        "phase midpoint. Phase boundaries use the archived altitude record.",
        "equations": "s=a-k*q, V=(s+sqrt(s^2-4*r*P))/2, I=P/V. Marginal q and I training ranges enforced.",
        "interpretation": "Conditional on observed charge. These values are estimates at average phase load; they "
                          "are not instantaneous measurements, flight forecasts from initial SOC, motor currents, "
                          "nor a dynamic feedback path. No model states are propagated and no accuracy is inferred.",
        "flights": len(saved["cohorts"]["train"]["flight_ids"]), "rows": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "sha256": {fit_path.name: digest(fit_path), table.name: digest(table), destination.name: digest(destination),
                   "flights.zip": external.SHA256["flights.zip"], "external.py": digest(Path(external.__file__)),
                   "cmu_pack_operating_points.py": digest(Path(__file__))},
    }
    (output / "pack_operating_points.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    extract(parser.parse_args().output)
