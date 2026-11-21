"""Calibrate preflight estimates of CMU airborne pack power, duration and energy."""

from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd

from .external import PHASES


SCOPES = (*PHASES, "airborne")
COMMANDS = ("payload_kg", "commanded_speed_mps", "commanded_altitude_agl_m")
OBSERVATIONS = {
    "mean_power_W": "observed_mean_power_W",
    "duration_s": "covered_duration_s",
    "energy_Wh": "observed_energy_Wh",
}
REFERENCE = "published_phase_preflight"
OFFSET_ONLY = "offset_only_phase_preflight"
CALIBRATED = "calibrated_phase_preflight"
MEAN = "train_mean"
REFERENCE_SPEED_MPS = 8.0


def _observations(flight: dict) -> dict:
    values = {
        phase: {name: float(flight["scopes"][phase][source]) for name, source in OBSERVATIONS.items()}
        for phase in SCOPES
    }
    for phase, targets in values.items():
        if not all(math.isfinite(value) and value > 0 for value in targets.values()):
            raise ValueError(f"Invalid airborne measurement for flight {flight['flight_id']}, {phase}")
    return values


def _fit_line(x: np.ndarray, y: np.ndarray, *, nonnegative: bool = False) -> tuple[float, float, int]:
    design = np.column_stack((np.ones(len(x)), x))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2:
        return float(y.mean()), 0.0, int(rank)
    intercept, slope = map(float, coefficients)
    # For a line with a positive predictor, the constrained least-squares
    # solution is on one of the axes when its unconstrained optimum is infeasible.
    if nonnegative and slope < 0:
        return float(y.mean()), 0.0, int(rank)
    if nonnegative and intercept < 0:
        return 0.0, float(np.dot(x, y) / np.dot(x, x)), int(rank)
    return intercept, slope, int(rank)


def _duration_feature(phase, speed, altitude):
    return 1.0 / speed if phase == "cruise" else altitude


def predict(calibration: dict, *, base_phase_power_W: dict, commanded_speed_mps: float,
            commanded_altitude_agl_m: float) -> dict:
    """Predict from planned commands and externally computed reference powers alone."""
    speed, altitude = float(commanded_speed_mps), float(commanded_altitude_agl_m)
    if not (math.isfinite(speed) and speed > 0 and math.isfinite(altitude) and altitude > 0):
        raise ValueError("A delivery flight requires positive, finite speed and altitude commands")
    forecasts = {REFERENCE: {}, OFFSET_ONLY: {}, CALIBRATED: {}}
    for phase in PHASES:
        coefficients = calibration["coefficients"][phase]
        clock = coefficients["duration"]
        duration = clock["intercept_s"] + clock["slope"] * _duration_feature(phase, speed, altitude)
        base = float(base_phase_power_W[phase])
        offset_only = base + coefficients["offset_only_W"]
        correction = coefficients["power_correction"]
        adjusted = base + correction["offset_W"] + correction["slope_W_per_mps"] * (
            speed - calibration["reference_speed_mps"]
        )
        for model, power in ((REFERENCE, base), (OFFSET_ONLY, offset_only), (CALIBRATED, adjusted)):
            if not all(math.isfinite(value) and value > 0 for value in (power, duration)):
                raise ValueError(f"Nonphysical preflight prediction for {phase} in {model}")
            forecasts[model][phase] = {
                "mean_power_W": power, "duration_s": duration, "energy_Wh": power * duration / 3600.0,
            }
    for phases in forecasts.values():
        elapsed = sum(phases[p]["duration_s"] for p in PHASES)
        energy = sum(phases[p]["energy_Wh"] for p in PHASES)
        phases["airborne"] = {"mean_power_W": 3600.0 * energy / elapsed, "duration_s": elapsed, "energy_Wh": energy}
    forecasts[MEAN] = {scope: values.copy() for scope, values in calibration["training_means"].items()}
    return forecasts


def _metrics(rows: pd.DataFrame) -> dict:
    scores = {}
    for (split, scope, model), group in rows.groupby(["split", "scope", "model"], sort=False):
        metrics = {"flights": len(group)}
        for target, unit in (("mean_power_W", "W"), ("duration_s", "s"), ("energy_Wh", "Wh")):
            errors = group[f"residual_{target}"].to_numpy(dtype=float)
            absolute = np.abs(errors)
            metrics.update({
                f"MAE_{unit}": float(absolute.mean()), f"RMSE_{unit}": float(np.sqrt(np.mean(errors**2))),
                f"bias_{unit}": float(errors.mean()), f"P95_abs_{unit}": float(np.quantile(absolute, 0.95)),
            })
        scores.setdefault(split, {}).setdefault(scope, {})[model] = metrics
    return scores


def fit(flights: list[dict]) -> tuple[dict, pd.DataFrame]:
    """Fit on complete training flights, then score the supplied development splits."""
    splits = {f["split"] for f in flights}
    if "train" not in splits or not splits <= {"train", "validation"}:
        raise ValueError("Expected train or validation records and at least one training flight")
    ordered = sorted(flights, key=lambda f: int(f["flight_id"]))
    if len({int(f["flight_id"]) for f in ordered}) != len(ordered):
        raise ValueError("Flight IDs must be unique across development splits")
    for flight in ordered:
        payload, speed, altitude = (float(flight[key]) for key in COMMANDS)
        reference_powers = [float(flight["base_phase_power_W"][phase]) for phase in PHASES]
        if not (math.isfinite(payload) and payload >= 0 and
                all(math.isfinite(v) and v > 0 for v in (speed, altitude, *reference_powers))):
            raise ValueError(f"Invalid command or reference phase power for flight {flight['flight_id']}")

    train = [f for f in ordered if f["split"] == "train"]
    observed_train = [_observations(f) for f in train]
    speeds = np.array([float(f["commanded_speed_mps"]) for f in train])
    altitudes = np.array([float(f["commanded_altitude_agl_m"]) for f in train])
    coefficients = {}
    for phase in PHASES:
        base = np.array([float(f["base_phase_power_W"][phase]) for f in train])
        power = np.array([m[phase]["mean_power_W"] for m in observed_train])
        power_residual = power - base
        offset, speed_slope, power_rank = _fit_line(speeds - REFERENCE_SPEED_MPS, power_residual)
        durations = np.array([m[phase]["duration_s"] for m in observed_train])
        time_offset, time_slope, time_rank = _fit_line(
            _duration_feature(phase, speeds, altitudes), durations, nonnegative=True,
        )
        coefficients[phase] = {
            "offset_only_W": float(power_residual.mean()),
            "power_correction": {"offset_W": offset, "slope_W_per_mps": speed_slope, "design_rank": power_rank},
            "duration": {"intercept_s": time_offset, "slope": time_slope, "design_rank": time_rank,
                         "predictor": "1 / commanded_speed_mps" if phase == "cruise" else "commanded_altitude_agl_m",
                         "slope_unit": "m" if phase == "cruise" else "s/m"},
        }

    domain = {key: sorted({float(f[key]) for f in train}) for key in COMMANDS}
    training_commands = {tuple(float(f[key]) for key in COMMANDS) for f in train}
    route_counts = {split: dict(sorted(Counter(str(f["route"]) for f in ordered if f["split"] == split).items()))
                    for split in sorted(splits)}
    summary = {
        "schema": "cmu_preflight_phase_calibration_v1", "fit_split": "train", "reference_speed_mps": REFERENCE_SPEED_MPS,
        "preflight_inputs": list(COMMANDS), "coefficients": coefficients,
        "split_flight_counts": {split: sum(route_counts[split].values()) for split in route_counts},
        "target": "Mean pack electrical power, duration and integrated pack energy over the CMU airborne phase windows.",
        "power_definition": "P_cal = base_phase_power_W + offset_W + slope_W_per_mps * (commanded_speed_mps - 8). "
                            "The reference is the supplied original phase model at 15 C using planned altitude and payload.",
        "power_coefficient_order": ["offset_W", "slope_W_per_mps"],
        "duration_definition": "duration_s = intercept_s + slope * predictor. Both coefficients are nonnegative. "
                               "The cruise slope estimates an effective distance for the campaign route mix.",
        "duration_coefficient_order": ["intercept_s", "slope"],
        "fitting": "Unweighted least squares on one observation per training flight and phase. A constant predictor "
                   "gives the training mean response and a zero unidentified slope. Validation is scored after fitting.",
        "aggregation": "Phase Wh = predicted W * predicted s / 3600. Airborne Wh and s sum takeoff, cruise "
                       "and landing. Airborne W = 3600 * airborne Wh / airborne s. All three phase models share predicted durations.",
        "models": {REFERENCE: "Original supplied phase power with calibrated preflight durations.",
                   OFFSET_ONLY: "Original phase power plus the training mean of observed minus original phase power.",
                   CALIBRATED: "Corrected phase power with the same preflight durations.",
                   MEAN: "Separate arithmetic training mean for each scope and target, including airborne."},
        "training_means": {scope: {key: float(np.mean([m[scope][key] for m in observed_train])) for key in OBSERVATIONS}
                           for scope in SCOPES},
        "training_domain": domain, "training_distinct_command_settings": len(training_commands), "route_coverage": route_counts,
        "route_limit": "Route identifies coverage only. Duration depends on planned speed or altitude. "
                       "Duration and energy estimates depend on the training campaign's route mix and turns.",
        "error_definition": "Predicted minus observed. P95_abs is the linear-interpolated 95th percentile of absolute flight errors.",
    }
    rows = []
    for flight in ordered:
        settings = tuple(float(flight[key]) for key in COMMANDS)
        route = str(flight["route"])
        metadata = {"flight_id": int(flight["flight_id"]), "split": flight["split"], "route": route,
                    **dict(zip(COMMANDS, settings, strict=True)), "exact_command_in_train": settings in training_commands,
                    "within_train_command_ranges": all(domain[key][0] <= v <= domain[key][-1]
                                                       for key, v in zip(COMMANDS, settings, strict=True)),
                    "route_seen_in_train": route in route_counts["train"]}
        forecasts = predict(summary, base_phase_power_W=flight["base_phase_power_W"],
                            commanded_speed_mps=settings[1], commanded_altitude_agl_m=settings[2])
        observed = _observations(flight)
        for model, phases in forecasts.items():
            for scope, targets in phases.items():
                row = {**metadata, "scope": scope, "model": model}
                for key, predicted in targets.items():
                    value = observed[scope][key]
                    row.update({f"observed_{key}": value, f"predicted_{key}": predicted, f"residual_{key}": predicted - value})
                rows.append(row)
    residuals = pd.DataFrame.from_records(rows)
    summary["metrics"] = _metrics(residuals)
    return summary, residuals
