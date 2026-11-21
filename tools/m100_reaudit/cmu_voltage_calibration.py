"""Fit a flight-weighted terminal-voltage model to measured CMU discharge current."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .external import MAX_GAP_S


def _loaded_observations(flight):
    """Return [current A, net discharged Ah, voltage V] and quadrature weights in seconds."""
    samples = flight["samples"]
    t, current, voltage = samples[["time", "battery_current", "battery_voltage"]].to_numpy(dtype=float).T
    start, level, descent, end = flight["bounds"]
    if not (0 <= start < level < descent < end < len(t)):
        raise ValueError(f"Invalid airborne indices for flight {flight['flight_id']}")
    dt = np.diff(t)
    if (not np.isfinite(t).all() or not np.isfinite(current).all() or not np.isfinite(voltage).all()
            or np.any(dt <= 0) or np.any(dt > MAX_GAP_S + 1e-10)):
        raise ValueError(f"Invalid sample or time continuity in flight {flight['flight_id']}")

    # Relative net charge starts at the first recording timestamp, including preflight.
    q = np.r_[0.0, np.cumsum((current[:-1] + current[1:]) * dt / 7200.0)]
    window = slice(start + 1, end)  # Same first and last airborne sample as external.measured_scopes.
    i = current[window]
    airborne_dt = np.diff(t[window])
    half_interval = 0.5 * airborne_dt * ((i[:-1] > 0) & (i[1:] > 0))
    seconds = np.r_[half_interval, 0.0] + np.r_[0.0, half_interval]
    keep = seconds > 0
    obs = np.column_stack((i, q[window], voltage[window]))
    return obs[keep], seconds[keep], float(airborne_dt.sum())


def _errors(residual, weights):
    """Weighted errors, with positive bias meaning overprediction."""
    if residual is None or not len(residual):
        return dict(mae_V=None, rmse_V=None, bias_V=None, p95_abs_V=None)
    w = weights / weights.sum()
    absolute = np.abs(residual)
    order = np.argsort(absolute)
    quantile_index = np.searchsorted(np.cumsum(w[order]), 0.95, side="left")
    return dict(mae_V=float(w @ absolute), rmse_V=float(np.sqrt(w @ np.square(residual))),
                bias_V=float(w @ residual), p95_abs_V=float(absolute[order[quantile_index]]))


def _range(values):
    return [float(values.min()), float(values.max())] if len(values) else [None, None]


def _minimum_errors(errors):
    if not errors:
        return dict(flights=0, mae_V=None, bias_V=None, median_error_V=None)
    return dict(flights=len(errors), mae_V=float(np.mean(np.abs(errors))),
                bias_V=float(np.mean(errors)), median_error_V=float(np.median(errors)))


def _design(obs):
    return np.column_stack((np.ones(len(obs)), -obs[:, 0], -obs[:, 1]))


def fit(flights: list[dict]) -> tuple[dict, pd.DataFrame]:
    """Fit V = a - r I - k q on train flights and report frozen development-validation errors."""
    ids = [int(f["flight_id"]) for f in flights]
    if len(set(ids)) != len(ids) or any(f["split"] not in {"train", "validation"} for f in flights):
        raise ValueError("Supply unique flights from train and validation only")
    records = [(int(f["flight_id"]), f["split"], *_loaded_observations(f)) for f in flights]
    training = [(obs, sec) for _, split, obs, sec, _ in records if split == "train" and len(sec)]
    if not training:
        raise ValueError("No train flight with a positive-discharge airborne interval")

    obs_train = np.concatenate([obs for obs, _ in training])
    train_weight = np.concatenate([sec / sec.sum() for _, sec in training])
    constant_v = float(np.average(obs_train[:, 2], weights=train_weight))
    weight = np.sqrt(train_weight)
    design = _design(obs_train)
    coefs, _, rank, singular = np.linalg.lstsq(design * weight[:, None], obs_train[:, 2] * weight, rcond=None)
    identified = rank == design.shape[1]
    if not identified:
        status = "RANK_DEFICIENT_NO_COEFFICIENTS"
    elif np.any(coefs <= 0):
        status = "NONPOSITIVE_COEFFICIENT_DESCRIPTIVE_ONLY"
    else:
        status = "FULL_RANK_EXPECTED_SIGNS"
    train_low, train_high = obs_train[:, :2].min(axis=0), obs_train[:, :2].max(axis=0)

    metrics, baseline_metrics, minimum_metrics, coverage, rows = {}, {}, {}, {}, []
    for split in ("train", "validation"):
        split_errors, split_weights, fractions = [], [], []
        baseline_errors, model_min_errors, baseline_min_errors = [], [], []
        for flight_id, name, obs, sec, airborne_s in records:
            if name != split:
                continue
            t_loaded = float(sec.sum())
            row = dict(flight_id=flight_id, split=name, loaded_samples=len(sec),
                       airborne_duration_s=airborne_s, loaded_duration_s=t_loaded,
                       observed_airborne_load_fraction=t_loaded / airborne_s)
            for column, key in enumerate(("current_A", "net_discharged_Ah", "observed_voltage_V")):
                row["min_" + key], row["max_" + key] = _range(obs[:, column])
            pred = _design(obs) @ coefs if identified and len(sec) else None
            residual = pred - obs[:, 2] if pred is not None else None
            row.update(_errors(residual, sec))
            row["min_predicted_voltage_V"] = float(pred.min()) if pred is not None else None
            row["predicted_at_observed_min_V"] = float(pred[obs[:, 2].argmin()]) if pred is not None else None
            row["minimum_voltage_error_V"] = None
            row["baseline_minimum_voltage_error_V"] = None
            row["fraction_in_training_ranges"] = None
            if len(sec):
                w = sec / t_loaded
                split_weights.append(w)
                baseline_errors.append(constant_v - obs[:, 2])
                row["baseline_minimum_voltage_error_V"] = constant_v - row["min_observed_voltage_V"]
                baseline_min_errors.append(row["baseline_minimum_voltage_error_V"])
                inside = np.all((obs[:, :2] >= train_low) & (obs[:, :2] <= train_high), axis=1)
                row["fraction_in_training_ranges"] = float(w @ inside)
                fractions.append(row["fraction_in_training_ranges"])
                if residual is not None:
                    split_errors.append(residual)
                    row["minimum_voltage_error_V"] = row["min_predicted_voltage_V"] - row["min_observed_voltage_V"]
                    model_min_errors.append(row["minimum_voltage_error_V"])
            rows.append(row)
        weights = np.concatenate(split_weights) if split_weights else None
        metrics[split] = _errors(np.concatenate(split_errors) if split_errors else None, weights)
        baseline_metrics[split] = _errors(np.concatenate(baseline_errors) if baseline_errors else None, weights)
        minimum_metrics[split] = dict(model=_minimum_errors(model_min_errors),
                                      baseline=_minimum_errors(baseline_min_errors))
        coverage[split] = dict(flights_supplied=sum(f["split"] == split for f in flights),
                               flights_with_loaded_intervals=len(fractions),
                               time_weighted_fraction_in_training_ranges=float(np.mean(fractions)) if fractions else None)

    summary = dict(
        model="V_terminal = a_V - r_V_per_A * measured_current_A - k_V_per_Ah * net_discharged_Ah",
        target="Measured Mauch PL-200 pack-output voltage during positive discharge in CMU airborne windows.",
        coefficients=dict(zip(("a_V", "r_V_per_A", "k_V_per_Ah"), map(float, coefs))) if identified else None,
        fit_status=status, design_rank=int(rank), design_columns=3,
        condition_number_in_raw_units=float(singular[0] / singular[-1]) if identified else None,
        training_predictor_ranges=dict(current_A=_range(obs_train[:, 0]), net_discharged_Ah=_range(obs_train[:, 1])),
        coverage_definition="Inclusive marginal train minima and maxima of I and q; joint combinations may be unseen.",
        weights="Each flight contributes equal mass. Within a flight each positively loaded adjacent airborne interval "
                "contributes half its elapsed seconds to each endpoint. P95 is the weighted empirical 95th percentile "
                "of absolute voltage error. Bias is predicted minus measured voltage.",
        charge_origin="Signed trapezoidal integral of measured current from the first recording sample, with discharge positive. "
                      "Initial charge and absolute SOC are unknown.",
        interpretation="Observational load-voltage slopes conditional on measured current. Pack identity and temperature "
                       "are unresolved. Coefficients do not identify TB47D OCV, internal resistance or capacity. "
                       "No validation or test observations enter coefficient estimation or the training ranges.",
        metrics=metrics, coverage=coverage,
        baseline=dict(method="Constant time-weighted mean voltage from training flights, with equal flight weight.",
                      mean_training_V=constant_v, metrics=baseline_metrics),
        flight_minimum_voltage=dict(definition="One residual per flight: minimum predicted minus minimum measured "
                                    "over the same loaded samples. Minima may occur at different times.",
                                    metrics=minimum_metrics),
    )
    return summary, pd.DataFrame(rows).sort_values(["split", "flight_id"], ignore_index=True)
