"""Select a limited native steady-flight power correction using flight-grouped CMU data."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .cruise_native_inputs import reference_grid
from .direct_airborne_evidence import digest
from .evaluate_gong_cruise import write_csv
from .mission_campaign import save


RESULTS = Path(__file__).resolve().parents[2] / "results/m100_reaudit"
OUT = RESULTS / "cruise_power_decision"
SPEEDS = (4., 6., 8.)
PAYLOADS = (0., .25, .5)
FEATURES = {
    "baseline": (), "native_offset": ("offset_W",),
    "native_payload": ("offset_W", "payload_slope_W_per_kg"),
    "legacy_speed": ("offset_W", "slope_W_per_mps"),
    "native_speed": ("offset_W", "slope_W_per_mps"),
    "native_payload_speed": ("offset_W", "slope_W_per_mps", "payload_slope_W_per_kg"),
}
EPS = 1e-9


def features(row):
    return {"offset_W": 1., "slope_W_per_mps": row["speed_mps"] - 6.,
            "payload_slope_W_per_kg": row["payload_kg"] - .25}


def coefficients(model, rows):
    names = FEATURES[model]
    if not names:
        return {}
    matrix = np.array([[features(r)[k] for k in names] for r in rows])
    source = "field_rodrigues_W" if model == "legacy_speed" else "baseline_W"
    target = [r["measured_W"] - r[source] for r in rows]
    values, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != len(names):
        return None
    return dict(zip(names, map(float, values), strict=True))


def watts(params, row):
    x = features(row)
    return row["baseline_W"] + sum(weight * x[name] for name, weight in params.items())


def predict(policy, row):
    """Apply the frozen residual to steady native base power within its empirical domain."""
    if (row["payload_kg"] not in policy["enabled_payloads_kg"]
            or row["speed_mps"] not in SPEEDS or not 25 <= row["altitude_m"] <= 100
            or row.get("wind_speed_mps", 0.) != 0. or row.get("temperature_c", 15.) != 15.):
        return row["baseline_W"]
    return watts(policy["coefficients"], row)


def metrics(y, predictions):
    errors = np.asarray(predictions) - np.asarray(y)
    return {"mae_W": float(np.mean(abs(errors))), "rmse_W": float(np.sqrt(np.mean(errors**2))),
            "bias_W": float(errors.mean())}


def validate_fit(rows):
    if not rows or any(r["split"] != "fit" for r in rows):
        raise ValueError("Only development flights may enter calibration")
    if len({r["flight_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate development flight")
    if len({r["date"] for r in rows}) < 2 or any(not r["date"] for r in rows):
        raise ValueError("At least two complete flight dates are required")
    for r in rows:
        if (r["speed_mps"] not in SPEEDS or r["payload_kg"] not in PAYLOADS
                or not 25 <= r["altitude_m"] <= 100
                or not all(math.isfinite(r[k]) and r[k] > 0 for k in
                           ("baseline_W", "measured_W", "field_rodrigues_W"))):
            raise ValueError(f"Unsupported or non-finite fit input for flight {r['flight_id']}")


def date_folds(rows):
    for date in sorted({r["date"] for r in rows}):
        yield date, [r for r in rows if r["date"] != date], [i for i,r in enumerate(rows) if r["date"] == date]


def cluster_se(paired, dates):
    residuals = paired - np.mean(paired)
    groups = np.unique(dates)
    sums = np.array([residuals[dates == date].sum() for date in groups])
    return float(np.sqrt(len(groups) / (len(groups)-1) * np.dot(sums, sums)) / len(paired))


def fit_policy(fit_rows):
    validate_fit(fit_rows)
    dates = np.array([r["date"] for r in fit_rows])
    y = np.array([r["measured_W"] for r in fit_rows])
    oof = {name: np.full(len(y), np.nan) for name in FEATURES}
    for _, train, ids in date_folds(fit_rows):
        for name, values in oof.items():
            params = coefficients(name, train)
            if params is not None:
                values[ids] = [watts(params, fit_rows[i]) for i in ids]
    oof = {name: values for name, values in oof.items() if np.isfinite(values).all()}
    loss = {name: np.abs(values-y) for name, values in oof.items()}
    best = min(loss, key=lambda name: float(loss[name].mean()))
    eligible = []
    scores = {}
    for name, errors in loss.items():
        difference = errors-loss[best]
        se = cluster_se(difference, dates)
        scores[name] = dict(**metrics(y, oof[name]), paired_se_vs_best_W=se)
        if difference.mean() <= se + EPS:
            eligible.append(name)
    chosen = min(eligible, key=lambda name: (len(FEATURES[name]), scores[name]["mae_W"]))
    enabled, gates = [], []
    for load in PAYLOADS:
        ids = [i for i,r in enumerate(fit_rows) if r["payload_kg"] == load]
        if not ids:
            continue
        base, candidate = (metrics(y[ids], oof[name][ids]) for name in ("baseline", chosen))
        use = (chosen != "baseline" and len(set(dates[ids])) >= 2
               and candidate["mae_W"] <= base["mae_W"] + EPS and candidate["rmse_W"] <= base["rmse_W"] + EPS)
        if use:
            enabled.append(load)
        gates.append(dict(payload_kg=load, n=len(ids), dates=len(set(dates[ids])), enabled=use,
                          baseline_oof=base, candidate_oof=candidate))
    return dict(model=chosen, coefficients=coefficients(chosen, fit_rows), enabled_payloads_kg=enabled,
                best_oof_model=best, one_se_eligible=eligible, development_oof=scores, payload_gates=gates)


def nested_validation(development):
    predictions = []
    for date, train, ids in date_folds(development):
        policy = fit_policy(train)
        legacy = coefficients("legacy_speed", train)
        for i in ids:
            r = development[i]
            predictions.append(dict(flight_id=r["flight_id"], date=date, payload_kg=r["payload_kg"],
                                    speed_mps=r["speed_mps"], altitude_m=r["altitude_m"],
                                    measured_W=r["measured_W"], baseline_W=r["baseline_W"],
                                    legacy_W=watts(legacy, r), selected_W=predict(policy,r),
                                    outer_selected_model=policy["model"],
                                    outer_payload_enabled=r["payload_kg"] in policy["enabled_payloads_kg"]))
    return sorted(predictions, key=lambda r:r["flight_id"])


def load_rows(protocol):
    for filename, expected in protocol["fixed_sources_sha256"].items():
        if digest(RESULTS/filename) != expected:
            raise ValueError(f"Registered source hash changed: {filename}")
    grid = reference_grid()
    rows = []
    with (RESULTS/"gong_cruise/flights.csv").open(newline="") as stream:
        for flight in csv.DictReader(stream):
            data = {k:float(flight[k]) for k in ("payload_kg", "altitude_m", "speed_mps", "measured_W", "duration_s")}
            native = grid[data["payload_kg"],data["altitude_m"],data["speed_mps"]]
            rows.append(dict(flight_id=int(flight["flight_id"]), split=flight["split"], date=flight["date"],
                             **data, field_rodrigues_W=float(flight["rodrigues_W"]),
                             baseline_W=native["baseline_W"], legacy_W=native["legacy_W"],
                             density_kg_m3=native["density_kg_m3"]))
    if len({r["flight_id"] for r in rows}) != len(rows):
        raise ValueError("Overlapping or repeated field identities")
    return sorted(rows,key=lambda r:r["flight_id"])


def scores(rows):
    return {name:metrics([r["measured_W"] for r in rows], [r[name+"_W"] for r in rows])
            for name in ("baseline","legacy","selected")}


def supervisor_coverage():
    """Annotate the archived supervisor design; previous energy values stay in their original release."""
    release = RESULTS / "supervisor_rerun"
    plan = json.loads((release/"protocol.json").read_bytes())["cases"]
    with (release/"M100_69_experiments_rerun.csv").open(encoding="utf-8-sig", newline="") as stream:
        previous = {r["case_id"]:r for r in csv.DictReader(stream)}
    if {c["case_id"] for c in plan} != previous.keys() or len(plan) != len(previous):
        raise ValueError("Supervisor design and archived flight IDs do not match")
    evidence = {0:"HOVER_SEPARATE_PHASE", 5:"INTERPOLATION_NOT_ACTIVE",
                10:"MEASURED_COMMAND_INSUFFICIENT_STABLE_SECONDS",
                15:"OUTSIDE_CMU_COMMANDS", 20:"OUTSIDE_CMU_COMMANDS"}
    rows = []
    for case in plan:
        source = previous[case["case_id"]]
        conditions = {k:case[k] for k in ("payload_kg","altitude_m","cruise_speed_mps","wind_speed_mps")}
        if any(float(source[k]) != value for k,value in conditions.items()):
            raise ValueError("Archived supervisor settings disagree with the design")
        command = case["cruise_speed_mps"]
        rows.append(dict(case_id=case["case_id"], archived_run_id=source["run_id"], **conditions,
                         archived_whole_flight_label_present=bool(source["estimated_airborne_energy_Wh"]),
                         archived_training_eligible=source["training_eligible"],
                         in_calibrated_steady_speed_commands=command in SPEEDS,
                         steady_speed_evidence=evidence[command]))
    return rows


def run(out=OUT):
    from .cruise_condition_scores import export_scores

    protocol = json.loads((out/"protocol.json").read_bytes())
    rows = load_rows(protocol)
    train = [r for r in rows if r["split"]=="fit"]
    held = [r for r in rows if r["split"]=="evaluation"]
    if len(train)!=protocol["fit_count"] or len(held)!=protocol["evaluation_count"] or len(train)+len(held)!=len(rows):
        raise ValueError("Locked development/evaluation counts changed")
    policy = fit_policy(train)
    cv = nested_validation(train)
    fitted_legacy = coefficients("legacy_speed", train)
    for row in rows:
        if not math.isclose(watts(fitted_legacy,row), row["legacy_W"], abs_tol=1e-7, rel_tol=0.):
            raise ValueError("Training-only reconstruction differs from the previously installed residual")
        row["selected_W"] = predict(policy, row)
        row["selected_adjustment_W"] = row["selected_W"]-row["baseline_W"]
    inspected_ids = set(protocol["previously_inspected_matching_flights"])
    previously_seen = [r for r in held if r["flight_id"] in inspected_ids]
    if {r["flight_id"] for r in previously_seen} != inspected_ids:
        raise ValueError("Previously inspected sensitivity cohort changed")
    full, earlier, outer = scores(held), scores(previously_seen), scores(cv)
    gates = dict(held_mae_le_baseline=full["selected"]["mae_W"]<=full["baseline"]["mae_W"]+EPS,
                 held_mae_lt_legacy=full["selected"]["mae_W"]<full["legacy"]["mae_W"]-EPS,
                 held_rmse_le_legacy=full["selected"]["rmse_W"]<=full["legacy"]["rmse_W"]+EPS,
                 outer_date_mae_le_refit_legacy=outer["selected"]["mae_W"]<=outer["legacy"]["mae_W"]+EPS,
                 prior_five_mae_le_legacy=earlier["selected"]["mae_W"]<=earlier["legacy"]["mae_W"]+EPS)
    decision = dict(schema="M100_NATIVE_STEADY_POWER_DECISION_V1", protocol_sha256=digest(out/"protocol.json"),
                    fit_flight_ids=[r["flight_id"] for r in train], held_out_flight_ids=[r["flight_id"] for r in held],
                    policy=policy, gates=gates, deploy=all(gates.values()),
                    heldout=full, previously_inspected_five=earlier, nested_development=outer,
                    previously_unmatched_28=scores([r for r in held if r["flight_id"] not in inspected_ids]))
    write_csv(out/"flights.csv", rows)
    write_csv(out/"cv_flights.csv", cv)
    write_csv(out/"supervisor_coverage.csv", supervisor_coverage())
    export_scores(rows,out)
    save(out/"decision.json", decision)
    return decision


if __name__=="__main__":
    print(json.dumps({k:v for k,v in run().items() if k in ("gates","deploy","heldout")}, indent=2))
