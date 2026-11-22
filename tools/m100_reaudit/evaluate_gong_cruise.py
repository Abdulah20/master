"""Paired, offline Gong/Rodrigues evaluation on the locked CMU stable-cruise split."""
import argparse
import csv
from dataclasses import asdict
import hashlib
from itertools import groupby
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
GONG_NAMES = ("gong_basis_profile_hover", "gong_basis_profile_forward_product",
              "gong_basis_induced", "gong_basis_parasite")
MODELS = ("rodrigues", "rodrigues_offset", "rodrigues_stable", "gong_refit", "gong_frozen", "rodrigues_same_isa")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def matched_flights(samples, summary, protocol, selector, calibration, gong, speed_fit):
    # Reuse the actual selectors. Independently reconstruct every signed W from V and I.
    expected = protocol["stable_cruise"]
    fit = set(expected["fit_flight_ids"])
    held = set(expected["evaluation_flight_ids"])
    pool = set(protocol["fit_flight_ids"]) | set(protocol["evaluation_flight_ids"])
    if fit & held or not fit <= set(protocol["fit_flight_ids"]) or not held <= set(protocol["evaluation_flight_ids"]):
        raise ValueError("Stable IDs disagree with the locked split")
    visited = set()
    flights = []
    with samples.open(newline="") as stream:
        groups = groupby(csv.DictReader(stream), key=lambda r: int(r["flight_id"]))
        for fid, group in groups:
            if fid not in pool:
                continue
            if fid in visited:
                raise ValueError(f"Non-contiguous input for flight {fid}")
            visited.add(fid)
            rows = list(group)
            for r in rows:
                power = float(r["voltage_v"]) * float(r["current_a"])
                if not np.isclose(power, float(r["power_real_signed_w"]), rtol=1e-7, atol=1e-7):
                    raise ValueError(f"Signed battery power mismatch in flight {fid}")
                r["power_real_signed_w"] = power
            window = selector.summarize_real_stable(rows)
            qualifies = window is not None and window["duration_s"] >= expected["window_selection"]["minimum_eligible_interval_sum_s"]
            if qualifies != (fid in fit or fid in held):
                raise ValueError(f"Published stable selection differs for flight {fid}")
            if not qualifies:
                continue
            fields = summary[fid]
            speed = float(fields["speed"])
            payload = float(fields["payload"]) / 1000
            stable = calibration.summarize_stable_flight(rows, payload)
            for field in ("duration_s", "intervals", "mean_power_w", "energy_wh"):
                if not np.isclose(window[field], stable[field], rtol=1e-12, atol=1e-10):
                    raise ValueError(f"Paired windows disagree for flight {fid}: {field}")
            # Field Rodrigues protocol uses the published flight mass/density feature.
            # Also compare both formulas under the same pointwise ISA density as Gong.
            base = 68.9 * float(fields["mass_rho"]) + 16.8
            mass = 3.715 + payload
            for r in rows:
                rho = gong.isa_density_kg_m3(float(r["altitude_asl_m"]))
                r["power_real_signed_w"] = 68.9 * mass**1.5 / np.sqrt(rho) + 16.8
            same_isa = selector.summarize_real_stable(rows)["mean_power_w"]
            flights.append(dict(flight_id=fid, split="fit" if fid in fit else "evaluation",
                                date=rows[0]["date"], speed_mps=speed, payload_kg=payload,
                                altitude_m=float(fields["altitude"]), duration_s=window["duration_s"],
                                eligible_intervals=window["intervals"], measured_W=window["mean_power_w"],
                                measured_Wh=window["energy_wh"], mean_ground_speed_mps=window["mean_speed_mps"],
                                rodrigues_W=base, rodrigues_same_isa_W=same_isa,
                                rodrigues_stable_W=base + speed_fit["offset_W"] + speed_fit["slope_W_per_mps"] * (speed-speed_fit["reference_speed_mps"]),
                                **dict(zip(GONG_NAMES, stable["mean_basis"], strict=True))))
    if visited != pool or {r["flight_id"] for r in flights} != fit | held:
        raise ValueError("Missing CMU flights")
    return sorted(flights, key=lambda r: r["flight_id"])


def score(rows, model):
    errors = {}
    for unit in ("W", "Wh"):
        residuals = np.array([r[f"{model}_{unit}"] - r[f"measured_{unit}"] for r in rows])
        errors.update({f"MAE_{unit}": float(np.abs(residuals).mean()),
                       f"RMSE_{unit}": float(np.sqrt(np.mean(residuals**2))),
                       f"bias_{unit}": float(residuals.mean())})
    return errors


def grouped_metrics(rows):
    groups = [("all", None, None)]
    speeds = (4, 5, 6, 8, 10, 12, 15, 20)
    payloads = (0, .25, .5)
    groups += [("speed", s, None) for s in speeds]
    groups += [("payload", None, p) for p in payloads]
    groups += [("speed_payload", s, p) for s in speeds for p in payloads]
    metrics = []
    for cohort, sample in rows.items():
        for grouping, speed, payload in groups:
            subset = [r for r in sample if (speed is None or r["speed_mps"] == speed)
                      and (payload is None or r["payload_kg"] == payload)]
            for model in MODELS:
                if model == "gong_frozen" and cohort != "evaluation_unseen_by_frozen_fit":
                    continue
                item = dict(cohort=cohort, grouping=grouping, speed_mps=speed, payload_kg=payload,
                            model=model, n=len(subset))
                if subset:
                    item.update(score(subset, model))
                else:
                    item.update(dict.fromkeys(("MAE_W", "RMSE_W", "bias_W", "MAE_Wh", "RMSE_Wh", "bias_Wh")))
                metrics.append(item)
    return metrics


def paired_change(rows, candidate, reference):
    # Paired cluster bootstrap of evaluation dates. No holdout data are used for fitting.
    dates = sorted({r["date"] for r in rows})
    rng = np.random.default_rng(20261122)
    draw = rng.integers(len(dates), size=(10000, len(dates)))
    counts = np.array([sum(r["date"] == day for r in rows) for day in dates])
    result = {"dates": len(dates), "flights": len(rows)}
    for unit in ("W", "Wh"):
        def delta(r):
            truth = r[f"measured_{unit}"]
            return abs(r[f"{candidate}_{unit}"] - truth) - abs(r[f"{reference}_{unit}"] - truth)
        sums = np.array([sum(delta(r) for r in rows if r["date"] == day) for day in dates])
        draws = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
        result[unit] = {"delta_MAE": sums.sum()/len(rows), "percentile_95": np.quantile(draws, [.025, .975]).tolist()}
    return result


def run(project, out):
    package = project / "ros2_ws/src/drone_nav"
    analysis = project / "analysis/m100_energy_calibration"
    field = project / "analysis/m100_rodrigues_official_reproduction_v1"
    sys.path[:0] = [str(package), str(analysis)]
    from drone_nav import gong_power_model as gong
    import calibrate_gong_power_model as calibration
    import calibrate_aircraft_level_energy as selector

    paths = dict(samples=project/"outputs/m100_real_dataset_v2/m100_calibration_ready_samples.csv",
                 summary=field/"energy_summary_v1.csv", official_sample=field/"official_sample_v1.csv",
                 qc=project/"outputs/m100_real_dataset_v2/m100_flight_audit_summary.csv",
                 split=project/"outputs/m100_cruise_speed_validation_v1/evaluation.json",
                 speed_config=package/"config/m100_stable_cruise_speed_v1.yaml",
                 gong_config=package/"config/m100_gong_power_model.yaml", gong_code=Path(gong.__file__),
                 solver=Path(calibration.__file__), selector=Path(selector.__file__),
                 old_fit=project/"archives/m100_records_before_20261118T182538Z/project/outputs/m100_gong_model/m100_gong_summary.json")
    sha = {k: digest(p) for k, p in paths.items()}
    protocol = json.loads(paths["split"].read_bytes())
    speed_fit = yaml.safe_load(paths["speed_config"].read_text())
    selection = protocol["stable_cruise"]["window_selection"]
    frozen_hashes = {"split": speed_fit["evidence_sha256"], "samples": selection["input_samples_sha256"],
                     "selector": selection["selector_sha256"], "old_fit": "d98406fe2026f204806e7d49daefb7b908f10943cdc752b064a51d60ab91b82d"}
    for key in ("summary", "official_sample", "qc"):
        frozen_hashes[key] = protocol["source_sha256"][paths[key].name]
    if any(sha[k] != v for k,v in frozen_hashes.items()):
        raise ValueError("Evidence hashes changed")
    with paths["summary"].open(newline="") as f:
        summary = {int(r["flight"]): r for r in csv.DictReader(f)}
    geometry, frozen, config = gong.load_gong_configuration(paths["gong_config"])
    legacy = json.loads(paths["old_fit"].read_bytes())
    if (asdict(frozen) != legacy["fit"]["parameters"] or geometry != calibration.GEOMETRY
            or config["geometry"]["base_aircraft_mass_kg"] != calibration.BASE_AIRCRAFT_MASS_KG):
        raise ValueError("The installed Gong model differs from its historical fit")
    old_ids = set(map(int, legacy["lineage"]["calibration_flight_ids"]))
    flights = matched_flights(paths["samples"], summary, protocol, selector, calibration, gong, speed_fit)
    fit = [r for r in flights if r["split"] == "fit"]
    held = [r for r in flights if r["split"] == "evaluation"]
    clean = [r for r in held if r["flight_id"] not in old_ids]
    clean_ids = {r["flight_id"] for r in clean}
    theta, diagnostics = calibration.fit_positive_parameters([dict(r, real_mean_power_w=r["measured_W"]) for r in fit])
    offset = float(np.mean([r["measured_W"] - r["rodrigues_W"] for r in fit]))
    for r in flights:
        basis = np.array([r.pop(name) for name in GONG_NAMES])
        r["historical_gong_fit"] = r["flight_id"] in old_ids
        r["rodrigues_offset_W"] = r["rodrigues_W"] + offset
        r["gong_refit_W"] = float(basis @ theta.linear_groups)
        r["gong_frozen_W"] = float(basis @ frozen.linear_groups) if r["flight_id"] in clean_ids else None
        for model in MODELS:
            power = r[f"{model}_W"]
            r[f"{model}_Wh"] = None if power is None else power * r["duration_s"] / 3600
    cohorts = {"fit": fit, "evaluation": held, "evaluation_unseen_by_frozen_fit": clean}
    output = grouped_metrics(cohorts)
    checks = protocol["stable_cruise"]["published_evaluation_split"]
    for model, published in [("rodrigues", "published_baseline"), ("rodrigues_offset", "stable_offset_only"),
                             ("rodrigues_stable", "stable_offset_and_speed")]:
        actual = score(held, model)
        for metric in ("MAE_W", "RMSE_W"):
            if abs(actual[metric] - checks[published][metric]) > 1e-5:
                raise ValueError("Reference scores not reproduced")
    changes = {}
    for name, sample, candidate in [("refit",held,"gong_refit"), ("frozen",clean,"gong_frozen")]:
        changes[name] = {m: paired_change(sample, candidate, m) for m in
                         ("rodrigues", "rodrigues_offset", "rodrigues_stable", "rodrigues_same_isa")}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"flights.csv", flights)
    write_csv(out/"metrics.csv", output)
    receipt = dict(inputs={k: {"path": str(p.relative_to(project)), "sha256": sha[k]} for k,p in paths.items()},
                   cohorts={k:[r["flight_id"] for r in v] for k,v in cohorts.items()},
                   historical_fit_overlap=sorted(old_ids & {r["flight_id"] for r in held}),
                   theta_frozen=asdict(frozen), theta_refit=asdict(theta), constant_offset_W=offset,
                   fit_diagnostics=diagnostics,
                   paired_date_bootstrap_candidate_minus_reference=changes)
    (out/"evaluation.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.home()/"Master/Drone/project")
    parser.add_argument("--out", type=Path, default=ROOT/"results/m100_reaudit/gong_cruise")
    args = parser.parse_args()
    run(args.project.resolve(), args.out)
