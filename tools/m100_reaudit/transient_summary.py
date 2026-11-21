"""Summarize airborne coverage and compare with the existing manager power trace.

Only the historical comparison fills unsupported intervals. The calibrated
ledger and its energy fields retain those intervals as missing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .external import MAX_GAP_S, ROOT

SOURCE = ROOT / "results/m100_reaudit/pack_drive_three"
OUTPUT = ROOT / "results/m100_reaudit/transient_airborne"


def manager_integrals(trace, starts, ends):
    """Integrate the last published, fresh power up to the next recorded callback."""
    t = trace.sim_time_s.to_numpy()
    dt = np.diff(t)
    if len(t) < 2 or np.any(dt <= 0) or starts.min() < t[0] or ends.max() > t[-1]:
        raise ValueError("Manager clock does not bracket the airborne ledger")
    power = trace.power_w.to_numpy()
    fresh = trace.fresh.eq(1).to_numpy()
    valid = fresh[:-1] & fresh[1:] & np.isfinite(power[:-1]) & (power[:-1] > 0) & (dt <= MAX_GAP_S)
    seconds = np.where(valid, dt, 0.0)
    watt_hours = np.where(valid, power[:-1] * dt / 3600.0, 0.0)

    def integral(over):
        cumulative = np.r_[0.0, np.cumsum(over)]
        return np.interp(ends, t, cumulative) - np.interp(starts, t, cumulative)

    return integral(seconds), integral(watt_hours)


def conditional_reference(ledger, trace, case):
    start, end = ledger.start_s.to_numpy(), ledger.end_s.to_numpy()
    duration = end - start
    if (np.any(duration <= 0) or not np.allclose(start[1:], end[:-1], atol=1e-8, rtol=0)
            or not np.allclose(duration, ledger.duration_s, atol=1e-8, rtol=0)):
        raise ValueError(f"{case}: invalid interval partition")
    unknown = ledger.status.eq("unresolved").to_numpy()
    if not np.array_equal(ledger.qualified_Wh.isna().to_numpy(), unknown):
        raise ValueError(f"{case}: invalid missing energy flag")
    historical_s, historical_wh = manager_integrals(trace, start, end)
    known_wh = ledger.qualified_Wh.sum()
    missing_s = duration[unknown].sum()
    fallback_s = historical_s[unknown].sum()
    fallback_wh = historical_wh[unknown].sum()
    full_fallback = np.isclose(fallback_s, missing_s, rtol=0, atol=1e-7)
    old_wh = historical_wh.sum() if np.isclose(historical_s.sum(), duration.sum(), rtol=0, atol=1e-7) else None
    return dict(case_id=case, run_id=ledger.run_id.iloc[0], airborne_duration_s=duration.sum(),
                native_and_cmu_qualified_Wh=known_wh, unresolved_s=missing_s,
                prior_manager_available_in_unresolved_s=fallback_s,
                prior_manager_estimated_Wh_in_those_intervals=fallback_wh,
                conditional_whole_airborne_Wh=known_wh+fallback_wh if full_fallback else None,
                complete_qualified_airborne_Wh=known_wh if not unknown.any() else None,
                prior_manager_whole_airborne_Wh=old_wh)


def field_power_context(output, conditions):
    """Summarize measured manoeuvre power at the flight altitudes, without filling any gap."""
    model = json.loads((output / "model.json").read_text())
    support = output / model["support_file"]
    with support.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != model["support_sha256"]:
            raise ValueError("Field support does not match the frozen CMU calibration")
    with np.load(support, allow_pickle=False) as archive:
        observations = pd.DataFrame(archive["observations"], columns=model["columns"])
    rows = []
    for altitude, payload in sorted(conditions):
        selected = observations.loc[observations.commanded_altitude_m.eq(altitude) & observations.payload_kg.eq(payload)]
        for code, name in enumerate(model["regimes"]):
            group = selected[selected.regime_code.eq(code)].copy()
            group["watt_seconds"] = group.power_w * group.weight_s
            per_flight = group.groupby("flight_id")[["watt_seconds", "weight_s"]].sum()
            power = per_flight.watt_seconds / per_flight.weight_s
            rows.append(dict(commanded_altitude_m=altitude, payload_kg=payload, manoeuvre=name,
                             field_flights=len(power), field_ids="|".join(str(int(x)) for x in power.index),
                             field_time_s=per_flight.weight_s.sum(), equal_flight_measured_mean_W=power.mean(),
                             minimum_flight_mean_W=power.min(), maximum_flight_mean_W=power.max()))
    pd.DataFrame(rows).to_csv(output / "field_power_at_flight_altitude.csv", index=False, float_format="%.12g")


def matched_flight_context(source, examples, gates):
    archive = source.parent / "cmu_calibration"
    observations = archive / "flight_residuals.csv"
    fit = json.loads((archive / "fit.json").read_text())
    with observations.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != fit["table_sha256"][observations.name]:
            raise ValueError("CMU airborne observation table differs from its archived digest")
    field = pd.read_csv(observations)
    # The same field target is repeated for each comparator model.
    field = field[field.scope.eq("airborne") & field.model.eq("published_phase_preflight")
                  & field.split.isin(["train", "validation"])]
    if field.flight_id.duplicated().any():
        raise ValueError("Expected one measured airborne energy per field flight")
    rows = []
    for run in examples:
        peers = field[field.payload_kg.eq(run["payload_kg"])
                      & field.commanded_speed_mps.eq(run["commanded_speed_mps"])
                      & field.commanded_altitude_agl_m.eq(run["commanded_altitude_agl_m"])]
        energy = run["estimated_airborne_energy_Wh"]
        duration = gates[run["case_id"]]
        simulated_w = energy * 3600 / duration if energy is not None else np.nan
        field_w = peers.observed_mean_power_W.mean()
        rows.append(dict(case_id=run["case_id"], payload_kg=run["payload_kg"], cmu_flights=len(peers),
                         cmu_flight_ids="|".join(map(str, peers.flight_id)), practical_airborne_Wh=energy,
                         cmu_mean_airborne_Wh=peers.observed_energy_Wh.mean(), simulated_airborne_s=duration,
                         min_cmu_airborne_s=peers.observed_duration_s.min(), max_cmu_airborne_s=peers.observed_duration_s.max(),
                         simulated_airborne_mean_W=simulated_w, cmu_flight_equal_airborne_mean_W=field_w,
                         min_cmu_airborne_mean_W=peers.observed_mean_power_W.min(),
                         max_cmu_airborne_mean_W=peers.observed_mean_power_W.max(),
                         descriptive_sim_minus_field_mean_power_pct=100*(simulated_w/field_w-1)))
    return pd.DataFrame(rows)


def draw_coverage(cases, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"native_pack_model": "#315d76", "cmu_local": "#168069", "cmu_pooled": "#e4a446",
              "recorded_manager": "#a54e40", "unavailable": "#b4b6b7"}
    labels = {"native_pack_model": "Native pack model", "cmu_local": "Local CMU estimates",
              "cmu_pooled": "Pooled CMU estimates", "recorded_manager": "Archived fallback",
              "unavailable": "Unavailable"}
    phases = (("STABILIZING", "Stabilization and outward acceleration"),
              ("RETURN_BRAKING", "Braking before return"))
    fig = plt.figure(figsize=(10.0, 6.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.22,1.0])
    budget = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1, 0])]
    axes.append(fig.add_subplot(grid[1, 1], sharex=axes[0], sharey=axes[0]))
    available = {source for _, ledger in cases for source in ledger.source.unique()}
    for row, (_, ledger) in enumerate(cases):
        left = 0.0
        for source, color in colors.items():
            subset = ledger[ledger.source.eq(source)]
            if source == "unavailable" or subset.empty:
                continue
            value = subset.estimated_Wh.sum()
            budget.barh(row, value, left=left, color=color, height=.51)
            left += value
        suffix = " (partial)" if ledger.estimated_Wh.isna().any() else ""
        budget.annotate(f"{left:.2f} Wh{suffix}", (left,row), xytext=(6,0), textcoords="offset points",
                        va="center", fontsize=10, fontweight="bold")
    budget.set_xlabel("Estimated pack energy (Wh)")
    budget.set_title("Takeoff to landing", loc="left", fontsize=12)
    budget.set_xlim(0, budget.get_xlim()[1]*1.15)
    budget.set_yticks(range(len(cases)), [f"{p:g} kg" for p, _ in cases])
    budget.invert_yaxis()
    for ax, (phase, title) in zip(axes, phases, strict=True):
        for row, (payload, ledger) in enumerate(cases):
            part = ledger[ledger.phase.eq(phase)].copy()
            if part.empty:
                continue
            # Coalesce adjacent classifications before drawing; preserve any genuinely separate phases.
            changes = part.source.ne(part.source.shift()) | (part.start_s - part.end_s.shift()).abs().gt(1e-8)
            for _, block in part.groupby(changes.cumsum(), sort=False):
                start = block.start_s.iloc[0] - part.start_s.iloc[0]
                ax.barh(row, block.end_s.iloc[-1]-block.start_s.iloc[0], left=start, height=.51,
                        color=colors[block.source.iloc[0]], linewidth=0)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("Time since phase start (s)")
    for ax in [budget, *axes]:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", linewidth=.5, alpha=.4)
    axes[0].set_yticks(range(len(cases)), [f"{payload:g} kg" for payload, _ in cases])
    axes[0].invert_yaxis()
    fig.suptitle("Matrice 100  |  Estimated energy on three recorded flights", x=.085, ha="left", y=.963,
                 fontweight="bold", fontsize=14)
    fig.legend(handles=[Patch(color=colors[k], label=label) for k, label in labels.items() if k in available],
               loc="upper left", bbox_to_anchor=(.077,.913), ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(top=.79, bottom=.10, left=.09, right=.98, wspace=.17, hspace=.75)
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


def summarize(output, source):
    release = pd.read_csv(source / "summary.csv")
    protocol = json.loads((source / "protocol.json").read_text())
    field_power_context(output, {(c["altitude_m"], c["payload_kg"]) for c in protocol["cases"]})
    configs = {c["case_id"]: c for c in protocol["cases"]}
    comparison, cases, examples = [], [], []
    for flight in release.itertuples():
        cid = flight.case_id
        ledger = pd.read_csv(output / "data" / f"{cid}.intervals.csv.gz")
        if not (ledger.case_id.eq(cid).all() and ledger.run_id.eq(flight.run_id).all()):
            raise ValueError(f"{cid}: ledger identity differs from the recorded flight")
        trace = pd.read_csv(source / "evidence" / cid / "flight_trace.csv.gz",
                            usecols=["sim_time_s", "fresh", "power_w"])
        comparison.append(dict(payload_kg=flight.payload_kg, **conditional_reference(ledger, trace, cid)))
        practical = pd.read_csv(output / "data" / f"{cid}.practical.csv.gz")
        # The practical ledger can subdivide an interval if the legacy callback
        # becomes unavailable partway through it. It must still span the same flight.
        if not (practical.case_id.eq(cid).all() and practical.run_id.eq(flight.run_id).all()
                and np.isclose(ledger.start_s.iloc[0], practical.start_s.iloc[0], atol=1e-8, rtol=0)
                and np.isclose(ledger.end_s.iloc[-1], practical.end_s.iloc[-1], atol=1e-8, rtol=0)
                and np.allclose(practical.start_s.to_numpy()[1:], practical.end_s.to_numpy()[:-1], atol=1e-8, rtol=0)):
            raise ValueError(f"{cid}: practical budget does not span the original airborne window")
        config = configs[cid]
        covered = practical.estimated_Wh.notna().all()
        examples.append(dict(run_id=flight.run_id, case_id=cid, label_source="CMU_calibrated_simulation",
                             payload_kg=config["payload_kg"], commanded_speed_mps=config["cruise_speed_mps"],
                             commanded_altitude_agl_m=config["altitude_m"],
                             planned_outbound_measurement_m=config["measurement_distance_m"],
                             commanded_wind_mps=config["wind_speed_mps"], ambient_reference_C=config["temperature_c"],
                             estimated_airborne_energy_Wh=practical.estimated_Wh.sum() if covered else None))
        cases.append((flight.payload_kg, practical))
    pd.DataFrame(comparison).to_csv(output / "conditional_reference.csv", index=False, float_format="%.12g")
    pd.DataFrame(examples).to_csv(output / "training_estimates.csv", index=False, float_format="%.12g")
    gates = {row["case_id"]: row["airborne_duration_s"] for row in comparison}
    matched_flight_context(source, examples, gates).to_csv(output / "matched_field_flights.csv", index=False, float_format="%.12g")
    draw_coverage(cases, output / "coverage.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summarize(args.out, args.source)
