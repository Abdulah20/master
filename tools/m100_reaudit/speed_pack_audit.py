"""Verify the registered speed-to-pack intervention using original Gazebo packets and PX4 ULogs."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import yaml

from . import direct_airborne_evidence as evidence
from .mission_audit import MODEL, pack_check
from .mission_campaign import ROOT, RUNTIME, read, save
from .mission_observations import flight_topics, table, trajectory
from .mission_phases import MAX_NATIVE_US, fresh_request, manager_observations, microseconds
from .pack_drive.m100_airborne_training import POWER_SOURCES
from .speed_pack_field import level_mask, straight_level_mask

OUT = ROOT / "results/m100_reaudit/speed_pack_link"
ARCHIVE = Path("/media/abdulah/New Volume/M100_Speed_Pack_Link_2026-11-23")
BATTERY_FIT = ROOT / "results/m100_reaudit/mission_150m/battery_fit.json"
CONFIG = RUNTIME / "ros2_ws/src/drone_nav/config/m100_stable_cruise_speed_v1.yaml"
SPEED_ID = "M100_STABLE_CRUISE_SPEED_CMU_V1"
TOLERANCE_W = 1e-7
PAIR_CONDITIONS = ("seed", "physics_step_s", "payload_kg", "cruise_speed_mps", "altitude_m",
                   "wind_speed_mps", "wind_direction_toward_deg_ENU", "temperature_c",
                   "initial_soc_percent", "mission_route", "measurement_distance_m", "hover_duration_s")


def expected_delta(case, fit):
    speed = case["cruise_speed_mps"]
    if (speed not in fit["eligible_commanded_speeds_mps"] or case["payload_kg"] not in (0., .25, .50)
            or abs(case["wind_speed_mps"]) > 1e-6 or case["temperature_c"] != 15.
            or not 25. <= case["altitude_m"] <= 100.
            or case["measurement_distance_m"] / speed < fit["min_nominal_outbound_duration_s"]):
        return None
    return fit["offset_W"] + fit["slope_W_per_mps"] * (speed - fit["reference_speed_mps"])


def check_wires(records, case, run, candidate):
    """Require agreement across proposal, Gazebo's received demand and its selected, effective source."""
    enabled = case["mode"] == "on"
    result = []
    counts = dict(fresh_ack_count=0, native_pack_publications_without_fresh_ack=0,
                  native_corrected_count=0, direct_request_corrected_count=0,
                  native_phase_recovery_corrected_count=0, candidate_count=0, gated_request_count=0,
                  max_request_identity_residual_W=0., max_pack_proposal_residual_W=0.)
    for packet in records:
        n = packet["native"]
        matched = fresh_request(packet, run)
        record = dict(ack=None, base=None, delta=None, phase=None, used=False)
        result.append(record)
        if matched is None:
            if n["effective_demand_source"]=="native_pack_model":
                counts["native_pack_publications_without_fresh_ack"]+=1
            continue
        req = packet["acknowledged_request"]
        counts["fresh_ack_count"] += 1
        evidence.confirm(req["stable_cruise_speed_enabled"] is enabled,
                         "Registered on/off treatment differs from the actual manager request")
        for key, target in (("payload_kg", "payload_kg"), ("commanded_speed_mps", "cruise_speed_mps"),
                            ("commanded_altitude_agl_m", "altitude_m"), ("wind_speed_mps", "wind_speed_mps"),
                            ("temperature_c", "temperature_c")):
            evidence.confirm(req[key] == case[target], f"Changed request condition {key}")
        phase = req["phase"]
        eligible = req["stable_cruise_speed_eligible_adjustment_w"]
        applied = req["stable_cruise_speed_adjustment_w"]
        base = req["phase_base_power_w"]
        proposed = req["phase_power_w"]
        measurement = req["measurement_active"]
        evidence.confirm(type(measurement) is bool, "Missing measurement state")
        if eligible is not None:
            evidence.confirm(phase == "CRUISE" and measurement and candidate is not None
                             and abs(eligible - candidate) < TOLERANCE_W, "Proposal outside calibrated gate")
            counts["candidate_count"] += 1
        if base is not None and phase == "CRUISE" and measurement and candidate is not None:
            evidence.confirm(eligible is not None, "Eligible steady proposal missing its fit")
        using_adjustment = enabled and eligible is not None and base is not None
        evidence.confirm((applied is not None) == using_adjustment
                         and req["stable_cruise_speed_model_id"] == (SPEED_ID if using_adjustment else None),
                         "Missing, spurious or mislabeled speed adjustment")
        evidence.confirm((base is None) == (proposed is None), "Phase power existence differs from unadjusted phase power")
        delta = 0. if applied is None else applied
        if base is not None:
            residual = abs(proposed - (base + delta))
            counts["max_request_identity_residual_W"] = max(counts["max_request_identity_residual_W"], residual)
            evidence.confirm(residual < TOLERANCE_W and
                             (not using_adjustment or abs(delta - candidate) < TOLERANCE_W),
                             "Missing or double-counted speed adjustment on native phase proposal")
        wire, echoed = req["power_w"], n["demand_power_w"]
        evidence.confirm(wire == echoed, "Acknowledged direct proposal differs from native echo")
        if wire is not None and wire > 0.:
            evidence.confirm(proposed is not None and abs(wire - proposed) < TOLERANCE_W,
                             "Conflicting direct and phase demand")
        pack_phase = n["effective_demand_source"] == "native_pack_model" and proposed is not None
        # The native PX4 land detector can recover the phase demand even while the
        # manager's direct signal is unavailable or reflects an older disarmed heartbeat.
        if pack_phase:
            residual = abs(n["effective_demand_power_w"] - proposed)
            counts["max_pack_proposal_residual_W"] = max(counts["max_pack_proposal_residual_W"], residual)
            evidence.confirm(residual < TOLERANCE_W, "Gazebo selected a different demand")
            if using_adjustment:
                counts["native_corrected_count"] += 1
                key = ("direct_request_corrected_count" if wire is not None and wire > 0.
                       else "native_phase_recovery_corrected_count")
                counts[key] += 1
        if not using_adjustment:
            counts["gated_request_count"] += 1
        record.update(ack=matched, base=base, delta=delta, phase=phase, used=using_adjustment and pack_phase,
                      measurement=measurement)
    evidence.confirm(counts["fresh_ack_count"] > 0, "No fresh run-matched pack acknowledgements")
    return result, counts


def kinematics(topics, times, start, case):
    """Use native times on the original PX4 truth timeline, with bounded interpolation."""
    ground = topics["vehicle_local_position_groundtruth"]
    t = np.asarray(ground["timestamp_sample"], dtype=np.float64) / 1e6
    evidence.confirm(len(t) > 1 and np.all(np.diff(t) > 0), "Ground-truth timeline is nonmonotonic")
    east, north, down, vn, ve, vd = (np.asarray(ground[k], dtype=float) for k in ("y", "x", "z", "vx", "vy", "vz"))
    agl = np.interp(start, t, down) - down
    v = np.hypot(ve, vn)
    cmd, height = case["cruise_speed_mps"], case["altitude_m"]
    moving = np.isfinite(v) & np.isfinite(agl) & np.isfinite(vd)
    locked = moving & level_mask(v, agl, vd, cmd, height)
    strict = straight_level_mask(t, v, agl, -vd, east, north, cmd, height)
    evidence.confirm(np.all(~strict | locked), "Strict selector exceeds the original CMU kinematic domain")

    # At interval edges include original observations on both sides; no data
    # are inferred across a dropped ground-truth sample or invalid mask point.
    left = np.searchsorted(t, times, side="right") - 1
    right = np.searchsorted(t, times, side="left")
    inside = (left >= 0) & (right < len(t))
    left = left.clip(0, len(t)-1)
    right = right.clip(0, len(t)-1)
    velocity = np.interp(times, t, v)
    altitude = np.interp(times, t, agl)
    pos = topics["vehicle_local_position"]
    tp = np.asarray(pos["timestamp_sample"], dtype=float) / 1e6
    # At this stage these samples describe realized flight, not online features.
    pv = np.hypot(pos["vx"].astype(float), pos["vy"].astype(float))
    pxvalid = np.isfinite(pv) & pos["v_xy_valid"].astype(bool)
    ix = np.searchsorted(tp, times, side="right")-1
    pxcovered = (ix >= 0) & (ix+1 < len(tp))
    ix = ix.clip(0, len(tp)-2)
    pxcovered &= pxvalid[ix] & pxvalid[ix+1] & (tp[ix+1]-tp[ix] <= .5)
    estimate = np.where(pxcovered, np.interp(times, tp, pv), np.nan)
    tracks = dict(vel=velocity, alt=altitude, east=np.interp(times,t,east), est=estimate)
    def qualify(mask):
        reject_prefix = np.r_[0, np.cumsum(~mask)]
        gap_prefix = np.r_[0, np.cumsum(np.diff(t) > .500000001)]
        return (inside[:-1] & inside[1:] &
                (reject_prefix[right[1:]+1] == reject_prefix[left[:-1]]) &
                (gap_prefix[right[1:]] == gap_prefix[left[:-1]]))
    return tracks, qualify(locked), qualify(strict)


def interval_stats(records, links, times, tracks, selected, phase_window, name, authority=None):
    """Trapezoids of terminal V*I, with eligibility and coverage independent of treatment."""
    sums = {key: 0. for key in ("pack_W", "baseline_W", "applied_delta_W", "mean_groundspeed_mps",
                                 "mean_altitude_agl_m", "mean_voltage_V", "mean_current_A")}
    coverage = {key: 0. for key in POWER_SOURCES}
    uncovered=0.
    duration = motion = applied = est_integral = est_s = 0.
    count = 0
    for i in np.flatnonzero(selected & phase_window):
        j = i+1
        dt = times[j]-times[i]
        a, b = records[i]["native"], records[j]["native"]
        left, right = links[i], links[j]
        motion += dt
        seconds = {k: b["airborne_budget"]["sources"][k]["seconds"] -
                   a["airborne_budget"]["sources"][k]["seconds"] for k in POWER_SOURCES}
        lost=b["airborne_budget"].get("uncovered_s",0.)-a["airborne_budget"].get("uncovered_s",0.)
        evidence.confirm(lost>=-1e-8 and all(s >= -1.e-8 for s in seconds.values())
                         and abs(sum(seconds.values())+lost - dt) < 1.e-7, "Nonpartitioned stable source ledger")
        uncovered+=lost
        for key in coverage:
            coverage[key] += seconds[key]
        if ((authority is not None and not authority[i]) or abs(seconds["native_pack_model"] - dt) > 1e-8
                or any(n["effective_demand_source"] != "native_pack_model" or n["command_limited"]
                       or n["voltage_v"] is None or n["current_a"] is None for n in (a,b))
                or left["base"] is None or right["base"] is None or left["used"] != right["used"]):
            continue
        vs, amps = [a["voltage_v"],b["voltage_v"]], [a["current_a"], b["current_a"]]
        for key, (first,last) in dict(pack_W=(vs[0]*amps[0], vs[1]*amps[1]),
                baseline_W=(left["base"],right["base"]), applied_delta_W=(left["delta"],right["delta"]),
                mean_voltage_V=vs,mean_current_A=amps,
                mean_groundspeed_mps=(tracks["vel"][i],tracks["vel"][j]),
                mean_altitude_agl_m=(tracks["alt"][i],tracks["alt"][j])).items():
            sums[key] += .5*(first+last)*dt
        if np.isfinite(tracks["est"][[i,j]]).all():
            est_integral += .5*(tracks["est"][i]+tracks["est"][j])*dt
            est_s += dt
        duration += dt
        applied += dt * left["used"]
        count += 1
    stats = dict(duration_s=duration, kinematic_ledger_s=motion, vi_native_intervals=count,
                 integrated_pack_Wh=sums["pack_W"]/3600 if duration else None,
                 unavailable_or_mixed_vi_s=motion-duration, kinematic_uncovered_s=uncovered,
                 applied_time_fraction=applied/duration if duration else None,
                 px4_estimator_mean_groundspeed_mps=est_integral/est_s if est_s else None,
                 px4_estimator_coverage_s=est_s)
    stats.update({key: value/duration if duration else None for key, value in sums.items()})
    stats.update({key+"_time_fraction": v/motion if motion else None for key,v in coverage.items()})
    return {name+"_"+k: v for k, v in stats.items()}


def window_stats(records, links, manager, topics, case, start, end):
    times = np.array([r["time_s"] for r in records])
    tracks, locked, strict = kinematics(topics, times, start, case)
    common = np.zeros(len(records)-1, dtype=bool)
    slots=[i for i,value in enumerate(manager.values) if value==("CRUISE",True)]
    evidence.confirm(slots and slots[-1]-slots[0]+1==len(slots), "Manager measurement has multiple outbound episodes")
    open_s, close_s=(manager.times[i]/1.e6 for i in (slots[0],slots[-1]))
    phase_window=(times[:-1]>=open_s)&(times[1:]<=close_s)
    evidence.confirm(start<=open_s<close_s<=end, "Manager measurement outside the PX4 airborne gate")
    measurement = []
    for i, (a,b) in enumerate(zip(links, links[1:])):
        dt = times[i+1]-times[i]
        if (not start <= times[i] < times[i+1] <= end or dt > MAX_NATIVE_US/1e6 + 1e-9
                or any(s["phase"] != "CRUISE" or not s.get("measurement") for s in (a,b))
                or any(records[k]["native"]["airborne_budget"]["status"] != "AIRBORNE" for k in (i,i+1))):
            continue
        values, gap = manager.bracket(microseconds(times[i]),microseconds(times[i+1]))
        if gap is not None and gap <= MAX_NATIVE_US and all(phase == "CRUISE" and active for phase,active in values):
            common[i] = True
            measurement.append(i)
    evidence.confirm(bool(measurement), "No native/manager corroborated cruise measurement interval")
    out = {**interval_stats(records, links, times, tracks, locked, phase_window, "stable",common),
           **interval_stats(records, links, times, tracks, strict, phase_window, "strict",common)}
    used = np.array([s["used"] for s in links])
    out["cmu_duration_eligible"] = bool(out["stable_duration_s"] >= 20. - 1e-8)
    out["strict_cmu_duration_eligible"] = bool(out["strict_duration_s"] >= 20. - 1e-8)
    span = np.diff(times)
    out["measurement_manager_elapsed_s"] = close_s-open_s
    out["measurement_native_bracketed_s"] = float(span[phase_window].sum())
    out["measurement_corroborated_s"] = float(span[common].sum())
    out["applied_corroborated_outside_cmu_kinematics_s"] = float(span[common & ~locked & used[:-1] & used[1:]].sum())
    out["measurement_start_native_s"] = float(times[measurement[0]])
    out["measurement_end_native_s"] = float(times[measurement[-1]+1])
    out["measurement_groundtruth_east_extent_m"] = float(tracks["east"][measurement[-1]+1] -
                                                         tracks["east"][measurement[0]])
    return out


def manager_fallback_check(trace, records, run, coefficient, regression_mass):
    """Recover the separately recorded manager channel at its own clock and local air density."""
    with gzip.open(trace, "rt", newline="") as source:
        by_time={microseconds(row["sim_time_s"]):row for row in csv.DictReader(source)}
    count=0
    matched_candidates=0
    echo_error=model_error=0.
    for packet in records:
        match=fresh_request(packet,run)
        if match is None:
            continue
        req=packet["acknowledged_request"]
        row=by_time.get(match[0])
        if req["stable_cruise_speed_eligible_adjustment_w"] is not None and row is not None:
            evidence.confirm(row["phase"]=="CRUISE" and row["measuring"]=="1",
                             "Candidate speed correction outside the manager's observed measurement gate")
            matched_candidates+=1
        if (match[1]!="CRUISE" or not req["measurement_active"] or req["manager_power_w"] is None
                or row is None or row["phase"]!="CRUISE" or row["measuring"]!="1" or not row["power_w"]
                or not row["rho_kg_m3"]):
            continue
        base=coefficient["b1"]*regression_mass**1.5/math.sqrt(float(row["rho_kg_m3"]))+coefficient["b0"]
        manager=req["manager_power_w"]
        delta=req["stable_cruise_speed_eligible_adjustment_w"] or 0.
        echo_error=max(echo_error,abs(float(row["power_w"])-manager))
        model_error=max(model_error,abs(base+delta-manager))
        count+=1
    evidence.confirm(count>0 and echo_error<TOLERANCE_W and model_error<TOLERANCE_W,
                     "Independent manager fallback audit failed for the steady window")
    return dict(manager_candidate_corroborated_publications=matched_candidates,
                manager_fallback_corroborated_publications=count,
                manager_fallback_echo_max_residual_W=echo_error,
                manager_fallback_single_adjustment_max_residual_W=model_error)


def audit_original(directory, raw, case, acquisition, protocol):
    cid, run = case["case_id"], acquisition["run_id"]
    receipt = read(raw/(cid+".json"))
    doc = read(raw/(run+".provenance.json"))
    info = doc["airborne_training"]
    evidence.confirm(read(raw/"request.json") == [case] and receipt["run_id"] == run and receipt["outcome"] == "COMPLETE",
                     "Original request or flight receipt differs from acquisition")
    mapping = {"receipt.json": raw/(cid+".json"), "provenance.json": raw/(run+".provenance.json"),
               "flight.csv": raw/(run+".csv"), "training.csv": raw/info["relative_path"]}
    for name,path in mapping.items():
        (directory/name).symlink_to(path)
    ulog, = raw.glob("state/rootfs/log/*/*.ulg")
    drops = evidence.ulog_extract(ulog, directory/"ulog_states.csv")
    sha = evidence.digest(ulog)
    save(directory/"manifest.json",dict(files_sha256={}, case_id=cid, run_id=run, attempt=acquisition["attempt"],
         ulog_dropout_events=drops, source_ulog_sha256=sha, selected_for_training=info["training_eligible"],
         launcher_sha256=(raw/"launcher_sha256").read_text().strip()))
    wire = raw/(run+".pack_drive.jsonl")
    hashes = protocol["source_sha256"]
    direct = evidence.audit_flight(directory,case,hashes,raw_trace_path=wire)
    evidence.confirm(direct["ulog_and_native_edges_match"] and direct["native_source_time_partition_verified"]
                     and direct["ulog_airborne_failsafe_s"] == 0, "PX4 and Gazebo airborne gate mismatch or failsafe")
    full_target = direct[evidence.TARGET]
    evidence.confirm((direct["native_budget_status"]=="COMPLETE") == (full_target is not None)
                     and (full_target is None or direct["native_uncovered_s"]==0), "Ambiguous direct energy target")
    evidence.confirm(all(doc["source_files"][field]["sha256"] == hashes[key] for key,field in (
        ("stable_cruise","m100_stable_cruise_speed_config"),("transient","m100_transient_power_config"),
        ("speed_fit_implementation","m100_stable_cruise_speed"),("calibration_envelope","m100_contract"),
        ("wind","m100_wind_simulation_config"),("drag_plugin","gazebo_drag_plugin"))),
        "Runtime source hash differs from preregistered configuration")
    cfg=doc["config"]
    conditions = doc["wind_environment_application"]
    angle = math.radians(case["wind_direction_toward_deg_ENU"])
    expected_wind = np.array([math.cos(angle), math.sin(angle), 0.])*case["wind_speed_mps"]
    evidence.confirm(conditions["verified"] and conditions["method"] == "APPLIED_GAZEBO_FORCE_PARAMETERS_SERVICE_RESPONSE"
                     and np.allclose(expected_wind,conditions["wind_vector_enu_mps"],atol=1e-8,rtol=0)
                     and not doc["energy_summary"]["wind_power_runtime_applied"]
                     and case["mission_route"] == receipt["mission_route"] == "measurement"
                     and cfg["route_definition"] == "ONE_WAY_PLUS_X_ENU_MEASUREMENT_DISTANCE"
                     and doc["native_pack_drive"]["stable_cruise_speed_enabled"] is (case["mode"]=="on")
                     and abs(cfg["total_mass_kg"]-(3.68+case["payload_kg"])) < 1e-8,
                     "Actual mass, route or environmental application differs")
    with wire.open() as source:
        records = [json.loads(line) for line in source]
    fit=read(BATTERY_FIT)
    evidence.confirm(evidence.digest(BATTERY_FIT) == hashes["battery_fit"] and evidence.digest(CONFIG) == hashes["stable_cruise"],
                     "Battery or speed reference hash differs")
    with mapping["training.csv"].open() as source:
        row, = csv.DictReader(source)
    evidence.confirm(row["battery_dynamics_model"] == MODEL and row["battery_parameter_sha256"] == fit["parameter_sha256"]
                     and row["battery_soc_physical_calibration"] == "False", "Training row battery metadata differs")
    pack=pack_check(records,case,cfg,fit)
    delta=expected_delta(case,yaml.safe_load(CONFIG.read_text()))
    links, checks = check_wires(records,case,run,delta)
    topics=flight_topics(ulog)
    start,end = direct["native_start_s"], direct["native_end_s"]
    route=trajectory(topics,start,end,[],case["altitude_m"])
    trace=raw/(run+".trace.csv.gz")
    with gzip.open(trace,"rt",newline="") as src:
        manager = manager_observations(src)
    stats=window_stats(records,links,manager,topics,case,start,end)
    ref_density=conditions["density_kg_m3"]
    energy_config = CONFIG.with_name("m100_energy_model_v2.yaml")
    evidence.confirm(evidence.digest(energy_config)==hashes["energy_config"], "Phase model identity differs")
    model=yaml.safe_load(energy_config.read_text())["model"]
    coeff=model["coefficients"]["cruise"]
    regression_mass=model["mass_basis"]["published_regression_baseline_kg"]+case["payload_kg"]
    reference = coeff["b1"]*regression_mass**1.5/math.sqrt(ref_density) + coeff["b0"]
    manager_check=manager_fallback_check(trace,records,run,coeff,regression_mass)
    evidence.confirm(stats["stable_baseline_W"] is not None and abs(reference-stats["stable_baseline_W"])<TOLERANCE_W,
                     "Unadjusted wire power differs from the independent phase equation")
    limits=read(ROOT/"config/m100_reaudit/acceptance.json")
    reported_distance=doc["energy_summary"]["measurement_window_distance_m"]
    evidence.confirm(reported_distance >= case["measurement_distance_m"]-limits["native_route_shortfall_tolerance_m"],
                     "Manager measurement distance falls short of the planned outbound distance")
    # Cadence can crop up to one native and one manager publication per edge.
    allowance = limits["native_route_shortfall_tolerance_m"] + case["cruise_speed_mps"]*(2*MAX_NATIVE_US/1e6+
                                                                                        limits["native_endpoint_allowance_s"])
    evidence.confirm(abs(stats["measurement_groundtruth_east_extent_m"]-reported_distance) <= allowance
                     and route["landing_error_m"] <= 2.,
                     "PX4 truth does not corroborate the outbound measurement displacement and return")
    if delta is not None and case["mode"] == "on":
        evidence.confirm(checks["native_corrected_count"] > 0, "No active native publication on eligible flight")
    evidence.confirm(stats["cmu_duration_eligible"] and abs(stats["stable_pack_W"] - stats["stable_baseline_W"]
                     - stats["stable_applied_delta_W"]) < TOLERANCE_W, "Native V*I window misses the model intervention")
    data=dict(**case, attempt=acquisition["attempt"], run_id=run, raw_archive=acquisition["archive"],
              raw_archive_sha256=acquisition["archive_sha256"], ulog_sha256=sha, native_wire_sha256=evidence.digest(wire),
              manager_trace_sha256=evidence.digest(trace), direct_training_sha256=info["sha256"],
              observed_seed=receipt["observed_seed"], observed_physics_step_s=receipt["observed_physics_step_s"],
              battery_model=MODEL, battery_parameter_sha256=fit["parameter_sha256"], battery_calibration=fit["classification"],
              training_label_source=row["label_source"], training_status=row["training_status"],
              airborne_energy_Wh=full_target, airborne_duration_s=direct["duration_s"],
              airborne_training_eligible=direct["training_eligible"], airborne_budget_status=direct["native_budget_status"],
              airborne_budget_reason=direct["budget_reason"], airborne_known_subtotal_Wh=direct["native_known_Wh"],
              airborne_takeoff_sim_s=start, airborne_landing_sim_s=end, airborne_uncovered_s=direct["native_uncovered_s"],
              airborne_native_ulog_edges_match=direct["ulog_and_native_edges_match"], ulog_dropout_count=drops,
              airborne_failsafe_s=direct["ulog_airborne_failsafe_s"], airborne_groundtruth_distance_m=route["groundtruth_airborne_distance_m"],
              landing_error_m=route["landing_error_m"], measurement_manager_east_extent_m=reported_distance,
              predicted_fit_delta_W=delta, independent_cruise_base_W=reference, fixed_reference_density_kg_m3=ref_density,
              pack_power_balance_max_residual_W=pack["pack_power_balance_max_residual_W"],
              native_2rc_equation_max_residual_V=pack["native_2rc_equation_max_residual_V"],
              control_scope="NATIVE_PACK_ONLY_MANAGER_RETAINS_EXISTING_CALIBRATION",
              **checks, **stats, **manager_check)
    for name, item in direct["source_bins"].items():
        data["airborne_"+name+"_s"] = item["seconds"]
        data["airborne_"+name+"_Wh"] = item["energy_Wh"]
    return data


def audit_zip(case, acq, protocol, archive_dir):
    archive = archive_dir/acq["archive"]
    evidence.confirm(acq["archive"] == f"{case['case_id']}/attempt_{acq['attempt']}.zip"
                     and evidence.digest(archive) == acq["archive_sha256"], "Unexpected archive path or SHA-256")
    with tempfile.TemporaryDirectory(prefix="m100-speed-audit-") as scratch:
        root=Path(scratch)
        raw=root/"raw"
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                entry=Path(name)
                evidence.confirm(not entry.is_absolute() and ".." not in entry.parts, "Invalid archive member")
                if len(entry.parts)==1 or name.startswith(("training/","state/rootfs/log/")):
                    z.extract(name, raw)
        return audit_original(root,raw,case,acq,protocol)


def pair_flights(flights):
    pairs = []
    for pid in sorted({r["pair_id"] for r in flights}):
        values = [r for r in flights if r["pair_id"] == pid]
        if len(values) < 2:
            continue
        by_mode = {r["mode"]:r for r in values}
        evidence.confirm(len(values)==2 and by_mode.keys()=={"on","off"}, "Ambiguous on/off pairing")
        on,off = by_mode["on"],by_mode["off"]
        evidence.confirm(all(off[k]==on[k] for k in PAIR_CONDITIONS) and off["run_id"] != on["run_id"]
                         and on["observed_seed"]==off["observed_seed"] and on["battery_parameter_sha256"]==off["battery_parameter_sha256"],
                         "Changed conditions between native controls")
        item=dict(pair_id=pid, **{k:on[k] for k in PAIR_CONDITIONS},
                  on_case_id=on["case_id"],off_case_id=off["case_id"],
                  both_cmu_duration_eligible=on["cmu_duration_eligible"] and off["cmu_duration_eligible"])
        for k in ("stable_duration_s","stable_mean_groundspeed_mps","stable_pack_W","stable_baseline_W","stable_applied_delta_W",
                  "stable_mean_voltage_V","stable_mean_current_A","strict_duration_s","strict_pack_W","airborne_energy_Wh","airborne_duration_s"):
            item["on_"+k],item["off_"+k] = on[k],off[k]
            item["on_minus_off_"+k] = on[k]-off[k] if on[k] is not None and off[k] is not None else None
        pairs.append(item)
    return pairs


def run(out=OUT, archive_dir=ARCHIVE, require_complete=False):
    proto_file=out/"protocol.json"
    protocol=read(proto_file)
    proto_hash=evidence.digest(proto_file)
    cases={c["case_id"]:c for c in protocol["cases"]}
    flights, attempts=[],[]
    for p in sorted((out/"acquisition").glob("*/attempt_*.json")):
        a=read(p)
        case=cases[a["case_id"]]
        record={k:a[k] for k in ("case_id","run_id","attempt","outcome","archive","archive_sha256")}
        evidence.confirm(a["protocol_sha256"]==proto_hash, "Acquisition protocol digest differs")
        if a["outcome"] == "COMPLETE":
            try:
                data=audit_zip(case,a,protocol,archive_dir)
            except (ValueError, KeyError, OSError) as exc:
                record.update(accepted=False,reason=str(exc))
            else:
                flights.append(data)
                record.update(accepted=True,reason="")
        else:
            record.update(accepted=False,reason=a.get("reason") or "NO_COMPLETED_MISSION")
        save(out/"audits"/case["case_id"]/f"attempt_{a['attempt']}.json", record)
        attempts.append(record)
    evidence.confirm(len({r["case_id"] for r in flights}) == len(flights), "Multiple accepted attempts for a case")
    paired=pair_flights(flights)
    for name,records in (("native_attempts.csv",attempts),("native_flights.csv",flights),("native_pairs.csv",paired)):
        if records:
            table(out/name,records)
    missing=sorted(set(cases) - {r["case_id"] for r in flights})
    save(out/"native_audit.json",dict(protocol_sha256=proto_hash,verified_attempts=len(attempts),accepted_flights=len(flights),
         matched_pairs=len(paired),missing_accepted_case_ids=missing,tests_of_field_equivalence=False,
         complete_original_airborne_targets=sum(r["airborne_training_eligible"] for r in flights),
         complete_primary_power_windows=sum(r["cmu_duration_eligible"] for r in flights),
         primary_selector="Original CMU speed/height/vertical thresholds, intersected with fresh outbound manager and native packets; native-pack V*I on unambiguously supported intervals.",
         strict_selector="Secondary straight, level and nearly steady kinematics on original PX4 truth.",
         scope="Native terminal estimates and original training labels. The complete-window source ledger preserves manager fallback and uncovered time. Terminal V*I comparison uses pure native-pack intervals only."))
    if require_complete:
        evidence.confirm(not missing, f"Missing accepted flights {missing}")
    return flights,attempts,paired


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--archive",type=Path,default=ARCHIVE)
    parser.add_argument("--require-complete",action="store_true")
    args=parser.parse_args()
    flight_rows,attempt_rows,pair_rows=run(args.out,args.archive,args.require_complete)
    for a in attempt_rows:
        print(a["case_id"],a["accepted"],a["reason"])
    print(f"Accepted {len(flight_rows)} flights; matched {len(pair_rows)} pairs.")
