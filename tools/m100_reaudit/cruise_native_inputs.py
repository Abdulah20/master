"""Reproduce the native, calm outbound pack proposal for prediction comparisons.

From the thesis root import reference_grid or run
`python3 -m tools.m100_reaudit.cruise_native_inputs --check-traces`.
The grid uses the archived runtime functions and configs and needs no ROS,
Gazebo or live checkout. The optional raw-trace check reads existing ZIPs.
"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from .runtime_snapshot.ros2_ws.src.drone_nav.drone_nav import (
    m100_atmosphere as atmosphere,
    m100_contract as contract,
    m100_published_energy_v2 as published,
    m100_stable_cruise_speed as speed_fit,
)


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "results/m100_reaudit/speed_pack_link"
PACKAGE = Path(published.__file__).resolve().parents[1]
SOURCES = {
    "m100_atmosphere": Path(atmosphere.__file__),
    "m100_published_energy_v2": Path(published.__file__),
    "m100_energy_v2_config": PACKAGE / "config/m100_energy_model_v2.yaml",
    "m100_stable_cruise_speed": Path(speed_fit.__file__),
    "m100_stable_cruise_speed_config": PACKAGE / "config/m100_stable_cruise_speed_v1.yaml",
}


def reference_grid(payloads=(0., .25, .5), altitudes=(25., 50., 75., 100.),
                   speeds=(4., 5., 6., 8., 10., 12., 15., 20.), *, route_length_m=200.):
    """Return (payload kg, commanded altitude m, commanded m/s) -> pack inputs.

    The manager commits compare_density_to_15c_reference at APPLY, using
    config.altitude_m in the zero-MSL-datum Gazebo world. Native phase power
    uses that fixed reference density throughout the run. The regressors
    use 3.715 kg + payload, while the physical airframe is 3.680 kg + payload.
    legacy_W is the power proposed with the old outbound correction enabled.
    Ineligible cells have legacy_adjustment_W=None and retain the baseline.
    """
    energy = published.PublishedEnergyConfig.load(SOURCES["m100_energy_v2_config"])
    fit = speed_fit.StableCruiseSpeedFit.load(SOURCES["m100_stable_cruise_speed_config"])
    grid = {}
    for altitude in altitudes:
        state = atmosphere.compare_density_to_15c_reference(15., altitude).reference
        for payload in payloads:
            base = energy.estimate_power("CRUISE", payload, state.density_kg_m3)
            for speed in speeds:
                if speed <= 0:
                    raise ValueError("A steady outbound cruise reference requires a positive commanded speed")
                config = contract.build_config(
                    temperature_c=15., payload_kg=payload, wind_speed_mps=0., wind_direction_deg=0.,
                    cruise_speed_mps=speed, altitude_m=altitude, route_length_m=route_length_m,
                    hover_duration_s=60.,
                )
                delta = fit.adjustment_w(config)
                legacy = energy.estimate_power("CRUISE", payload, state.density_kg_m3,
                                                stable_cruise_speed_adjustment_w=delta)
                grid[payload, altitude, speed] = {
                    "baseline_W": base.power_w, "legacy_W": legacy.power_w,
                    "legacy_adjustment_W": delta, "density_kg_m3": state.density_kg_m3,
                    "regression_mass_kg": energy.published_regression_mass_kg(payload),
                    "baseline_model_id": base.model_id, "legacy_model_id": legacy.model_id,
                }
    return grid


def _sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_traces(archive_root=None):
    """Check each accepted original cruise demand, effective load, and V × I."""
    from .mission_phases import fresh_request

    protocol = json.loads((CAMPAIGN / "protocol.json").read_text())
    archive_root = Path(archive_root or protocol["raw_archive_root"])
    hashes = {key: _sha256(path) for key, path in SOURCES.items()}
    grid = reference_grid()
    cases = protocol["cases"]
    if sorted((case["cruise_speed_mps"], case["mode"]) for case in cases) != [
            (speed, mode) for speed in (4, 6, 8) for mode in ("off", "on")]:
        raise ValueError("Expected the frozen three paired speed cases")
    checks = []
    for case in cases:
        case_id = case["case_id"]
        accepted = [json.loads(path.read_text()) for path in sorted((CAMPAIGN / "audits" / case_id).glob("attempt_*.json"))]
        [receipt] = [entry for entry in accepted if entry["accepted"]]
        path = archive_root / receipt["archive"]
        if _sha256(path) != receipt["archive_sha256"]:
            raise ValueError(f"{case_id}: archive digest mismatch")
        run = receipt["run_id"]
        data = grid[case["payload_kg"], case["altitude_m"], case["cruise_speed_mps"]]
        enabled = case["mode"] == "on"
        target = data["legacy_W"] if enabled else data["baseline_W"]
        maxima = dict(phase_base_W=0., phase_proposal_W=0., native_effective_W=0., terminal_VI_W=0.)
        packets = terminal = 0
        with ZipFile(path) as archive:
            provenance = json.loads(archive.read(run + ".provenance.json"))
            source_files = provenance["source_files"]
            if any(source_files[key]["sha256"] != digest for key, digest in hashes.items()) or (
                source_files["m100_experiment_manager"]["sha256"] != protocol["source_sha256"]["manager"]
            ):
                raise ValueError(f"{case_id}: native estimator or manager differs from frozen source")
            actual = provenance["atmosphere"]
            if (actual["world_datum_msl_m"] != 0 or actual["reference"]["temperature_c_at_datum"] != 15
                    or actual["reference"]["altitude_m_msl"] != case["altitude_m"]
                    or actual["reference"]["density_kg_m3"] != data["density_kg_m3"]):
                raise ValueError(f"{case_id}: APPLY reference atmosphere differs")
            with archive.open(run + ".pack_drive.jsonl") as source:
                for line in source:
                    packet = json.loads(line)
                    matched = fresh_request(packet, run)
                    if matched is None or matched[1] != "CRUISE":
                        continue
                    req = packet["acknowledged_request"]
                    if not req["measurement_active"]:
                        continue
                    if (req["stable_cruise_speed_enabled"] is not enabled or
                            req["stable_cruise_speed_eligible_adjustment_w"] != data["legacy_adjustment_W"] or
                            req["stable_cruise_speed_adjustment_w"] != (data["legacy_adjustment_W"] if enabled else None)):
                        raise ValueError(f"{case_id}: wrong native speed mode or adjustment")
                    native = packet["native"]
                    if native["effective_demand_source"] != "native_pack_model":
                        raise ValueError(f"{case_id}: effective source differs on a fresh cruise request")
                    errors = dict(phase_base_W=abs(req["phase_base_power_w"] - data["baseline_W"]),
                                  phase_proposal_W=abs(req["phase_power_w"] - target),
                                  native_effective_W=abs(native["effective_demand_power_w"] - target))
                    packets += 1
                    if native["voltage_v"] is not None and native["current_a"] is not None:
                        errors["terminal_VI_W"] = abs(native["voltage_v"] * native["current_a"] - target)
                        terminal += 1
                    for key, error in errors.items():
                        if not error <= 1e-7:
                            raise ValueError(f"{case_id}: {key} differs by {error} W")
                        maxima[key] = max(maxima[key], error)
        if not terminal:
            raise ValueError(f"{case_id}: no accepted terminal cruise observations")
        checks.append(dict(case_id=case_id, run_id=run, archive=receipt["archive"],
                           archive_sha256=receipt["archive_sha256"], payload_kg=case["payload_kg"],
                           altitude_m=case["altitude_m"], speed_mps=case["cruise_speed_mps"], mode=case["mode"],
                           reference_density_kg_m3=data["density_kg_m3"], baseline_W=data["baseline_W"],
                           legacy_W=data["legacy_W"], expected_W=target,
                           acknowledged_cruise_packets=packets, terminal_packets=terminal,
                           maximum_absolute_residual_W=maxima))
    return dict(source_protocol="results/m100_reaudit/speed_pack_link/protocol.json",
                manager_sha256=protocol["source_sha256"]["manager"], estimator_sha256=hashes,
                reference_atmosphere="Fixed at APPLY: commanded altitude above the zero-MSL Gazebo datum, 15 C at datum.",
                phase_model="Published CRUISE regression, 3.715 kg + payload, pre-battery aircraft-level electrical power.",
                traces=checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-traces", action="store_true")
    parser.add_argument("--archive-root", type=Path)
    args = parser.parse_args()
    result = (check_traces(args.archive_root) if args.check_traces else
              [dict(payload_kg=p, altitude_m=h, speed_mps=v, **cell)
               for (p, h, v), cell in reference_grid().items()])
    print(json.dumps(result, indent=2, allow_nan=False))
