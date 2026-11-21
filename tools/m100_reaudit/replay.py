"""Replay all accepted historical M100 observer trajectories at their real sample times.

Retain the archived acceleration terms, body motion, and wind policy. Replace
only the density dependence of the published and empirical phase equations.
This estimates a change to energy accounting conditional on the observed path.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from contextlib import ExitStack
import csv
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics

if __package__:
    from . import historical as hist
    from .physics import Model
else:
    import historical as hist
    from physics import Model

# Historical status and energy each publish on the same 0.1-s manager tick, but
# only energy has a simulator stamp. Receive-time pairing within half a tick
# avoids pretending that a neighboring publication is the same observation.
PAIR_LIMIT_S = 0.05
PHASES = {'TAKEOFF', 'CLIMB', 'STABILIZING', 'HOVER', 'CRUISE', 'RETURN_BRAKING', 'RETURNING', 'LANDING'}
TRANSIENT = {'STABILIZING', 'RETURN_BRAKING', 'HOVER'}
TRACE_FIELDS = ('t_s', 'phase', 'paired_nav_phase', 'phase_source', 'x_enu_m', 'y_enu_m', 'z_agl_m', 'vx_enu_mps', 'vy_enu_mps',
                'vz_enu_mps', 'roll_deg', 'pitch_deg', 'yaw_deg', 'paired_status_abs_skew_s', 'wind_e_mps', 'wind_n_mps',
                'ground_speed_mps', 'air_speed_mps', 'rho_kg_m3', 'power_archived_W', 'power_replay_W',
                'archived_wind_adjustment_W', 'cumulative_replay_Wh', 'battery_voltage_archived_V', 'battery_current_archived_A',
                'archive_measurement_counter_Wh', 'archive_measurement_distance_m', 'in_measurement', 'interval_status')


@dataclass
class Observed:
    arrival: float
    t: float
    power: float | None
    speed: float | None
    airspeed: float | None
    voltage: float | None
    current: float | None
    measured: float | None
    measured_x: float | None
    applied_wind_w: float | None
    roll: float | None
    pitch: float | None
    energy_model: str | None


@dataclass
class Nav:
    arrival: float
    phase: str
    position: tuple
    velocity: tuple
    yaw: float | None


def number_or_none(v):
    return float(v) if isinstance(v, (float, int)) and math.isfinite(v) else None


def observed_records(path, run_id):
    nav, energy = [], []
    duplicates = 0
    hashed = hashlib.sha256()
    with path.open('rb') as stream:
        for data in stream:
            hashed.update(data)
            entry = json.loads(data)
            p, clock = entry['payload'], float(entry['monotonic_s'])
            if entry['stream'] == 'status':
                nav.append(Nav(clock, p['state'], tuple(p['pos']), tuple(p['vel']), p.get('yaw_deg')))
            elif entry['stream'] == 'energy' and p.get('label') == run_id and p.get('recording'):
                stamp = number_or_none(p.get('energy_timestamp_s'))
                if stamp is None:
                    continue
                if energy and stamp == energy[-1].t:
                    duplicates += 1
                    continue
                hist.require(not energy or stamp > energy[-1].t, f'Clock reset in {path.name}')
                energy.append(Observed(clock, stamp, *[number_or_none(p.get(k)) for k in
                              ('power_w', 'ground_horizontal_speed_mps', 'air_relative_speed_mps',
                               'battery_voltage_v', 'battery_current_a', 'measurement_energy_wh',
                               'measurement_distance_m', 'wind_power_adjustment_w', 'roll_deg', 'pitch_deg')], p.get('energy_model_operational')))
    hist.require(energy and nav, 'Missing observer streams')
    hist.require(all(a.arrival <= b.arrival for a, b in zip(nav, nav[1:])), 'Disordered navigation receive times')
    return energy, nav, duplicates, hashed.hexdigest()


def closest_nav(times, records, when):
    i = bisect_left(times, when)
    candidates = [j for j in (i-1, i) if 0 <= j < len(times)]
    if not candidates:
        return None, None
    best = min(candidates, key=lambda j: abs(times[j] - when))
    skew = abs(times[best] - when)
    return (records[best] if skew <= PAIR_LIMIT_S else None), skew


def density_slope(phase, transient, ground, air, wind_on, published_coeff):
    """dP/dM where M = m**1.5/sqrt(rho); all other local fitted terms cancel."""
    speed_slope = transient['mass_feature_times_horizontal_speed_squared']
    if phase in TRANSIENT:
        base = transient['mass_over_density_feature'] + speed_slope * ground**2
    else:
        base = published_coeff[phase]
    return base + (speed_slope * (air**2 - ground**2) if wind_on else 0.)


def integrate_interval(prev, cur, max_gap):
    dt = cur['t_s'] - prev['t_s']
    if not 0 < dt <= max_gap:
        return 0., 'TIME_GAP'
    if prev['power_replay_W'] is None or cur['power_replay_W'] is None:
        return 0., 'MISSING_POWER_OR_NAV'
    return dt * (prev['power_replay_W'] + cur['power_replay_W']) / 7200, 'INTEGRATED'


def paired_rows(energy, nav, case, meta, model, coef, phase_by_stamp):
    nt = [row.arrival for row in nav]
    charge = 0.
    previous = None
    density0 = meta['atmosphere']['case']['density_kg_m3']
    mass = meta['energy_summary']['published_regression_mass_kg']
    trans = meta['transient_power_model']['coefficients']
    wind_x, wind_y, _ = meta['config']['wind_vector_enu_mps']
    wind_on = case['wind_speed_mps'] > 0
    target_phase = 'CRUISE' if case['cruise_speed_mps'] else 'HOVER'
    for e in energy:
        n, skew = closest_nav(nt, nav, e.arrival)
        nav_phase = n.phase if n else 'UNPAIRED'
        if nav_phase == 'MEASURING':
            nav_phase = target_phase
        # The 1-Hz manager CSV logs the phase from the very same energy estimate.
        # Use this synchronized observation whenever it is available.
        phase = phase_by_stamp.get(e.t, nav_phase)
        phase_source = 'SYNCHRONOUS_CSV' if e.t in phase_by_stamp else ('PAIRED_STATUS' if n else 'UNPAIRED')
        pos = n.position if n else (None,) * 3
        vel = n.velocity if n else (None,) * 3
        height = pos[2]
        rho = None
        # The ordinary corrected manager uses surface density for EKF excursions
        # within 0.5 m below the local ground; preserve the observed height itself.
        if height is not None and -.5 <= height <= 5000:
            rho = model.atmosphere(case['temperature_c'], max(0., height)).density_kg_m3
        ground, air = e.speed, e.airspeed
        if ground is None and all(v is not None for v in vel[:2]):
            ground = math.hypot(*vel[:2])
        if air is None and all(v is not None for v in vel[:2]):
            air = math.hypot(vel[0]-wind_x, vel[1]-wind_y)
        power = None
        if (phase in PHASES and e.power is not None and rho is not None and ground is not None
                and air is not None):
            # All historical hovers use general CMU transient transfer. Assert this
            # dispatch for a calm hover instead of applying its formula by guess.
            if phase == 'HOVER' and not wind_on:
                hist.require(e.energy_model == 'M100_CMU_TRANSIENT_POWER_V1', 'Unexpected historical hover model')
            wind_applied = e.applied_wind_w is not None
            hist.require(wind_on or not wind_applied, 'Wind adjustment on a calm source record')
            slope = density_slope(phase, trans, ground, air, wind_applied, coef)
            delta_feature = mass**1.5 * (1 / math.sqrt(rho) - 1 / math.sqrt(density0))
            power = e.power + slope * delta_feature
            hist.require(power > 0 and math.isfinite(power), 'Invalid replay power')
        row = dict(t_s=e.t, phase=phase, paired_nav_phase=nav_phase, phase_source=phase_source, x_enu_m=pos[0], y_enu_m=pos[1], z_agl_m=height,
                   vx_enu_mps=vel[0], vy_enu_mps=vel[1], vz_enu_mps=vel[2], roll_deg=e.roll, pitch_deg=e.pitch, yaw_deg=n.yaw if n else None,
                   paired_status_abs_skew_s=skew, wind_e_mps=wind_x, wind_n_mps=wind_y,
                   ground_speed_mps=ground, air_speed_mps=air, rho_kg_m3=rho, power_archived_W=e.power,
                   power_replay_W=power, archived_wind_adjustment_W=e.applied_wind_w, cumulative_replay_Wh=charge, battery_voltage_archived_V=e.voltage,
                   battery_current_archived_A=e.current, archive_measurement_counter_Wh=e.measured,
                   archive_measurement_distance_m=e.measured_x, in_measurement=int(phase == target_phase and e.measured is not None),
                   interval_status='START')
        if previous:
            amount, row['interval_status'] = integrate_interval(previous, row, model.energy.maximum_valid_gap_s)
            charge += amount
            row['cumulative_replay_Wh'] = charge
        yield row
        previous = row


def point_distance(a, b):
    coordinates = ['x_enu_m', 'y_enu_m', 'z_agl_m']
    if any(a[k] is None or b[k] is None for k in coordinates):
        return None
    return math.dist([a[k] for k in coordinates], [b[k] for k in coordinates])


def phase_budgets(rows, case_id, sidecar):
    by_phase = {}
    for a, b in zip(rows, rows[1:]):
        dt = b['t_s'] - a['t_s']
        mid = a['t_s'] + .5 * dt
        distance = point_distance(a, b)
        for point, start, stop in [(a, a['t_s'], mid), (b, mid, b['t_s'])]:
            phase = point['phase']
            if phase not in by_phase:
                by_phase[phase] = dict(case_id=case_id, phase=phase, first_boundary_s=start, last_boundary_s=stop,
                                       observed_duration_s=0., integrated_duration_s=0., path_observed_m=0.,
                                       path_coverage_s=0., archive_quadrature_Wh=0., replay_quadrature_Wh=0.,
                                       archived_manager_phase_Wh=sidecar.get(phase))
            row = by_phase[phase]
            row['last_boundary_s'] = stop
            row['observed_duration_s'] += .5 * dt
            if distance is not None and b['interval_status'] != 'TIME_GAP':
                row['path_observed_m'] += .5 * distance
                row['path_coverage_s'] += .5 * dt
            if b['interval_status'] == 'INTEGRATED':
                row['integrated_duration_s'] += .5 * dt
                row['archive_quadrature_Wh'] += point['power_archived_W'] * dt / 7200
                row['replay_quadrature_Wh'] += point['power_replay_W'] * dt / 7200
    # A brief takeoff can fall between two observer publications while the
    # motor-callback sidecar still contains its energy. Retain both facts.
    for phase, energy in sidecar.items():
        if phase not in by_phase:
            by_phase[phase] = dict(case_id=case_id, phase=phase, first_boundary_s=None,
                                   last_boundary_s=None, observed_duration_s=0., integrated_duration_s=0.,
                                   path_observed_m=0., path_coverage_s=0., archive_quadrature_Wh=0.,
                                   replay_quadrature_Wh=0., archived_manager_phase_Wh=energy)
    return list(by_phase.values())


def measurement(rows, case):
    selected = [idx for idx, row in enumerate(rows) if row['in_measurement']]
    hist.require(len(selected) >= 3, 'Insufficient observed measurement')
    track = rows[selected[0]:selected[-1]+1]
    first, last = track[0], track[-1]
    t0, t1 = first['t_s'], last['t_s']
    energy = original = covered_s = 0.
    duration_weighted = {k: 0. for k in ('ground_speed_mps', 'air_speed_mps', 'z_agl_m')}
    horizontal_path = spatial_path = 0.
    for a, b in zip(track, track[1:]):
        if a['in_measurement'] and b['in_measurement'] and b['interval_status'] == 'INTEGRATED':
            dur = b['t_s'] - a['t_s']
            covered_s += dur
            original += dur * (a['power_archived_W'] + b['power_archived_W']) / 7200
            energy += dur * (a['power_replay_W'] + b['power_replay_W']) / 7200
            for name in duration_weighted:
                duration_weighted[name] += dur * (a[name] + b[name]) / 2
            horizontal_path += math.hypot(b['x_enu_m'] - a['x_enu_m'], b['y_enu_m'] - a['y_enu_m'])
            spatial_path += point_distance(a, b)
    full_span = t1-t0
    complete = math.isclose(covered_s, full_span, abs_tol=1e-8, rel_tol=0.)
    counter_change = last['archive_measurement_counter_Wh'] - first['archive_measurement_counter_Wh']
    segment = [r for r in track if r['in_measurement'] and r['power_replay_W'] is not None]
    max_speed_error = max(abs(r['ground_speed_mps'] - case['cruise_speed_mps']) for r in segment)
    max_height_error = max(abs(r['z_agl_m'] - case['altitude_m']) for r in segment)
    displacement = last['x_enu_m'] - first['x_enu_m'] if case['cruise_speed_mps'] else None
    result = dict(measurement_observations=len(segment), measurement_observed_first_s=t0,
                  measurement_observed_last_s=t1, measurement_observed_span_s=full_span,
                  measurement_integrated_s=covered_s, measurement_clock_complete=complete,
                  measurement_achieved_mean_ground_speed_mps=duration_weighted['ground_speed_mps'] / covered_s if complete else None,
                  measurement_achieved_mean_air_speed_mps=duration_weighted['air_speed_mps'] / covered_s if complete else None,
                  measurement_achieved_mean_altitude_m=duration_weighted['z_agl_m'] / covered_s if complete else None,
                  measurement_achieved_min_ground_speed_mps=min(r['ground_speed_mps'] for r in segment),
                  measurement_achieved_max_ground_speed_mps=max(r['ground_speed_mps'] for r in segment),
                  measurement_achieved_min_altitude_m=min(r['z_agl_m'] for r in segment),
                  measurement_achieved_max_altitude_m=max(r['z_agl_m'] for r in segment),
                  measurement_horizontal_path_m=horizontal_path if complete and displacement is not None else None,
                  measurement_spatial_path_m=spatial_path if complete and displacement is not None else None,
                  measurement_abs_speed_error_max_mps=max_speed_error,
                  measurement_abs_height_error_max_m=max_height_error,
                  measurement_control_pass=(max_speed_error <= max(.3, .05 * case['cruise_speed_mps']) and max_height_error <= .5),
                  measurement_replay_energy_Wh=energy if complete else None,
                  measurement_replay_mean_W=3600*energy/covered_s if complete else None,
                  measurement_partial_integral_Wh=energy,
                  measurement_original_same_observer_window_Wh=original if complete else None,
                  measurement_archived_counter_change_Wh=counter_change,
                  measurement_archive_quadrature_minus_counter_Wh=(original-counter_change if complete else None),
                  measurement_outbound_displacement_m=displacement,
                  measurement_replay_Wh_per_km=(1000 * energy / displacement if complete and displacement is not None and displacement > 0 else None))
    # Keep endpoints fixed for every coarsening. Use the same covered trajectory
    # at all three rates. Coarsening is diagnostic, not a second physics timestep.
    if complete:
        for stride in (2, 4):
            reduced = segment[::stride]
            if reduced[-1] is not segment[-1]:
                reduced.append(segment[-1])
            times = [r['t_s'] for r in reduced]
            value = hist.trapezoid(times, [r['power_replay_W'] for r in reduced]) / 3600
            result[f'coarsen_{stride}x_delta_Wh'] = value-energy
            result[f'coarsen_{stride}x_max_dt_s'] = max(b-a for a,b in zip(times,times[1:]))
    else:
        for stride in (2,4):
            result[f'coarsen_{stride}x_delta_Wh'] = None
            result[f'coarsen_{stride}x_max_dt_s'] = None
    return result


def write_trace(path, rows):
    tmp = path.with_suffix('.tmp')
    try:
        with ExitStack() as stack:
            raw = stack.enter_context(tmp.open('wb'))
            compressed = stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6))
            handle = stack.enter_context(io.TextIOWrapper(compressed, encoding='ascii', newline=''))
            writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS, lineterminator='\n')
            writer.writeheader()
            # 11 significant figures preserve sub-millisecond clock precision and
            # sub-microwatt power precision while keeping the shareable traces small.
            for row in rows:
                writer.writerow({k:(format(v, '.11g') if isinstance(v, float) else v) for k,v in row.items()})
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def replay_case(case, record, model, coef, study, flight_dir, output, save_trace):
    identifier = case['case_id']
    source, provenance, meta = hist.flight_sources(study, record, flight_dir)
    if record.get('attempt_directory'):
        directory = study / Path(record['attempt_directory']).name
        candidates = list(directory.glob('status_*.result.json'))
    else:
        # The earliest runner version did not retain an attempt_directory key.
        candidates = list(study.glob(identifier + '*/status_*.result.json'))
    verified = [r for r in candidates if Path(str(hist.load_json(r).get('flight_csv'))).name == source.name]
    hist.require(len(verified) == 1, f'{identifier}: need one runner receipt for {source.name}')
    receipt = hist.load_json(verified[0])
    hist.require(receipt['outcome'] == 'COMPLETE', f'{identifier}: rejected acquisition receipt')
    directory = verified[0].parent
    trace = directory / Path(receipt['status_log']).name
    energy, nav, duplicates, trace_sha = observed_records(trace, record['metrics']['run_id'])
    _, source_observations = hist.csv_table(source)
    phase_by_stamp = {float(row['energy_timestamp_s']): row['flight_phase'] for row in source_observations}
    rows = list(paired_rows(energy, nav, case, meta, model, coef, phase_by_stamp))
    phase_data = phase_budgets(rows, identifier, meta['energy_summary']['phase_energy_by_phase_wh'])
    m = measurement(rows, case)
    original_supported = record['metrics']['supported_phase_mission_energy_Wh']
    total = math.fsum(p['replay_quadrature_Wh'] for p in phase_data)
    original_sampled = math.fsum(p['archive_quadrature_Wh'] for p in phase_data)
    hist.check_close(total, rows[-1]['cumulative_replay_Wh'], 'phase allocation conservation', 1e-8)
    statuses = Counter(row['interval_status'] for row in rows)
    lookup = {r['t_s']: r for r in rows}
    matches = [(v, lookup[float(v['energy_timestamp_s'])]) for v in source_observations
               if float(v['energy_timestamp_s']) in lookup]
    # The frozen logger exported one row for each shared clock. Reject an
    # observer log from a different or incomplete acquisition.
    hist.require(matches and len(matches) == len(source_observations), f'{identifier}: incomplete overlap with original source CSV')
    power_pairs = [(float(v['electrical_power_W']), w['power_archived_W']) for v,w in matches
                   if v['electrical_power_W'] and w['power_archived_W'] is not None]
    max_archive_power_error = max((abs(v-w) for v,w in power_pairs), default=None)
    hist.require(max_archive_power_error is not None, f'{identifier}: no powered overlap')
    absence_disagreements = sum(bool(v['electrical_power_W']) != (w['power_archived_W'] is not None)
                                for v,w in matches)
    hist.require(absence_disagreements == 0 and max_archive_power_error <= 1e-8,
                 f'{identifier}: observer power differs from the original synchronized CSV')
    phase_disagreements = sum(v['flight_phase'] != w['paired_nav_phase'] for v,w in matches)
    vi_residuals = [abs(r['power_archived_W'] - r['battery_voltage_archived_V'] * r['battery_current_archived_A'])
                    for r in rows if all(r[k] is not None for k in ('power_archived_W', 'battery_voltage_archived_V', 'battery_current_archived_A'))]
    transient_coeff = meta['transient_power_model']['coefficients']
    mass_feature = meta['energy_summary']['published_regression_mass_kg']**1.5 / math.sqrt(meta['atmosphere']['case']['density_kg_m3'])
    wind_residuals = []
    for row in rows:
        g, a, w = row['ground_speed_mps'], row['air_speed_mps'], row['archived_wind_adjustment_W']
        if g is not None and a is not None and w is not None:
            expected = (transient_coeff['mass_feature_times_horizontal_speed_squared'] * mass_feature * (a**2 - g**2)
                        + transient_coeff['horizontal_speed_cubed'] * (a**3 - g**3))
            wind_residuals.append(abs(expected - w))
    hist.require(max(wind_residuals, default=0.) <= 1e-8, f'{identifier}: original wind policy does not reproduce archived correction')
    output_hash = None
    if save_trace:
        compressed = output / 'traces' / (identifier+'.csv.gz')
        write_trace(compressed, rows)
        output_hash = hist.digest(compressed)
    unsupported = sum(row['power_archived_W'] is None for row in rows)
    excluded = sum(row['power_archived_W'] is not None and row['power_replay_W'] is None for row in rows)
    protected, empirical = model.assess(case['payload_kg'],case['cruise_speed_mps'],case['wind_speed_mps'],
                                        case['altitude_m'],case['temperature_c'],case['measurement_distance_m'],
                                        case['hover_duration_s'],case['wind_direction_toward_deg_ENU']),record['metrics']['energy_evidence_status']
    meta_d = dict(case_id=identifier, run_id=record['metrics']['run_id'], **{k:case[k] for k in hist.INPUTS},
                  replay_scope='OFFLINE_FIXED_OBSERVED_SITL_TRAJECTORY', original_evidence=empirical,
                  manufacturer_exclusions='|'.join(protected.stock_reasons), evidence_flags='|'.join(protected.evidence_flags),
                  original_source_sha256=hist.digest(source), original_provenance_sha256=hist.digest(provenance),
                  source_trace=f'{directory.name}/{trace.name}', source_trace_sha256=trace_sha, saved_trace_sha256=output_hash,
                  crosscheck_1Hz_matching_timestamps=len(matches),crosscheck_1Hz_source_rows=len(source_observations),
                  crosscheck_1Hz_paired_powers=len(power_pairs),crosscheck_1Hz_abs_power_difference_max_W=max_archive_power_error,
                  crosscheck_1Hz_nearest_nav_phase_disagreements=phase_disagreements,crosscheck_1Hz_power_availability_disagreements=absence_disagreements,
                  archived_voltage_current_pairs=len(vi_residuals),
                  archived_voltage_current_max_abs_error_W=max(vi_residuals, default=None),
                  archived_wind_adjustment_checks=len(wind_residuals),
                  archived_wind_formula_max_abs_error_W=max(wind_residuals, default=None),
                  unique_energy_samples=len(rows), duplicate_clocks_dropped=duplicates, unsupported_archived_samples=unsupported,
                  archived_power_samples_without_replay=excluded, status_unpaired=sum(row['paired_nav_phase']=='UNPAIRED' for row in rows),
                  maximum_status_arrival_skew_s=max(row['paired_status_abs_skew_s'] or 0 for row in rows),
                  reference_gap_limit_s=model.energy.maximum_valid_gap_s,
                  intervals_over_gap_limit=statuses['TIME_GAP'], covered_intervals=statuses['INTEGRATED'],
                  missing_power_or_nav_intervals=statuses['MISSING_POWER_OR_NAV'],
                  max_observer_dt_s=max(b['t_s']-a['t_s'] for a,b in zip(rows, rows[1:])),
                  archived_manager_supported_Wh=original_supported, observer_archive_supported_Wh=original_sampled,
                  observer_replay_supported_Wh=total, observer_density_change_Wh=total-original_sampled,
                  replay_phase_sum_error_Wh=total-rows[-1]['cumulative_replay_Wh'],
                  archive_measurement_full_Wh=record['metrics']['measurement_energy_Wh'], **m)
    return meta_d, phase_data


def paired_wind_cases(summaries, phases):
    keys = ('payload_kg','cruise_speed_mps','altitude_m','temperature_c')
    calm = {tuple(c[k] for k in keys):c for c in summaries if c['wind_speed_mps']==0}
    table = []
    for c in summaries:
        if not c['wind_speed_mps']:
            continue
        base = calm.get(tuple(c[k] for k in keys))
        if base is None:
            continue
        bph = {r['phase']:r for r in phases if r['case_id']==base['case_id']}
        cph = {r['phase']:r for r in phases if r['case_id']==c['case_id']}
        for phase in sorted(PHASES):
            f, s = bph.get(phase, {}), cph.get(phase, {})
            def change(name):
                # Zero here is this phase's *allocation* when absent. A phase
                # without observations has no measurable duration or mean power.
                return (s.get(name) or 0.) - (f.get(name) or 0.)
            table.append(dict(calm_id=base['case_id'], wind_id=c['case_id'], phase=phase,
                              wind_toward_deg=c['wind_direction_toward_deg_ENU'], wind_mps=c['wind_speed_mps'],
                              calm_phase_observed=f.get('first_boundary_s') is not None,
                              windy_phase_observed=s.get('first_boundary_s') is not None,
                              delta_original_manager_phase_Wh=change('archived_manager_phase_Wh'),
                              delta_observer_duration_s=change('observed_duration_s'),
                              delta_replay_phase_Wh=change('replay_quadrature_Wh'),
                              delta_observer_path_m=change('path_observed_m')))
    return table


def run(args):
    study = hist.original_study(args.study)
    out = hist.safe_destination(args.out, [study, args.flight_dir or study.parent/'m100_csv'])
    model = Model.load(args.runtime)
    protocol, records, sidecar = hist.read_study(study)
    # Recover the rounded coefficients used for the source acquisition. The
    # current patched manager is optional; the original coefficient file must match.
    import yaml
    config = model.runtime_root/'ros2_ws/src/drone_nav/config/m100_energy_model_v2.yaml'
    frozen = hist.load_json(study/'frozen_source_sha256.json')
    hash_expected, = [v for k,v in frozen.items() if Path(k).name==config.name]
    hist.require(hist.digest(config)==hash_expected, 'Published energy configuration differs from original')
    published = yaml.safe_load(config.read_bytes())['model']
    slopes = {p:published['coefficients'][family]['b1'] for p,family in published['phase_mapping'].items()}
    (out/'traces').mkdir(exist_ok=True)
    summaries, phases = [], []
    for case in protocol['cases']:
        if args.only and case['case_id'] not in args.only:
            continue
        # Keep full high-rate files for representative calm, hover and windy runs by
        # default, with every case independently processed for all metrics.
        save = args.all_traces or case['case_id'] in {'case_001','case_007','case_035','case_069'}
        summary, per_phase = replay_case(case,records[case['case_id']],model,slopes,study,args.flight_dir,out,save)
        summaries.append(summary)
        phases.extend(per_phase)
    hist.require(bool(summaries), 'No chosen cases')
    hist.write_csv(out/'cases.csv', summaries)
    hist.write_csv(out/'phases.csv', phases)
    wind = paired_wind_cases(summaries, phases)
    if wind: hist.write_csv(out/'paired_wind.csv', wind)
    source_files = {
        'replay.py': Path(__file__).resolve(),
        'historical.py': Path(hist.__file__).resolve(),
        'physics.py': Path(__file__).with_name('physics.py'),
        'physics.json': Path(__file__).resolve().parents[2] / 'config/m100_reaudit/physics.json',
        'm100_atmosphere.py': Path(model.atmosphere_module.__file__),
        'frozen_source_sha256.json': study / 'frozen_source_sha256.json',
    }
    stats = dict(schema_version=1, implementation_sha256={name: hist.digest(path) for name, path in source_files.items()},
                 model_scope='Offline energy accounting replay conditioned on the observed SITL path and original wind-transfer policy.',
                 study_hash=sidecar['csv_sha256'], energy_config_sha256=hash_expected, sample_pairing_max_wall_s=PAIR_LIMIT_S,
                 guard_max_energy_gap_s=model.energy.maximum_valid_gap_s, source_cases=len(summaries),
                 source_trace_digests_recorded=len({s['source_trace_sha256'] for s in summaries}),
                 unique_energy_observations=sum(s['unique_energy_samples'] for s in summaries),
                 overlap_with_1Hz_stamps=sum(s['crosscheck_1Hz_matching_timestamps'] for s in summaries),
                 overlap_with_1Hz_power_samples=sum(s['crosscheck_1Hz_paired_powers'] for s in summaries),
                 overlap_max_abs_power_difference_W=max(s['crosscheck_1Hz_abs_power_difference_max_W'] for s in summaries),
                 overlap_nearest_nav_phase_disagreements=sum(s['crosscheck_1Hz_nearest_nav_phase_disagreements'] for s in summaries),
                 overlap_power_availability_disagreements=sum(s['crosscheck_1Hz_power_availability_disagreements'] for s in summaries),
                 archived_voltage_current_pairs=sum(s['archived_voltage_current_pairs'] for s in summaries),
                 archived_voltage_current_max_abs_error_W=max((s['archived_voltage_current_max_abs_error_W'] for s in summaries if s['archived_voltage_current_pairs']), default=None),
                 archived_wind_formula_checks=sum(s['archived_wind_adjustment_checks'] for s in summaries),
                 archived_wind_formula_max_abs_error_W=max((s['archived_wind_formula_max_abs_error_W'] for s in summaries if s['archived_wind_adjustment_checks']), default=None),
                 stored_high_rate_examples=sum(bool(s['saved_trace_sha256']) for s in summaries),
                 complete_measurement_windows=sum(s['measurement_clock_complete'] for s in summaries),
                 passing_achieved_control_cases=sum(s['measurement_control_pass'] for s in summaries),
                 archived_power_samples_without_replay=sum(s['archived_power_samples_without_replay'] for s in summaries),
                 intervals_over_gap_limit=sum(s['intervals_over_gap_limit'] for s in summaries),
                 max_abs_replay_phase_sum_error_Wh=max(abs(s['replay_phase_sum_error_Wh']) for s in summaries),
                 max_abs_archive_observer_quadrature_error_on_complete_measurement_Wh=max((abs(s['measurement_archive_quadrature_minus_counter_Wh']) for s in summaries if s['measurement_clock_complete']),default=None),
                 max_abs_2x_coarsen_delta_Wh=max((abs(s['coarsen_2x_delta_Wh']) for s in summaries if s['measurement_clock_complete']),default=None),
                 max_abs_4x_coarsen_delta_Wh=max((abs(s['coarsen_4x_delta_Wh']) for s in summaries if s['measurement_clock_complete']),default=None),
                 median_observer_density_change_Wh=statistics.median(s['observer_density_change_Wh'] for s in summaries),
                 observer_density_change_range_Wh=[min(s['observer_density_change_Wh'] for s in summaries),max(s['observer_density_change_Wh'] for s in summaries)])
    (out/'summary.json').write_text(json.dumps(stats, indent=2, allow_nan=False)+'\n')
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--runtime',type=Path,help='M100 runtime source root (M100_RUNTIME_ROOT also supported)')
    parser.add_argument('--flight-dir',type=Path,help='Relocated m100_csv raw files and provenance')
    canonical_out = Path(__file__).resolve().parents[2]/'results/m100_reaudit/historical/replay'
    parser.add_argument('--out',type=Path,default=canonical_out)
    parser.add_argument('--all-traces',action='store_true',help='Save compact 10 Hz traces for all 69 runs')
    parser.add_argument('--only',nargs='+',help='Subset, e.g. case_001; use a separate --out')
    options = parser.parse_args()
    if options.only and options.out.resolve() == canonical_out:
        parser.error('Choose a separate --out for a subset to preserve the complete summary.')
    print(json.dumps(run(options),indent=2))
