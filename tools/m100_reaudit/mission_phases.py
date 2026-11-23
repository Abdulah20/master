"""Attribute the accepted airborne budget to contiguous observed mission phases.

The native source counters are authoritative at publications and at the two
PX4 detector edges. A counter increment crossing a phase change is kept in an
unresolved row in its entirety. PX4 navigation mode is independent context.
No power, time or distance is divided in proportion to an assumed phase duration.
"""
import argparse
from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass
import gzip
import hashlib
import io
from itertools import groupby
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from pyulog import ULog

from .direct_airborne_evidence import confirm, digest, vi_integral
from .mission_campaign import ARCHIVE, OUT, read
from .mission_observations import NAV, US_PER_S, table
from .pack_drive.m100_airborne_training import BOUNDARY_SOURCE, POWER_SOURCES

# Match the release's native sample, PX4 status and ground-truth gap limits.
# At most two 0.30-s streams can independently support a phase label. An
# explicit absent or invalid acknowledgement overrides surrounding evidence.
MAX_NATIVE_US = 300_000
MAX_ACK_AGE_US = 500_000
MAX_STATUS_US = 1_500_000
MAX_GROUND_US = 500_000
FLIGHT_PHASES = {'TAKEOFF', 'CLIMB', 'STABILIZING', 'CRUISE', 'HOVER',
                 'RETURN_BRAKING', 'RETURNING', 'LANDING'}


def microseconds(seconds):
    return round(float(seconds) * US_PER_S)


class Observations:
    """Closed brackets include a sample on each side, without extrapolation."""

    def __init__(self, times, values):
        self.times = times
        self.values = values
        confirm(len(times) == len(values) and all(a < b for a, b in zip(times, times[1:])),
                'Observation clock is not strictly increasing')

    def bracket(self, left, right):
        i = bisect_right(self.times, left) - 1
        j = bisect_left(self.times, right)
        if i < 0 or j == len(self.times):
            return (), None
        gap = max(self.times[k+1] - self.times[k] for k in range(i, j))
        return self.values[i:j+1], gap


def sequence(values):
    return tuple(value for value, _ in groupby(values))


def fresh_request(packet, run):
    native = packet['native']
    req = packet.get('acknowledged_request') or {}
    if (packet.get('demand_fresh') is not True or native.get('demand_fresh') is not True
            or req.get('run_id') != run or req.get('producer') != 'm100_experiment_manager'
            or req.get('model') != native.get('model') or not req.get('phase')
            or req.get('time_s') is None or native.get('demand_time_s') is None):
        return None
    stamp = microseconds(req['time_s'])
    if (stamp != microseconds(native['demand_time_s'])
            or not 0 <= microseconds(packet['time_s']) - stamp <= MAX_ACK_AGE_US):
        return None
    return stamp, req['phase']


def source_counters(budget):
    sources = budget['sources']
    confirm(set(sources) == set(POWER_SOURCES), 'Native source counter schema differs')
    return np.array([[sources[key]['seconds'], sources[key]['energy_Wh']] for key in POWER_SOURCES])


def native_observations(packets, run, budget):
    confirm(budget['run_id'] == run and budget['status'] == 'COMPLETE' and
            budget['uncovered_s'] == 0 and budget['boundary_source'] == BOUNDARY_SOURCE,
            'Airborne budget is incomplete or uses different detector edges')
    start, end = microseconds(budget['start_s']), microseconds(budget['end_s'])
    times, ledgers = [start], [np.zeros((len(POWER_SOURCES), 2))]
    ack_times, ack_phases, invalid_acks = [], [], []
    vi_samples = []
    final = source_counters(budget)
    saw_terminal = False
    prior_time, epoch = None, None
    for packet in packets:
        native = packet['native']
        # Preserve adjacent publications, including unsupported ones. Removing
        # them before the trapezoid check would bridge interruptions in V or I.
        vi_samples.append({k: native[k] for k in ('time_s', 'effective_demand_source', 'command_limited',
                                                  'voltage_v', 'current_a', 'airborne_budget')})
        stamp = microseconds(native['time_s'])
        confirm(packet['run_id'] == run and microseconds(packet['time_s']) == stamp and
                (prior_time is None or stamp > prior_time) and
                (epoch is None or epoch == packet['epoch']), 'Native run, epoch or clock differs')
        prior_time, epoch = stamp, packet['epoch']
        ack = fresh_request(packet, run)
        if ack is None:
            invalid_acks.append(stamp)
        elif not ack_times or ack[0] != ack_times[-1]:
            ack_times.append(ack[0])
            ack_phases.append(ack[1])
        else:
            confirm(ack_phases[-1] == ack[1], 'Conflicting phases for one acknowledged request')
        current = native['airborne_budget']
        if current.get('run_id') != run:
            confirm(not start < stamp < end, 'Foreign budget during accepted airborne window')
            continue
        if current['status'] == 'COMPLETE':
            confirm(stamp >= end and microseconds(current['end_s']) == end and
                    microseconds(current['start_s']) == start and current['uncovered_s'] == 0 and
                    np.array_equal(source_counters(current), final), 'Terminal wire and recorder budgets differ')
            saw_terminal = True
        if start < stamp < end:
            # The PX4 detector sends a stamped event. Until the delivery of its
            # first airborne edge the plugin can still publish an empty
            # pre-takeoff ledger. The first AIRBORNE ledger retrospectively
            # covers the time from the detector edge; the waiting zero is not
            # a counter observation inside the flight.
            if current['status'] == 'WAITING_FOR_TAKEOFF' and len(times) == 1:
                confirm(current['start_s'] is None and current['end_s'] is None and not source_counters(current).any(),
                        'Nonempty budget while waiting for the stamped takeoff edge')
                continue
            confirm(current['status'] == 'AIRBORNE' and microseconds(current['start_s']) == start and
                    current['uncovered_s'] == 0, 'Incomplete native budget inside the accepted flight')
            times.append(stamp)
            ledgers.append(source_counters(current))
    confirm(saw_terminal and start < end, 'Missing matching native terminal publication')
    times.append(end)
    ledgers.append(final)
    counters = np.stack(ledgers)
    increments = np.diff(counters, axis=0)
    elapsed = np.diff(times) / US_PER_S
    confirm(np.isfinite(counters).all() and (increments >= -1e-10).all() and
            np.allclose(increments[:, :, 0].sum(axis=1), elapsed, rtol=0, atol=1e-8) and
            math.isclose(float(final[:, 1].sum()), budget['estimated_airborne_energy_Wh'],
                         rel_tol=0, abs_tol=1e-9), 'Native counter reversed or failed time/energy closure')
    vi = vi_integral(vi_samples, budget['start_s'], budget['end_s'])
    return (times, counters, Observations(ack_times, ack_phases), invalid_acks), vi


def manager_observations(source):
    times, values = [], []
    for row in csv.DictReader(source):
        confirm(row['measuring'] in ('0', '1'), 'Invalid manager measurement gate')
        times.append(microseconds(row['sim_time_s']))
        values.append((row['phase'], row['measuring'] == '1'))
    return Observations(times, values)


def navigation(status):
    times = [int(v) for v in status['timestamp']]
    event_times, modes = [], []
    for published, onset, mode in zip(times, status['nav_state_timestamp'], status['nav_state'], strict=True):
        mode = int(mode)
        if not modes or mode != modes[-1]:
            confirm(0 < onset <= published and (not event_times or event_times[-1] < onset),
                    'Invalid PX4 navigation-mode transition clock')
            event_times.append(int(onset))
            modes.append(mode)
    # PX4 can publish several messages with one timestamp. Keep the transition
    # clock above and deduplicate publication stamps only for the gap check.
    confirm(times and all(a <= b for a, b in zip(times, times[1:])), 'PX4 status clock reversed')
    publications = list(dict.fromkeys(times))
    return Observations(publications, publications), event_times, modes


def px4_gate(landed):
    stamps = landed['timestamp']
    confirm(len(stamps) > 0 and all(a <= b for a, b in zip(stamps, stamps[1:])),
            'PX4 detector clock reversed')
    transitions = [(int(stamp), int(level)) for i, (stamp, level) in
                   enumerate(zip(stamps, landed['landed'], strict=True))
                   if i == 0 or level != landed['landed'][i-1]]
    confirm([level for _, level in transitions] == [1, 0, 1], 'Expected one complete PX4 airborne episode')
    return transitions[1][0], transitions[2][0]


def ground_track(ground, edges):
    """Interpolate cumulative horizontal arc length along the logged ENU polyline."""
    t = np.asarray(ground['timestamp_sample'], dtype=np.int64)
    left = int(np.searchsorted(t, edges[0], 'right'))-1
    right = int(np.searchsorted(t, edges[-1], 'left'))
    confirm(left >= 0 and right < len(t), 'Ground truth does not bracket the detector edges')
    section = slice(left, right+1)
    t = t[section]
    gaps = np.diff(t)
    confirm(len(gaps) > 0 and np.all((gaps > 0) & (gaps <= MAX_GROUND_US)) and
            all(len(set(ground[key][section])) == 1 for key in ('xy_reset_counter', 'z_reset_counter')),
            'Ground-truth gap or position reset')
    # PX4 local coordinates are north/east. Cast before differencing small
    # drifts, including at the two off-grid detector edges.
    east, north, down = (np.asarray(ground[k][section], dtype=float) for k in ('y', 'x', 'z'))
    confirm(all(np.isfinite(v).all() for v in (east, north, down)), 'Nonfinite ground-truth position')
    distance = np.r_[0., np.cumsum(np.hypot(np.diff(east), np.diff(north)))]
    path, east_edges, north_edges = (np.interp(edges, t, values) for values in (distance, east, north))
    starts = np.searchsorted(t, edges[:-1], 'right')-1
    stops = np.searchsorted(t, edges[1:], 'left')
    bracket_gaps = [int(gaps[a:b].max()) for a, b in zip(starts, stops, strict=True)]
    down_edges = np.interp(edges, t, down)
    return path, east_edges, north_edges, down_edges[0]-down_edges, bracket_gaps


def route_direction(phase, gate):
    if phase in ('CLIMB', 'TAKEOFF'):
        return 'TAKEOFF_AT_HOME'
    if phase == 'LANDING':
        return 'LANDING_AT_HOME'
    if phase == 'RETURNING':
        return 'HOMEWARD'
    if gate == 'ACTIVE':
        return 'OUTBOUND_WAYPOINT_SEQUENCE'
    if gate == 'INACTIVE' and phase == 'STABILIZING':
        return 'STABILIZING_BEFORE_OUTBOUND'
    if gate == 'INACTIVE' and phase == 'RETURN_BRAKING':
        return 'BRAKING_BEFORE_HOMEWARD'
    return 'UNRESOLVED'


def nav_context(nav, left, right):
    published, times, values = nav
    _, gap = published.bracket(left, right)
    i = bisect_right(times, left)-1
    j = bisect_left(times, right)
    if gap is None or gap > MAX_STATUS_US or i < 0:
        return 'UNKNOWN', 'PX4_STATUS_GAP'
    modes = [NAV.get(code, str(code)) for code in values[i:j]]
    return '|'.join(modes), 'PX4_MODE_CHANGE_INSIDE_COUNTER_BRACKET' if len(modes)>1 else 'PX4_STAMPED_MODE'


@dataclass(frozen=True)
class SegmentLabel:
    phase: str
    attribution: str
    candidate_phases: str
    measurement_gate: str
    px4_modes: str
    px4_mode_basis: str


def interval_label(left, right, acks, invalid, manager, nav):
    requests, ack_gap = acks.bracket(left, right)
    observed, trace_gap = manager.bracket(left, right)
    phases = sequence(p for p, measuring in observed)
    gates = sequence(measuring for p, measuring in observed)
    trace_ok = trace_gap is not None and trace_gap <= MAX_NATIVE_US
    gate = ('ACTIVE' if gates[0] else 'INACTIVE') if trace_ok and len(gates) == 1 else 'UNRESOLVED'
    candidates = tuple(dict.fromkeys([*phases, *requests]))
    phase = 'UNRESOLVED'
    if right-left > MAX_NATIVE_US:
        attribution = 'NATIVE_PUBLICATION_GAP'
    elif bisect_left(invalid, left) != bisect_right(invalid, right):
        attribution = 'MISSING_OR_INVALID_ACKNOWLEDGEMENT'
    elif ack_gap is None or ack_gap > MAX_NATIVE_US:
        attribution = 'ACKNOWLEDGEMENTS_DO_NOT_BRACKET_INTERVAL'
    elif not trace_ok:
        attribution = 'MANAGER_TRACE_GAP'
    elif len(phases) > 1 or len(sequence(requests)) > 1:
        attribution = 'PHASE_TRANSITION_IN_OBSERVATION_BRACKET'
    elif len(candidates) != 1:
        attribution = 'ACK_AND_MANAGER_PHASE_DISAGREE'
    elif candidates[0] not in FLIGHT_PHASES:
        attribution = 'UNRECOGNIZED_AIRBORNE_PHASE'
    else:
        phase = candidates[0]
        attribution = 'AGREEING_FRESH_ACKS_AND_MANAGER_SAMPLES'
    mode, mode_basis = nav_context(nav, left, right)
    return SegmentLabel(phase, attribution, '|'.join(candidates), gate, mode, mode_basis), ack_gap, trace_gap


def segments(flight, native, manager, ground, status):
    edges, counters, acks, invalid = native
    distance, east, north, height, ground_gaps = ground_track(ground, edges)
    nav = navigation(status)
    labels = [interval_label(l, r, acks, invalid, manager, nav) for l, r in zip(edges, edges[1:])]
    changes = [(manager.times[k-1], manager.times[k], manager.values[k-1][0], manager.values[k][0])
               for k in range(1, len(manager.times)) if manager.values[k-1][0] != manager.values[k][0]]
    rows = []
    for label, group in groupby(range(len(labels)), key=lambda k: labels[k][0]):
        indices = list(group)
        i, j = indices[0], indices[-1]+1
        increments = counters[j]-counters[i]
        energy = float(increments[:, 1].sum())
        duration = (edges[j]-edges[i])/US_PER_S
        unknown = label.phase == 'UNRESOLVED'
        # The state changed after the earlier trace sample and at or before the
        # later one. Retain those actual sample stamps even at an energy edge.
        observed_changes = [dict(last_old_sim_s=a/US_PER_S, first_new_sim_s=b/US_PER_S,
                                 from_phase=old, to_phase=new)
                            for a, b, old, new in changes if a < edges[j] and b > edges[i]]
        row = dict(case_id=flight['case_id'], run_id=flight['run_id'], attempt=flight['attempt'],
                   segment_index=len(rows)+1, flight_phase=label.phase, route_direction=route_direction(
                       label.phase, label.measurement_gate) if not unknown else 'UNRESOLVED',
                   manager_measurement_gate=label.measurement_gate, observed_phase_candidates=label.candidate_phases,
                   phase_attribution_method=label.attribution, phase_unresolved=unknown,
                   manager_phase_change_bracket_present=bool(observed_changes),
                   manager_phase_change_brackets_json=json.dumps(observed_changes, separators=(',', ':')),
                   px4_mode=label.px4_modes, px4_mode_basis=label.px4_mode_basis,
                   start_sim_s=edges[i]/US_PER_S, end_sim_s=edges[j]/US_PER_S,
                   start_basis='PX4_TAKEOFF_DETECTOR' if i == 0 else 'NATIVE_COUNTER_PUBLICATION',
                   end_basis='PX4_LANDING_DETECTOR' if j == len(edges)-1 else 'NATIVE_COUNTER_PUBLICATION',
                   duration_s=duration, estimated_energy_Wh=energy, mean_power_W=3600*energy/duration,
                   horizontal_distance_m=float(distance[j]-distance[i]), net_east_m=float(east[j]-east[i]),
                   net_north_m=float(north[j]-north[i]), phase_unattributed_s=duration if unknown else 0.,
                   start_height_from_takeoff_m=float(height[i]), end_height_from_takeoff_m=float(height[j]),
                   phase_unattributed_Wh=energy if unknown else 0., energy_uncovered_s=0.,
                   native_counter_intervals=j-i,
                   max_counter_interval_s=max(edges[k+1]-edges[k] for k in indices)/US_PER_S,
                   max_groundtruth_sample_gap_s=max(ground_gaps[i:j])/US_PER_S,
                   energy_source='DIFFERENCE_OF_NATIVE_AIRBORNE_SOURCE_COUNTERS',
                   distance_source='PX4_SITL_GROUNDTRUTH_HORIZONTAL_POLYLINE',
                   raw_archive=flight['raw_archive'], raw_archive_sha256=flight['raw_archive_sha256'])
        for source, (seconds, wh) in zip(POWER_SOURCES, increments, strict=True):
            row[source+'_s'] = float(seconds)
            row[source+'_Wh'] = float(wh)
        for key, column in ((1,'max_acknowledged_request_gap_s'), (2,'max_manager_sample_gap_s')):
            gaps = [labels[k][key] for k in indices]
            row[column] = max(gaps)/US_PER_S if all(g is not None for g in gaps) else None
        rows.append(row)
    for result, metric, tolerance in [('estimated_energy_Wh', 'estimated_airborne_energy_Wh', 1e-9),
                                      ('duration_s', 'airborne_duration_s', 1e-8),
                                      ('horizontal_distance_m', 'airborne_horizontal_distance_m', 1e-6)]:
        confirm(abs(math.fsum(r[result] for r in rows)-float(flight[metric])) <= tolerance,
                f'Segments do not reproduce released flight {metric}')
    for source in POWER_SOURCES:
        for segment_field, flight_field in [('_s', '_coverage_s'), ('_Wh', '_energy_Wh')]:
            confirm(abs(math.fsum(r[source+segment_field] for r in rows)-float(flight[source+flight_field])) < 1e-8,
                    f'Segments do not reproduce released {source}{flight_field}')
    return rows


def flight_checks(flight, rows, vi):
    duration = math.fsum(r['duration_s'] for r in rows)
    energy = math.fsum(r['estimated_energy_Wh'] for r in rows)
    distance = math.fsum(r['horizontal_distance_m'] for r in rows)
    supported_s = vi['eligible_seconds']
    confirm(0 <= supported_s <= duration+1e-8, 'V×I publication coverage exceeds the accepted flight')
    supported = vi['eligible_intervals'] > 0
    return dict(case_id=flight['case_id'], run_id=flight['run_id'], attempt=flight['attempt'],
                segments=len(rows), phase_unresolved_segments=sum(r['phase_unresolved'] for r in rows),
                px4_takeoff_sim_s=rows[0]['start_sim_s'], px4_landing_sim_s=rows[-1]['end_sim_s'],
                duration_s=duration, duration_minus_flight_s=duration-float(flight['airborne_duration_s']),
                segmented_energy_Wh=energy, energy_minus_flight_Wh=energy-float(flight['estimated_airborne_energy_Wh']),
                segmented_horizontal_distance_m=distance,
                distance_minus_flight_m=distance-float(flight['airborne_horizontal_distance_m']),
                phase_unattributed_s=math.fsum(r['phase_unattributed_s'] for r in rows),
                phase_unattributed_Wh=math.fsum(r['phase_unattributed_Wh'] for r in rows), energy_uncovered_s=0.,
                vi_eligible_intervals=vi['eligible_intervals'], vi_eligible_s=supported_s,
                vi_coverage_fraction=supported_s/duration,
                published_VI_trapezoids_Wh=vi['published_VI_trapezoids_Wh'] if supported else None,
                same_intervals_native_Wh=vi['same_intervals_native_Wh'] if supported else None,
                published_minus_native_Wh=vi['published_minus_native_Wh'] if supported else None,
                vi_scope='UNMIXED_NATIVE_OR_CMU_SOURCE_VALID_SIMULATED_VI_NO_COMMAND_LIMIT',
                source_audit=flight['source_audit'], raw_archive=flight['raw_archive'],
                raw_archive_sha256=flight['raw_archive_sha256'])


def analyze_flight(flight, release_dir, archive_root):
    audit = read(release_dir/flight['source_audit'])
    confirm(audit['accepted'] is True and audit['case_id'] == flight['case_id'] and
            audit['run_id'] == flight['run_id'] and int(flight['attempt']) == audit['attempt'],
            'Unaccepted flight or attempt identity mismatch')
    path = archive_root/flight['raw_archive']
    confirm(digest(path) == flight['raw_archive_sha256'] == audit['archive_sha256'],
            'Acquisition archive checksum differs')
    run = flight['run_id']
    with zipfile.ZipFile(path) as raw:
        check = audit['checks']
        provenance = json.loads(raw.read(run+'.provenance.json'))
        confirm(provenance['run_id'] == run and provenance['config']['route_definition'] == 'FIXED_DESTINATION_FROM_TAKEOFF',
                'Unexpected recording or route convention')
        ulog_name, = [n for n in raw.namelist() if n.startswith('state/rootfs/log/') and n.endswith('.ulg')]
        binary = raw.read(ulog_name)
        confirm(hashlib.sha256(binary).hexdigest() == check['ulog_sha256'], 'PX4 log checksum differs')
        topics = ['vehicle_status','vehicle_land_detected','vehicle_local_position_groundtruth']
        log = ULog(io.BytesIO(binary), topics)
        gate = px4_gate(log.get_dataset('vehicle_land_detected').data)
        wire = raw.read(run+'.pack_drive.jsonl')
        confirm(hashlib.sha256(wire).hexdigest() == check['native_wire_sha256'], 'Native trace checksum differs')
        budget = provenance['airborne_training']['native_budget']
        native, vi = native_observations((json.loads(line) for line in io.BytesIO(wire)), run, budget)
        confirm(gate == (native[0][0], native[0][-1]) == (check['px4_takeoff_us'], check['px4_landing_us']),
                'Native ledger edges do not match independently logged PX4 detector')
        with raw.open(run+'.trace.csv.gz') as compressed, gzip.open(compressed, 'rt', newline='') as stream:
            manager = manager_observations(stream)
    rows = segments(flight, native, manager, log.get_dataset('vehicle_local_position_groundtruth').data,
                    log.get_dataset('vehicle_status').data)
    return rows, flight_checks(flight, rows, vi)


def write_release(release_dir, archive_root, output, checks_output):
    flight_table = release_dir/'mission_flights.csv'
    confirm(len({p.resolve() for p in (flight_table, output, checks_output)}) == 3,
            'Flight, phase and diagnostics tables must have distinct paths')
    snapshot = flight_table.read_bytes()
    flights = list(csv.DictReader(io.StringIO(snapshot.decode('utf-8'))))
    # Concurrent isolated runs can have the same UTC-second recorder name in
    # different cases. Their archive identities and accepted attempts are distinct.
    confirm(flights and len({f['case_id'] for f in flights}) == len(flights),
            'Empty accepted-flight table or more than one accepted attempt for a case')
    rows, checks = [], []
    for flight in flights:
        try:
            flight_rows, check = analyze_flight(flight, release_dir, archive_root)
        except ValueError as exc:
            raise ValueError(f"{flight['case_id']}, attempt {flight['attempt']}: {exc}") from exc
        rows.extend(flight_rows)
        checks.append(check)
    confirm(flight_table.read_bytes() == snapshot, 'Flight table changed during phase extraction; rerun on the closed release')
    table(output, rows)
    table(checks_output, checks)
    print(f'{len(flights)} flights, {len(rows)} segments, '
          f'{math.fsum(r["phase_unattributed_s"] for r in rows):.3f} s of unresolved phase across all flights')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, default=OUT)
    parser.add_argument('--archive-root', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checks-output', type=Path)
    args = parser.parse_args()
    out = args.output or args.release_dir/'mission_phases.csv'
    write_release(args.release_dir, args.archive_root, out, args.checks_output or out.with_name('mission_phase_checks.csv'))
