"""Account for existing M100 flight time using native pack VI and supported CMU transients.

All clocks are integer microseconds on the common PX4 boot / Gazebo simulation axis.
The calibrated result is an offline estimate of whole-aircraft power and never inferred
per-motor work. Each time interval has precisely one energy source or an unresolved reason.
"""
import argparse
from bisect import bisect_right
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'results/m100_reaudit/pack_drive_three'
OUT = ROOT / 'results/m100_reaudit/transient_airborne'
RAW = ROOT / 'tmp/m100_reaudit_pack_three'
RELEASED_FLAG_SHA256 = 'bf56088f523b3393602a36e7719aa7563bc536ff05000b761b418410d8033c44'
TRANSIENT = {'STABILIZING', 'RETURN_BRAKING'}
MAX_AGE_US = 500_000  # Existing native acquisition gap tolerance in config/m100_reaudit/acceptance.json.
US = 1_000_000


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def us(seconds):
    return int(round(float(seconds) * US))


def json_file(path):
    return json.loads(path.read_text())


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Input hash mismatch: {path}')


def save_table(path, rows):
    # Path-independent gzip header, deterministic native clocks, full float precision.
    frame = pd.DataFrame(rows)
    if path.suffix != '.gz':
        frame.to_csv(path, index=False, float_format='%.15g')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as out, gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as stream:
            frame.to_csv(stream, index=False, float_format='%.15g')


def flight_flags(index, release, raw, out):
    """Retain every ULog Boolean transition and the last supported message per field."""
    saved = out / 'ulog_flight_flags.csv'
    fallback = saved if saved.is_file() else OUT / saved.name
    old = None
    archive = release / 'm100_three_flights_2026-11-22.zip'
    rows = []
    for item in index:
        case = item['case_id']
        digest = item['ulog_sha256']
        path = raw / item['ulog_relative_to_work']
        if path.is_file():
            verify(path, digest)
            source = str(path)
        elif archive.is_file():
            with zipfile.ZipFile(archive) as bundle:
                data = bundle.read(item['ulog_in_download'])
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f'{case}: archived ULog differs from release')
            source = io.BytesIO(data)
        else:
            if old is None:
                if not fallback.is_file():
                    raise FileNotFoundError(f'{case}: both original ULog and released flight flags unavailable')
                verify(fallback, RELEASED_FLAG_SHA256)
                with fallback.open() as source_flags:
                    old = list(csv.DictReader(source_flags))
            selected = [r for r in old if r['case_id'] == case and r['ulog_sha256'] == digest]
            if not selected:
                raise FileNotFoundError(f'{case}: original ULog and attributed flag extract unavailable')
            rows.extend(selected)
            continue
        from pyulog import ULog
        log = ULog(source, ['vehicle_land_detected', 'vehicle_status'])
        for topic, keys in [('vehicle_land_detected', ('landed', 'ground_contact', 'maybe_landed')),
                            ('vehicle_status', ('arming_state',))]:
            data = log.get_dataset(topic).data
            clock = data['timestamp'].astype('int64')
            if np.any(np.diff(clock) < 0):
                raise ValueError(f'{case}: non-monotone ULog {topic}')
            for key in keys:
                values = data[key]
                indices = np.unique(np.r_[0, np.flatnonzero(np.diff(values.astype(int)) != 0)+1, len(values)-1])
                for i in indices:
                    rows.append(dict(case_id=case, ulog_sha256=digest, topic=topic, field=key,
                                     timestamp_us=int(clock[i]), value=int(values[i])))
    save_table(saved, rows)
    return rows


def change_events(times, values):
    return [(int(t), value) for i, (t, value) in enumerate(zip(times, values, strict=True))
            if i == 0 or value != values[i-1]]


def one_air_gate(events):
    """Exactly one continuously airborne episode bracketed by positive landed samples."""
    if [v for _, v in events] != [1, 0, 1]:
        raise ValueError(f'Expected a single airborne session; found transitions {events}')
    return events[1][0], events[2][0]


def flag_series(rows, field):
    pairs = [(int(r['timestamp_us']), int(r['value'])) for r in rows if r['field'] == field]
    times, values = zip(*pairs, strict=True)
    return change_events(times, values), times[-1]


def kinematics(trace, attitude, takeoff_us, landing_us):
    from .transient_cmu import Kinematics

    series = trace.loc[trace.px4_time_s.ne(trace.px4_time_s.shift())]
    history = Kinematics()
    records = []
    # The released compact trajectory used a past-only ULog attitude join at the
    # displayed PX4 position time, with a 0.1 s maximum quaternion age.
    tilt = {}
    for r in attitude.itertuples():
        if math.isfinite(r.roll_deg) and math.isfinite(r.pitch_deg):
            cos_tilt = math.cos(math.radians(r.roll_deg))*math.cos(math.radians(r.pitch_deg))
            tilt[(us(r.sim_time_s),us(r.px4_time_s))] = math.degrees(math.acos(min(1.,max(-1.,cos_tilt))))
    for r in series.itertuples():
        stamp, arrival = us(r.px4_time_s), us(r.sim_time_s)
        if stamp < takeoff_us or arrival >= landing_us:
            continue
        # In one log a PX4 report is 4 ms ahead of the callback that saw it. Wait
        # until both clocks have reached the report before exposing its features.
        available = max(arrival, stamp)
        valid = (r.fresh and 0 <= r.local_position_age_wall_s <= MAX_AGE_US/US and
                 all(math.isfinite(v) for v in [r.vx_mps,r.vy_mps,r.vz_mps,r.z_m]))
        if valid:
            features = history.update(stamp/US, math.hypot(r.vx_mps, r.vy_mps), r.vz_mps)
        else:
            history = Kinematics()
            features = None
        records.append((available, stamp, features, float(r.z_m) if valid else None, tilt.get((arrival,stamp)), arrival))
    return records


def valid_native_pair(left, right):
    v = [left.voltage_v, left.current_a, right.voltage_v, right.current_a]
    return (left.electrical_pair_qualified and right.electrical_pair_qualified and
            not left.command_limited and not right.command_limited and all(math.isfinite(x) and x > 0 for x in v))


def interval_energy(left, right, start, end):
    """Interpolate endpoint power V*I; exactly preserves the release trapezoid on splits."""
    t0, t1 = us(left.time_s), us(right.time_s)
    p0, p1 = left.voltage_v*left.current_a, right.voltage_v*right.current_a
    p_start = p0 + (p1-p0) * ((start-t0) / (t1-t0))
    p_end = p0 + (p1-p0) * ((end-t0) / (t1-t0))
    return p_start, p_end, (p_start+p_end)*0.5*(end-start)/(US*3600)


def state_on_span(row, left, right, start, end, phase, prediction):
    """Select native VI, qualified field power in unsupported transient phases, or an explicit hole."""
    if left is None or right is None:
        return 'unresolved', 'native_observations_do_not_bracket_interval', None
    if left.command_limited:
        return 'unresolved', 'rotor_command_limited_delivered_power_unknown', None
    if right.previous_interval_covered:
        if not valid_native_pair(left, right):
            raise ValueError('Qualified native interval conflicts with released endpoint evidence')
        p0, p1, wh = interval_energy(left, right, start, end)
        row.update(power_start_W=p0, power_end_W=p1, native_Wh=wh)
        return 'native_vi', 'qualified_native_VI', wh
    # An incompletely covered native interval cannot be split into covered/missing portions
    # by guessing the location of the hidden transition from the cumulative time difference.
    if right.native_covered_s - left.native_covered_s > 1e-7:
        return 'unresolved', 'incomplete_native_interval_or_unqualified_endpoints', None
    if row['manager_landed']:
        return 'unresolved', 'airborne_flag_not_yet_received_by_manager', None
    if phase not in TRANSIENT:
        if phase != left.phase or phase != right.phase:
            return 'unresolved', 'native_gap_at_phase_boundary', None
        return 'unresolved', 'native_gap_outside_target_phases', None
    if not (left.status == 'POWER_UNAVAILABLE_CAP_HELD' and left.demand_source == 'UNSUPPORTED_PHASE'
            and left.phase == phase):
        return 'unresolved', 'phase_boundary_or_demand_unavailable', None
    if not (left.raw_command_fresh and left.demand_fresh and left.armed and not left.q_domain_exhausted
            and 0 < us(right.time_s) - us(left.time_s) <= MAX_AGE_US):
        return 'unresolved', 'native_controls_or_operating_domain_unconfirmed', None
    power = prediction['power_w']
    if power is None:
        return 'unresolved', prediction['reason'], None
    if not math.isfinite(power) or power <= 0:
        raise ValueError('Calibration returned invalid accepted power')
    wh = power * (end-start)/(US*3600)
    row.update(power_start_W=power, power_end_W=power, calibrated_Wh=wh)
    return 'cmu_transient', prediction['reason'], wh


def flight_ledger(item, case, flag_rows, release, model):
    from .transient_cmu import predict

    cid = case['case_id']
    root = release / 'evidence' / cid
    trace = pd.read_csv(root/'flight_trace.csv.gz')
    pack_frame = pd.read_csv(release/'data'/f'{cid}.pack.csv.gz')
    if not (set(pack_frame.run_id) == {item['run_id']} and pack_frame.epoch.nunique() == 1):
        raise ValueError(f'{cid}: native run identity mismatch')
    packs = list(pack_frame.itertuples())
    pt = [us(p.time_s) for p in packs]
    sim = np.rint(trace.sim_time_s.to_numpy() * US).astype('int64')
    if np.any(np.diff(pt) <= 0) or np.any(np.diff(sim) <= 0):
        raise ValueError(f'{cid}: clock discontinuity in source')
    ulog_landed, last_land_stamp = flag_series(flag_rows, 'landed')
    air0, air1 = one_air_gate(ulog_landed)
    contacts, _ = flag_series(flag_rows, 'ground_contact')
    contact0, contact1 = one_air_gate(contacts)
    maybe, _ = flag_series(flag_rows, 'maybe_landed')
    maybe0, maybe1 = one_air_gate(maybe)
    if not (air0 <= maybe0 <= contact0 < contact1 <= maybe1 <= air1):
        raise ValueError(f'{cid}: contact-clear, maybe-landed and airborne windows conflict')
    armed, last_arm_stamp = flag_series(flag_rows, 'arming_state')
    if [value for _, value in armed] != [1,2,1] or not (armed[1][0] <= air0 < air1 <= armed[2][0]):
        raise ValueError(f'{cid}: airborne duration conflicts with the armed transitions')
    if min(last_land_stamp, last_arm_stamp) < air1 or sim[0] > air0 or sim[-1] < air1:
        raise ValueError(f'{cid}: unbracketed airborne flag interval')
    cb0, cb1 = one_air_gate(change_events(sim, trace.landed.to_numpy()))
    phases = change_events(sim, trace.phase.to_numpy())
    phase_times = [t for t,_ in phases]
    attitude = pd.read_csv(release/'data'/f'{cid}.trajectory.csv.gz',usecols=['sim_time_s','px4_time_s','roll_deg','pitch_deg'])
    nav = kinematics(trace, attitude, air0, air1)
    arrivals = [i[0] for i in nav]
    # Freshness cutoffs matter only when an update is not replaced first.
    expiries = [min(stamp+MAX_AGE_US, arrival+MAX_AGE_US) for i, (arrival, stamp, _, _, _, _) in enumerate(nav)
                if i+1 == len(nav) or stamp+MAX_AGE_US < arrivals[i+1]]
    edges = sorted({air0,air1} | {int(s) for s in [*pt,*arrivals,*phase_times,*expiries,contact0,contact1,maybe0,maybe1,cb0,cb1]
                                  if air0 < s < air1})
    predictions = {}
    rows = []
    missing = {'power_w': None, 'reason': 'missing_or_stale_kinematics', 'regime': 'unclassified',
               'support_flights': None, 'support_ids': []}
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        i = bisect_right(pt, start)-1
        left = packs[i] if 0 <= i < len(packs) else None
        right = packs[i+1] if 0 <= i+1 < len(packs) else None
        phase = phases[bisect_right(phase_times, start)-1][1]
        j = bisect_right(arrivals,start)-1
        available, stamp, features, alt, tilt, arrival = nav[j] if j >= 0 else (None,None,None,None,None,None)
        if stamp is None or end-stamp > MAX_AGE_US:
            features = None
        prediction = missing
        if phase in TRANSIENT and features is not None:
            if j not in predictions:
                inputs = {k:features[k] for k in ('speed_mps','along_accel_mps2','vertical_up_mps','low_speed_dwell_s')}
                predictions[j] = predict(model, payload_kg=case['payload_kg'], altitude_agl_m=alt, tilt_deg=tilt, **inputs)
            prediction = predictions[j]
        ids = prediction.get('support_ids') or []
        row = dict(case_id=cid, run_id=item['run_id'], start_s=start/US, end_s=end/US, duration_s=(end-start)/US,
                   phase=phase, manoeuvre=prediction['regime'] if phase in TRANSIENT else phase.lower(),
                   contact_possible=not contact0 <= start < contact1, maybe_landed=not maybe0 <= start < maybe1,
                   manager_landed=not cb0 <= start < cb1,
                   native_left_s=left.time_s if left else None, native_right_s=right.time_s if right else None,
                   command_limited=left.command_limited if left else None, native_status=left.status if left else None,
                   native_attributed_phase=left.phase if left else None,
                   previous_interval_covered=right.previous_interval_covered if right else False,
                   px4_time_s=stamp/US if stamp is not None else None, position_seen_sim_s=arrival/US if arrival is not None else None,
                   position_usable_sim_s=available/US if available is not None else None,
                   altitude_agl_m=alt, tilt_deg=tilt, support_flights=prediction.get('support_flights'),
                   support_ids='|'.join(map(str,ids)), calibration_reason=prediction['reason'] if phase in TRANSIENT else None,
                   native_Wh=None, calibrated_Wh=None, power_start_W=None, power_end_W=None)
        for key in ('speed_mps','along_accel_mps2','vertical_up_mps','low_speed_dwell_s'):
            row[key] = features[key] if features else None
        status, reason, wh = state_on_span(row, left, right, start, end, phase, prediction)
        row.update(status=status, reason=reason, qualified_Wh=wh)
        rows.append(row)
    ledger = pd.DataFrame(rows)
    # Independent native-only reference: one integral per original release interval on the
    # airborne gate before partitioning by phase, position or any calibration information.
    check_wh, check_us = 0., 0
    for prev, curr in zip(packs[:-1],packs[1:],strict=True):
        a,b = max(us(prev.time_s),air0),min(us(curr.time_s),air1)
        if a < b and curr.previous_interval_covered:
            check_wh += interval_energy(prev,curr,a,b)[2]
            check_us += b-a
    qualified = ledger[ledger.status.eq('native_vi')]
    if (abs(qualified.native_Wh.sum()-check_wh)>1e-9 or us(qualified.duration_s.sum()) != check_us
            or us(ledger.duration_s.sum()) != air1-air0):
        raise ValueError(f'{cid}: source energy or duration changed during splitting')
    details = dict(case_id=cid, run_id=item['run_id'], aircraft_payload_kg=case['payload_kg'],
                   airborne_start_s=air0/US, airborne_end_s=air1/US, elapsed_s=(air1-air0)/US,
                   callback_airborne_start_s=cb0/US, callback_airborne_end_s=cb1/US,
                   liftoff_flag_delay_s=(cb0-air0)/US, touchdown_flag_delay_s=(cb1-air1)/US,
                   contact_clear_start_s=contact0/US, contact_clear_end_s=contact1/US,
                   nominal_outbound_gate_m=json_file(root/'provenance.json')['energy_summary']['measurement_window_distance_m'],
                   position_reports_with_ahead_timestamp=sum(1 for available,stamp,_,_,_,arrival in nav if stamp>arrival),
                   largest_position_deferral_s=max((available-arrival for available,_,_,_,_,arrival in nav), default=0)/US,
                   native_unsplit_Wh=check_wh, native_unsplit_s=check_us/US, max_native_observation_gap_s=max(np.diff(pt))/US)
    return ledger, details


def totals(frame):
    seconds = lambda status: round(float(frame.loc[frame.status.eq(status),'duration_s'].sum()), 6)
    missing = seconds('unresolved')
    primary = float(frame.native_Wh.sum())
    added = float(frame.calibrated_Wh.sum())
    return dict(duration_s=round(float(frame.duration_s.sum()),6), intervals=len(frame), native_s=seconds('native_vi'),
                native_Wh=primary, calibrated_s=seconds('cmu_transient'), calibrated_Wh=added, unresolved_s=missing,
                qualified_subtotal_Wh=primary+added, complete=missing == 0., total_Wh=primary+added if missing == 0. else None)


def run(args):
    dest = args.out
    dest.mkdir(parents=True,exist_ok=True)
    locked = dict((name,digest) for digest,name in (l.split(maxsplit=1) for l in (args.release/'SHA256SUMS').read_text().splitlines()))
    for path in ['evidence/index.json','protocol.json']:
        verify(args.release/path, locked[path])
    index = json_file(args.release/'evidence/index.json')
    case_map = {c['case_id']:c for c in json_file(args.release/'protocol.json')['cases']}
    flags = flight_flags(index,args.release,args.raw_root,dest)
    if args.flags_only:
        return
    from .transient_cmu import load_model
    model_file = args.model or (dest/'model.json' if (dest/'model.json').is_file() else OUT/'model.json')
    model = load_model(model_file)
    flights, groups, checks, input_hashes = [], [], [], {}
    for item in index:
        cid = item['case_id']
        for name in ('flight_trace.csv.gz','provenance.json','receipt.json'):
            path = f'evidence/{cid}/{name}'
            verify(args.release/path, locked[path]); input_hashes[path]=locked[path]
        for suffix in ('pack','trajectory'):
            path = f'data/{cid}.{suffix}.csv.gz'
            verify(args.release/path,locked[path]); input_hashes[path]=locked[path]
        ledger, gate = flight_ledger(item,case_map[cid],[r for r in flags if r['case_id']==cid],args.release,model)
        path = dest/'data'/f'{cid}.intervals.csv.gz'
        save_table(path,ledger)
        summary = totals(ledger)
        sensitivity = totals(ledger.loc[~ledger.contact_possible])
        flights.append({**gate,**summary,**{'contact_clear_'+k:v for k,v in sensitivity.items()}})
        for level, keys in [('phase',['phase']),('manoeuvre',['phase','manoeuvre']),('reason',['phase','status','reason'])]:
            for values, subset in ledger.groupby(keys,sort=False,dropna=False):
                groups.append(dict(case_id=cid,level=level,**dict(zip(keys,values,strict=True)),**totals(subset)))
        reasons = ledger[ledger.status.eq('unresolved')].groupby('reason').duration_s.sum()
        checks.append(dict(case_id=cid, ledger_sha256=sha(path), rows=len(ledger), independent_native_Wh=gate['native_unsplit_Wh'],
                           native_residual_Wh=summary['native_Wh']-gate['native_unsplit_Wh'],
                           partition_residual_s=round(summary['duration_s']-gate['elapsed_s'],6),
                           unresolved_seconds_by_reason={str(k):round(float(v),6) for k,v in reasons.items()}))
    save_table(dest/'transient_airborne_flights.csv',flights)
    save_table(dest/'transient_airborne_segments.csv',groups)
    audit = dict(schema='M100_TRANSIENT_AIRBORNE_OFFLINE_V1',
                 air_gate='Half-open [first ULog vehicle_land_detected.landed=false, final landed=true); PX4 armed throughout. Contact-clear sensitivity requires ground_contact=false as well. These are estimator flags, not independent geometric contact measurements.',
                 clocks='PX4 SITL boot (microseconds) and Gazebo simulation time share a clock. Each position report becomes available at the later of its own PX4 timestamp and the first callback that saw it (one observed 4 ms lead). Kinematic derivatives use original PX4 report timestamps. Tilt is from the matched past-only attitude join with maximum ULog age 0.1 s. The native command_limited Boolean is held from each observation until the next event.',
                 interval_method='Exhaustive non-overlapping segments at airborne, contact, phase, native-event and received PX4 kinematic updates. A released previous_interval_covered flag qualifies the preceding native interval. Interpolate endpoint V*I linearly only inside that interval. For newly modeled transient power, hold the prior locally supported CMU estimate forward. Exclude native partial coverage, absent input, ambiguous intervals and command saturation. No inferred power enters the native battery ledger.',
                 missing_policy='Missing energy stays blank. Subtotals sum only justified native and modeled intervals. The total for a gate is reported only when it is fully covered.',
                 ground_scope='Simulation time preceding liftoff and at or after touchdown is excluded. Ground-contact and maybe-landed flags are retained during airborne detector transition periods.',
                 inputs={'release_manifest_sha256':sha(args.release/'SHA256SUMS'),'protocol_sha256':locked['protocol.json'],
                         'evidence_index_sha256':locked['evidence/index.json'],'model_sha256':sha(model_file),
                         'field_support_file':model['support_file'], 'field_support_sha256':model['support_sha256'],
                         'kinematics_and_predictor_source_sha256':sha(Path(__file__).with_name('transient_cmu.py')),
                         'integrator_source_sha256':sha(Path(__file__)),'flag_extract_sha256':sha(dest/'ulog_flight_flags.csv'),
                         'ulog_sha256':{r['case_id']:r['ulog_sha256'] for r in index},'flight_file_sha256':input_hashes},
                 flights=checks)
    (dest/'transient_airborne_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,default=RELEASE)
    parser.add_argument('--raw-root',type=Path,default=RAW)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--flags-only',action='store_true',help='Extract the original ULog ground and arming transitions only.')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
