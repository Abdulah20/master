"""Estimate whole airborne pack energy from the three recorded M100 missions, offline.

Run after the strict airborne ledger and the pooled CMU model are published with
``python3 -B -m tools.m100_reaudit.transient_practical``. The strict ledger is a read-only input.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .transient_airborne import OUT, RELEASE, TRANSIENT, US, json_file, save_table, sha, verify
from .transient_summary import manager_integrals


SOURCE_COLUMNS = {
    'native_pack_model': 'native', 'cmu_local': 'local_cmu', 'cmu_pooled': 'pooled',
    'recorded_manager': 'recorded_manager',
}
POOLED_DOMAIN = {
    'no_pooled_transfer_flag': 'no_flag', 'long_or_unknown_dwell': 'dwell',
    'bounded_dynamic_inputs_or_power': 'bounded', 'dwell_and_bounded_dynamics': 'dwell_and_bounded',
    'other_transfer_flag': 'other_flag',
}
DWELL_FLAGS = {'LONG_LOW_SPEED_DWELL_TRANSFER', 'UNKNOWN_LOW_SPEED_DWELL_TRANSFER'}
KINEMATICS = ('speed_mps', 'along_accel_mps2', 'vertical_up_mps', 'low_speed_dwell_s')


def domain_for(flags):
    dwell = bool(DWELL_FLAGS.intersection(flags))
    bounded = any('_CLAMPED_' in flag for flag in flags)
    if dwell and bounded:
        return 'dwell_and_bounded_dynamics'
    if dwell:
        return 'long_or_unknown_dwell'
    if bounded:
        return 'bounded_dynamic_inputs_or_power'
    return 'other_transfer_flag' if flags else 'no_pooled_transfer_flag'


def read_partition(path, gate):
    ledger = pd.read_csv(path)
    if not (ledger.case_id.eq(gate.case_id).all() and ledger.run_id.eq(gate.run_id).all()):
        raise ValueError(f'{gate.case_id}: strict ledger identity mismatch')
    start, end, duration = [np.rint(US*ledger[c]).to_numpy(dtype=np.int64)
                            for c in ('start_s', 'end_s', 'duration_s')]
    if (len(start) == 0 or np.any(end <= start) or not np.array_equal(end - start, duration)
            or not np.array_equal(end[:-1], start[1:]) or start[0] != round(US*gate.airborne_start_s)
            or end[-1] != round(US*gate.airborne_end_s)):
        raise ValueError(f'{gate.case_id}: strict rows do not partition the published airborne window')
    if (not ledger.status.isin(['native_vi', 'cmu_transient', 'unresolved']).all()
            or not ledger.qualified_Wh.isna().eq(ledger.status.eq('unresolved')).all()):
        raise ValueError(f'{gate.case_id}: incomplete or contradictory strict source labels')
    return ledger


def manager_coverage(ledger, trace):
    """Split a partially fresh fallback interval at its real manager callback times, if present."""
    starts = ledger.start_s.to_numpy()
    ends = ledger.end_s.to_numpy()
    seconds, watt_hours = manager_integrals(trace, starts, ends)
    partial = (ledger.status.eq('unresolved').to_numpy() & (seconds > 1e-7) & (ends-starts-seconds > 1e-7))
    if not partial.any():
        return ledger, seconds, watt_hours
    trace_us = np.rint(trace.sim_time_s.to_numpy()*US).astype(np.int64)
    positions, new_starts, new_ends = [], [], []
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        a, b = round(start*US), round(end*US)
        inner = trace_us[(trace_us > a) & (trace_us < b)] if partial[i] else []
        edges = [a, *inner, b]
        positions.extend([i]*(len(edges)-1))
        new_starts.extend(edges[:-1])
        new_ends.extend(edges[1:])
    frame = ledger.iloc[positions].reset_index(drop=True)
    start, end = np.asarray(new_starts)/US, np.asarray(new_ends)/US
    frame['start_s'], frame['end_s'], frame['duration_s'] = start, end, end-start
    seconds, watt_hours = manager_integrals(trace, start, end)
    if ((frame.status.eq('unresolved').to_numpy()) & (seconds > 1e-7) & (end-start-seconds > 1e-7)).any():
        raise ValueError('Manager coverage changes between its own callback timestamps')
    return frame, seconds, watt_hours


def fallback_eligible_for_pooled(row):
    # The strict reason equals its CMU rejection only after the observed controls and phase gates passed.
    return (row.phase in TRANSIENT and row.reason == row.calibration_reason and not row.command_limited
            and not row.manager_landed and row.native_status == 'POWER_UNAVAILABLE_CAP_HELD'
            and row.native_attributed_phase == row.phase)


def estimate_flight(ledger, trace, payload, model):
    from .transient_pooled import predict_kinematics

    ledger, manager_s, manager_wh = manager_coverage(ledger, trace)
    rows = []
    for row, seconds, historical_wh in zip(ledger.itertuples(), manager_s, manager_wh, strict=True):
        wh, pooled_reason, pooled_regime, flags = row.qualified_Wh, None, None, []
        if row.status == 'native_vi':
            source, domain = 'native_pack_model', 'native_release_covered'
        elif row.status == 'cmu_transient':
            source, domain = 'cmu_local', 'local_field_neighbours'
        else:
            source, domain, wh = 'unavailable', 'no_available_power', None
            if fallback_eligible_for_pooled(row):
                kinematics = {name: getattr(row, name) for name in KINEMATICS}
                pred = predict_kinematics(model, payload, kinematics, row.altitude_agl_m, tilt_deg=row.tilt_deg)
                pooled_reason, pooled_regime = pred['reason'], pred['regime']
                flags = sorted(set(pred['transfer_flags']))
                if pred['power_w'] is not None:
                    source, domain = 'cmu_pooled', domain_for(flags)
                    wh = pred['power_w']*row.duration_s/3600
            if wh is None and abs(seconds-row.duration_s) < 1e-7:
                source, domain, wh = 'recorded_manager', 'historical_manager_only', historical_wh
        if wh is not None and (not math.isfinite(wh) or wh <= 0):
            raise ValueError(f'{row.case_id} at {row.start_s}: invalid interval energy from {source}')
        rows.append(dict(case_id=row.case_id, run_id=row.run_id, start_s=row.start_s, end_s=row.end_s,
                         duration_s=row.duration_s, phase=row.phase, manoeuvre=row.manoeuvre,
                         source=source, domain_group=domain, transfer_flags='|'.join(flags), pooled_regime=pooled_regime,
                         pooled_reason=pooled_reason, strict_status=row.status, strict_reason=row.reason,
                         command_limited=row.command_limited, manager_landed=row.manager_landed, contact_possible=row.contact_possible,
                         strict_local_support_flights=row.support_flights, px4_time_s=row.px4_time_s,
                         altitude_agl_m=row.altitude_agl_m, tilt_deg=row.tilt_deg,
                         **{key:getattr(row,key) for key in KINEMATICS},
                         strict_qualified_Wh=row.qualified_Wh, estimated_Wh=wh,
                         estimated_mean_power_W=wh*3600/row.duration_s if wh is not None else None))
    return pd.DataFrame(rows)


def subtotal(frame, mask):
    return round(float(frame.loc[mask, 'duration_s'].sum()), 6), float(frame.loc[mask, 'estimated_Wh'].sum())


def totals(frame, detailed=False):
    out = {}
    for source, prefix in SOURCE_COLUMNS.items():
        out[prefix+'_s'], out[prefix+'_Wh'] = subtotal(frame, frame.source.eq(source))
    out['duration_s'] = round(float(frame.duration_s.sum()), 6)
    out['numeric_gap_s'] = round(float(frame.loc[frame.source.eq('unavailable'), 'duration_s'].sum()), 6)
    out['numeric_coverage_s'] = round(out['duration_s']-out['numeric_gap_s'], 6)
    out['numeric_coverage_fraction'] = out['numeric_coverage_s']/out['duration_s']
    out['strict_unresolved_s'] = round(float(frame.loc[frame.strict_status.eq('unresolved'), 'duration_s'].sum()), 6)
    out['strict_subtotal_Wh'] = float(frame.strict_qualified_Wh.sum())
    out['numeric_subtotal_Wh'] = float(frame.estimated_Wh.sum())
    out['practical_total_Wh'] = out['numeric_subtotal_Wh'] if out['numeric_gap_s'] == 0 else None
    out['practical_mean_power_W'] = (out['practical_total_Wh']*3600/out['duration_s']
                                     if out['practical_total_Wh'] is not None else None)
    if detailed:
        for domain, label in POOLED_DOMAIN.items():
            out['pooled_'+label+'_s'], out['pooled_'+label+'_Wh'] = subtotal(frame, frame.domain_group.eq(domain))
        if abs(sum(out['pooled_'+g+'_s'] for g in POOLED_DOMAIN.values()) - out['pooled_s']) > 1e-7:
            raise ValueError('Pooled interval transfer groups do not exhaust the pooled duration')
    return out


def run(strict_dir, source, model_path, output):
    from . import transient_pooled

    audit_path = strict_dir/'transient_airborne_audit.json'
    flights_path = strict_dir/'transient_airborne_flights.csv'
    strict = json_file(audit_path)
    digests = {case['case_id']:case['ledger_sha256'] for case in strict['flights']}
    gates = pd.read_csv(flights_path)
    if set(digests) != set(gates.case_id) or not gates.case_id.is_unique:
        raise ValueError('Strict flight list differs from its audit')
    model = transient_pooled.load_model(model_path)
    output.mkdir(parents=True, exist_ok=True)
    flights, groups, inputs = [], [], {}
    for gate in gates.itertuples():
        cid = gate.case_id
        primary = strict_dir/'data'/f'{cid}.intervals.csv.gz'
        trace_name = f'evidence/{cid}/flight_trace.csv.gz'
        verify(primary, digests[cid])
        trace_hash = strict['inputs']['flight_file_sha256'][trace_name]
        verify(source/trace_name, trace_hash)
        trace = pd.read_csv(source/trace_name, usecols=['sim_time_s', 'fresh', 'power_w'])
        ledger = read_partition(primary, gate)
        frame = estimate_flight(ledger, trace, gate.aircraft_payload_kg, model)
        summary = totals(frame, detailed=True)
        if (round(US*summary['duration_s']) != round(US*gate.elapsed_s)
                or abs(gate.qualified_subtotal_Wh-summary['strict_subtotal_Wh']) > 1e-9
                or abs(gate.unresolved_s-summary['strict_unresolved_s']) > 1e-7
                or abs(sum(summary[k+'_s'] for k in SOURCE_COLUMNS.values())-summary['numeric_coverage_s']) > 1e-7
                or abs(sum(summary[k+'_Wh'] for k in SOURCE_COLUMNS.values())-summary['numeric_subtotal_Wh']) > 1e-9):
            raise ValueError(f'{cid}: time, source energy, or strict coverage identity failed')
        path = output/'data'/f'{cid}.practical.csv.gz'
        save_table(path, frame)
        inputs[cid] = dict(strict_ledger_sha256=digests[cid], manager_trace_sha256=trace_hash,
                           practical_ledger_sha256=sha(path), rows=len(frame))
        flights.append(dict(case_id=cid,run_id=gate.run_id,payload_kg=gate.aircraft_payload_kg,
                            airborne_start_s=gate.airborne_start_s,airborne_end_s=gate.airborne_end_s,**summary))
        unknown = frame[frame.strict_status.eq('unresolved')]
        for level, subset, keys in [('phase',frame,['phase']),('manoeuvre',frame,['phase','manoeuvre']),
                                    ('source_domain',frame,['phase','source','domain_group']),
                                    ('pooled_flag_set',frame[frame.source.eq('cmu_pooled')],['domain_group','transfer_flags']),
                                    ('strict_gap_reason',unknown,['phase','strict_reason','source','domain_group','transfer_flags'])]:
            for key, part in subset.groupby(keys,sort=False,dropna=False):
                values = key if isinstance(key,tuple) else (key,)
                groups.append(dict(case_id=cid,level=level,**dict(zip(keys,values,strict=True)),**totals(part)))
    save_table(output/'practical_flights.csv', flights)
    save_table(output/'practical_segments.csv', groups)
    audit = dict(schema='M100_PRACTICAL_AIRBORNE_V1', scope='Existing recorded simulation missions, after-flight estimate at observed motion.',
                 air_gate=strict['air_gate'], units='s, W, Wh; mean interval power = Wh * 3600 / duration_s.',
                 precedence=['native_pack_model: preserve released pack-model VI energy',
                             'cmu_local: preserve qualified local CMU whole-aircraft pack-power estimate',
                             'cmu_pooled: pooled training-flight regression on the received kinematics in STABILIZING and RETURN_BRAKING',
                             'recorded_manager: integrate the previously recorded manager power wherever the preceding sources are unavailable',
                             'unavailable: no valid recorded power or permitted estimate'],
                 model_transfer='Local support gaps remain recorded in strict_status and strict_reason regardless of the practical energy source. Pooled power estimates use joint motion after the flight. Model flags describe marginal transfer or clipping of the field regression. A pooled prediction with zero flags does not establish a measured joint analogue.',
                 domain_partition='For pooled intervals classify as dwell_and_bounded_dynamics if both dwell and clamp flags occur, long_or_unknown_dwell for dwell only, bounded_dynamic_inputs_or_power for other clamps, other_transfer_flag for remaining flags and no_pooled_transfer_flag otherwise. Exact flag combinations remain in each row and in pooled_flag_set. The categories are disjoint. Different summary levels repeat the same elapsed time and must not be added together.',
                 fallback='Use the last fresh manager power at its actual recorded callback time until the next valid callback, via manager_integrals(). Clipped controls, partial native coverage, unconfirmed states and phase boundaries remain separate reasons. Manager energy estimates contextual pack demand; no delivered motor current is observed during clipping. No battery state or thrust is recomputed.',
                 missing='Numeric missing time stays identifiable; practical_total_Wh is populated only for a fully covered airborne window.',
                 inputs=dict(strict_audit_sha256=sha(audit_path),strict_flights_sha256=sha(flights_path),
                             pooled_model_sha256=sha(model_path),pooled_predictor_sha256=sha(Path(transient_pooled.__file__)),
                             integrator_sha256=sha(Path(__file__)),manager_integration_file_sha256=sha(Path(__file__).with_name('transient_summary.py')),
                             flights=inputs))
    (output/'practical_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict',type=Path,default=OUT,help='Published strict ledger and gate audit.')
    parser.add_argument('--source',type=Path,default=RELEASE,help='The original three-flight release.')
    parser.add_argument('--model',type=Path,default=OUT/'pooled_model.json')
    parser.add_argument('--out',type=Path,default=OUT)
    args = parser.parse_args()
    run(args.strict, args.source, args.model, args.out)
