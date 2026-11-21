"""Audit three recorded PX4/Gazebo pack drives. This command only reads completed acquisitions.

The native state clock is the time axis. Motor observations are joined causally on their own
Gazebo stamps. Electrical energy covers only intervals whose two acknowledged, unclipped
pack-model operating points and full native coverage counter agree. Current is not a measured
motor current. Rotor speeds are simulated joint observations scaled by Gazebo's slowdown.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/m100_reaudit/pack_drive_three'
RAW = ROOT / 'tmp/m100_reaudit_pack_three'
FIT_FILE = ROOT / 'results/m100_reaudit/cmu_calibration/fit.json'
PORTABLE = ROOT / 'tools/m100_reaudit/runtime_snapshot/px4/Tools/simulation/gz/models/m100_cmu_v2/model.sdf'
MOTORS = range(4)
NAMES = tuple(f'rotor_{i}_joint' for i in MOTORS)
LOADED = {'LOADED_FIT', 'LOADED_FIT_STARTUP_Q_FLOOR'}
EPS = 1e-8


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def vector(x):
    return isinstance(x, list) and len(x) == 4 and all(number(v) for v in x)


def close(x, y):
    return number(x) and number(y) and math.isclose(x, y, rel_tol=1e-10, abs_tol=EPS)


def current(now, t, max_age):
    return number(t) and -EPS <= now-t <= max_age + EPS


def constants(doc, supplied_sdf):
    receipt = doc['source_files']['m100_canonical_base_sdf']
    path = supplied_sdf or Path(receipt['path'])
    if path.is_file():
        source = path.read_bytes()
    else:
        # Reconstruct the recorded SDF from the release without changing any file.
        delta = (ROOT/'tools/m100_reaudit/pack_drive/px4.patch').read_bytes().split(b'\ndiff --git ',1)[0]
        source = subprocess.run(['patch','--batch','--silent','--output=-',str(PORTABLE)],
                                input=delta+b'\n',check=True,capture_output=True).stdout
    if hashlib.sha256(source).hexdigest() != receipt['sha256']:
        raise ValueError('Base-model SDF differs from the flight receipt')
    model = ET.fromstring(source)
    plugin = model.find(".//plugin[@name='m100::PackDrive']")
    if plugin is None:
        raise ValueError('Recorded model has no native pack drive')
    param = {tag:float(plugin.findtext(tag)) for tag in ('voltageIntercept','currentCoefficient','dischargeCoefficient',
        'minCurrent','maxCurrent','minDischargedAh','maxDischargedAh','maxDemandAge','referenceVoltage','nominalMaxRotVelocity')}
    rotor = model.findall(".//plugin[@name='gz::sim::systems::MulticopterMotorModel']")
    if len(rotor)!=4 or {motor.findtext('motorNumber') for motor in rotor} != {'0','1','2','3'}:
        raise ValueError('Four indexed stock rotors are required')
    for key in ('motorConstant','rotorVelocitySlowdownSim'):
        unique = {float(motor.findtext(key)) for motor in rotor}
        if len(unique)!=1:
            raise ValueError(f'Unmatched M100 rotors for {key}')
        param[key] = unique.pop()
    return param


def navigator(path):
    # Use the high-rate manager trace only for flight phase, kinematics and the measurement gate.
    keys = ('px4_time_s','x_m','y_m','z_m','vx_mps','vy_mps','vz_mps')
    times, samples = [], []
    with gzip.open(path,'rt',newline='') as f:
        for r in csv.DictReader(f):
            t = float(r['sim_time_s'])
            if times and t < times[-1]:
                raise ValueError('Rewound navigation timeline: dynamic epochs need separate navigation clocks')
            vals = {k:float(r[k]) if r[k] else None for k in keys}
            vals.update(flight_phase=r['phase'], measuring=r['measuring']=='1', nav_fresh=r['fresh']=='1')
            times.append(t)
            samples.append(vals)
    return times,samples


def preceding(index, t, age):
    times, values = index
    i = bisect_right(times,t + EPS) - 1
    return (times[i],values[i]) if i >= 0 and current(t,times[i],age) else (None,None)


def joint_index(records):
    epochs = defaultdict(dict)
    for row in records:
        obs = row.get('latest_joint_observation')
        if not obs:
            continue
        t = obs['time_s']
        speeds = obs['simulated_rotor_speed_rad_s']
        if not number(t) or not vector(speeds) or min(speeds)<0 or tuple(obs['joint_names'])!=NAMES:
            raise ValueError('Invalid stamped Gazebo rotor observation')
        found = epochs[row['epoch']].setdefault(t,speeds)
        if found != speeds:
            raise ValueError('Conflicting motor observations at identical joint time')
    result = {}
    for epoch, observations in epochs.items():
        times = sorted(observations)
        result[epoch] = (times,[observations[t] for t in times])
    return result


def flatten(case, records, doc, nav, p, reference, errors):
    joints = joint_index(records)
    mass = doc['energy_summary']['physical_total_mass_kg']
    entity = doc['temperature_application']['runtime_model'] + '_0'
    age = min(0.5,p['maxDemandAge'])
    cfg = doc['config']
    expected_cond = {'payload_kg':cfg['payload_kg'],'commanded_speed_mps':cfg['cruise_speed_mps'],
                     'commanded_altitude_agl_m':cfg['altitude_m'],'temperature_c':cfg['temperature_c'],'wind_speed_mps':cfg['wind_speed_mps']}
    phase_terms = {'TAKEOFF':'takeoff','CLIMB':'takeoff','CRUISE':'cruise','RETURNING':'cruise','LANDING':'landing'}
    mass_density = (reference['regression_baseline_kg']+cfg['payload_kg'])**1.5/math.sqrt(doc['atmosphere']['reference']['density_kg_m3'])
    samples = []
    for row in records:
        n = row['native']; t = row['time_s']; epoch = row['epoch']
        if (row['schema']!='M100_NATIVE_PACK_DRIVE_RECORD_V1' or row['run_id']!=doc['run_id'] or
                n['model']!=entity or not close(t,n['time_s'])):
            raise ValueError('Native wire identity or clock mismatch')
        if samples and (samples[-1]['epoch']!=epoch or samples[-1]['time_s']>=t):
            raise ValueError('This single-spawn protocol requires one ordered recording epoch per flight')
        native_fresh = n['demand_fresh'] is True and current(t,n['demand_time_s'],age)
        if (n['demand_fresh'] is True) != native_fresh:
            errors['demand_age_flag_mismatches'] += 1
        request = row.get('acknowledged_request')
        matched = bool(request and native_fresh and request['model']==entity and close(request['time_s'],n['demand_time_s'])
                       and request['armed'] is n['armed'] and (request['power_w'] == n['demand_power_w'] or
                       close(request['power_w'],n['demand_power_w'])))
        if (row['demand_fresh'] is True)!=matched:
            errors['ack_mismatches'] += 1
        ack = matched and row['demand_fresh'] is True
        if any(row['conditions'].get(k)!=v or (ack and request.get(k)!=v) for k,v in expected_cond.items()):
            errors['condition_mismatches'] += 1
        if ack and request['source']==reference['model'] and number(request['power_w']):
            coeff=reference['coefficients_by_phase'][phase_terms[request['phase']]]
            errors['max_phase_demand_model_residual_w'] = max(errors['max_phase_demand_model_residual_w'],
                                                             abs(request['power_w']-(coeff['b1']*mass_density+coeff['b0'])))
        raw_fresh = n['raw_command_fresh'] is True and current(t,n['raw_command_time_s'],age) and vector(n['raw_motor_rad_s'])
        if (n['raw_command_fresh'] is True)!=raw_fresh:
            errors['raw_age_flag_mismatches'] += 1
        raw = n['raw_motor_rad_s'] if raw_fresh else [None]*4
        applied = n['applied_motor_rad_s']
        if not vector(applied):
            raise ValueError('Missing native bounded command vector')
        disarmed = native_fresh and n['armed'] is False
        ceiling = n['omega_limit_rad_s']; cap_v = n['cap_voltage_v']
        expected_cap = p['nominalMaxRotVelocity']*max(0,min(1,cap_v/p['referenceVoltage']))
        errors['max_cap_error_rad_s'] = max(errors['max_cap_error_rad_s'],abs(ceiling-expected_cap))
        errors['max_thrust_fraction_error'] = max(errors['max_thrust_fraction_error'],abs(n['thrust_limit_fraction']-(expected_cap/p['nominalMaxRotVelocity'])**2))
        for i in MOTORS:
            expected = min(max(raw[i],0),ceiling) if raw_fresh and not disarmed else 0
            errors['max_bounded_command_error_rad_s'] = max(errors['max_bounded_command_error_rad_s'],abs(expected-applied[i]))
        should_limit = raw_fresh and not disarmed and max(raw)>ceiling
        if bool(n['command_limited'])!=should_limit:
            errors['limited_flag_mismatches'] += 1
        level = n['demand_power_w'] if native_fresh else None
        v, amps = n['voltage_v'], n['current_a']
        rv, ri = n['requested_load_voltage_v'], n['requested_pack_current_a']
        if (v is None)!=(amps is None) or (rv is None)!=(ri is None):
            errors['partial_vi_pairs'] += 1
        if number(rv) and number(ri) and number(level):
            errors['max_requested_vi_error_w'] = max(errors['max_requested_vi_error_w'],abs(rv*ri-level))
            fit_v = p['voltageIntercept']-p['currentCoefficient']*ri-p['dischargeCoefficient']*n['fit_q_ah']
            errors['max_requested_fit_error_v'] = max(errors['max_requested_fit_error_v'],abs(rv-fit_v))
            if not p['minCurrent']-EPS <= ri <= p['maxCurrent']+EPS:
                errors['fit_current_domain_violations'] += 1
        q = n['discharged_ah']
        if not close(n['fit_q_ah'],max(p['minDischargedAh'],min(p['maxDischargedAh'],q))):
            errors['fit_q_domain_violations'] += 1
        pair = (native_fresh and n['armed'] is True and raw_fresh and not n['command_limited'] and
                n['status'] in LOADED and number(v) and number(amps) and number(level) and level>0)
        if number(v) and number(amps):
            if not pair:
                errors['unsupported_vi_pairs'] += 1
            if number(level):
                errors['max_terminal_vi_error_w'] = max(errors['max_terminal_vi_error_w'],abs(v*amps-level))
            if not close(v,rv) or not close(amps,ri):
                errors['terminal_request_pair_mismatches'] += 1
        if n['status'] in LOADED or n['status']=='COMMAND_LIMITED_AT_REQUESTED_VOLTAGE':
            if not number(rv):
                errors['missing_supported_request_voltage'] += 1
            else:
                errors['max_active_cap_minus_requested_v'] = max(errors['max_active_cap_minus_requested_v'],abs(cap_v-rv))
        if n['delivered_power_w'] is not None:
            errors['unexpected_delivered_power_claims'] += 1
        if should_limit and n['status']=='COMMAND_LIMITED_AT_REQUESTED_VOLTAGE' and number(rv):
            mode = 'SUPPORTED_REQUEST_CEILING'
        else:
            mode = 'HELD_CEILING' if should_limit else 'NONE'
        jt, speed = preceding(joints.get(epoch,([],[])),t,age)
        nt, flight = preceding(nav,t,age)
        flight = flight or {}
        phase = request['phase'] if ack else flight.get('flight_phase','UNKNOWN')
        r = dict(case_id=case,run_id=doc['run_id'],epoch=epoch,time_s=t,phase=phase,
                 phase_source='ACKNOWLEDGED_REQUEST' if ack else 'FLIGHT_OBSERVER' if nt is not None else 'UNAVAILABLE',
                 demand_phase=request['phase'] if ack else None,demand_source=request['source'] if ack else None,
                 flight_phase=flight.get('flight_phase'),nav_time_s=nt,px4_time_s=flight.get('px4_time_s'),
                 measuring=flight.get('measuring'),nav_fresh=flight.get('nav_fresh'),
                 z_m=flight.get('z_m') if flight.get('nav_fresh') else None,
                 horizontal_speed_mps=math.hypot(flight['vx_mps'],flight['vy_mps']) if flight.get('nav_fresh') and
                 number(flight.get('vx_mps')) and number(flight.get('vy_mps')) else None,
                 status=n['status'],armed=n['armed'] if native_fresh else None,native_demand_fresh=native_fresh,
                 demand_fresh=ack,demand_time_s=n['demand_time_s'],demand_power_w=level,
                 requested_load_voltage_v=rv if native_fresh else None,requested_pack_current_a=ri if native_fresh else None,
                 voltage_v=v if pair else None,current_a=amps if pair else None,electrical_pair_qualified=bool(pair and ack),
                 discharged_ah=q,fit_q_ah=n['fit_q_ah'],q_at_lower_boundary=n['q_at_lower_boundary'],
                 q_domain_exhausted=n['q_domain_exhausted'],loaded_charge_coverage_complete=n['loaded_charge_coverage_complete'],
                 native_covered_s=n['covered_load_time_s'],native_unaccounted_s=n['unaccounted_load_time_s'],
                 native_idle_s=n['unmodelled_idle_time_s'],cap_voltage_v=cap_v,omega_limit_rad_s=ceiling,
                 thrust_limit_fraction=n['thrust_limit_fraction'],command_limited=bool(n['command_limited']),limit_mode=mode,
                 raw_command_time_s=n['raw_command_time_s'],raw_command_fresh=raw_fresh,drive_disarmed=disarmed,
                 joint_time_s=jt,joint_age_s=t-jt if jt is not None else None,joint_fresh=jt is not None)
        for i in MOTORS:
            r[f'raw_motor_{i}_rad_s']=raw[i]; r[f'applied_motor_{i}_rad_s']=applied[i]
            r[f'joint_motor_{i}_rad_s']=speed[i] if speed else None
        r['mean_joint_motor_rad_s'] = statistics.fmean(speed) if speed else None
        r['gross_rotor_thrust_over_weight'] = p['motorConstant']*sum(x*x for x in speed)/(mass*9.80665) if speed else None
        r['peak_raw_minus_applied_while_limited_rad_s'] = max(x-y for x,y in zip(raw,applied)) if should_limit else None
        samples.append(r)
    return samples


def intervals(samples, age, errors):
    # The observer is event/20 Hz. A positive change smaller than elapsed in the native
    # physics-step coverage counter exposes a hidden hole even between two valid endpoints.
    totals = defaultdict(Counter)
    episode = 0
    previous = None
    for r in samples:
        t = r['time_s']
        dt = t-previous['time_s'] if previous else None
        near = dt is not None and 0 < dt <= age+EPS
        clipped = r['limit_mode']!='NONE'
        new_episode = clipped and (not near or previous['limit_mode']!=r['limit_mode'])
        if new_episode:
            episode += 1
            totals['ALL'][r['limit_mode']+'_episodes'] += 1
            totals[r['phase']][r['limit_mode']+'_episodes'] += 1
        r.update(clip_episode_id=episode if clipped else None, since_previous_s=dt,
                 previous_interval_covered=False,previous_interval_energy_Wh=None,previous_interval_charge_Ah=None)
        if previous:
            dcovered = r['native_covered_s']-previous['native_covered_s']
            dq = r['discharged_ah']-previous['discharged_ah']
            if min(dcovered, dq)<-EPS or dcovered>dt+EPS:
                errors['native_counter_violations'] += 1
            if near and previous['command_limited']:
                mode=previous['limit_mode']
                for name in ('ALL',previous['phase']):
                    totals[name][mode+'_left_hold_s'] += dt
                    if mode == r['limit_mode']:
                        totals[name][mode+'_between_clipped_observations_s'] += dt
            if (near and r['electrical_pair_qualified'] and previous['electrical_pair_qualified']
                    and close(dcovered,dt)):
                lo, hi = sorted((previous['current_a']*dt/3600,r['current_a']*dt/3600))
                outside = max(lo-dq,dq-hi,0.0)
                errors['max_interval_q_outside_endpoint_bounds_Ah'] = max(errors['max_interval_q_outside_endpoint_bounds_Ah'],outside)
                if outside > EPS:
                    errors['charge_interval_outside_endpoint_bounds'] += 1
                if close(previous['demand_power_w'],r['demand_power_w']) and (
                        r['voltage_v']>previous['voltage_v']+EPS or r['current_a']<previous['current_a']-EPS):
                    errors['constant_demand_discharge_trend_violations'] += 1
                wh = ah = 0.0
                for endpoint in (previous,r):
                    partial_ah = endpoint['current_a'] * dt / 7200
                    partial_wh = endpoint['voltage_v'] * partial_ah
                    for name in ('ALL',endpoint['phase']):
                        totals[name]['covered_s'] += dt/2
                        totals[name]['integrated_Wh'] += partial_wh
                        totals[name]['integrated_Ah'] += partial_ah
                    wh += partial_wh; ah += partial_ah
                r.update(previous_interval_covered=True,previous_interval_energy_Wh=wh,
                         previous_interval_charge_Ah=ah)
                totals['ALL']['native_dq_on_same_intervals_Ah'] += dq
                errors['max_interval_q_residual_Ah'] = max(errors['max_interval_q_residual_Ah'],abs(dq-ah))
        previous=r
    return totals


def stats(items):
    if not items:
        return (None,)*4
    return (min(items),statistics.median(items),max(items),statistics.pstdev(items))


def characterize(rows, accu):
    paired = [r for r in rows if r['electrical_pair_qualified']]
    levels = sorted({round(r['demand_power_w'],6) for r in paired})
    result = dict(native_state_rows=len(rows),qualified_pair_rows=len(paired),
                  with_ack_rows=sum(r['demand_fresh'] for r in rows),positive_qualified_levels_1e_6_W=len(levels),
                  positive_qualified_levels_W='|'.join(f'{v:.6f}' for v in levels),
                  distinct_native_statuses='|'.join(sorted({r['status'] for r in rows})),
                  raw_absent_or_stale_rows=sum(not r['raw_command_fresh'] for r in rows),
                  disarmed_rows=sum(r['drive_disarmed'] for r in rows),joint_absent_or_stale_rows=sum(not r['joint_fresh'] for r in rows),
                  covered_s=accu['covered_s'],covered_electrical_Wh=accu['integrated_Wh'] if accu['covered_s'] else None,
                  covered_electrical_Ah=accu['integrated_Ah'] if accu['covered_s'] else None,
                  covered_time_mean_w=accu['integrated_Wh']*3600/accu['covered_s'] if accu['covered_s'] else None,
                  first_state_s=rows[0]['time_s'],last_state_s=rows[-1]['time_s'],
                  first_discharged_ah=rows[0]['discharged_ah'],last_discharged_ah=rows[-1]['discharged_ah'])
    joint = {r['joint_time_s']:r for r in rows if r['joint_fresh']}
    result['distinct_joint_stamps']=len(joint)
    result['max_accepted_joint_age_s']=max((r['joint_age_s'] for r in rows if r['joint_fresh']),default=None)
    for name in ('demand_power_w','voltage_v','current_a'):
        values = [r[name] for r in paired]
        for kind,value in zip(('min','median','max','sd'),stats(values)):
            result[f'{name}_{kind}']=value
    if paired:
        for field in ('voltage_v','current_a'):
            result[f'{field}_last_minus_first'] = paired[-1][field]-paired[0][field]
    for field in ('z_m','horizontal_speed_mps'):
        gathered=[r[field] for r in rows if r[field] is not None]
        result[f'{field}_median']=statistics.median(gathered) if gathered else None
    for name in ('mean_joint_motor_rad_s','gross_rotor_thrust_over_weight'):
        for kind,value in zip(('min','median','max','sd'),stats([r[name] for r in joint.values()])):
            result[f'{name}_{kind}']=value
    limited=[r for r in rows if r['command_limited']]
    for mode, label in [('HELD_CEILING','held'),('SUPPORTED_REQUEST_CEILING','supported')]:
        selected=[r for r in limited if r['limit_mode']==mode]
        result.update({f'clip_{label}_rows':len(selected),f'clip_{label}_episodes':accu[mode+'_episodes'],
                       f'clip_{label}_left_hold_s':accu[mode+'_left_hold_s'],
                       f'clip_{label}_between_observations_s':accu[mode+'_between_clipped_observations_s'],
                       f'clip_{label}_peak_reduction_rad_s':max((r['peak_raw_minus_applied_while_limited_rad_s'] for r in selected),default=None)})
    return result


def write_csv(path, records):
    fields=list(dict.fromkeys(key for r in records for key in r))
    if not fields:
        return
    def emit(text):
        writer=csv.DictWriter(text,fieldnames=fields,lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    if path.suffix=='.gz':
        with path.open('wb') as out, gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0) as gz:
            with io.TextIOWrapper(gz,encoding='utf-8',newline='') as text:
                emit(text)
    else:
        with path.open('w',encoding='utf-8',newline='') as out:
            emit(out)


def analyze_case(case, directory, spec, data_dir, sdf=None):
    cid=case['case_id']
    receipt_path=directory/(cid+'.json')
    receipt=load(receipt_path)
    for k in ('payload_kg','cruise_speed_mps','wind_speed_mps','temperature_c','altitude_m','measurement_distance_m'):
        if receipt[k]!=case[k]:
            raise ValueError(f'{cid} differs from planned {k}')
    summary=dict(case_id=cid,run_id=receipt.get('run_id'),outcome=receipt['outcome'],reason=receipt.get('reason'),
                 landed=receipt.get('landed'),armed_at_end=receipt.get('armed'),**{k:v for k,v in case.items() if k!='case_id'},
                 case_receipt_sha256=digest(receipt_path))
    if not receipt.get('run_id'):
        return dict(**summary,evidence_status='NO_RECORDED_FLIGHT'),[]
    rid=receipt['run_id']; doc_path=directory/(rid+'.provenance.json'); doc=load(doc_path)
    if doc['run_id']!=rid:
        raise ValueError('Provenance run is not the planned flight')
    info=doc['native_pack_drive']; receipt_pack=info['trace']; wire=directory/receipt_pack['filename']
    if (receipt_pack['io_error'] or digest(wire)!=receipt_pack['sha256'] or
            info['cmu_development_fit_sha256']!=digest(FIT_FILE)):
        raise ValueError(f'{cid} native pack recording or identified fit failed its checksum')
    sources=doc['source_files']
    for k,file_key in [('plugin','gazebo_m100_pack_drive_plugin'),('base_sdf','m100_canonical_base_sdf'),
                       ('manager','m100_experiment_manager'),('energy_config','m100_energy_v2_config')]:
        if spec['input_sha256'][k]!=sources[file_key]['sha256']:
            raise ValueError(f'{cid} acquisition differs from the registered {k} hash')
    nav_path=directory/(rid+'.trace.csv.gz'); nav=navigator(nav_path)
    with wire.open() as f:
        records=[json.loads(line) for line in f]
    if len(records)!=receipt_pack['rows'] or not records:
        raise ValueError(f'{cid} has an incomplete pack trace')
    p=constants(doc,sdf); age=min(0.5,p['maxDemandAge']); fit=load(FIT_FILE); cal=fit['loaded_voltage']
    coeff=cal['coefficients']; train=cal['training_predictor_ranges']
    if not all(close(p[key],value) for key,value in zip(('voltageIntercept','currentCoefficient','dischargeCoefficient',
            'minCurrent','maxCurrent','minDischargedAh','maxDischargedAh'),(coeff['a_V'],coeff['r_V_per_A'],coeff['k_V_per_Ah'],
            *train['current_A'],*train['net_discharged_Ah']))):
        raise ValueError('Native SDF pack parameters differ from the registered electrical fit')
    checked=('demand_age_flag_mismatches','ack_mismatches','raw_age_flag_mismatches','limited_flag_mismatches',
             'partial_vi_pairs','fit_current_domain_violations','fit_q_domain_violations','unsupported_vi_pairs',
             'terminal_request_pair_mismatches','unexpected_delivered_power_claims','native_counter_violations',
             'missing_supported_request_voltage','charge_interval_outside_endpoint_bounds','constant_demand_discharge_trend_violations','condition_mismatches')
    residuals=('max_cap_error_rad_s','max_thrust_fraction_error','max_bounded_command_error_rad_s',
               'max_requested_vi_error_w','max_requested_fit_error_v','max_terminal_vi_error_w',
               'max_active_cap_minus_requested_v','max_interval_q_residual_Ah','max_interval_q_outside_endpoint_bounds_Ah',
               'max_phase_demand_model_residual_w')
    errors=Counter(dict.fromkeys((*checked,*residuals),0))
    samples=flatten(cid,records,doc,nav,p,fit['published_power_reference'],errors); sums=intervals(samples,age,errors)
    phases=defaultdict(list)
    for r in samples:
        phases[r['phase']].append(r)
    phase_rows=[dict(case_id=cid,run_id=rid,payload_kg=case['payload_kg'],phase=k,**characterize(group,sums[k])) for k,group in phases.items()]
    first,last=samples[0],samples[-1]; whole=characterize(samples,sums['ALL']); q_native=sums['ALL']['native_dq_on_same_intervals_Ah']
    wh_residual=math.fsum(r['covered_electrical_Wh'] or 0 for r in phase_rows)-sums['ALL']['integrated_Wh']
    duration_residual=math.fsum(r['covered_s'] for r in phase_rows)-sums['ALL']['covered_s']
    accurate=(not any(errors[k] for k in checked) and all(errors[k]<1e-7 for k in residuals if k!='max_interval_q_residual_Ah')
              and abs(wh_residual)<EPS and abs(duration_residual)<EPS)
    start_q=first['discharged_ah']; end_q=last['discharged_ah']; native_s=last['native_covered_s']-first['native_covered_s']
    native_gap=last['native_unaccounted_s']-first['native_unaccounted_s']; native_idle=last['native_idle_s']-first['native_idle_s']
    file=data_dir/(cid+'.pack.csv.gz'); write_csv(file,samples)
    mass=doc['energy_summary']['physical_total_mass_kg']; source_end=doc['recording_end']['native_pack_drive']
    # This comparison uses native q on exactly the same intervals as the decimated quadrature.
    # Absolute elapsed flight q also contains short, valid physics intervals excluded here.
    receipt_dt=source_end['time_s']-last['time_s'] if number(source_end.get('time_s')) else None
    receipt_dq=source_end['discharged_ah']-end_q if number(source_end.get('discharged_ah')) else None
    if close(receipt_dt,0) and not close(receipt_dq,0):
        accurate=False
    summary.update(whole,evidence_status='PACK_ARITHMETIC_CONSISTENT' if accurate and whole['qualified_pair_rows'] else 'CHECK_PACK_EVIDENCE',
       **errors,native_covered_delta_s=native_s,native_unaccounted_delta_s=native_gap,native_idle_delta_s=native_idle,
       observed_span_s=last['time_s']-first['time_s'],native_q_delta_ah=end_q-start_q,
       native_q_delta_same_intervals_ah=q_native,charge_residual_same_intervals_ah=sums['ALL']['integrated_Ah']-q_native,
       covered_fraction_of_native_covered=whole['covered_s']/native_s if native_s else None,
       terminal_receipt_state_s=source_end.get('time_s'),terminal_receipt_q_ah=source_end.get('discharged_ah'),
       terminal_receipt_minus_last_state_s=receipt_dt,terminal_receipt_minus_last_q_ah=receipt_dq,
       sum_phases_minus_total_Wh=wh_residual,sum_phases_minus_total_duration_s=duration_residual,
       recorded_final_q_coverage_complete=last['loaded_charge_coverage_complete'],recorded_final_q_domain_exhausted=last['q_domain_exhausted'],
       nominal_hover_speed_rad_s=math.sqrt(mass*9.80665/(4*p['motorConstant'])),physical_total_mass_kg=mass,
       thrust_coefficient_N_per_rad_s_squared=p['motorConstant'],joint_slowdown_factor=p['rotorVelocitySlowdownSim'],
       speed_reference_voltage_v=p['referenceVoltage'],nominal_max_motor_rad_s=p['nominalMaxRotVelocity'],maximum_signal_age_s=age,
       model=records[0]['native']['model'],physics_step_observed_s=receipt['observed_physics_step_s'],seed_observed=receipt.get('observed_seed'),
       energy_trace_sha256=receipt_pack['sha256'],provenance_sha256=digest(doc_path),kinematics_trace_gz_sha256=digest(nav_path),
       fit_sha256=info['cmu_development_fit_sha256'],plugin_sha256=sources['gazebo_m100_pack_drive_plugin']['sha256'],
       base_sdf_sha256=sources['m100_canonical_base_sdf']['sha256'],series_file='data/'+file.name,series_gz_sha256=digest(file))
    return summary,phase_rows


def csv_rows(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def verify_saved(out, spec):
    """Independently audit the published compact files without the acquisition workspace."""
    summaries=csv_rows(out/'electrical.csv')
    phases=csv_rows(out/'phases.csv')
    if [case['case_id'] for case in spec['cases']]!=[r['case_id'] for r in summaries]:
        raise ValueError('Exported flights differ from the preregistered order')
    for summary in summaries:
        if summary['evidence_status']=='NO_RECORDED_FLIGHT':
            continue
        path=out/summary['series_file']; cid=summary['case_id']
        if digest(path)!=summary['series_gz_sha256']:
            raise ValueError(f'{cid}: compressed series checksum mismatch')
        samples=csv_rows(path)
        age=float(summary['maximum_signal_age_s'])
        previous=None; increments=[]; durations=[]; qs=[]; sums_phase=defaultdict(list)
        for r in samples:
            t=float(r['time_s']); omega=float(r['omega_limit_rad_s'])
            fraction=min(1.0,float(r['cap_voltage_v'])/float(summary['speed_reference_voltage_v']))
            if (r['case_id']!=cid or r['run_id']!=summary['run_id'] or
                    not close(omega,float(summary['nominal_max_motor_rad_s'])*fraction)
                    or not close(float(r['thrust_limit_fraction']),fraction**2)):
                raise ValueError(f'{cid}: invalid sample identity or rotor ceiling at {t}')
            if r['joint_fresh']=='True':
                if not current(t,float(r['joint_time_s']),age) or any(float(r[f'joint_motor_{i}_rad_s'])<0 for i in MOTORS):
                    raise ValueError(f'{cid}: stale or invalid speed measurement at {t}')
            elif any(r[f'joint_motor_{i}_rad_s'] for i in MOTORS):
                raise ValueError(f'{cid}: stale joint speed exposed at {t}')
            for i in MOTORS:
                raw=r[f'raw_motor_{i}_rad_s']
                expected=0.0 if not raw or r['drive_disarmed']=='True' else max(0.0,min(omega,float(raw)))
                if not close(expected,float(r[f'applied_motor_{i}_rad_s'])):
                    raise ValueError(f'{cid}: motor {i} missed its ceiling at {t}')
            qualified=r['electrical_pair_qualified']=='True'
            power=float(r['voltage_v'])*float(r['current_a']) if qualified else None
            if qualified and (not close(power,float(r['demand_power_w'])) or r['demand_fresh']!='True' or r['command_limited']=='True'):
                raise ValueError(f'{cid}: unverified operating point at {t}')
            include=False
            if previous:
                old_time,old_power,old=previous; dt=t-old_time
                if dt<=0:
                    raise ValueError(f'{cid}: pack clock failed to advance')
                include=(qualified and old_power is not None and dt<=age+EPS and
                         close(float(r['native_covered_s'])-float(old['native_covered_s']),dt))
                if include:
                    wh=(old_power+power)*dt/7200
                    dq=(float(old['current_a'])+float(r['current_a']))*dt/7200
                    if not close(wh,float(r['previous_interval_energy_Wh'])) or not close(dq,float(r['previous_interval_charge_Ah'])):
                        raise ValueError(f'{cid}: wrong interval energy or charge at {t}')
                    increments.append(wh); qs.append(dq); durations.append(dt)
                    sums_phase[old['phase']].append(old_power*dt/7200)
                    sums_phase[r['phase']].append(power*dt/7200)
            if include!=(r['previous_interval_covered']=='True'):
                raise ValueError(f'{cid}: coverage flag differs from native counter at {t}')
            previous=(t,power,r)
        if (len(samples)!=int(summary['native_state_rows']) or not close(math.fsum(increments),float(summary['covered_electrical_Wh']))
                or not close(math.fsum(durations),float(summary['covered_s'])) or not close(math.fsum(qs),float(summary['covered_electrical_Ah']))):
            raise ValueError(f'{cid}: integrated flight totals changed')
        for row in (phase for phase in phases if phase['case_id']==cid):
            if not close(math.fsum(sums_phase[row['phase']]),float(row['covered_electrical_Wh'] or 0)):
                raise ValueError(f"{cid}: energy differs for {row['phase']}")
    print(f'Verified {len(summaries)} recorded case summaries and their compact pack series')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,default=RAW)
    parser.add_argument('--out','--root',type=Path,default=OUT,help='Output directory containing protocol.json')
    parser.add_argument('--protocol',type=Path,help='Read the preregistered design here instead of OUT/protocol.json')
    parser.add_argument('--sdf',type=Path,help='Exact canonical base SDF used in the acquisition')
    parser.add_argument('--verify-saved',action='store_true',help='Read and verify the exported time series and tables without original simulator logs')
    args=parser.parse_args(); spec=load(args.protocol or args.out/'protocol.json')
    if args.verify_saved:
        verify_saved(args.out,spec)
        return
    args.out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out,prefix='.pack-analysis-') as scratch:
        tmp=Path(scratch); (tmp/'data').mkdir(); reports=[]; phases=[]
        for case in spec['cases']:
            summary,by_phase=analyze_case(case,args.raw_root/case['case_id']/'attempt_1',spec,tmp/'data',args.sdf)
            reports.append(summary); phases.extend(by_phase)
        write_csv(tmp/'electrical.csv',reports); write_csv(tmp/'phases.csv',phases)
        (args.out/'data').mkdir(exist_ok=True)
        for f in sorted((tmp/'data').glob('*.csv.gz')):
            f.replace(args.out/'data'/f.name)
        for name in ('phases.csv','electrical.csv'):
            if (tmp/name).is_file(): (tmp/name).replace(args.out/name)
    print(f"{len(reports)} planned cases, {len(phases)} case/phase rows: {args.out}")


if __name__=='__main__':
    main()
