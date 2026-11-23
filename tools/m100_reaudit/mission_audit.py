"""Audit closed native flights, reconcile each recorder label with PX4, and export flight observations."""
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from . import direct_airborne_evidence as evidence
from .mission_campaign import ARCHIVE, OUT, ROOT, read, save
from .mission_observations import flight_topics, online_series, table, trajectory

INTERVAL_CACHE=ROOT/'tmp/m100_closure/verified_intervals'
MODEL='CMU_LOADED_THEVENIN_2RC_V1'
TARGET=evidence.TARGET


def check_temperature(raw, doc, case, hashes):
    app=doc['temperature_application']
    if case['temperature_c']==15:
        evidence.confirm(app['reference_identity'] is True,'Reference density model differs')
        return hashes
    evidence.confirm(app['battery_temperature_physical'] is False and app['reference_identity'] is False,
                     'Invalid ambient temperature application')
    model=app['runtime_model']
    folder=raw/'state/temperature_models'/model
    manifest_file=folder/'manifest.json'
    manifest=read(manifest_file)
    key='base_sdf' if not case['payload_kg'] else f"payload{round(100*case['payload_kg']):03d}"
    ratio=doc['atmosphere']['density_ratio']
    evidence.confirm(manifest['temperature_c']==case['temperature_c'] and manifest['reference_altitude_m']==case['altitude_m']
                     and manifest['source_base_sdf_sha256']==hashes['base_sdf']
                     and manifest['source_payload_sdf_sha256']==hashes[key] and manifest['density_ratio']==ratio
                     and manifest['model_name']==model and manifest['case_sha256']==app['generated_case_sha256']
                     and evidence.digest(manifest_file)==app['generated_manifest_sha256']
                     and evidence.digest(folder/'model.sdf')==manifest['model_sdf_sha256']==app['generated_sdf_sha256']
                     and evidence.digest(folder.parent/manifest['scaled_base_model_name']/'model.sdf')
                         ==manifest['scaled_base_sdf_sha256']==app['generated_scaled_base_sdf_sha256']
                     and all(math.isclose(v*ratio,manifest['effective_coefficients'][k],rel_tol=1e-12)
                             for k,v in manifest['source_coefficients'].items()),'Generated temperature model differs')
    return {**hashes, 'applied_sdf':manifest['model_sdf_sha256']}


def pack_check(records, case, config, fit):
    samples=[r['native'] for r in records]
    evidence.confirm(all(s['battery_dynamics_model']==MODEL and s['battery_parameter_sha256']==fit['parameter_sha256']
                         and s['initial_nominal_soc_percent']==case['initial_soc_percent'] for s in samples),
                     'Nominal SOC or native battery identity differs')
    supported=[s for s in samples if s.get('voltage_v') is not None and s.get('current_a') is not None]
    stats=dict(native_vi_publications=len(supported), native_2rc_equation_max_residual_V=None,
               pack_power_balance_max_residual_W=None, min_supported_load_voltage_V=None,max_supported_current_A=None,
               max_fast_polarization_V=max(s['polarization_fast_v'] for s in samples),
               max_slow_polarization_V=max(s['polarization_slow_v'] for s in samples),
               low_charge_domain_exhausted=any(s['q_domain_exhausted'] for s in samples),
               peak_unaccounted_pack_load_s=max(s['unaccounted_load_time_s'] for s in samples),
               motor_command_limited_publications=sum(s['command_limited'] for s in samples))
    p=fit['parameters']
    for key,name in (('r0_ohm','r0'),('r1_ohm','r1'),('r2_ohm','r2')):
        evidence.confirm(p[key]>0, f'Nonpositive {name}')
    for s in samples:
        if s['raw_command_fresh'] and not s['awaiting_post_reset_zero_command'] and s['armed'] is True:
            evidence.confirm(all(a<=s['omega_limit_rad_s']+1e-8 and abs(a-min(max(r,0),s['omega_limit_rad_s']))<1e-7
                                 for a,r in zip(s['applied_motor_rad_s'],s['raw_motor_rad_s'],strict=True)),
                             'Actuator ceiling differs from applied motor commands')
        evidence.confirm(s['initial_nameplate_discharge_offset_ah']==(100-case['initial_soc_percent'])*4.5/100,
                         'Starting charge offset differs')
    if supported:
        err=max(abs(s['voltage_v']-(s['reference_source_v']-s['polarization_fast_v']-s['polarization_slow_v']
                    -p['r0_ohm']*s['current_a'])) for s in supported)
        pw=max(abs(s['current_a']*s['voltage_v']-s['effective_demand_power_w']) for s in supported)
        evidence.confirm(err<1e-8 and pw<1e-7,'2RC terminal voltage or power equation differs')
        stats.update(native_2rc_equation_max_residual_V=err, pack_power_balance_max_residual_W=pw,
                     min_supported_load_voltage_V=min(s['voltage_v'] for s in supported),
                     max_supported_current_A=max(s['current_a'] for s in supported),
                     max_supported_load_sag_V=max(s['reference_source_v']-s['voltage_v'] for s in supported),
                     min_supported_available_thrust_fraction=min(s['thrust_limit_fraction'] for s in supported))
    evidence.confirm(config['initial_nominal_soc_percent']==case['initial_soc_percent'],'Manager starting SOC differs')
    return stats


def plan_fields(case, config, protocol):
    points=protocol['route_waypoints_enu_m'][case['mission_route']]
    heading=math.radians(case['wind_direction_toward_deg_ENU'])
    legs=[]
    for a,b in zip([[0,0], *points], [*points,[0,0]],strict=True):
        de,dn=b[0]-a[0],b[1]-a[1]
        length=math.hypot(de,dn)
        we=case['wind_speed_mps']*math.cos(heading)
        wn=case['wind_speed_mps']*math.sin(heading)
        legs.append(dict(start_enu_m=a, end_enu_m=b,length_m=length,wind_along_mps=(we*de+wn*dn)/length,
                         wind_cross_left_mps=(wn*de-we*dn)/length))
    planned_out=sum(leg['length_m'] for leg in legs[:-1])
    evidence.confirm(config['mission_route_id']==case['mission_route'] and config['route_definition']=='FIXED_DESTINATION_FROM_TAKEOFF'
                     and config['planned_waypoints_enu_m']==points and config['planned_destination_east_m']==150
                     and config['planned_destination_north_m']==0 and math.isclose(config['planned_outbound_path_m'],planned_out)
                     and abs(config['total_mass_kg']-(3.68+case['payload_kg']))<1e-9,
                     'Recorded destination, geometry or platform mass differs')
    return dict(base_uav_mass_kg=3.68,total_mass_kg=config['total_mass_kg'], planned_outbound_m=planned_out,
                planned_return_m=150, destination_east_m=150,destination_north_m=0, intermediate_stops=len(points)-1,
                planned_xy_accel_limit_mps2=1.5, planned_return_speed_cap_mps=4., leg_conditions_json=json.dumps(legs,separators=(',',':')))


def inspect(raw, case, attempt, protocol, out):
    cid=case['case_id']
    receipt_file=raw/(cid+'.json')
    receipt=read(receipt_file) if receipt_file.exists() else {}
    if receipt.get('outcome')!='COMPLETE':
        return dict(accepted=False,reason=receipt.get('reason') or 'NO_COMPLETED_MISSION',outcome=receipt.get('outcome'))
    run=receipt['run_id']
    doc=read(raw/f'{run}.provenance.json')
    hashes=check_temperature(raw,doc,case,protocol['source_sha256'])
    for key,source in [('stable_cruise','m100_stable_cruise_speed_config'),('transient','m100_transient_power_config'),
                       ('wind','m100_wind_simulation_config'),('drag_plugin','gazebo_drag_plugin')]:
        evidence.confirm(doc['source_files'][source]['sha256']==hashes[key],f'Runtime source identity differs for {source}')
    for key,source in (('stable_cruise_implementation','m100_stable_cruise_speed'),
                       ('stable_cruise_decision','m100_stable_cruise_decision')):
        if key in hashes:
            recorded = doc['source_files'].get(source, {})
            evidence.confirm(recorded.get('sha256') == hashes[key], f'Cruise calibration identity differs for {source}')
    info=doc['airborne_training']
    wire=raw/f'{run}.pack_drive.jsonl'
    row_file=raw/info['relative_path']
    ulog,=raw.glob('state/rootfs/log/*/*.ulg')
    evidence.confirm(receipt['armed'] is False and receipt['landed'] is True and all(case[k]==receipt[k] for k in case)
                     and read(raw/'request.json')==[case], 'Receipt no longer matches registered flight')
    phase_settings=plan_fields(case,doc['config'],protocol)
    with tempfile.TemporaryDirectory() as stage:
        directory=Path(stage)
        for name,path in {'receipt.json':receipt_file,'provenance.json':raw/f'{run}.provenance.json',
                          'flight.csv':raw/f'{run}.csv','training.csv':row_file}.items():
            (directory/name).symlink_to(path.resolve())
        drops=evidence.ulog_extract(ulog,directory/'ulog_states.csv')
        ulog_hash=evidence.digest(ulog)
        save(directory/'manifest.json',dict(case_id=cid,run_id=run,attempt=attempt,
             files_sha256={},selected_for_training=info['training_eligible'],source_ulog_sha256=ulog_hash,
             ulog_dropout_events=drops,launcher_sha256=(raw/'launcher_sha256').read_text().strip()))
        audit=evidence.audit_flight(directory,case,hashes,raw_trace_path=wire)
    checks=dict(recorder_status=info['status'], recorder_energy_Wh=info[TARGET], training_row_sha256=info['sha256'],
                native_wire_sha256=audit['original_trace_sha256'],ulog_sha256=ulog_hash,ulog_dropout_events=drops,
                independent_ulog_rejection=audit.get('independent_ulog_rejection',[]),
                native_uncovered_s=audit['native_uncovered_s'],native_budget_reason=audit['budget_reason'])
    if (audit[TARGET] is None or not audit['native_source_time_partition_verified'] or audit['ulog_airborne_failsafe_s']!=0
            or audit['native_uncovered_s']!=0):
        why=audit['budget_reason'] or '|'.join(checks['independent_ulog_rejection']) or 'AIRBORNE_GATE_OR_ENERGY_INCOMPLETE'
        return dict(accepted=False,reason=why,run_id=run,checks=checks)
    protocol_waypoints=protocol['route_waypoints_enu_m'][case['mission_route']]
    controller_waypoints=doc['waypoints_reached']
    evidence.confirm([[w['target_east_m'],w['target_north_m']] for w in controller_waypoints]==protocol_waypoints
                     and all(math.hypot(w['observed_east_m']-w['target_east_m'],w['observed_north_m']-w['target_north_m'])<=.151
                             and w['horizontal_speed_mps']<=.151 for w in controller_waypoints),
                     'Waypoint hold was not confirmed by flight controller')
    wind=doc['wind_environment_application']
    angle=math.radians(case['wind_direction_toward_deg_ENU'])
    expected=[case['wind_speed_mps']*math.cos(angle),case['wind_speed_mps']*math.sin(angle),0]
    evidence.confirm(wind['verified'] and wind['method']=='APPLIED_GAZEBO_FORCE_PARAMETERS_SERVICE_RESPONSE'
                     and all(abs(a-b)<1e-8 for a,b in zip(wind['wind_vector_enu_mps'],expected,strict=True)),
                     'World wind vector differs from requested wind')
    topics=flight_topics(ulog)
    path=trajectory(topics,audit['native_start_s'],audit['native_end_s'],protocol_waypoints,case['altitude_m'],
                    protocol['independent_waypoint_tolerance_horizontal_m'],protocol['independent_waypoint_tolerance_vertical_m'])
    with wire.open() as src:
        records=[json.loads(line) for line in src]
    fit=read(OUT/'battery_fit.json')
    pack=pack_check(records,case,doc['config'],fit)
    row,=csv.DictReader(io.StringIO(row_file.read_text()))
    evidence.confirm(row['battery_dynamics_model']==MODEL and row['battery_parameter_sha256']==fit['parameter_sha256']
                     and row['route_field_validation']=='SIMULATION_ONLY' and row['mission_route_id']==case['mission_route']
                     and float(row['initial_nominal_soc_percent'])==case['initial_soc_percent']
                     and row['battery_soc_physical_calibration']=='False','Direct recorder fields differ')
    conditions=dict(**case,nameplate_Wh=protocol['nameplate_Wh'])
    series=online_series(cid,run,records,topics,info['native_budget'],conditions)
    ground=topics['vehicle_local_position_groundtruth']
    stamps=ground['timestamp_sample'].astype(np.float64)*1e-6
    outbound=(stamps>=audit['native_start_s']) & (stamps<=path['waypoints'][-1]['time_s'])
    inbound=(stamps>path['waypoints'][-1]['time_s']) & (stamps<=audit['native_end_s'])
    speed=np.hypot(ground['vx'],ground['vy'])
    cruise=[r for r in series if r['flight_phase']=='CRUISE' and r['horizontal_speed_mps'] is not None]
    cruise_time=sum(r['interval_s'] for r in cruise)
    file=out/cid/f'attempt_{attempt}.csv'
    table(file,series)
    energy,duration=audit[TARGET],audit['duration_s']
    nominal=protocol['nameplate_Wh']
    usable=nominal*(case['initial_soc_percent']-protocol['nominal_reserve_percent'])/100
    mean=energy*3600/duration
    remaining=nominal*case['initial_soc_percent']/100-energy
    metric=dict(**case,**phase_settings,run_id=run,attempt=attempt,estimated_airborne_energy_Wh=energy,
                outbound_peak_groundspeed_mps=float(np.max(speed[outbound])),
                return_peak_groundspeed_mps=float(np.max(speed[inbound])),
                observed_cruise_phase_interval_s=cruise_time,
                observed_cruise_mean_groundspeed_mps=(sum(r['horizontal_speed_mps']*r['interval_s'] for r in cruise)
                                                     /cruise_time if cruise_time else None),
                airborne_mean_power_W=mean,airborne_duration_s=duration,airborne_horizontal_distance_m=path['groundtruth_airborne_distance_m'],
                nominal_remaining_energy_Wh=remaining,nominal_remaining_soc_percent=100*remaining/nominal,
                nominal_reserve_percent=protocol['nominal_reserve_percent'], nominal_reserve_margin_Wh=usable-energy,
                nominal_endurance_to_reserve_min=60*usable/mean,
                nominal_horizontal_range_to_reserve_km=path['groundtruth_airborne_distance_m']*.001*usable/energy,
                initial_nominal_soc_percent=case['initial_soc_percent'], altitude_reference='AGL', nameplate_Wh=nominal,
                flight_energy_source=row['label_source'],original_training_eligible=info['training_eligible'],
                route_field_validation='SIMULATION_ONLY',battery_calibration='LIMITED_EMPIRICAL_PROXY',
                wind_electrical_calibration=doc['wind_simulation_model']['classification'],
                wind_power_runtime_applied=doc['energy_summary']['wind_power_runtime_applied'],
                physical_battery_temperature=False,battery_dynamics_model=MODEL, battery_parameter_sha256=fit['parameter_sha256'],
                measured_usable_TB47D_capacity=False,energy_uncovered_s=audit['native_uncovered_s'],
                landing_offset_m=path['landing_error_m'], training_row_sha256=info['sha256'],**pack)
    for source,values in audit['source_bins'].items():
        metric[source+'_energy_Wh']=values['energy_Wh']
        metric[source+'_coverage_s']=values['seconds']
    return dict(accepted=True,reason='',outcome='COMPLETE',run_id=run,metrics=metric,
                checks=dict(**checks,px4_takeoff_us=audit['ulog_takeoff_us'],px4_landing_us=audit['ulog_landing_us'],
                            px4_failsafe_s=audit['ulog_airborne_failsafe_s'],waypoints=path['waypoints'],
                            intervals=len(series), intervals_sha256=evidence.digest(file), interval_energy_sum_Wh=sum(
                                s['future_interval_energy_Wh'] for s in series)))


def audit_closed():
    base=read(OUT/'protocol.json')
    extractor_hash=evidence.digest(Path(__file__).with_name('mission_observations.py'))
    audit_hash=evidence.digest(Path(__file__))
    cases={c['case_id']:(c,OUT/'protocol.json') for c in base['cases']}
    confirmation=OUT/'confirmation_protocol.json'
    if confirmation.is_file():
        cases.update({c['case_id']:(c,confirmation) for c in read(confirmation)['cases']})
    for acq in sorted((OUT/'acquisition').glob('*/attempt_*.json')):
        receipt=read(acq)
        case,file=cases[receipt['case_id']]
        num=receipt['attempt']
        target=OUT/'audits'/case['case_id']/f'attempt_{num}.json'
        previous=read(target) if target.exists() else None
        if previous:
            evidence.confirm(previous['archive_sha256']==receipt['archive_sha256'],
                             'A previous audit points to a different acquisition')
            if not previous['accepted']:
                continue
            cache=INTERVAL_CACHE/case['case_id']/f'attempt_{num}.csv'
            checks=previous['checks']
            if (checks.get('series_extractor_sha256')==extractor_hash and checks.get('audit_code_sha256')==audit_hash and cache.is_file()
                    and evidence.digest(cache)==checks['intervals_sha256']):
                continue
        evidence.confirm(receipt['protocol_sha256']==evidence.digest(file),'Acquisition protocol identity differs')
        archive=ARCHIVE/receipt['archive']
        evidence.confirm(evidence.digest(archive)==receipt['archive_sha256'],'Archive digest differs')
        with tempfile.TemporaryDirectory(prefix='m100-audit-',dir=ROOT/'tmp') as scratch:
            raw=Path(scratch)
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    entry=Path(name)
                    if entry.is_absolute() or '..' in entry.parts:
                        raise ValueError('Invalid archive member')
                    # Original bootstrap binaries and caches are archived but do not enter the audit.
                    if name.startswith(('state/rootfs/log/','state/temperature_models/','training/')) or len(entry.parts)==1:
                        z.extract(name,raw)
            result=inspect(raw,case,num,base,INTERVAL_CACHE)
        if previous:
            evidence.confirm(result['accepted'] and result['metrics'][TARGET]==previous['metrics'][TARGET],
                             'Feature re-extraction changed the original direct flight label')
        if result['accepted']:
            result['checks']['series_extractor_sha256']=extractor_hash
            result['checks']['audit_code_sha256']=audit_hash
        save(target,dict(**result,case_id=case['case_id'],attempt=num,
                         protocol_sha256=receipt['protocol_sha256'],archive_sha256=receipt['archive_sha256']))
        print(case['case_id'],result['accepted'],result['reason'],flush=True)


if __name__=='__main__':
    audit_closed()
