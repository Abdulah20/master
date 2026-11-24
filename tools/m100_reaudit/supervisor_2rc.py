"""Rerun the frozen supervisor flight matrix with the installed 2RC model."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import yaml

from . import mission_campaign as mission
from .direct_airborne_evidence import TARGET, confirm, digest
from .mission_audit import MODEL, pack_check
from .supervisor_results import mission_summary


ROOT = mission.ROOT
OUT = ROOT / 'results/m100_reaudit/supervisor_2rc'
WORK = ROOT / 'tmp/m100_supervisor_2rc'
FLIGHT_WORK = Path(os.environ.get('M100_SUPERVISOR_2RC_FLIGHT_WORK', WORK / 'flights'))
OLD = OUT.with_name('supervisor_rerun')
ORIGINAL = ROOT / 'tools/m100_reaudit/runtime_snapshot/original_study/M100_69_experiments_summary.csv'
REFERENCE = OLD / 'M100_69_experiments_rerun.csv'
ARCHIVE = Path(os.environ.get('M100_SUPERVISOR_2RC_ARCHIVE',
               '/media/abdulah/New Volume/M100_Supervisor_2RC_2026-11-24'))
PROTOCOL = OUT / 'protocol.json'
CASE_FIELDS = ('payload_kg', 'cruise_speed_mps', 'wind_speed_mps', 'wind_direction_toward_deg_ENU',
               'altitude_m', 'temperature_c')


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def bootstrap(registered):
    target = WORK / 'bootstrap_rootfs'
    expected = registered['initial_parameters']
    canonical = mission.PX4 / 'build/px4_sitl_default/rootfs'
    if not target.exists():
        old_parameter_paths = (ROOT / 'tmp').glob('**/parameters*.bson')
        source = next((p for p in old_parameter_paths if p.is_file() and digest(p) == expected), None)
        confirm(source is not None, 'The originally registered PX4 parameter bytes are unavailable')
        pending = Path(tempfile.mkdtemp(dir=WORK, prefix='bootstrap-'))
        for name in ('parameters.bson', 'parameters_backup.bson'):
            shutil.copyfile(source, pending / name)
        for name in ('gz_env.sh', 'dataman'):
            shutil.copy2(canonical / name, pending / name)
        shutil.copytree(canonical / 'eeprom', pending / 'eeprom')
        pending.rename(target)
    confirm(digest(target / 'gz_env.sh') == registered['gz_environment']
            and digest(target / 'dataman') == digest(canonical / 'dataman')
            and all(digest(target / name) == expected for name in ('parameters.bson', 'parameters_backup.bson')),
            'Saved private PX4 startup files do not match the registered settings')
    return target


def register():
    previous = mission.read(OLD / 'protocol.json')
    models = mission.read(OLD / 'model_identities.json')
    confirm(models['protocol_sha256'] == digest(OLD / 'protocol.json'), 'Payload model register changed')
    latest = {r['case_id']: r for r in rows(REFERENCE)}
    original = {r['run_id']: r for r in rows(ORIGINAL)}
    confirm(len(latest) == len(original) == len(previous['cases']) == 69, 'Expected the 69 original unique flights')
    for case in previous['cases']:
        last = latest[case['case_id']]
        first = original[last['historical_run_id']]
        confirm(all(float(r[k]) == case[k] for k in CASE_FIELDS for r in (first, last)), 'Historical condition mismatch')
        confirm(last['mission_outcome'] == 'COMPLETE' and float(last[TARGET])>0 and float(last['uncovered_s'])==0,
                'The previous full-flight baseline is incomplete')
    confirm(len({r['historical_run_id'] for r in latest.values()})==69, 'Original runs were reused across physical cases')
    old_hashes = previous['source_sha256']
    initial = bootstrap(old_hashes)
    hashes = mission.sources()
    for model, identity in models['models'].items():
        key = 'payload'+f"{round(100*identity['payload_kg']):03d}" if identity['payload_kg'] else 'base_sdf'
        hashes[key] = digest(mission.PX4 / 'Tools/simulation/gz/models' / model / 'model.sdf')
        confirm(hashes[key] == identity['model_sdf_sha256'], f'Historical aircraft model differs for {model}')
    for key in ('px4_binary', 'energy_config', 'fit'):
        confirm(hashes[key] == old_hashes[key], f'Historical runtime input differs for {key}')
    hashes.update(initial_parameters=digest(initial / 'parameters.bson'), gz_environment=digest(initial / 'gz_env.sh'),
                  initial_dataman=digest(initial / 'dataman'))
    config_dir = mission.RUNTIME / 'ros2_ws/src/drone_nav/config'
    confirm(hashes['stable_cruise'] == digest(config_dir / 'm100_stable_cruise_speed_v2.yaml'),
            'Expected the installed native V2 cruise calibration')
    cases = [dict(c, mission_route='measurement', initial_soc_percent=100) for c in previous['cases']]
    protocol = dict(schema='M100_SUPERVISOR_2RC_REPEAT_V1', previous_protocol_sha256=digest(OLD / 'protocol.json'),
                    previous_csv_sha256=digest(REFERENCE), original_csv_sha256=digest(ORIGINAL),
                    model_register_sha256=digest(OLD / 'model_identities.json'), cases=cases, groups=previous['groups'],
                    assumptions=previous['assumptions'], source_sha256=hashes, battery_model=MODEL,
                    battery_parameter_sha256=mission.read(mission.OUT / 'battery_fit.json')['parameter_sha256'],
                    stable_cruise_profile='native', raw_archive_root=str(ARCHIVE), max_attempts=mission.MAX_ATTEMPTS)
    if PROTOCOL.exists():
        confirm(mission.read(PROTOCOL) == protocol, 'Registered inputs or installed runtime changed')
    else:
        mission.save(PROTOCOL, protocol)
    return protocol, initial


def measurement_trace(path, case, expected_wh, maximum_gap_s):
    """Independently integrate the exact samples inside the recorder's measurement gate."""
    gate = False
    opens = 0
    points = []
    phases = Counter()
    with gzip.open(path, 'rt', newline='') as file:
        for row in csv.DictReader(file):
            active = row['measuring']=='1'
            if not active:
                gate = False
                continue
            if not gate:
                opens += 1
                gate = True
            power = float(row['power_w']) if row['power_w'] else None
            points.append((float(row['sim_time_s']), power))
            phases[row['phase']] += 1
    if len(points)<2:
        return dict(complete=False, reason='MEASUREMENT_TRACE_EMPTY_OR_TOO_SHORT', samples=len(points))
    pieces = []
    gaps = []
    for (left, p0), (right, p1) in zip(points, points[1:]):
        dt = right-left
        if 0<dt<=maximum_gap_s and p0 is not None and p1 is not None and all(math.isfinite(p) and p>=0 for p in (p0,p1)):
            pieces.append(dt*(p0+p1)/7200)
        else:
            gaps.append(dt)
    energy = math.fsum(pieces)
    confirm(expected_wh is not None and math.isclose(energy, expected_wh, abs_tol=1e-8, rel_tol=0),
            f'The measurement trace does not reproduce the frozen recorder energy in {case["case_id"]}')
    expected_phase = 'CRUISE' if case['cruise_speed_mps'] else 'HOVER'
    duration = points[-1][0]-points[0][0]
    correct_phase = phases[expected_phase]>=.99*len(points)
    correct_duration = duration>0 and (expected_phase=='CRUISE' or abs(duration-case['hover_duration_s'])<=maximum_gap_s)
    complete = opens==1 and not gate and not gaps and correct_phase and correct_duration
    return dict(complete=complete, reason='' if complete else 'MEASUREMENT_GATE_OR_POWER_INCOMPLETE',
                start_s=points[0][0], end_s=points[-1][0], duration_s=duration, phase_samples=dict(phases),
                samples=len(points), reconstructed_energy_Wh=energy, mismatch_Wh=energy-expected_wh,
                integrated_mean_power_W=3600*energy/duration if duration>0 else None,
                invalid_intervals=len(gaps), measurement_gate_opens=opens, gap_limit_s=maximum_gap_s,
                trace_sha256=digest(path))


def gap_limit():
    reference = ROOT / 'tools/m100_reaudit/runtime_snapshot/ros2_ws/src/drone_nav/config/m100_energy_model_v2.yaml'
    return yaml.safe_load(reference.read_bytes())['integration']['maximum_valid_gap_s']


def audit(case, attempt, protocol):
    cid = case['case_id']
    receipt_path = OUT / 'acquisition' / cid / f'attempt_{attempt}.json'
    destination = OUT / 'audits' / cid / receipt_path.name
    if destination.is_file():
        return mission.read(destination)
    receipt = mission.read(receipt_path)
    confirm(receipt['case_id']==cid and receipt['attempt']==attempt and receipt['protocol_sha256']==digest(PROTOCOL),
            'Acquisition receipt does not match the registered flight')
    archive = ARCHIVE / receipt['archive']
    confirm(digest(archive)==receipt['archive_sha256'], 'Raw archive differs from its receipt')
    hashes = protocol['source_sha256']
    FLIGHT_WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=FLIGHT_WORK, prefix='audit-') as path, zipfile.ZipFile(archive) as source:
        raw = Path(path)
        for name in source.namelist():
            member = Path(name)
            confirm(not member.is_absolute() and '..' not in member.parts, 'Unsafe archive member')
            if len(member.parts)==1 or name.startswith(('state/rootfs/log/', 'training/')):
                source.extract(name, raw)
        confirm(mission.read(raw / 'request.json')==[case], 'Archived flight plan changed')
        outcome = mission_summary(raw, case, attempt, hashes)
        metric, evidence = outcome['metrics'], outcome['audit']
        run = metric['run_id']
        if metric['mission_outcome']=='COMPLETE':
            provenance = mission.read(raw / f'{run}.provenance.json')
            config = provenance['config']
            pack = provenance['native_pack_drive']
            with (raw / provenance['airborne_training']['relative_path']).open(newline='', encoding='utf-8') as f:
                training, = csv.DictReader(f)
            confirm(not config.get('mission_route_id') and not provenance.get('waypoints_reached')
                    and config['route_definition']=='ONE_WAY_PLUS_X_ENU_MEASUREMENT_DISTANCE'
                    and config['total_mass_kg']==3.68+case['payload_kg'],
                    'Acquisition did not fly the original after-acceleration measurement protocol')
            # Empty targets may lack the battery identity if the recorder closed
            # before matching a terminal native message. The raw trace is still checked.
            if metric[TARGET] is not None:
                confirm(training['battery_dynamics_model']==MODEL
                        and training['battery_parameter_sha256']==protocol['battery_parameter_sha256'],
                        'Direct recorder did not identify the registered 2RC model')
            used = provenance['source_files']
            for key, source_name in (('stable_cruise','m100_stable_cruise_speed_config'),
                    ('stable_cruise_implementation','m100_stable_cruise_speed'),
                    ('stable_cruise_decision','m100_stable_cruise_decision'),('transient','m100_transient_power_config'),
                    ('wind','m100_wind_simulation_config'),('drag_plugin','gazebo_drag_plugin')):
                confirm(used[source_name]['sha256']==hashes[key], f'Installed source changed: {source_name}')
            confirm(pack['stable_cruise_speed_enabled'] and provenance['stable_cruise_speed_model'] is None,
                    'A speed correction unexpectedly applied to the supervisor matrix')
            with (raw / f'{run}.pack_drive.jsonl').open() as f:
                records = [json.loads(line) for line in f]
            requests = [r['acknowledged_request'] for r in records if r.get('acknowledged_request')]
            confirm(any(r['measurement_active'] for r in requests)
                    and all(r['stable_cruise_speed_adjustment_w'] is None
                            and r['manager_stable_cruise_speed_adjustment_w'] is None for r in requests),
                    'Runtime measurement correction differs from the expected speed domain')
            pack_stats = pack_check(records, case, config, mission.read(mission.OUT / 'battery_fit.json'))
            confirm(pack_stats['native_vi_publications'] > 0, 'No supported native 2RC voltage/current samples')
            metric.update(pack_stats, battery_dynamics_model=MODEL, cruise_correction_applied=False)
            outcome['measurement_trace'] = measurement_trace(raw / f'{run}.trace.csv.gz', case,
                                                             metric['measurement_energy_Wh'], gap_limit())
        airborne_ok = (metric[TARGET] is not None and metric['uncovered_s']==0
                       and evidence.get('native_source_time_partition_verified') and evidence.get('ulog_airborne_failsafe_s')==0)
        segment = outcome.get('measurement_trace', {})
        planned_distance = not case['cruise_speed_mps'] or (metric.get('measurement_distance_m') or 0)>=case['measurement_distance_m']
        segment_ok = segment.get('complete', False) and metric.get('measurement_frozen', False) and planned_distance
        verified = airborne_ok and segment_ok
        reason = '' if verified else (metric.get('rejection_reason') or 'AIRBORNE_FAILSAFE_OR_INCOMPLETE_COVERAGE'
                                      if not airborne_ok else segment.get('reason') or 'MEASUREMENT_LENGTH_NOT_REACHED')
    result = dict(**outcome, accepted=bool(verified), verified_airborne=bool(airborne_ok), verified_segment=bool(segment_ok),
                  reason=reason, archive_sha256=receipt['archive_sha256'])
    mission.save(destination, result)
    print(cid, f'attempt {attempt}', 'VERIFIED' if verified else reason, metric[TARGET], flush=True)
    return result


def acquire(case, attempt, protocol, initial):
    mission.run_one(case, PROTOCOL, attempt, release_dir=OUT, work_root=FLIGHT_WORK, archive_root=ARCHIVE,
                    bootstrap_rootfs=initial, extra_env={'M100_STABLE_CRUISE_PROFILE': 'native',
                                                       'M100_PACK_STABLE_CRUISE_CORRECTION': '1'})
    return audit(case, attempt, protocol)


def run(protocol, initial, workers, case_ids):
    cases = {c['case_id']: c for c in protocol['cases']}
    if case_ids:
        confirm(set(case_ids)<=cases.keys() and len(set(case_ids))==len(case_ids), 'Unknown or duplicate case ID')
        cases = {cid: cases[cid] for cid in case_ids}
    pending = cases
    for attempt in range(1, protocol['max_attempts']+1):
        retry = {}
        with ThreadPoolExecutor(max_workers=max(1, workers-attempt+1)) as pool:
            futures = {pool.submit(acquire, case, attempt, protocol, initial): cid for cid, case in pending.items()}
            for future in as_completed(futures):
                result = future.result()
                cid = futures[future]
                if not result['accepted']:
                    retry[cid] = pending[cid]
        pending = retry
        if not pending:
            break
    outcomes = [mission.read(p) for cid in cases for p in (OUT / 'audits' / cid).glob('attempt_*.json')]
    print('Verified', len(cases)-len(pending), 'of', len(cases), 'cases;', len(outcomes), 'attempts;',
          dict(Counter(a['reason'] for a in outcomes if not a['accepted'])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, choices=range(1, 5), default=3)
    p.add_argument('--case', action='append', dest='case_ids')
    p.add_argument('--register-only', action='store_true')
    args = p.parse_args()
    confirm(ARCHIVE.parent.is_dir(), f'Archive volume is unavailable at {ARCHIVE.parent}')
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, initial = register()
        if not args.register_only:
            run(protocol, initial, args.workers, args.case_ids)


if __name__=='__main__':
    main()
