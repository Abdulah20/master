"""Capture unchanged native pack messages during a fresh flight and verify both 2-RC rest responses."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time

from .direct_airborne_evidence import confirm, digest, gate_from_ulog, ulog_extract
from .mission_campaign import OUT, ROOT, make_case, read, save, sources


HERE = Path(__file__).resolve().parent
TOPIC = '/m100/pack_drive/state'
TRACE = 'native_pack_state_original.jsonl'
MODEL = 'CMU_LOADED_THEVENIN_2RC_V1'
REST = 'polarization_confirmed_rest_s'
RAW_TIMEOUT_S = 0.30
LAND_TIMEOUT_S = 1.50
WINDOW_S = 8.5
BRANCHES = (('polarization_fast_v', 'tau1_s'), ('polarization_slow_v', 'tau2_s'))


def patch_once(content, before, after):
    confirm(content.count(before) == 1, f'Expected one launcher anchor: {before}')
    return content.replace(before, after)


def prepare(raw, step, seed):
    confirm(raw.is_dir() and not any(raw.iterdir()), 'Choose a new empty directory for this flight')
    confirm(step in (0.004, 0.002) and 0 <= seed <= 4294967295, 'Invalid physics step or seed')
    case = make_case('rest_check', 0, 6, 25, 'direct', seed)
    case['case_id'] += f'_dt{round(step * 1000)}ms'
    case['physics_step_s'] = step
    save(raw / 'request.json', [case])
    fit = read(OUT / 'battery_fit.json')
    manifest = dict(schema='M100_NATIVE_REST_VERIFICATION_V1', created_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=sources(), verifier_sha256=digest(Path(__file__)),
                    entrypoint_sha256=digest(HERE / 'mission_rest_acquire.sh'),
                    portable_pack_source_sha256=digest(HERE / 'pack_drive/M100PackDrive.cc'),
                    portable_pack_header_sha256=digest(HERE / 'pack_drive/M100TheveninFit.hh'),
                    parameter_sha256=fit['parameter_sha256'], parameters=fit['parameters'],
                    request_sha256=digest(raw / 'request.json'), requested_topic=TOPIC,
                    temporary_launcher_sha256={})
    folder = raw / 'launchers'
    folder.mkdir()
    for name in ('isolated_native.sh', 'run_native.sh'):
        original = (HERE / name).read_text()
        modified = patch_once(original, 'root=$(cd "$(dirname "$0")/../.." && pwd)', 'root=${M100_REST_ROOT:?}')
        if name == 'isolated_native.sh':
            modified = patch_once(modified, 'exec "$1/tools/m100_reaudit/run_native.sh" "$2" "$3"',
                                  'exec "${M100_REST_INNER_LAUNCHER:?}" "$2" "$3"')
        else:
            modified = patch_once(modified, 'cleanup() {', '''rest_observer_pid=""
cleanup() {
  if [[ -n "$rest_observer_pid" ]]; then kill -TERM "$rest_observer_pid" 2>/dev/null || true; fi''')
            collect = '/usr/bin/python3 -m tools.m100_reaudit.native collect "$run_dir" --plan "$plan"'
            modified = patch_once(modified, collect, '''/usr/bin/python3 -m tools.m100_reaudit.mission_rest_check monitor "$run_dir" >"$run_dir/rest_observer.log" 2>&1 &
rest_observer_pid=$!
''' + collect + '''
wait "$rest_observer_pid"
rest_observer_pid=""''')
        path = folder / name
        path.write_text(modified)
        path.chmod(0o755)
        manifest['temporary_launcher_sha256'][name] = digest(path)
    save(raw / 'rest_protocol.json', manifest)


def require_install(raw):
    protocol = read(raw / 'rest_protocol.json')
    hashes = protocol['source_sha256']
    confirm(hashes == sources(), 'Runtime changed after this run was prepared')
    confirm(hashes['pack_source'] == protocol['portable_pack_source_sha256'] and
            hashes['pack_header'] == protocol['portable_pack_header_sha256'],
            'Install the reviewed pack source and fitted header before starting a flight')
    from .mission_campaign import RUNTIME
    plugin = RUNTIME / 'ros2_ws/install/m100_gazebo_gui/lib/m100_gazebo_gui/libM100PackDrive.so'
    confirm(REST.encode() in plugin.read_bytes(), 'Installed plugin lacks the confirmed-rest telemetry field')


def process_witnesses():
    """Inspect only processes in this fresh acquisition's private PID namespace."""
    witnesses = {}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            exe = (proc / 'exe').resolve(strict=True)
            if exe.name == 'px4' and 'px4_binary' not in witnesses:
                witnesses['px4_binary'] = (exe, proc)
            if 'plugin' not in witnesses:
                for line in (proc / 'maps').read_text().splitlines():
                    parts = line.split(maxsplit=5)
                    if len(parts) == 6 and 'x' in parts[1] and parts[5].endswith('/libM100PackDrive.so'):
                        witnesses['plugin'] = (Path(parts[5]).resolve(strict=True), proc)
                        break
        except (OSError, RuntimeError):
            continue
    return {name: dict(path=str(path), sha256=digest(path), pid_in_namespace=int(proc.name),
                       source='executable mapping' if name == 'plugin' else 'process executable')
            for name, (path, proc) in witnesses.items()}


def is_rest(row):
    t = row['time_s']
    land = row.get('px4_land_time_s')
    return (row.get('model') == 'm100_cmu_v2_0' and row.get('battery_dynamics_model') == MODEL
            and row.get('status') == 'DISARMED_ZERO_DEMAND_CAP_HELD'
            and row.get('armed') is False and row.get('px4_landed') is True
            and land is not None and -1e-8 <= t-land <= LAND_TIMEOUT_S+1e-8
            and row.get('demand_fresh') is True and row.get('demand_power_w') == 0
            and row.get('effective_demand_power_w') == 0 and row.get('raw_command_fresh') is True
            and row.get('awaiting_post_reset_zero_command') is False
            and row.get('raw_motor_rad_s') == [0]*4 and row.get('applied_motor_rad_s') == [0]*4
            and row.get('current_a') in (None, 0)
            and row.get('q_domain_exhausted') is False and row.get('airborne_budget', {}).get('status') == 'COMPLETE')


def monitor(raw):
    import rclpy
    from std_msgs.msg import String

    confirm(os.environ.get('M100_AUDIT_ISOLATED') == '1' and
            Path('/proc/1/ns/pid').readlink() == Path('/proc/self/ns/pid').readlink(),
            'Monitor must use the native acquisition PID namespace')
    rclpy.init()
    node = rclpy.create_node('m100_rest_evidence_recorder')
    case = read(raw / 'request.json')[0]
    receipt = raw / (case['case_id'] + '.json')
    start = time.monotonic()
    finished = None
    saw_run = None
    rows = 0
    witnesses = {}
    last_scan = 0.
    segment_start = None
    previous = None
    segment_duration = 0.
    try:
        with (raw / TRACE).open('xb') as stream:
            def capture(msg):
                nonlocal previous, segment_start, segment_duration, saw_run, rows
                data = msg.data
                stream.write(data.encode('utf-8') + b'\n')
                row = json.loads(data)
                rows += 1
                run = row['airborne_budget']['run_id']
                t = row['time_s']
                if run and is_rest(row):
                    if run != saw_run or previous is None or not 0 < t-previous <= RAW_TIMEOUT_S:
                        segment_start = t
                    segment_duration = t - segment_start
                    previous = t
                else:
                    previous = None
                    segment_duration = 0.
                if run:
                    saw_run = run

            subscription = node.create_subscription(String, TOPIC, capture, 256)
            while time.monotonic() - start < 1100:
                rclpy.spin_once(node, timeout_sec=.1)
                now = time.monotonic()
                if saw_run and len(witnesses) < 2 and now-last_scan >= 2:
                    witnesses.update(process_witnesses())
                    last_scan = now
                if receipt.exists():
                    if finished is None:
                        result = read(receipt)
                        confirm(result.get('outcome') == 'COMPLETE', f'Native flight failed: {result.get("reason")}')
                        finished = now
                    if saw_run == result.get('run_id') and segment_duration >= 9.5:
                        break
                    confirm(now-finished < 80, 'Missing continuous post-disarm stream within 80 s of receipt')
            else:
                raise ValueError('Timed out before the guarded flight and post-disarm recording completed')
            node.destroy_subscription(subscription)
    finally:
        save(raw / 'capture_identity.json', dict(topic=TOPIC, wire_format='original gz.msgs.StringMsg data via ROS String',
             captured_rows=rows, process_witnesses=witnesses, run_id=saw_run))
        node.destroy_node()
        rclpy.shutdown()


def rest_window(rows, run, landing, disarm):
    selected = []
    for row in rows:
        t = row['time_s']
        if not (row.get('airborne_budget', {}).get('run_id') == run and t >= disarm and
                is_rest(row) and abs(row['px4_land_transition_s']-landing) <= 0.5e-6):
            selected = []
            continue
        if selected and not 0 < t-selected[-1]['time_s'] <= RAW_TIMEOUT_S:
            selected = []
        selected.append(row)
        if t-selected[0]['time_s'] >= WINDOW_S:
            return selected
    raise ValueError('No uninterrupted 8.5 s native rest window after this ULog disarm')


def decay_metrics(rows, protocol):
    first, last = rows[0], rows[-1]
    rest_delta = last[REST] - first[REST]
    elapsed = last['time_s'] - first['time_s']
    confirm(8 <= rest_delta <= elapsed+1e-7 and 8 <= elapsed <= 10, 'Confirmed rest duration out of range')
    confirm(all(-1e-8 <= b[REST]-a[REST] <= b['time_s']-a['time_s']+1e-8 for a,b in zip(rows,rows[1:])),
            'Confirmed rest counter moved backwards or advanced faster than simulated time')
    confirm(elapsed-rest_delta <= .02, 'Too much simulated time without confirmed rest in selected window')
    confirm(all(row['battery_parameter_sha256'] == protocol['parameter_sha256'] and
                row['initial_nominal_soc_percent'] == 100 and row['initial_nameplate_discharge_offset_ah'] == 0
                and row['px4_land_sequence'] == first['px4_land_sequence']
                and row['airborne_budget']['end_s'] == first['px4_land_transition_s'] for row in rows),
            'Battery identity, initial charge or PX4 landing edge changed in the rest window')
    charge = [row['discharged_ah'] for row in rows]
    confirm(all(math.isfinite(q) for q in charge) and min(charge)>0 and max(charge)-min(charge)<=1e-12,
            'Discharged charge changed during confirmed zero load or showed no flight discharge')
    energy = first['airborne_budget']['estimated_airborne_energy_Wh']
    confirm(math.isfinite(energy) and energy > 0 and
            all(row['airborne_budget']['estimated_airborne_energy_Wh'] == energy for row in rows),
            'Airborne energy changed after landing')
    branches = {}
    for key, tau_key in BRANCHES:
        tau = protocol['parameters'][tau_key]
        initial = first[key]
        confirm(math.isfinite(initial) and initial > 1e-6 and tau > 0, f'Missing measurable initial {key}')
        errors = [abs(row[key] - initial*math.exp(-(row[REST]-first[REST])/tau)) for row in rows]
        confirm(all(math.isfinite(e) and e <= 1e-9 for e in errors), f'Exponential mismatch in {key}')
        branches[key] = dict(tau_s=tau, start_v=initial, end_v=last[key],
                             expected_end_v=initial*math.exp(-rest_delta/tau), max_abs_residual_v=max(errors))
    return dict(start_sim_s=first['time_s'], end_sim_s=last['time_s'], samples=len(rows), elapsed_s=elapsed,
                confirmed_rest_delta_s=rest_delta, unconfirmed_s=max(0., elapsed-rest_delta),
                max_sample_gap_s=max(b['time_s']-a['time_s'] for a,b in zip(rows,rows[1:])),
                discharged_ah_start=charge[0], discharged_ah_end=charge[-1], max_charge_variation_ah=max(charge)-min(charge),
                estimated_airborne_energy_Wh=energy, px4_landing_sequence=first['px4_land_sequence'], branches=branches)


def flight_edges(raw):
    logs = list(raw.glob('state/rootfs/log/*/*.ulg'))
    confirm(len(logs) == 1, f'Expected one original PX4 ULog; found {len(logs)}')
    ulog = logs[0]
    before = digest(ulog)
    extract = raw / 'rest_ulog_states.csv'
    dropouts = ulog_extract(ulog, extract)
    confirm(before == digest(ulog) and dropouts == 0, 'ULog not stable or contains a recorded dropout')
    takeoff, landing, armed_duration, failsafe_duration = gate_from_ulog(extract)
    confirm(armed_duration == landing-takeoff and not failsafe_duration, 'PX4 arming or failsafe during flight')
    with extract.open(newline='') as f:
        states = [r for r in csv.DictReader(f) if r['topic'] == 'vehicle_status']
    edges = [(int(row['timestamp_us']), row['arming_state']) for i,row in enumerate(states)
             if i == 0 or row['arming_state'] != states[i-1]['arming_state']]
    confirm([level for _,level in edges] == ['1','2','1'] and edges[1][0] <= takeoff < landing <= edges[2][0],
            'ULog must show one complete arm–flight–land–disarm sequence')
    return dict(path=str(ulog.relative_to(raw)), sha256=before, dropout_events=dropouts, takeoff_us=takeoff,
                landing_us=landing, disarm_us=edges[2][0], last_status_us=int(states[-1]['timestamp_us']),
                extracted_states_sha256=digest(extract))


def analyze(raw):
    protocol = read(raw / 'rest_protocol.json')
    case = read(raw / 'request.json')[0]
    confirm(digest(raw / 'request.json') == protocol['request_sha256'], 'Request differs from preflight identity')
    for name, sha in protocol['temporary_launcher_sha256'].items():
        confirm(digest(raw/'launchers'/name) == sha, f'Prepared launcher differs: {name}')
    result_file = raw / (case['case_id'] + '.json')
    result = read(result_file)
    confirm(all(result[k] == v for k,v in case.items()) and result.get('outcome') == 'COMPLETE'
            and result.get('landed') is True and result.get('armed') is False
            and result['observed_seed'] == case['seed'] and abs(result['observed_physics_step_s']-case['physics_step_s']) < 1e-9,
            'Unsuccessful flight or requested conditions not observed')
    run = result['run_id']
    provenance = raw / (run + '.provenance.json')
    doc = read(provenance)
    config = doc['config']
    for config_key, case_key in (('payload_kg', 'payload_kg'), ('cruise_speed_mps','cruise_speed_mps'),
                                ('altitude_m','altitude_m'), ('route_length_m','measurement_distance_m'),
                                ('mission_route_id','mission_route'), ('temperature_c','temperature_c'),
                                ('wind_speed_mps','wind_speed_mps'), ('wind_direction_deg','wind_direction_toward_deg_ENU')):
        confirm(config[config_key] == case[case_key], f'Unexpected native setting: {config_key}')
    confirm(doc['run_id'] == run and config['planned_waypoints_enu_m'] == [[150,0]], 'Wrong mission or destination')
    capture = read(raw / 'capture_identity.json')
    confirm(capture['topic'] == TOPIC and capture['run_id'] == run, 'Observer run or topic differs')
    identities = protocol['source_sha256']
    for name in ('plugin', 'px4_binary'):
        confirm(capture['process_witnesses'][name]['sha256'] == identities[name], f'Running {name} differs')
    for key, source_name in (('gazebo_m100_pack_drive_plugin','plugin'), ('gazebo_drag_plugin','drag_plugin'),
                            ('px4_executable_with_stamped_land_state','px4_binary'), ('m100_experiment_manager','manager'),
                            ('m100_canonical_base_sdf','base_sdf'), ('m100_airborne_training','training_writer')):
        confirm(doc['source_files'][key]['sha256'] == identities[source_name], f'Provenance differs for {key}')
    manager_trace = doc['native_pack_drive']['trace']
    confirm(manager_trace['filename'] == run + '.pack_drive.jsonl' and manager_trace['io_error'] is None and
            digest(raw / manager_trace['filename']) == manager_trace['sha256'], 'Native manager trace identity differs')
    train = doc['airborne_training']
    confirm(train['training_eligible'] is True and train['relative_path'] == 'training/' + run + '.csv'
            and digest(raw / train['relative_path']) == train['sha256'], 'Recorder training row is missing or changed')
    edges = flight_edges(raw)
    takeoff, landing, disarm = (edges[k]/1e6 for k in ('takeoff_us', 'landing_us', 'disarm_us'))
    budget = train['native_budget']
    confirm(budget['run_id'] == run and abs(budget['start_s']-takeoff) <= 0.5e-6
            and abs(budget['end_s']-landing) <= 0.5e-6 and
            budget['status'] == 'COMPLETE' and budget['uncovered_s'] == 0, 'PX4 edges disagree with the native budget')
    original = raw / TRACE
    with original.open() as stream:
        rows = [json.loads(line) for line in stream]
    confirm(len(rows) == capture['captured_rows'], 'Native observer stream is incomplete')
    attributable = [row for row in rows if row['airborne_budget']['run_id'] == run]
    confirm(all(a['time_s'] <= b['time_s'] for a,b in zip(attributable,attributable[1:])), 'Pack clock went backwards')
    confirm(any(row['time_s'] < takeoff and row['initial_nominal_soc_percent'] == 100
                and row['initial_nameplate_discharge_offset_ah'] == 0 and row['discharged_ah'] == 0 and
                row['polarization_fast_v'] == 0 and row['polarization_slow_v'] == 0 and row['armed'] is False
                for row in attributable), 'Fresh pack with zero discharge and polarization not seen before takeoff')
    confirm(any(takeoff <= row['time_s'] <= landing and row['armed'] is True and row['px4_landed'] is False
                and (row.get('current_a') or 0) > 0 for row in attributable), 'No loaded native pack in flight')
    metrics = decay_metrics(rest_window(rows, run, landing, disarm), protocol)
    confirm(metrics['estimated_airborne_energy_Wh'] == train['estimated_airborne_energy_Wh'],
            'Airborne energy in post-disarm messages differs from the direct training row')
    return dict(status='PASS', run_id=run, case=case, observed_physics_step_s=result['observed_physics_step_s'],
                observed_seed=result['observed_seed'], ulog=edges, rest=metrics, landing_to_rest_start_s=metrics['start_sim_s']-landing,
                disarm_to_rest_start_s=metrics['start_sim_s']-disarm, raw_stream_sha256=digest(original),
                training_csv_sha256=train['sha256'], manager_trace_sha256=manager_trace['sha256'],
                native_receipt_sha256=digest(result_file), flight_provenance_sha256=digest(provenance),
                protocol_sha256=digest(raw / 'rest_protocol.json'), capture_identity=capture, parameter_sha256=protocol['parameter_sha256'],
                scope='Numerical recovery of both transferred 2-RC states, conditional on native confirmed zero load.')


def publish_pair(coarse, fine):
    """Publish two independently audited flights and an observed, paired step-size difference."""
    from .mission_audit import inspect
    from .mission_campaign import ARCHIVE, archive
    from .mission_prediction import ENERGY, TIME, dataframe, number

    base = read(OUT / 'protocol.json')
    checked = []
    for folder, step in ((coarse, .004), (fine, .002)):
        rest = analyze(folder)
        settings = read(folder / 'rest_protocol.json')
        case = rest['case']
        confirm(case['physics_step_s'] == step, 'Fine and coarse recordings have unexpected physics steps')
        confirm(rest == read(folder / 'rest_verification.json'), 'Saved rest report differs from original observations')
        with tempfile.TemporaryDirectory() as scratch:
            flight = inspect(folder, case, 1, {**base, 'source_sha256':settings['source_sha256']}, Path(scratch))
        confirm(flight['accepted'] and flight['metrics'][ENERGY] == rest['rest'][ENERGY], 'Independent airborne checks failed')
        checked.append((folder, case, settings, rest, flight))
    ignored = {'case_id','physics_step_s'}
    reference = {k:v for k,v in checked[0][1].items() if k not in ignored}
    confirm(reference == {k:v for k,v in checked[1][1].items() if k not in ignored}, 'Conditions differ beyond the time step')
    hashes = checked[0][2]['source_sha256']
    confirm(hashes == checked[1][2]['source_sha256'], 'Software changed across the paired step-size checks')
    records = []
    for raw,case,settings,rest,flight in checked:
        key = case['case_id']
        name = f"engineering/{key}/{rest['run_id']}.zip"
        archive_hash = archive(raw, ARCHIVE / name)
        save(OUT/'postflight_checks'/f'{key}.json', dict(native_rest_check=rest, independent_airborne_check=flight,
             collected_with=settings, raw_archive=name, raw_archive_sha256=archive_hash))
        m = flight['metrics']
        records.append(dict(case_id=key, seed=case['seed'], step_s=case['physics_step_s'], run_id=rest['run_id'],
            report=f'postflight_checks/{key}.json', raw_archive=name, archive_sha256=archive_hash,
            energy_Wh=m[ENERGY], airborne_s=m[TIME], horizontal_m=m['airborne_horizontal_distance_m'],
            min_supported_voltage_V=m['min_supported_load_voltage_V'], max_supported_current_A=m['max_supported_current_A'],
            max_voltage_equation_error_V=m['native_2rc_equation_max_residual_V'], max_vi_error_W=m['pack_power_balance_max_residual_W'],
            confirmed_rest_s=rest['rest']['confirmed_rest_delta_s'], max_rest_charge_variation_Ah=rest['rest']['max_charge_variation_ah'],
            max_rest_branch_error_V=max(b['max_abs_residual_v'] for b in rest['rest']['branches'].values())))
    prior = [number(r, ENERGY) for r in dataframe(OUT/'mission_flights.csv') if r['group']=='factorial'
             and r['mission_route']=='direct' and number(r,'payload_kg')==0 and number(r,'cruise_speed_mps')==6
             and number(r,'altitude_m')==25]
    delta = records[1]['energy_Wh'] - records[0]['energy_Wh']
    save(OUT/'postflight_verification.json', dict(status='PASS', included_in_training=False, flights=records,
        corrected_plugin_sha256=hashes['plugin'], corrected_pack_source_sha256=hashes['pack_source'],
        corrected_pack_header_sha256=hashes['pack_header'], refinement_energy_change_Wh=delta,
        refinement_energy_change_percent=100*delta/records[0]['energy_Wh'],
        refinement_time_change_s=records[1]['airborne_s']-records[0]['airborne_s'],
        discovery_same_setting_energies_Wh=prior,
        interpretation='One paired full mission at two solver steps with the same Gazebo seed. Includes controller, timing and trajectory variation; does not bound global discretization error.'))


def self_test():
    protocol = dict(parameter_sha256='synthetic', parameters=dict(tau1_s=.5,tau2_s=8.))
    rows = []
    for i in range(191):
        elapsed = i*.05
        row = dict(time_s=100+elapsed, model='m100_cmu_v2_0', battery_dynamics_model=MODEL,
                   battery_parameter_sha256='synthetic', status='DISARMED_ZERO_DEMAND_CAP_HELD',
                   px4_land_time_s=100+elapsed, px4_land_transition_s=95., px4_land_sequence=3,
                   raw_command_fresh=True, demand_fresh=True, demand_power_w=0, effective_demand_power_w=0,
                   raw_motor_rad_s=[0]*4, applied_motor_rad_s=[0]*4, awaiting_post_reset_zero_command=False,
                   q_domain_exhausted=False, initial_nominal_soc_percent=100, initial_nameplate_discharge_offset_ah=0,
                   discharged_ah=.66, armed=False, px4_landed=True,
                   airborne_budget=dict(run_id='synthetic',status='COMPLETE',end_s=95.,estimated_airborne_energy_Wh=18.),
                   polarization_fast_v=.12*math.exp(-elapsed/.5), polarization_slow_v=.40*math.exp(-elapsed/8.))
        row[REST] = 3+elapsed
        rows.append(row)

    def evaluate(series, disarm=100.):
        return decay_metrics(rest_window(series,'synthetic',95.,disarm), protocol)

    confirm(evaluate(rows)['samples'] == 171, 'Healthy synthetic control failed')
    for name, change in ((REST, 2.), ('polarization_slow_v', .04), ('polarization_fast_v', .08),
                         ('discharged_ah', .6), ('raw_motor_rad_s', [0,1,0,0]), ('armed', True),
                         ('demand_fresh', False), ('px4_land_transition_s', 98.)):
        damaged = rows.copy()
        damaged[90] = dict(rows[90], **{name:change})
        try:
            evaluate(damaged)
        except ValueError:
            continue
        raise AssertionError(f'Synthetic corruption passed: {name}')
    for samples,start in ((rows[:60]+rows[80:],100.), (rows,110.)):
        try:
            evaluate(samples,start)
        except ValueError:
            continue
        raise AssertionError('Synthetic timestamp gap or disarm mismatch passed')
    print('Synthetic rest check passed (healthy control + 10 rejection cases).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    pair = subs.add_parser('publish-pair')
    pair.add_argument('coarse', type=Path)
    pair.add_argument('fine', type=Path)
    for mode in ('prepare','check-install','monitor','analyze','self-test'):
        sub = subs.add_parser(mode)
        if mode != 'self-test':
            sub.add_argument('directory', type=Path)
        if mode == 'prepare':
            sub.add_argument('--step', type=float, default=.004, choices=(.004,.002))
            sub.add_argument('--seed', type=int, default=3771)
    args = parser.parse_args()
    if args.mode == 'publish-pair':
        publish_pair(args.coarse.resolve(), args.fine.resolve())
        return
    if args.mode == 'self-test':
        self_test()
        return
    raw = args.directory.resolve()
    if args.mode == 'prepare':
        prepare(raw, args.step, args.seed)
    elif args.mode == 'check-install':
        require_install(raw)
    elif args.mode == 'monitor':
        monitor(raw)
    else:
        try:
            report = analyze(raw)
        except Exception as exc:
            save(raw / 'rest_verification.json', dict(status='FAIL', reason=f'{type(exc).__name__}: {exc}'))
            raise
        save(raw / 'rest_verification.json', report)
        print(f'PASS {raw / "rest_verification.json"}')


if __name__ == '__main__':
    main()
