"""Schedule the registered supervisor sorties in isolated native Gazebo/PX4 sessions."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'tmp/m100_reaudit_supervisor_rerun'
RELEASE = ROOT / 'results/m100_reaudit/supervisor_rerun'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def publish(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def existing_workers():
    active = set()
    for cmd in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = cmd.read_bytes().decode(errors='replace').rstrip('\0').split('\0')
        except OSError:
            continue
        if ('tools.m100_reaudit.direct_airborne_acquire' in args and '--work' in args
                and Path(args[args.index('--work') + 1]).resolve() == WORK):
            active.add((args[-1], int(args[args.index('--attempt') + 1])))
        if (args and (Path(args[0]).name == 'unshare' or 'tools.m100_reaudit.native' in args
                      or any(value.endswith('/run_native.sh') for value in args))):
            for value in args:
                if not value.startswith(str(WORK) + '/'):
                    continue
                relative = Path(value).relative_to(WORK)
                if len(relative.parts) == 2 and relative.name.startswith('attempt_'):
                    active.add((relative.parts[0], int(relative.name.split('_')[1])))
    return active


def check_model(case):
    name = 'm100_cmu_v2'
    if case['payload_kg']:
        name += f"_payload_{round(case['payload_kg'] * 100):03d}"
    px4 = Path(os.environ.get('M100_PX4', str(Path.home() / 'PX4-Autopilot')))
    digest = hashlib.sha256((px4 / 'Tools/simulation/gz/models' / name / 'model.sdf').read_bytes()).hexdigest()
    expected = read(RELEASE / 'model_identities.json')['models'][name]['model_sdf_sha256']
    if digest != expected:
        raise ValueError(f'Unregistered model for {case["case_id"]}')


def register_retry(cid, attempt, reason):
    if attempt < 2 or not reason:
        raise ValueError('Register the previous failed outcome and a reason before repeating it')
    previous = WORK / cid / f'attempt_{attempt-1}' / 'acquisition_exit.json'
    failed = read(previous)
    if failed['outcome'] == 'COMPLETE' and failed['direct_energy_Wh'] is not None:
        raise ValueError(f'{cid} already has a directly recorded label')
    base = RELEASE / 'protocol.json'
    protocol = read(base)
    case = next(c for c in protocol['cases'] if c['case_id'] == cid)
    protocol.update(schema='M100_SUPERVISOR_NATIVE_REPLACEMENT_V1', planned_utc=now(),
                    cases=[case], attempt=attempt, planned_replacement_cases=[cid],
                    original_protocol_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                    previous_attempt=failed, previous_exit_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
                    trigger=reason, workers=1)
    protocol['groups'] = {key: [cid] for key, members in protocol['groups'].items() if cid in members}
    folder = RELEASE / 'retries'
    folder.mkdir(exist_ok=True)
    dest = folder / f'{cid}_attempt_{attempt}.json'
    with dest.open('x') as out:
        json.dump(protocol, out, indent=2)
        out.write('\n')
    print(dest)


def record_exit(cid, attempt, protocol, code):
    directory = WORK / cid / f'attempt_{attempt}'
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / f'{cid}.json'
    state = read(receipt) if receipt.exists() else {}
    rid = state.get('run_id')
    row = directory / f'{rid}.provenance.json'
    training = read(row).get('airborne_training', {}) if row.exists() else {}
    result = dict(case_id=cid, attempt=attempt, run_id=rid,
                  protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                  finalized_utc=now(), worker_exit_code=code,
                  outcome=state.get('outcome', 'NO_ACQUISITION_RECEIPT'), reason=state.get('reason'),
                  training_status=training.get('status'), direct_energy_Wh=training.get('estimated_airborne_energy_Wh'))
    destination = directory / 'acquisition_protocol.json'
    if destination.exists() and destination.read_bytes() != protocol.read_bytes():
        raise ValueError(f'Conflicting protocol for {cid} attempt {attempt}')
    destination.write_bytes(protocol.read_bytes())
    publish(directory / 'acquisition_exit.json', result)
    print(cid, result['outcome'], result['training_status'], result['direct_energy_Wh'], flush=True)


def schedule(protocol_path, workers):
    protocol = read(protocol_path)
    attempt = protocol.get('attempt', 1)
    definitions = {case['case_id']: case for case in protocol['cases']}
    cases = protocol.get('planned_replacement_cases', [c['case_id'] for c in protocol['cases']])
    WORK.mkdir(parents=True, exist_ok=True)
    children = {}
    next_launch = 0
    status_path = WORK / 'acquisition_status.json'
    with (WORK / 'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            for cid, child in list(children.items()):
                code = child.poll()
                if code is not None:
                    record_exit(cid, attempt, protocol_path, code)
                    del children[cid]
            active = existing_workers() | {(cid, attempt) for cid in children}
            finished, queued, running = [], [], []
            for cid in cases:
                directory = WORK / cid / f'attempt_{attempt}'
                marker = directory / 'acquisition_exit.json'
                if marker.exists():
                    finished.append(cid)
                elif (cid, attempt) in active:
                    running.append(cid)
                elif directory.exists():
                    # A previously started worker may be supervised outside this process.
                    record_exit(cid, attempt, protocol_path, None)
                    finished.append(cid)
                else:
                    queued.append(cid)
            setting = WORK / 'max_workers'
            slots = (int(setting.read_text()) if setting.exists() else workers) if attempt == 1 else 1
            if not 1 <= slots <= 8:
                raise ValueError('Expected a positive local worker budget up to eight')
            free_gib = shutil.disk_usage(WORK).free / 2**30
            publish(status_path, dict(observed_utc=now(), attempt=attempt, completed=len(finished),
                                      running=running, pending=len(queued), max_workers=slots, free_GiB=round(free_gib, 3)))
            if not running and not queued:
                return
            if queued and len(running) < slots and time.monotonic() >= next_launch and free_gib > 0.5:
                cid = queued[0]
                check_model(definitions[cid])
                cmd = [sys.executable, '-u', '-m', 'tools.m100_reaudit.direct_airborne_acquire',
                       '--protocol', str(protocol_path), '--work', str(WORK), '--attempt', str(attempt), cid]
                children[cid] = subprocess.Popen(cmd, cwd=ROOT)
                next_launch = time.monotonic() + 10
            time.sleep(3)


def drain_registered_retries():
    initial = read(RELEASE / 'protocol.json')
    while any(not (WORK / case['case_id'] / 'attempt_1/acquisition_exit.json').exists() for case in initial['cases']):
        time.sleep(5)
    for protocol_path in sorted((RELEASE / 'retries').glob('case_*_attempt_*.json')):
        protocol = read(protocol_path)
        cid, = protocol['planned_replacement_cases']
        if (WORK / cid / f'attempt_{protocol["attempt"]}/acquisition_exit.json').exists():
            continue
        while True:
            try:
                schedule(protocol_path, 1)
                break
            except BlockingIOError:
                time.sleep(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=RELEASE / 'protocol.json')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--register-retry', metavar='CASE')
    parser.add_argument('--attempt', type=int, default=2, help='Attempt to preregister')
    parser.add_argument('--reason', help='Reason for repeating an incomplete acquisition')
    parser.add_argument('--drain-registered-retries', action='store_true', help='Wait for first pass, then run only preregistered replacements serially')
    args = parser.parse_args()
    if args.register_retry:
        register_retry(args.register_retry, args.attempt, args.reason)
    elif args.drain_registered_retries:
        drain_registered_retries()
    else:
        schedule(args.protocol.resolve(), args.workers)
