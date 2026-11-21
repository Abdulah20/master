"""Run isolated PX4 cases with their original mission inputs, remaining restartable by case."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .native_metrics import ROOT
from .physics import PORTABLE_STUDY


def case_plan(which):
    config = ROOT/'config/m100_reaudit'
    cases = []
    if which in ('pilots', 'all'):
        for name in ('native_pilots','native_seed_1701','native_seed_1702'):
            cases.extend(json.loads((config/f'{name}.json').read_text()))
    if which in ('originals', 'all'):
        original = json.loads((PORTABLE_STUDY/'protocol.json').read_text())['cases']
        if len(original) != 69:
            raise ValueError('invalid frozen 69-case protocol')
        cases.extend(dict(case,physics_step_s=.004) for case in original)
    ids = [c['case_id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate scenario ID in native acquisition plan')
    return cases


def experiment(case, root):
    cid = case['case_id']
    directory = root/cid
    directory.mkdir(parents=True,exist_ok=True)
    plan = directory/'plan.json'
    content = json.dumps([case],sort_keys=True,indent=2)+'\n'
    if plan.exists() and plan.read_text() != content:
        raise ValueError(f'{cid}: acquisition plan changed. Choose a new working directory.')
    plan.write_text(content)
    env = dict(os.environ)
    env.pop('M100_GAZEBO_SEED',None)
    if 'seed' in case:
        env['M100_GAZEBO_SEED'] = str(case['seed'])
    for attempt in ('attempt_1','attempt_2'):
        run_dir = directory/attempt
        receipt = run_dir/f'{cid}.json'
        if not receipt.exists():
            run_dir.mkdir(exist_ok=True)
            if any(run_dir.glob('m100_*.csv')):
                # Preserve interrupted observations and use a fresh namespace for the retry.
                receipt.write_text(json.dumps(dict(**case,outcome='INTERRUPTED',reason='acquisition ended before the terminal receipt'))+'\n')
            else:
                with (run_dir/'acquisition.log').open('a') as log:
                    ret = subprocess.run([str(ROOT/'tools/m100_reaudit/isolated_native.sh'),str(plan),str(run_dir)],
                                         stdout=log,stderr=subprocess.STDOUT,env=env)
                if not receipt.exists():
                    receipt.write_text(json.dumps(dict(**case,outcome='RUNNER_ERROR',reason=f'acquisition exit {ret.returncode} without a flight receipt'))+'\n')
        result = json.loads(receipt.read_text())
        if result.get('run_id') or result['outcome'] == 'COMPLETE':
            break
    return cid, result['outcome'],result.get('reason','')


def run_cases(which, work, jobs, only=None):
    cases = case_plan(which)
    if only:
        ids = set(only)
        cases = [c for c in cases if c['case_id'] in ids]
        if len(ids) != len(cases):
            raise ValueError('unknown case ID in requested subset')
    work = work.resolve()
    work.mkdir(exist_ok=True,parents=True)
    from .physics import DEFAULT_RUNTIME, ENERGY_MODULE
    project = Path(os.environ.get("M100_RUNTIME", DEFAULT_RUNTIME)).expanduser()
    px4 = Path(os.environ.get("M100_PX4", Path.home() / "PX4-Autopilot")).expanduser()
    inputs = {
        "linked_px4_manager": project / ENERGY_MODULE.parent / "m100_experiment_manager.py",
        "px4_sitl_executable": px4 / "build/px4_sitl_default/bin/px4",
        "native_pack_drive_library": project / "ros2_ws/install/m100_gazebo_gui/lib/m100_gazebo_gui/libM100PackDrive.so",
        "m100_cmu_v2/model.sdf": px4 / "Tools/simulation/gz/models/m100_cmu_v2/model.sdf",
        "gz_env.sh": px4 / "build/px4_sitl_default/rootfs/gz_env.sh",
    }
    training_writer = project / ENERGY_MODULE.parent / "m100_airborne_training.py"
    if training_writer.is_file():
        inputs["airborne_training_writer"] = training_writer
    for script in (Path(__file__), Path(__file__).with_name("native.py"),
                   Path(__file__).with_name("run_native.sh"), Path(__file__).with_name("isolated_native.sh")):
        inputs[str(script.relative_to(ROOT))] = script
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
    stamp=work/'acquisition_inputs.json'
    if stamp.exists() and json.loads(stamp.read_text())!=hashes:
        raise ValueError('native acquisition source differs from checkpoint. Choose a new working directory.')
    stamp.write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures=[pool.submit(experiment,c,work) for c in cases]
        for ix,f in enumerate(as_completed(futures),1):
            cid,outcome,reason=f.result();print(f'[{ix}/{len(cases)}] {cid} {outcome} {reason[:160]}',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('which',choices=['all','pilots','originals'])
    parser.add_argument('work',type=Path)
    parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--only',nargs='+')
    args=parser.parse_args()
    if args.jobs < 1: parser.error('at least one worker required')
    run_cases(args.which,args.work,args.jobs,args.only)
