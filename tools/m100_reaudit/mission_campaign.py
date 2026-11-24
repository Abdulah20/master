"""Register and acquire fixed-destination native PX4/Gazebo flights with lossless receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import zipfile

from .direct_airborne_evidence import digest

ROOT = Path(__file__).resolve().parents[2]
MAX_ATTEMPTS = 4
OUT = ROOT / 'results/m100_reaudit/mission_150m'
WORK = ROOT / 'tmp/m100_closure/registered_acquire'
LAUNCH = ROOT / 'tools/m100_reaudit/isolated_native.sh'
RUNTIME = Path(os.environ.get('M100_RUNTIME', '/home/abdulah/Master/Drone/project'))
PX4 = Path(os.environ.get('M100_PX4', str(Path.home()/'PX4-Autopilot')))
ARCHIVE = Path(os.environ.get('M100_MISSION_ARCHIVE', '/media/abdulah/New Volume/M100_Mission_150m_2026-11-23'))
START_LOCK = threading.Lock()
last_start = 0.0


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_bytes())


def make_case(group, mass, speed, height, route, seed, wind=0, direction=0, temp=15, soc=100):
    cid = (f'{group}_p{round(mass*100):03d}_v{speed}_h{height}_{route}'
           f'_w{wind}d{direction}_t{temp}_q{soc}_s{seed}')
    return dict(case_id=cid, group=group, mission_route=route, payload_kg=mass, cruise_speed_mps=speed,
                altitude_m=height, wind_speed_mps=wind, wind_direction_toward_deg_ENU=direction, temperature_c=temp,
                initial_soc_percent=soc, seed=seed, physics_step_s=0.004, measurement_distance_m=150, hover_duration_s=60)


def cases():
    masses = (0, .25, .5)
    design = []
    # Interleave masses and heights within a seed, so partial acquisitions also span the design.
    for seed in (1701, 1702, 1703):
        for i, j, k in itertools.product(range(3), repeat=3):
            design.append(make_case('factorial', masses[k], (6, 8, 10)[(j+k)%3],
                                    (25, 50, 100)[(i+j+k)%3], 'direct', seed))
    # Test a second geometry at all three speeds and masses at the low height,
    # and at the same middle condition used for weather/charge sensitivities.
    routes = [make_case('route', m, v, h, 'north_dogleg', 1701)
              for m in masses for v, h in ((6,25), (8,25), (10,25), (8,50))]
    settings = [(3, 0, 15, 100), (3, 90, 15, 100), (3, 180, 15, 100),
                (6, 0, 15, 100), (6, 90, 15, 100), (6, 180, 15, 100),
                (0, 0, 10, 100), (0, 0, 25, 100), (0, 0, 15, 90), (0, 0, 15, 80)]
    sensitivity = [make_case('sensitivity', .25, 8, 50, route, 1701, *setting)
                   for setting in settings for route in ('direct', 'north_dogleg')]
    return design[:1] + routes[:1] + sensitivity[-2:-1] + design[1:27] + routes[1:] + sensitivity[:-2] + sensitivity[-1:] + design[27:]


def active_cruise_config(package):
    # Import from the target runtime in isolation. The analysis process can
    # already have a historical drone_nav package loaded from a frozen archive.
    resolver = (
        "import os; from drone_nav import m100_stable_cruise_speed as speed; "
        "selector = getattr(speed, 'profile_config_name', None); "
        "print(selector(os.environ.get('M100_STABLE_CRUISE_PROFILE', 'native')) "
        "if selector is not None else 'm100_stable_cruise_speed_v1.yaml')"
    )
    name = subprocess.check_output([sys.executable, '-B', '-c', resolver], cwd=package, text=True).strip()
    return package/'config'/Path(name).name


def sources():
    src = RUNTIME/'ros2_ws/src'
    plugins = RUNTIME/'ros2_ws/install/m100_gazebo_gui/lib/m100_gazebo_gui'
    models = PX4/'Tools/simulation/gz/models'
    refs = dict(manager=src/'drone_nav/drone_nav/m100_experiment_manager.py',
                training_writer=src/'drone_nav/drone_nav/m100_airborne_training.py',
                trajectory_guard=RUNTIME/'analysis/m100_high_speed_stability/run_one_stability_case.py',
                plugin=plugins/'libM100PackDrive.so', drag_plugin=plugins/'libM100AerodynamicDrag.so',
                pack_source=src/'m100_gazebo_gui/propulsion/M100PackDrive.cc',
                pack_header=src/'m100_gazebo_gui/propulsion/M100TheveninFit.hh',
                px4_binary=PX4/'build/px4_sitl_default/bin/px4',
                energy_config=src/'drone_nav/config/m100_energy_model_v2.yaml',
                stable_cruise=active_cruise_config(src/'drone_nav'),
                wind=src/'drone_nav/config/m100_wind_simulation_v1.yaml',
                transient=src/'drone_nav/config/m100_transient_power_v1.yaml',
                base_sdf=models/'m100_cmu_v2/model.sdf', launcher=LAUNCH, runner=LAUNCH.with_name('native.py'),
                launch_script=LAUNCH.with_name('run_native.sh'), battery_fit=OUT/'battery_fit.json',
                fit=OUT.parent/'cmu_calibration/fit.json')
    for suffix in ('025','050'):
        refs['payload'+suffix] = models/f'm100_cmu_v2_payload_{suffix}/model.sdf'
    if refs['stable_cruise'].name == 'm100_stable_cruise_speed_v2.yaml':
        refs['stable_cruise_decision'] = refs['stable_cruise'].with_name('m100_stable_cruise_decision_v2.json')
        refs['stable_cruise_implementation'] = src/'drone_nav/drone_nav/m100_stable_cruise_speed.py'
    return {name:digest(path) for name,path in refs.items()}


def register():
    protocol = dict(schema='M100_FIXED_DESTINATION_2RC_V1', destination_enu_m=[150,0], return_home=True,
                    reference_initial_nominal_soc_percent=100, nominal_reserve_percent=20, nameplate_Wh=99.9,
                    altitude_reference='AGL', temperature_reference='ambient at world datum',
                    wind_convention='degrees toward, east = 0 and north = 90',
                    route_waypoints_enu_m={'direct': [[150,0]], 'north_dogleg': [[75,37.5],[150,0]]},
                    independent_waypoint_tolerance_horizontal_m=2, independent_waypoint_tolerance_vertical_m=2,
                    battery_parameter_sha256=read(OUT/'battery_fit.json')['parameter_sha256'], source_sha256=sources(),
                    cases=cases(), confirmation='Independent seeds 2711 and 2712, lowest estimated energy option '
                    'and reference direct at 8 m/s and 50 m for each payload. Constraint includes return and reserve.')
    file = OUT/'protocol.json'
    if file.exists():
        if read(file) != protocol:
            raise ValueError('Acquisition registration or runtime identity changed. Preserve the existing cohort.')
    else:
        save(file, protocol)
    return protocol


def archive(raw, destination):
    """Archive originals on the data volume, then verify every uncompressed byte before releasing scratch space."""
    selected = [p for p in raw.rglob('*') if p.is_file() and not p.is_symlink()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix('.pending')
    if destination.exists():
        with zipfile.ZipFile(destination) as saved:
            if set(saved.namelist()) != {str(p.relative_to(raw)) for p in selected}:
                raise FileExistsError(f'Different prior archive at {destination}')
            for path in selected:
                with saved.open(str(path.relative_to(raw))) as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest() != digest(path):
                        raise IOError(f'Archive recovery mismatch for {path}')
        return digest(destination)
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as out:
        for path in sorted(selected):
            out.write(path, path.relative_to(raw))
    with zipfile.ZipFile(tmp) as saved:
        for path in selected:
            with saved.open(str(path.relative_to(raw))) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest() != digest(path):
                    raise IOError(f'Archive verification failed for {path}')
    tmp.replace(destination)
    return digest(destination)


def run_one(case, protocol_path, attempt, *, release_dir=None, work_root=None, archive_root=None, extra_env=None,
            bootstrap_rootfs=None):
    global last_start
    release_dir = release_dir or OUT
    work_root = work_root or WORK
    archive_root = archive_root or ARCHIVE
    cid = case['case_id']
    receipt_path = release_dir/'acquisition'/cid/f'attempt_{attempt}.json'
    work = work_root/cid
    work.mkdir(parents=True, exist_ok=True)
    with (work/'case.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt_path.exists():
            return
        if attempt > 1:
            previous = read(release_dir/'audits'/cid/f'attempt_{attempt-1}.json')
            if previous['accepted'] or not previous['reason']:
                raise ValueError(f'{cid}: previous attempt did not require replacement')
        raw = work/f'attempt_{attempt}'
        plan = raw/'request.json'
        if raw.exists():
            # If the acquisition parent exited before installing its receipt, keep the
            # original bytes and distinguish a closed flight from an interrupted one.
            if not plan.is_file() or read(plan)!=[case]:
                raise ValueError(f'Unknown original acquisition directory: {raw}')
            for proc in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    argv=proc.read_bytes().decode(errors='replace').split('\0')
                except OSError:
                    continue
                if 'tools.m100_reaudit.native' in argv and str(raw) in argv:
                    raise RuntimeError(f'An isolated acquisition still owns {raw}')
            code=0 if (raw/(cid+'.json')).exists() else None
        else:
            raw.mkdir()
            save(plan,[case])
            if bootstrap_rootfs is not None:
                # The launcher uses a pre-seeded private rootfs when present.
                shutil.copytree(bootstrap_rootfs, raw/'state/rootfs', copy_function=shutil.copy2)
            env = dict(os.environ, M100_GAZEBO_SEED=str(case['seed']), M100_MISSION_ROUTE=case['mission_route'],
                       M100_INITIAL_SOC_PERCENT=str(case['initial_soc_percent']), M100_PACK_DYNAMICS='cmu_2rc')
            env.update(extra_env or {})
            env.pop('M100_BATTERY_RESEARCH_PROFILE', None)
            with START_LOCK:
                time.sleep(max(0, 6 - (time.monotonic()-last_start)))
                last_start = time.monotonic()
            with (raw/'launcher.log').open('wb') as output:
                code = subprocess.run([str(LAUNCH), str(plan), str(raw)], stdout=output, stderr=subprocess.STDOUT,
                                      cwd=ROOT, env=env).returncode
        receipt = read(raw/(cid+'.json')) if (raw/(cid+'.json')).is_file() else {}
        name = f'{cid}/attempt_{attempt}.zip'
        archive_sha = archive(raw, archive_root/name)
        save(receipt_path, dict(case_id=cid, attempt=attempt, run_id=receipt.get('run_id'), outcome=receipt.get('outcome'),
                                reason=receipt.get('reason'), launcher_exit=code, archive=name, archive_sha256=archive_sha,
                                recorded_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=digest(protocol_path)))
        shutil.rmtree(raw)
        print(f'{cid} attempt {attempt} {receipt.get("outcome")} {receipt.get("reason")}', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--case', action='append', dest='requested')
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--confirmation', action='store_true')
    parser.add_argument('--register-only', action='store_true')
    args=parser.parse_args()
    protocol=register()
    file=OUT/'protocol.json'
    if args.confirmation:
        file=OUT/'confirmation_protocol.json'
        confirm=read(file)
        if confirm['parent_sha256']!=digest(OUT/'protocol.json'):
            raise ValueError('Different discovery protocol')
        planned=confirm['cases']
    else:
        planned=protocol['cases']
    if args.register_only:
        print(len(planned))
        return
    if args.requested:
        planned=[c for c in planned if c['case_id'] in args.requested]
        if len(planned)!=len(set(args.requested)):
            raise ValueError('Unregistered case')
    if not ARCHIVE.parent.is_dir():
        raise FileNotFoundError(ARCHIVE.parent)
    if not 1<=args.workers<=4:
        raise ValueError('Choose one to four isolated simulation workers for this host')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda c: run_one(c,file,args.attempt), planned))


if __name__=='__main__':
    main()
