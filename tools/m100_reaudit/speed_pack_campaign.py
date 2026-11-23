"""Register and run a paired Gazebo check of the stable-cruise pack request."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from . import mission_campaign as mission
from .direct_airborne_evidence import digest


ROOT = mission.ROOT
OUT = ROOT / 'results/m100_reaudit/speed_pack_link'
WORK = ROOT / 'tmp/m100_speed_pack_link'
ARCHIVE = Path(os.environ.get('M100_SPEED_PACK_ARCHIVE',
               '/media/abdulah/New Volume/M100_Speed_Pack_Link_2026-11-23'))
PROTOCOL = OUT / 'protocol.json'


def register():
    src = mission.RUNTIME / 'ros2_ws/src/drone_nav/drone_nav'
    hashes = mission.sources()
    for name, path in {
        'initial_parameters': mission.PX4 / 'build/px4_sitl_default/rootfs/parameters.bson',
        'speed_fit_implementation': src / 'm100_stable_cruise_speed.py',
        'calibration_envelope': src / 'm100_contract.py',
        'field_evaluation': OUT.parent / 'gong_cruise/flights.csv',
    }.items():
        hashes[name] = digest(path)
    cases = []
    for mode in ('off', 'on'):
        for speed in (4, 6, 8):
            case = mission.make_case('steady', .25, speed, 50, 'measurement', 4101)
            pair = f'v{speed}_s4101'
            case.update(case_id=f'steady_{pair}_{mode}', pair_id=pair, mode=mode,
                        measurement_distance_m=200)
            cases.append(case)
    protocol = dict(
        schema='M100_STABLE_CRUISE_PACK_PAIR_V1', cases=cases, source_sha256=hashes,
        raw_archive_root=str(ARCHIVE), corrected_mode='on',
        control='Same software and seed. Off disables only the correction passed to the native pack.',
        geometry='Straight outbound 200 m measurement leg after acceleration, then return and land.',
        comparison='Time-weighted native terminal V*I in matched stable outbound flight. '
                   'Reference: held-out CMU periods at the same commanded speed, payload and altitude. '
                   'Simulation wind is fixed; CMU ambient wind is unmatched.',
        support='4, 6, 8 m/s only; the existing calm-reference and minimum-duration gates apply.',
        repetitions_per_mode_and_speed=1,
    )
    if PROTOCOL.exists():
        if mission.read(PROTOCOL) != protocol:
            raise ValueError('Runtime, reference data or registered design changed.')
    else:
        mission.save(PROTOCOL, protocol)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--attempt', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--case', action='append', dest='requested')
    parser.add_argument('--register-only', action='store_true')
    args = parser.parse_args()
    if not ARCHIVE.parent.is_dir():
        raise FileNotFoundError(ARCHIVE.parent)
    cases = register()
    if args.requested:
        cases = [case for case in cases if case['case_id'] in args.requested]
        if len(cases) != len(set(args.requested)):
            parser.error('Unknown case ID')
    if args.register_only:
        return

    def acquire(case):
        print('Acquiring', case['case_id'], flush=True)
        mission.run_one(case, PROTOCOL, args.attempt, release_dir=OUT, work_root=WORK,
                        archive_root=ARCHIVE, extra_env={
                            'M100_PACK_STABLE_CRUISE_CORRECTION': '1' if case['mode'] == 'on' else '0',
                        })

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(acquire, cases))


if __name__ == '__main__':
    main()
