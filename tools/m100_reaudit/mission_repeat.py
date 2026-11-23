"""Repeat the native 150 m study in a fresh directory, preserving the published flights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from .mission_campaign import ARCHIVE, MAX_ATTEMPTS, ROOT, read, save


RELEASE = Path('results/m100_reaudit/mission_150m')
INPUTS = ('battery_fit.json', 'column_dictionary.csv', 'field_holdout.json', 'field_holdout_by_flight.csv',
          'historical_preflight.csv', 'historical_intervals.csv.gz', 'historical_preflight_model',
          'historical_power_model', 'historical_provenance.json')


def prepare(archive_parent, workers):
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    scratch = ROOT / 'tmp'
    scratch.mkdir(exist_ok=True)
    archive_parent.mkdir(parents=True, exist_ok=True)
    study = Path(tempfile.mkdtemp(prefix='m100-repeat-', dir=scratch))
    paths = ['tools/m100_reaudit', 'config/m100_reaudit', 'docs/m100_reaudit/mission_training_ar.md',
             'docs/m100_reaudit/mission_2rc_ar.md', 'results/m100_reaudit/cmu_calibration/fit.json',
             *(str(RELEASE / name) for name in INPUTS)]
    with tempfile.TemporaryFile() as packed:
        subprocess.run(['git', 'archive', code, '--', *paths], cwd=ROOT, stdout=packed, check=True)
        packed.seek(0)
        with tarfile.open(fileobj=packed) as files:
            files.extractall(study, filter='data')
    raw = tempfile.mkdtemp(prefix='m100-repeat-', dir=archive_parent)
    save(study / 'repeat.json', dict(code_commit=code, archive=raw, workers=workers))
    save(study / RELEASE / 'retry_policy.json', dict(max_attempts=MAX_ATTEMPTS, same_seed_and_conditions=True,
         concurrency_by_attempt=[max(1, workers-attempt+1) for attempt in range(1, MAX_ATTEMPTS+1)]))
    return study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-parent', type=Path, default=ARCHIVE.parent, help='Storage volume for a new raw archive')
    p.add_argument('--workers', type=int, choices=range(1, 5), default=2, help='Workers for a new study')
    p.add_argument('--prepare-only', action='store_true', help='Register and check the installed inputs without flying')
    p.add_argument('--resume', type=Path, help='Continue a previously created study with its own archive and workers')
    args = p.parse_args()
    study = args.resume.resolve() if args.resume else prepare(args.archive_parent.resolve(), args.workers)
    settings = read(study / 'repeat.json')
    env = {**os.environ, 'M100_MISSION_ARCHIVE': settings['archive'], 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    print(json.dumps(dict(study=str(study), results=str(study / RELEASE), **settings), indent=2), flush=True)

    def run(module, *options):
        subprocess.run([sys.executable, '-m', 'tools.m100_reaudit.' + module, *options], cwd=study, env=env, check=True)

    run('mission_campaign', '--register-only')
    if args.prepare_only:
        return
    run('mission_workflow', '--workers', str(settings['workers']))
    run('mission_phases')
    run('mission_error_breakdown', '--cohort', 'all', '--output', str(RELEASE / 'error_breakdown'))
    run('mission_report')
    subprocess.run(['sha256sum', '--quiet', '-c', 'SHA256SUMS'], cwd=study / RELEASE, check=True)


if __name__ == '__main__':
    main()
