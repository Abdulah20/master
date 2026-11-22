"""Archive the original light-weight recorder files for a completed supervisor acquisition."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


def verify_export(bundle, release):
    with (release / 'M100_69_experiments_rerun.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {c['case_id'] for c in json.loads(bundle.read('protocol.json'))['cases']}
    if len(rows) != len(expected) or {row['case_id'] for row in rows} != expected:
        raise ValueError('The exported cases differ from the acquisition protocol')
    for row in rows:
        if not row['estimated_airborne_energy_Wh']:
            continue
        run_id = row['run_id']
        prefix = f"{row['case_id']}/attempt_{row['attempt']}"
        receipt = json.loads(bundle.read(f"{prefix}/{row['case_id']}.json"))
        if not (receipt['landed'] and not receipt['armed'] and receipt['outcome'] == 'COMPLETE'
                and receipt['run_id'] == run_id):
            raise ValueError(f'Original receipt does not confirm the selected flight in {prefix}')
        data = bundle.read(f'{prefix}/training/{run_id}.csv')
        original, = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
        if hashlib.sha256(data).hexdigest() != row['direct_row_sha256']:
            raise ValueError(f'Original recorder bytes differ in {prefix}')
        for field in ('run_id', 'training_status', 'training_eligible', 'label_source', 'estimated_airborne_energy_Wh'):
            if original[field] != row[field]:
                raise ValueError(f'{field} differs from the original recorder in {prefix}')
        if float(original['uncovered_s']) != 0 or float(original['airborne_duration_s']) <= 0:
            raise ValueError(f'Incomplete airborne time in {prefix}')


def archive_receipts(raw, release):
    protocol = json.loads((release / 'protocol.json').read_bytes())
    manifest = io.StringIO(newline='')
    index = csv.writer(manifest, lineterminator='\n')
    index.writerow(('case_id', 'attempt', 'run_id', 'outcome', 'archive_member', 'sha256'))
    seen = set()
    with TemporaryDirectory(dir=release.parent) as scratch:
        staged = Path(scratch) / 'recorder_files.zip'
        with ZipFile(staged, 'w', compression=ZIP_DEFLATED, compresslevel=5) as bundle:
            bundle.write(release / 'protocol.json', 'protocol.json')
            for case in protocol['cases']:
                cid = case['case_id']
                for folder in sorted((raw / cid).glob('attempt_*')):
                    receipt_file = folder / (cid + '.json')
                    if not receipt_file.is_file():
                        continue
                    receipt = json.loads(receipt_file.read_bytes())
                    if receipt.get('case_id') != cid:
                        raise ValueError(f'Wrong receipt in {folder}')
                    rid = receipt.get('run_id')
                    files = [receipt_file, folder / 'acquisition_protocol.json', folder / 'acquisition_exit.json']
                    if rid:
                        files.extend(p for p in (folder / f'{rid}.csv', folder / f'{rid}.provenance.json',
                                                  folder / 'training' / f'{rid}.csv') if p.is_file())
                    for member in files:
                        data = member.read_bytes()
                        name = member.relative_to(raw).as_posix()
                        bundle.writestr(name, data)
                        index.writerow((cid, folder.name, rid, receipt['outcome'], name, hashlib.sha256(data).hexdigest()))
                    seen.add(cid)
            if seen != {case['case_id'] for case in protocol['cases']}:
                raise ValueError('One or more planned cases still lack a receipt')
            bundle.writestr('files.csv', manifest.getvalue())
        with ZipFile(staged) as check:
            manifest.seek(0)
            for item in csv.DictReader(manifest):
                if hashlib.sha256(check.read(item['archive_member'])).hexdigest() != item['sha256']:
                    raise ValueError(item['archive_member'])
            verify_export(check, release)
        staged.replace(release / staged.name)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=root / 'tmp/m100_reaudit_supervisor_rerun')
    parser.add_argument('--release', type=Path, default=root / 'results/m100_reaudit/supervisor_rerun')
    args = parser.parse_args()
    archive_receipts(args.raw, args.release)
