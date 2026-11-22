"""Archive the original light-weight recorder files for a completed supervisor acquisition."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


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
                    files = [receipt_file]
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
        staged.replace(release / staged.name)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=root / 'tmp/m100_reaudit_supervisor_rerun')
    parser.add_argument('--release', type=Path, default=root / 'results/m100_reaudit/supervisor_rerun')
    args = parser.parse_args()
    archive_receipts(args.raw, args.release)
