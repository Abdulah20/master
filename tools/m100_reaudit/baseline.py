"""Archive the two pre-change SITL control receipts, accepted CSVs and source digests."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

from .historical import sampled_window


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pack(source_root, output):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for identifier, sub in [('pilot_calm','before'), ('pilot_hover','before_hover')]:
        [path] = (source_root / sub).glob('*.result.json')
        receipt = json.loads(path.read_text())
        csv_path = Path(receipt['flight_csv'])
        provenance = csv_path.with_suffix('.provenance.json')
        if receipt['outcome'] != 'COMPLETE' or receipt['last_status']['armed'] or not receipt['last_status']['landed']:
            raise ValueError(f'incomplete pre-change control {identifier}')
        payload = csv_path.read_bytes()
        (output / (identifier + '.csv.gz')).write_bytes(gzip.compress(payload, compresslevel=7, mtime=0))
        rows = list(csv.DictReader(io.StringIO(payload.decode())))
        phase = 'CRUISE' if receipt['speed_command_mps'] else 'HOVER'
        measures = sampled_window(rows,phase)
        raw = Path(receipt['status_log'])
        records.append(dict(case_id=identifier, run_id=csv_path.stem, outcome=receipt['outcome'],
                            landed=True, armed=False, csv_sha256=sha(csv_path), receipt_sha256=sha(path),
                            provenance_sha256=sha(provenance), status_jsonl_sha256=sha(raw), **measures))
    (output/'controls.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root',type=Path); parser.add_argument('output',type=Path)
    args=parser.parse_args(); pack(args.source_root, args.output)
