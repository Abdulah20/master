import gzip
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.m100_reaudit import verify_archive as check


def test_corrupt_saved_power_is_detected_even_when_counter_is_untouched():
    case = 'case_001'
    path = check.DEFAULT/'gazebo_batch/traces'/f'{case}.csv.gz'
    truth = pd.read_csv(path)
    sample = truth.index[truth.phase == 'CRUISE'][3]
    truth.loc[sample, 'power_W'] += 250
    with pytest.raises(ValueError, match='cumulative energy'):
        check.dart_phases(truth, case)


def test_before_control_hash_binds_decompressed_payload(tmp_path):
    source = check.DEFAULT/'before'
    for p in source.glob('*'):
        shutil.copyfile(p, tmp_path/p.name)
    path = tmp_path/'pilot_hover.csv.gz'
    path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes())+b'\n', mtime=0))
    with pytest.raises(ValueError, match='raw CSV hash'):
        check.verify_before(tmp_path)


def test_raw_target_and_reported_error_groups_reproduce_from_public_measurements():
    summary = check.verify_external(check.DEFAULT/'external', ROOT/'data/external/m100',
                                    ROOT/'tools/m100_reaudit/runtime_snapshot')
    assert summary == {'recordings':209, 'measured_complete_flights':170, 'continuous_W_VI_windows':680,
                       'error_groups_recomputed':260}
