"""CMU protocol, integration, group leakage and held-out numerical regressions."""
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.m100_reaudit import external as ex


def test_noncontiguous_irregular_signed_trapezoids():
    time = np.array([0.0, 0.25, 0.375, 0.5, 0.875])
    voltage = np.array([10., 12., 20., 22., 30.])
    current = np.array([-1., 2., 0., 2., 1.])
    usable = np.array([True, True, False, True, True])
    interval_mask = usable[1:] & usable[:-1]
    mean_power = ex.midpoint(voltage * current)
    result = ex.summarize_intervals(np.diff(time), mean_power, mean_power*0, mean_power*0, interval_mask)
    expected_j = ((-10 + 24)/2) * 0.25 + ((44 + 30)/2) * 0.375
    assert result['included_intervals'] == 2
    assert result['covered_duration_s'] == 0.625
    assert result['observed_energy_Wh'] == pytest.approx(expected_j / 3600)
    assert result['observed_mean_power_W'] == pytest.approx(expected_j / .625)


def test_whole_flight_qc_gap_and_reverse_time():
    f = pd.DataFrame({c: np.ones(4) for c in ex.COLUMNS})
    f['time'] = [0, 0.3, 0.3, 0.9]
    assert set(ex.quality_flags(f)) == {'non_increasing_time', 'gap_over_0.5s'}
    with pytest.raises(ValueError, match='cannot resolve'):
        ex.phase_boundaries(np.ones(120))


def test_measured_partition_and_new_regression_are_target_blind():
    m = pd.read_csv(ex.INPUT / 'parameters.csv').set_index('flight')
    c = pd.read_csv(ex.CONTEXT).set_index('flight_id')
    splits = ex.assign_splits(m, c)
    assert {fid for fid, s in splits.items() if s == 'test'} == set(c[c.published_role == 'holdout'].index)
    days = lambda s: {m.loc[f, 'date'] for f in splits if splits[f] == s}
    assert not days('train') & days('validation')
    x = pd.read_csv(ex.OUTPUT / 'measured_cruise.csv')
    fit = ex.preflight_reference(x)
    x.loc[x.split != 'train', 'observed_mean_power_W'] += 1000
    assert ex.preflight_reference(x)['coefficients'] == fit['coefficients']
    assert x.flight_id.is_unique and list(x.columns) == ['flight_id', 'split', *ex.FEATURES, 'observed_mean_power_W']


def test_real_flight_boundaries_and_raw_integral():
    m = pd.read_csv(ex.INPUT / 'parameters.csv').set_index('flight')
    c = pd.read_csv(ex.CONTEXT).set_index('flight_id')
    with ZipFile(ex.INPUT / 'flights.zip') as z, z.open('flights/2.csv') as stream:
        f = pd.read_csv(stream, usecols=ex.COLUMNS)
    windows, energy_error, time_error = ex.measured_scopes(f, m.loc[2], c.loc[2])
    airborne = windows['airborne']
    t = f.time.to_numpy()
    z = (t >= airborne['window_first_time_s']) & (t <= airborne['window_last_time_s'])
    p = (f.battery_current * f.battery_voltage).to_numpy()[z]
    manual_j = sum((a+b)/2 * (b_t - a_t) for a,b,a_t,b_t in zip(p[:-1], p[1:], t[z][:-1], t[z][1:]))
    assert airborne['observed_energy_Wh'] == pytest.approx(manual_j / 3600)
    assert energy_error < ex.ENERGY_ATOL_WH and time_error < ex.TIME_ATOL_S
    assert windows['stable_selected']['covered_duration_s'] < windows['stable_selected']['window_last_time_s'] - windows['stable_selected']['window_first_time_s']


def test_every_recording_and_locked_holdout_reproduce(tmp_path):
    import json

    summary = ex.evaluate(ex.INPUT, ex.CONTEXT, tmp_path)
    ex.check_hash(tmp_path / 'published_context.csv', ex.SHA256['published_context.csv'])
    saved = json.loads((ex.OUTPUT / 'validation.json').read_text())
    assert summary['source_sha256'] == saved['source_sha256']
    assert summary['actual_recordings'] == 209 and summary['included_complete_flights'] == 170
    assert summary['measured_train_validation_test'] == {'train': 75, 'validation': 35, 'test': 60}
    metrics = { (i['scope'],i['model']): i for i in summary['holdout'] }
    assert metrics['airborne', ex.BASELINE]['MAE_Wh'] == pytest.approx(0.3859066851576, abs=1e-7)
    assert metrics['cruise', ex.BASELINE]['RMSE_W'] == pytest.approx(20.6326280463, abs=1e-7)
    assert metrics['stable_selected', ex.SPEED_EXTENSION]['flights'] == 33
    assert metrics['stable_selected', ex.SPEED_EXTENSION]['MAE_W'] == pytest.approx(9.76926829856, abs=1e-7)
    for name in ('flight_manifest', 'phase_comparisons', 'grouped_errors', 'measured_cruise'):
        a = pd.read_csv(tmp_path / f'{name}.csv')
        b = pd.read_csv(ex.OUTPUT / f'{name}.csv')
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-8, rtol=1e-12)
