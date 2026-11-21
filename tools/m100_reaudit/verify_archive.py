"""Verify saved M100 research outputs without starting Gazebo, PX4, or ROS."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(REPO))
from tools.m100_reaudit.external import REPRODUCTION_SHA256, SHA256

DEFAULT = REPO / 'results/m100_reaudit'
WH_TOL = 1e-7
TIME_TOL = 1e-7
DIGIT_TOL = 1e-7
HISTORICAL_EXAMPLES = {'case_001', 'case_007', 'case_035', 'case_069'}
PHASES = ('takeoff', 'cruise', 'landing')
EXTERNAL_RELEASE = {
    'flight_manifest.csv': '171d8176bbddab97b25d19ada43c54c67d760a6e7680f0380233d54c08c044f3',
    'measured_cruise.csv': 'c0498b640244220bc5cda3c6de9d521641ed47ab30801edc076d2c18514a8f38',
    'phase_comparisons.csv': '302e2aa35b16ae0c755bc8a9baf3f193637b70b7d6a0df8fa36e08a4c44d1add',
    'grouped_errors.csv': 'f0f6ed3a13254be470bf70d81c9b51a9f06bfd7178c02a2d0f1e27974db932e9',
}


def require(condition, context):
    if not condition:
        raise ValueError(context)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_hash(path, expected):
    require(sha(path) == expected, f'SHA-256 mismatch: {path}')


def close(actual, expected, context, atol=DIGIT_TOL):
    # Summaries round to 11 or 12 significant digits. Tolerances are absolute.
    if not np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True):
        raise ValueError(f'{context}: numbers disagree')


def json_file(path):
    return json.loads(path.read_text())


def indexed(path, keys):
    table = pd.read_csv(path).set_index(keys)
    require(table.index.is_unique, f'duplicate key in {path}')
    return table


def inventory(directory, names):
    require({p.name for p in directory.glob('*.csv.gz')} == set(names), f'trace inventory: {directory}')


def dart_phases(trace, case_id):
    """Reconstruct all time, distance, energy and counters directly from adjacent samples."""
    require(set(trace.case_id) == {case_id}, case_id + ': wrong trace identity')
    t = trace.time_s.to_numpy()
    p = trace.power_W.to_numpy()
    labels = trace.phase.to_numpy()
    dt = np.diff(t)
    same = labels[1:] == labels[:-1]
    require(np.all((dt >= 0) & (dt <= .5 + 1e-8)), case_id + ': time gap')
    require(np.all(dt[~same] == 0), case_id + ': positive interval between phases')
    valid = same & (dt > 0) & np.isfinite(p[:-1]) & np.isfinite(p[1:])
    close(trace.incoming_interval_has_power, np.r_[False, valid], case_id + ': coverage')
    d_wh = np.where(valid, (p[1:] + p[:-1]) * dt / 7200, 0.)
    counter = np.r_[0., np.cumsum(d_wh)]
    close(counter, trace.cumulative_supported_Wh, case_id + ': cumulative energy', WH_TOL)
    close(99.9-counter, trace.nominal_remaining_Wh, case_id + ': nominal remaining energy', WH_TOL)
    for axis in ('e', 'n', 'u'):
        close(trace['air_' + axis + '_mps'], trace['v' + axis + '_mps'] - trace['wind_' + axis + '_mps'],
              case_id + ': air-relative velocity')
    require(trace[['voltage_V', 'current_A']].isna().all().all(), case_id + ': unexpected battery channels')
    step_distance = np.hypot(np.diff(trace.x_m), np.diff(trace.y_m))
    groups = {}
    for phase in pd.unique(labels):
        at = np.flatnonzero(labels == phase)
        first, last = at[0], at[-1]
        require(last - first + 1 == len(at), case_id + ': disjoint phase')
        intervals = slice(first, last)
        covered = math.fsum(dt[intervals][valid[intervals]])
        energy = math.fsum(d_wh[intervals])
        duration = t[last] - t[first]
        groups[phase] = dict(start_s=t[first], end_s=t[last], duration_s=duration, covered_s=covered,
                             energy_Wh=energy if covered else np.nan,
                             horizontal_path_m=math.fsum(step_distance[intervals]),
                             along_course_m=trace.x_m.iat[last] - trace.x_m.iat[first],
                             weighted_mean_power_W=energy*3600/covered if covered else np.nan,
                             covered_fraction=covered/duration if duration else np.nan)
    close(sum(row['duration_s'] for row in groups.values()), t[-1]-t[0], case_id + ': phase times')
    close(sum(r['energy_Wh'] for r in groups.values() if r['covered_s']), counter[-1], case_id + ': phase energies')
    return groups


def dart_totals(groups):
    airborne = [s for phase, s in groups.items() if phase not in ('GROUND', 'LANDED')]
    scored = groups['CRUISE'] if 'CRUISE' in groups else groups['HOVER']
    return dict(measurement_mean_power_W=scored['weighted_mean_power_W'], measurement_Wh=scored['energy_Wh'],
                measurement_duration_s=scored['duration_s'], mission_supported_Wh=sum(s['energy_Wh'] for s in airborne),
                airborne_duration_s=sum(s['duration_s'] for s in airborne), covered_s=sum(s['covered_s'] for s in airborne))


def verify_gazebo(folder):
    study = json_file(folder/'study.json')
    for name, digest in study['output_table_sha256'].items():
        verify_hash(folder/name, digest)
    cases = indexed(folder/'cases.csv', 'run_id')
    ptable = indexed(folder/'phases.csv', ['run_id', 'phase'])
    repeats = indexed(folder/'repeatability.csv', 'case_id')
    require((cases.loc[cases.engine_result == 'not_launched', 'status'] == 'excluded').all(),
            'DART unlaunched case marked as simulated')
    ran = cases[cases.engine_result != 'not_launched']
    runs = set(ran.index)
    inventory(folder/'traces', [s+'.csv.gz' for s in runs | {'repeat_'+c for c in repeats.index}])
    require(runs == set(ptable.index.get_level_values('run_id')), 'DART phase inventory')
    for run_id, row in ran.iterrows():
        path = folder/'traces'/(run_id+'.csv.gz')
        verify_hash(path, row.sha256)
        trace = pd.read_csv(path)
        require(len(trace) == row.samples, run_id + ': sample count')
        groups = dart_phases(trace, row.case_id)
        stored = ptable.loc[run_id]
        require(set(groups) == set(stored.index), run_id + ': phase labels')
        for phase, values in groups.items():
            for key, number in values.items():
                close(number, stored.at[phase, key], f'{run_id}/{phase}/{key}')
        for key, number in dart_totals(groups).items():
            close(number, row[key], run_id+'/'+key)
        exhaustion = trace.loc[trace.nominal_remaining_Wh <= 0, 'time_s']
        close(exhaustion.iat[0] if len(exhaustion) else np.nan, row.first_sample_nameplate_exhaustion_s,
              run_id + ': nameplate exhaustion')
    for name, row in repeats.iterrows():
        path = folder/'traces'/('repeat_'+name+'.csv.gz')
        verify_hash(path, row.repeat_trace_sha256)
        trace = pd.read_csv(path)
        require(len(trace) == row.repeated_sample_rows, name + ': repeat sample count')
        values = dart_totals(dart_phases(trace, name))
        base = cases.loc[name]
        for field, delta in [('measurement_mean_power_W', 'measurement_delta_W'), ('measurement_Wh', 'measurement_delta_Wh'),
                             ('mission_supported_Wh', 'mission_delta_Wh'), ('airborne_duration_s', 'airborne_delta_s')]:
            close(values[field] - base[field], row[delta], name + '/' + delta)
    require(len(ran) == study['actual_base_simulated'] + study['fine_runs'] and len(repeats) == study['fresh_repeat_runs'],
            'DART run counts')
    require(sum(ran.samples) == study['trace_rows'] and sum(repeats.repeated_sample_rows) == study['extra_repeat_trace_rows'],
            'DART total sample counts')
    return {'traces': len(ran)+len(repeats), 'samples': int(sum(ran.samples)+sum(repeats.repeated_sample_rows))}


def verify_before(folder):
    records = json_file(folder/'controls.json')
    require({r['case_id'] for r in records} == {'pilot_calm', 'pilot_hover'}, 'before control identities')
    inventory(folder, [r['case_id']+'.csv.gz' for r in records])
    for r in records:
        name = r['case_id']
        payload = gzip.decompress((folder/(name+'.csv.gz')).read_bytes())
        require(hashlib.sha256(payload).hexdigest() == r['csv_sha256'], name + ': raw CSV hash')
        require(r['outcome']=='COMPLETE' and r['landed'] and not r['armed'], name + ': outcome')
        frame = pd.read_csv(io.BytesIO(payload))
        m = frame.loc[frame.flight_phase == ('CRUISE' if name=='pilot_calm' else 'HOVER')]
        require(len(m) == r['samples'] and np.all(np.diff(m.index)==1), name + ': measurement samples')
        t = m.energy_timestamp_s.to_numpy()
        p = m.electrical_power_W.to_numpy()
        require(np.all(np.diff(t)>0), name + ': timestamp order')
        energy = math.fsum(np.diff(t)*(p[1:]+p[:-1])/7200)
        first, last = m.measurement_energy_Wh.iat[0], m.measurement_energy_Wh.iat[-1]
        measurements = dict(aligned_1Hz_energy_Wh=energy, aligned_high_rate_counter_change_Wh=last-first,
                            aligned_1Hz_minus_high_rate_Wh=energy-(last-first), aligned_mean_power_W=energy*3600/(t[-1]-t[0]),
                            counter_at_first_sample_Wh=first, counter_at_last_sample_Wh=last, sampled_first_s=t[0],
                            sampled_last_s=t[-1], sampled_duration_s=t[-1]-t[0])
        for field, value in measurements.items():
            close(value, r[field], name+'/'+field)
    return len(records)


def verify_historical(folder):
    cases = indexed(folder/'cases.csv', 'case_id')
    facts = json_file(folder/'summary.json')
    require(len(cases) == facts['source_cases'] == 69, 'historical case counts')
    saved = cases.dropna(subset=['saved_trace_sha256'])
    require(set(saved.index) == HISTORICAL_EXAMPLES, 'historical frozen trace identifiers')
    inventory(folder/'traces', [c+'.csv.gz' for c in HISTORICAL_EXAMPLES])
    for name, row in saved.iterrows():
        path = folder/'traces'/(name+'.csv.gz')
        verify_hash(path, row.saved_trace_sha256)
        trace = pd.read_csv(path)
        power = trace.power_replay_W.to_numpy()
        dt = np.diff(trace.t_s)
        valid = (trace.interval_status.to_numpy()[1:] == 'INTEGRATED')
        d_wh = np.where(valid, dt*(power[1:]+power[:-1])/7200, 0)
        close(np.r_[0.,np.cumsum(d_wh)], trace.cumulative_replay_Wh, name + ': replay counter')
        measured = trace.in_measurement.to_numpy().astype(bool)
        close(np.sum(d_wh[measured[1:] & measured[:-1]]), row.measurement_replay_energy_Wh, name + ': replay window')
    phases = pd.read_csv(folder/'phases.csv').groupby('case_id').replay_quadrature_Wh.sum()
    require(set(phases.index) == set(cases.index), 'historical phases missing')
    close(phases.loc[cases.index], cases.observer_replay_supported_Wh, 'historical phase closure')
    return len(saved)


def verify_published_errors(folder, comparisons):
    checked = 0
    for metric in pd.read_csv(folder/'grouped_errors.csv', dtype={'value': str}).itertuples():
        group = comparisons[(comparisons.split == metric.split) & (comparisons.scope == metric.scope) &
                            (comparisons.model == metric.model)]
        if metric.stratum == 'airflow_band_mps':
            lower, upper = {'<4': (-np.inf, 4), '4_to_<8': (4, 8), '8_to_<12': (8, 12),
                            '>=12': (12, np.inf)}[metric.value]
            group = group[(group.onboard_airflow_mps >= lower) & (group.onboard_airflow_mps < upper)]
        elif metric.stratum != 'all':
            group = group[group[metric.stratum] == float(metric.value)]
        require(group.flight_id.nunique() == len(group) == metric.flights, 'CMU error-group count')
        for field, unit in [('mean_power', 'W'), ('energy', 'Wh')]:
            residual = group[f'estimated_{field}_{unit}'] - group[f'observed_{field}_{unit}']
            scores = {'MAE': abs(residual).mean(), 'RMSE': math.sqrt(np.mean(residual**2)), 'bias': residual.mean()}
            for score, value in scores.items():
                close(value, getattr(metric, score+'_'+unit), 'CMU '+score+' '+unit)
        checked += 1
    return checked


def verify_external(folder, data_root, snapshot):
    meta = json_file(folder/'validation.json')
    require(meta['source_sha256'] == SHA256 and meta['context_parent_sha256'] == REPRODUCTION_SHA256,
            'CMU public release identities')
    for name, digest in SHA256.items():
        verify_hash((folder if name == 'published_context.csv' else data_root)/name, digest)
    for name, digest in EXTERNAL_RELEASE.items():
        verify_hash(folder/name, digest)
    for name, digest in meta['runtime_config_sha256'].items():
        verify_hash(snapshot/'ros2_ws/src/drone_nav/config'/name, digest)
    manifest = indexed(folder/'flight_manifest.csv', 'flight_id')
    measured = indexed(folder/'measured_cruise.csv', 'flight_id')
    table = pd.read_csv(folder/'phase_comparisons.csv')
    base = table[table.model == 'published_phase']
    require(set(measured.index) == set(manifest[manifest.included].index), 'CMU complete flights')
    require(measured.split.value_counts().to_dict() == meta['measured_train_validation_test'], 'CMU split counts')
    require(table.groupby('flight_id').split.nunique().max() == 1, 'CMU split leakage')
    require(len(manifest) == meta['actual_recordings'] and len(measured) == meta['included_complete_flights'], 'CMU counts')
    library = ZipFile(data_root/'flights.zip')
    windows = 0
    with library:
        for fid, rows in base.groupby('flight_id'):
            require((rows.split == measured.at[fid,'split']).all(), f'CMU flight {fid} split')
            if manifest.at[fid,'published_role'] == 'holdout':
                require(measured.at[fid, 'split'] == 'test', f'CMU locked test {fid}')
            with library.open(f'flights/{fid}.csv') as flight:
                observations = pd.read_csv(flight, usecols=['time','battery_voltage','battery_current'])
            t = observations.time.to_numpy()
            p = (observations.battery_voltage*observations.battery_current).to_numpy()
            got = {}
            for rec in rows[rows.contiguous_windows==1].itertuples():
                cut = np.flatnonzero((t >= rec.window_first_time_s-1e-8) & (t <= rec.window_last_time_s+1e-8))
                require(len(cut)==rec.included_intervals+1 and np.all(np.diff(cut)==1), f'CMU endpoints {fid}/{rec.scope}')
                tv, pv = t[cut], p[cut]
                energy = math.fsum(np.diff(tv)*(pv[1:]+pv[:-1])/7200)
                duration = tv[-1]-tv[0]
                close(energy, rec.observed_energy_Wh, f'CMU electrical energy {fid}/{rec.scope}')
                close(energy*3600/duration, rec.observed_mean_power_W, f'CMU electrical power {fid}/{rec.scope}')
                close(duration, rec.covered_duration_s, f'CMU duration {fid}/{rec.scope}')
                if rec.scope == 'cruise':
                    close(energy*3600/duration, measured.at[fid,'observed_mean_power_W'], f'CMU ML target {fid}')
                got[rec.scope] = energy
                windows += 1
            close(sum(got[phase] for phase in PHASES), got['airborne'], f'CMU phase closure {fid}')
    return {'recordings':len(manifest), 'measured_complete_flights':len(measured), 'continuous_W_VI_windows':windows,
            'error_groups_recomputed':verify_published_errors(folder, table)}


def verify(root=DEFAULT, repo=REPO):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    snapshot = repo/'tools/m100_reaudit/runtime_snapshot'
    source_files = (snapshot/'SHA256SUMS').read_text().splitlines()
    for line in source_files:
        checksum, name = line.split('  ', maxsplit=1)
        verify_hash(snapshot/name, checksum)
    return {'gazebo':verify_gazebo(root/'gazebo_batch'), 'before_controls':verify_before(root/'before'),
            'frozen_historical_examples':verify_historical(root/'historical/replay'),
            'public_CMU':verify_external(root/'external', repo/'data/external/m100', snapshot),
            'portable_source_files':len(source_files)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT, help='results/m100_reaudit directory')
    parser.add_argument('--repo', type=Path, default=REPO, help='portable checkout with public CMU and bundled source')
    args=parser.parse_args()
    print(json.dumps(verify(args.root, args.repo), indent=2))
