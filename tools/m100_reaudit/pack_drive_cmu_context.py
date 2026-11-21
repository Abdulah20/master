"""Measured CMU context for the calm, 15 C reference, 25 m / 6 m/s simulator plan.

Run from the thesis root with ``python3 -m tools.m100_reaudit.pack_drive_cmu_context``.
Only the three programmed variables (height, speed, payload) are matched. Weather,
pack identity, initial charge, flight direction and equivalent routes are unknown.
Per-flight results use original adjacent timestamps. Group summaries give equal
weight to each available flight and retain the observed range of flight means.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from . import external


ROOT = Path(__file__).resolve().parents[2]
FIT = ROOT / 'results/m100_reaudit/cmu_calibration/fit.json'
OUT = ROOT / 'results/m100_reaudit/pack_drive_three'
PAYLOADS = (0.0, 0.25, 0.5)
SPEED, HEIGHT, LENGTH = 6.0, 25.0, 100.0
CRUISE = 'rodrigues_cruise_including_turns'
SEGMENT = 'first_contiguous_steady_straight_100m_crossing'


def gnss_track(samples):
    """Local east/north offsets in m from recorded WGS84 longitude and latitude."""
    lon, lat = (np.radians(samples[k].to_numpy()) for k in ('position_x', 'position_y'))
    a = 6378137.0  # WGS84 semi-major axis in m
    e2 = 6.6943799901413165e-3
    denom = 1.0 - e2 * np.sin(lat[0])**2
    east = (lon - lon[0]) * a * np.cos(lat[0]) / np.sqrt(denom)
    north = (lat - lat[0]) * a * (1.0 - e2) / denom**1.5
    return np.r_[0.0, np.cumsum(np.hypot(np.diff(east), np.diff(north)))], east, north


def straight_segment(samples, level, descent, agl, track):
    """First acceptable whole-sample window. No voltage, current or power enters selection."""
    vx, vy, vz = (samples[k].to_numpy() for k in ('velocity_x', 'velocity_y', 'velocity_z'))
    speed = np.hypot(vx, vy)
    # Same speed, height and vertical velocity tolerances as external.measured_scopes.
    good = (abs(speed - SPEED) <= max(0.5, 0.1 * SPEED)) & (abs(agl - HEIGHT) <= max(3.0, 0.1 * HEIGHT)) & (abs(vz) <= 1.5)
    rejected = np.r_[0, np.cumsum(~good)]
    path, x, y = track
    for start in range(level, descent):
        if not good[start]:
            continue
        # Find the first recorded endpoint crossing 100 m of net displacement. Absolute
        # GNSS positions retain turns even when the flight velocity frame rotates.
        chord = np.hypot(x[start:descent + 1] - x[start], y[start:descent + 1] - y[start])
        crossing = np.flatnonzero(chord >= LENGTH)
        if not crossing.size:
            continue
        stop = start + int(crossing[0])
        if rejected[stop + 1] == rejected[start] and chord[crossing[0]] >= 0.99 * (path[stop] - path[start]):
            return start, stop
    return None


def moments(values, dt):
    """Time mean and population SD of the piecewise-linear samples on irregular dt."""
    mean = float(np.dot(dt, external.midpoint(values)) / dt.sum())
    a, b = values[:-1] - mean, values[1:] - mean
    variance = np.dot(dt, a*a + a*b + b*b) / (3.0 * dt.sum())
    return mean, float(np.sqrt(variance))


def measure(samples, bounds, agl, track):
    if bounds is None:
        return {'available': False}
    start, stop = bounds
    window = samples.iloc[start:stop + 1]
    t = window['time'].to_numpy()
    dt = np.diff(t)
    voltage = window['battery_voltage'].to_numpy()
    current = window['battery_current'].to_numpy()
    power = voltage * current
    v, v_sd = moments(voltage, dt)
    i, i_sd = moments(current, dt)
    p, p_sd = moments(power, dt)
    path, x, y = track
    length = float(path[stop] - path[start])
    return {
        'available': True, 'start_s': t[0], 'end_s': t[-1], 'seconds': dt.sum(), 'samples': len(t),
        'horizontal_path_m': length, 'horizontal_chord_m': np.hypot(x[stop] - x[start], y[stop] - y[start]),
        'mean_horizontal_speed_mps': moments(np.hypot(window['velocity_x'], window['velocity_y']).to_numpy(), dt)[0],
        'mean_height_agl_m': moments(agl[start:stop + 1], dt)[0],
        'mean_onboard_airflow_mps': moments(window['wind_speed'].to_numpy(), dt)[0],
        'mean_voltage_V': v, 'within_flight_sd_V': v_sd, 'mean_current_A': i, 'within_flight_sd_A': i_sd,
        'mean_power_W': p, 'within_flight_sd_W': p_sd, 'within_flight_power_cv_pct': 100*p_sd/p,
        'sample_min_power_W': power.min(), 'sample_max_power_W': power.max(), 'window_energy_Wh': p*dt.sum()/3600,
    }


def field_context():
    fit = json.loads(FIT.read_text())
    archive_path = external.INPUT / 'flights.zip'
    parameters_path = external.INPUT / 'parameters.csv'
    manifest_path = external.OUTPUT / 'flight_manifest.csv'
    for path in (archive_path, parameters_path, manifest_path):
        external.check_hash(path, fit['input_and_code_sha256'][str(path.relative_to(ROOT))])
    manifest = pd.read_csv(manifest_path).set_index('flight_id')
    params = pd.read_csv(parameters_path).set_index('flight')
    if not manifest.index.is_unique or not params.index.is_unique:
        raise ValueError('Duplicate flight ID in development metadata')
    result, traces = [], []
    # The frozen fit contains only development IDs. Filter public command metadata before opening any signal member.
    with ZipFile(archive_path) as archive:
        for split in ('train', 'validation'):
            for fid in fit['cohorts'][split]['flight_ids']:
                assignment, config = manifest.loc[fid], params.loc[fid]
                if assignment.split != split or assignment.published_role != 'fit' or not assignment.included:
                    raise ValueError(f'Inconsistent development membership for {fid}')
                payload = float(config.payload) / 1000.0
                if payload not in PAYLOADS or float(config.speed) != SPEED or float(config.altitude) != HEIGHT:
                    continue
                with archive.open(f'flights/{fid}.csv') as raw:
                    samples = pd.read_csv(raw, usecols=[*external.COLUMNS, 'position_x', 'position_y'])
                if flags := external.quality_flags(samples):
                    raise ValueError(f'Unexpected recording quality flags for {fid}: {flags}')
                t = samples['time'].to_numpy()
                z = samples['position_z'].to_numpy()
                agl = z - np.median(z[t <= 5])
                track = gnss_track(samples)
                _, level, descent, _ = external.phase_boundaries(z)
                straight = straight_segment(samples, level, descent, agl, track)
                if straight is not None:
                    a, b = straight
                    cut = samples.iloc[a:b + 1]
                    traces.append(pd.DataFrame({
                        'flight_id': fid, 'split': split, 'payload_kg': payload,
                        'elapsed_s': cut['time'] - t[a], 'source_time_s': cut['time'],
                        'measured_power_w': cut['battery_voltage'] * cut['battery_current'],
                        'measured_voltage_v': cut['battery_voltage'], 'measured_current_a': cut['battery_current'],
                    }))
                for scope, bounds in [(CRUISE, (level, descent)), (SEGMENT, straight)]:
                    result.append({'flight_id': fid, 'split': split, 'published_role': assignment.published_role,
                                   'publisher_date': config.date, 'route': config.route, 'payload_kg': payload,
                                   'commanded_speed_mps': SPEED, 'commanded_altitude_m': HEIGHT, 'scope': scope,
                                   **measure(samples, bounds, agl, track)})
    trace_columns = ['flight_id', 'split', 'payload_kg', 'elapsed_s', 'source_time_s',
                     'measured_power_w', 'measured_voltage_v', 'measured_current_a']
    curves = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame(columns=trace_columns)
    return pd.DataFrame(result), curves


def summarize(flights):
    rows = []
    for payload in PAYLOADS:
        for scope in (CRUISE, SEGMENT):
            selected = flights[(flights.payload_kg == payload) & (flights.scope == scope)]
            valid = selected[selected.available]
            row = {'payload_kg': payload, 'commanded_speed_mps': SPEED, 'commanded_altitude_m': HEIGHT,
                   'scope': scope, 'exact_command_flights': len(selected), 'flights_with_window': len(valid),
                   'train_ids': ' '.join(map(str, valid.loc[valid.split == 'train', 'flight_id'])),
                   'validation_ids': ' '.join(map(str, valid.loc[valid.split == 'validation', 'flight_id']))}
            for col in ('mean_power_W', 'mean_voltage_V', 'mean_current_A', 'within_flight_sd_W',
                        'within_flight_power_cv_pct', 'seconds', 'horizontal_path_m', 'window_energy_Wh'):
                row[f'flight_mean_{col}'] = valid[col].mean()
                row[f'flight_min_{col}'] = valid[col].min()
                row[f'flight_max_{col}'] = valid[col].max()
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    flights, curves = field_context()
    flights = flights.sort_values(['payload_kg', 'flight_id', 'scope'])
    OUT.mkdir(parents=True, exist_ok=True)
    flights.to_csv(OUT / 'cmu_context_flights.csv', index=False, float_format='%.10g')
    summarize(flights).to_csv(OUT / 'cmu_context.csv', index=False, float_format='%.10g')
    (OUT / 'data').mkdir(exist_ok=True)
    curves.sort_values(['payload_kg', 'flight_id', 'elapsed_s']).to_csv(
        OUT / 'data/cmu_windows.csv.gz', index=False, float_format='%.11g', compression={'method': 'gzip', 'mtime': 0})
