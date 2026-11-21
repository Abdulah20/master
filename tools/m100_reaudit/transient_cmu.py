"""Local CMU pack-power estimator for near-level airborne manoeuvres.

Build from archived development records with ``python3 -B -m tools.m100_reaudit.transient_cmu``.
The prediction API reads the frozen support table; it does not import ROS or fit a model.
"""

from collections import deque
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/m100_reaudit/transient_airborne'
FIT = ROOT / 'results/m100_reaudit/cmu_calibration/fit.json'
REGIMES = ('speed_up', 'braking', 'low_speed_settling', 'settling', 'nearly_steady_motion')
PHASES = ('takeoff', 'cruise', 'landing')
PAYLOADS = (0.0, 0.25, 0.5)
FEATURES = ('speed_mps', 'along_accel_mps2', 'vertical_up_mps', 'low_speed_dwell_s', 'altitude_agl_m', 'tilt_deg')
COLS = ('flight_id', 'payload_kg', *FEATURES, 'power_w', 'weight_s', 'heading_unresolved',
        'regime_code', 'phase_code', 'source_time_s', 'commanded_altitude_m')
FID, MASS, SPEED, ACC, UP, DWELL, AGL, TILT, POWER, WT, UNRES, REG, PHASE, TIME, H_CMD = range(len(COLS))
POLICY = {
    'max_gap_s': 0.5, 'causal_accel_lag_s': 1.0, 'low_speed_mps': 0.6,
    'max_low_speed_dwell_s': 5.0, 'small_accel_mps2': 0.30, 'settling_speed_mps': 1.5,
    'max_speed_mps': 13.5, 'max_abs_accel_mps2': 3.0, 'max_abs_vertical_up_mps': 0.8,
    'absolute_agl_m': [10.0, 115.0], 'minimum_support_flights': 3,
    'minimum_kernel_weight_s_per_flight': 0.15,
    'kernel_half_widths': [1.0, 0.75, 0.45, 1.0, 12.5, 7.0], 'local_tilt_edge_tolerance_deg': 1.0,
    'local_dwell_edge_tolerance_s': 0.20, 'local_agl_edge_tolerance_m': 3.0,
    'gnss_heading_leg_s': 0.75, 'gnss_min_leg_m': 1.0, 'maximum_bend_deg': 20.0,
    'unresolved_heading_max_speed_mps': 2.0,
}


def finite(value):
    return isinstance(value, (float, int, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


class Kinematics:
    """Causal scalar-speed derivative on original seconds. Reset at airborne entry/clock restart.

    Replayed or older reports never refresh the estimate. History restarts after a >0.5 s gap.
    A slow period already in progress at first observation has unknown duration until motion resumes.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = deque()
        self.slow_since = None
        self.previously_low_level = None

    def update(self, time_s, speed_mps, vertical_up_mps):
        if not all(map(finite, (time_s, speed_mps, vertical_up_mps))) or speed_mps < 0:
            self.reset()
            return None
        t, s = float(time_s), float(speed_mps)
        if self.history and t <= self.history[-1][0]:
            return None
        if self.history and t - self.history[-1][0] > POLICY['max_gap_s'] + 1e-9:
            self.reset()
        low_level = s < POLICY['low_speed_mps'] and abs(vertical_up_mps) <= POLICY['max_abs_vertical_up_mps']
        if not low_level:
            self.slow_since = None
        elif self.previously_low_level is False:
            self.slow_since = t  # First observed point meeting both conditions.
        self.previously_low_level = low_level
        self.history.append((t, s))
        lag = POLICY['causal_accel_lag_s']
        cutoff = t - lag
        while len(self.history) >= 3 and self.history[1][0] <= cutoff:
            self.history.popleft()
        if len(self.history) < 2 or self.history[0][0] > cutoff + 1e-9:
            return None
        (a, v0), (b, v1) = self.history[0], self.history[1]
        previous_speed = v0 + (v1 - v0) * (cutoff - a) / (b - a)
        dwell = (t - self.slow_since if self.slow_since is not None else None) if low_level else 0.0
        return {'time_s': t, 'speed_mps': s, 'along_accel_mps2': (s - previous_speed) / lag,
                'vertical_up_mps': float(vertical_up_mps), 'low_speed_dwell_s': dwell}


def regime(speed, accel):
    if accel >= POLICY['small_accel_mps2']:
        return 'speed_up'
    if accel <= -POLICY['small_accel_mps2']:
        return 'braking'
    if speed < POLICY['low_speed_mps']:
        return 'low_speed_settling'
    return 'settling' if speed < POLICY['settling_speed_mps'] else 'nearly_steady_motion'


def eligibility(payload, s, a, up, dwell, h, tilt):
    if not all(map(finite, (s, a, up))) or s < 0:
        return 'NONFINITE_KINEMATICS'
    if payload not in PAYLOADS:
        return 'UNSUPPORTED_PAYLOAD'
    if abs(up) > POLICY['max_abs_vertical_up_mps']:
        return 'VERTICAL_OUTSIDE_NEAR_LEVEL'
    if s > POLICY['max_speed_mps'] or abs(a) > POLICY['max_abs_accel_mps2']:
        return 'AGGRESSIVE_OR_OUTSIDE_SPEED_ACCEL'
    if not finite(dwell) or not 0.0 <= dwell <= POLICY['max_low_speed_dwell_s']:
        return 'LONG_OR_UNKNOWN_LOW_SPEED_DWELL'
    lo, hi = POLICY['absolute_agl_m']
    if not finite(h) or not lo <= h <= hi:
        return 'ALTITUDE_OUTSIDE_FIELD_SUPPORT'
    if not finite(tilt) or not 0 <= tilt < 90.0:
        return 'TILT_UNAVAILABLE_OR_INVALID'
    return None


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_model(path=OUT / 'model.json'):
    path = Path(path)
    model = json.loads(path.read_text())
    model['_support_directory'] = str(path.resolve().parent)
    return model


@lru_cache(maxsize=4)
def _index(path, digest):
    path = Path(path)
    if sha(path) != digest:
        raise ValueError('CMU field support checksum differs from the frozen model')
    with np.load(path, allow_pickle=False) as archive:
        rows = archive['observations']
    return {(p, g): group[np.argsort(group[:, SPEED], kind='stable')]
            for p in PAYLOADS for g, name in enumerate(REGIMES)
            if len(group := rows[(rows[:, MASS] == p) & (rows[:, REG] == g)])}


def predict(model, *, payload_kg, speed_mps, along_accel_mps2, vertical_up_mps, low_speed_dwell_s,
            altitude_agl_m=None, tilt_deg=None, _exclude_flight=None, _describe=True):
    s, a, up, dwell, h = speed_mps, along_accel_mps2, vertical_up_mps, low_speed_dwell_s, altitude_agl_m
    name = regime(s, a) if finite(s) and finite(a) else None
    result = {'power_w': None, 'reason': eligibility(payload_kg, s, a, up, dwell, h, tilt_deg), 'regime': name,
              'support_flights': 0, 'support_ids': []}
    if result['reason']:
        return result
    if model['policy'] != POLICY or tuple(model['columns']) != COLS:
        raise ValueError('Incompatible CMU transient model policy or support columns')
    path = Path(model.get('_support_directory', OUT)) / model['support_file']
    group = _index(str(path), model['support_sha256']).get((payload_kg, REGIMES.index(name)))
    result['reason'] = 'INSUFFICIENT_LOCAL_FLIGHTS'
    if group is None:
        return result
    widths = model['policy']['kernel_half_widths']
    lo, hi = np.searchsorted(group[:, SPEED], (s - widths[0], s + widths[0]))
    candidates = group[lo:hi]
    diff = np.abs((candidates[:, SPEED:TILT + 1] - (s, a, up, dwell, h, tilt_deg)) / widths)
    near = np.max(diff, axis=1) < 1.0
    if _exclude_flight is not None:
        near &= candidates[:, FID] != _exclude_flight
    candidates = candidates[near]
    if not len(candidates):
        return result
    # Compact product kernel (Epanechnikov per dimension) on *joint observed* operating points.
    weights = np.prod(1.0 - diff[near]**2, axis=1) * candidates[:, WT]
    ids = candidates[:, FID].astype(int)
    support_weight = np.bincount(ids, weights)
    flight_ids = np.flatnonzero(support_weight >= POLICY['minimum_kernel_weight_s_per_flight'])
    result['support_flights'] = len(flight_ids)
    if _describe:
        result['support_ids'] = flight_ids.tolist()
    if len(flight_ids) < POLICY['minimum_support_flights']:
        return result
    kept = np.isin(ids, flight_ids)
    tilt_min = float(candidates[kept, TILT].min())
    tilt_max = float(candidates[kept, TILT].max())
    slack = POLICY['local_tilt_edge_tolerance_deg']
    if _describe:
        result['support_tilt_deg'] = [tilt_min, tilt_max]
    if not tilt_min - slack <= tilt_deg <= tilt_max + slack:
        result['reason'] = 'OUTSIDE_LOCAL_TILT_RANGE'
        return result
    agl_range = [float(candidates[kept, AGL].min()), float(candidates[kept, AGL].max())]
    if not agl_range[0] - POLICY['local_agl_edge_tolerance_m'] <= h <= agl_range[1] + POLICY['local_agl_edge_tolerance_m']:
        result['reason'] = 'ALTITUDE_OUTSIDE_FIELD_SUPPORT'
        return result
    if s < POLICY['low_speed_mps'] and dwell > candidates[kept, DWELL].max() + POLICY['local_dwell_edge_tolerance_s']:
        result['reason'] = 'DWELL_EXCEEDS_LOCAL_FIELD'
        return result
    weighted_power = np.bincount(ids, weights * candidates[:, POWER])
    flight_estimates = weighted_power[flight_ids] / support_weight[flight_ids]
    result.update(power_w=float(flight_estimates.mean()), reason='SUPPORTED_LOCAL_FIELD')
    if _describe:
        unseen_heading = np.bincount(ids, weights * candidates[:, UNRES])
        result.update(support_actual_agl_m=agl_range,
                      support_planned_heights_m=np.unique(candidates[kept, H_CMD]).tolist(),
                      support_flight_estimate_range_w=[float(flight_estimates.min()), float(flight_estimates.max())],
                      heading_unresolved_weight_fraction=float(np.mean(unseen_heading[flight_ids] / support_weight[flight_ids])))
    return result


def predict_kinematics(model, payload_kg, kinematics, altitude_agl_m, tilt_deg=None):
    if kinematics is None:
        return {'power_w': None, 'reason': 'INSUFFICIENT_CAUSAL_HISTORY', 'regime': None, 'support_flights': 0, 'support_ids': []}
    return predict(model, payload_kg=payload_kg, altitude_agl_m=altitude_agl_m, tilt_deg=tilt_deg,
                   **{k: kinematics.get(k) for k in FEATURES[:4]})


def heading_filter(t, east, north):
    leg = POLICY['gnss_heading_leg_s']
    x1, x0 = np.interp(t - leg, t, east), np.interp(t - 2*leg, t, east)
    y1, y0 = np.interp(t - leg, t, north), np.interp(t - 2*leg, t, north)
    ax, ay, bx, by = x1 - x0, y1 - y0, east - x1, north - y1
    n1, n2 = np.hypot(ax, ay), np.hypot(bx, by)
    resolved = (t >= t[0] + 2*leg) & (n1 >= POLICY['gnss_min_leg_m']) & (n2 >= POLICY['gnss_min_leg_m'])
    bend = resolved & (ax*bx + ay*by < n1*n2*math.cos(math.radians(POLICY['maximum_bend_deg'])))
    return resolved, bend


def flight_rows(fid, config, samples):
    from . import external
    from .pack_drive_cmu_context import gnss_track

    duplicate_count = int(samples.duplicated().sum())
    samples = samples.drop_duplicates().reset_index(drop=True)
    if samples.time.duplicated().any():
        raise ValueError(f'Conflicting data at the same timestamp in flight {fid}')
    if flags := external.quality_flags(samples):
        raise ValueError(f'Unexpected source quality for flight {fid}: {flags}')
    t = samples.time.to_numpy()
    z = samples.position_z.to_numpy()
    h = z - np.median(z[t <= 5.0])
    s = np.hypot(samples.velocity_x, samples.velocity_y).to_numpy()
    up = samples.velocity_z.to_numpy()
    quaternion = samples[['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']].to_numpy()
    norm = np.linalg.norm(quaternion, axis=1)
    nonzero = norm > 1e-6
    tilt = np.full(len(t), np.nan)
    cosine = 1.0 - 2.0 * np.sum(quaternion[nonzero, :2]**2, axis=1) / norm[nonzero]**2
    tilt[nonzero] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    power = (samples.battery_voltage * samples.battery_current).to_numpy()  # Signed as recorded.
    _, east, north = gnss_track(samples)
    resolved, bend = heading_filter(t, east, north)
    start, level, descent, end = external.phase_boundaries(z)
    airborne = np.arange(start + 1, end)
    sign_witness = float(np.dot(external.midpoint(up)[start + 1:end - 1], np.diff(z)[start + 1:end - 1]))
    if sign_witness <= 0.0:
        raise ValueError(f'Unresolved upward vertical velocity sign in flight {fid}')
    # Correlate independent altitude change and the concomitant velocity over a 1 s trailing window.
    dz_1s = z - np.interp(t - 1.0, t, z)
    moving_z = airborne[np.abs(dz_1s[airborne]) > 0.7]
    rho = float(np.corrcoef(dz_1s[moving_z], up[moving_z])[0, 1]) if len(moving_z) >= 3 else None
    payload, height = float(config.payload) / 1000.0, float(config.altitude)
    kin = Kinematics()
    labels = np.full(len(t), -1, int)
    observations = np.zeros((len(t), len(COLS)))
    criteria = dict(near_level_candidates=0, definite_turn_samples=0, unresolved_moving_samples=0, nonpositive_pack_samples=0)
    for j in airborne:
        f = kin.update(t[j], s[j], up[j])
        if f is None or abs(h[j] - height) > max(3.0, 0.10 * height):
            continue
        acc, dwell = f['along_accel_mps2'], f['low_speed_dwell_s']
        if eligibility(payload, s[j], acc, up[j], dwell, h[j], tilt[j]):
            continue
        criteria['near_level_candidates'] += 1
        if bend[j]:
            criteria['definite_turn_samples'] += 1
            continue
        if not resolved[j] and s[j] >= POLICY['unresolved_heading_max_speed_mps']:
            criteria['unresolved_moving_samples'] += 1
            continue
        if power[j] <= 0:
            criteria['nonpositive_pack_samples'] += 1
            continue
        code = REGIMES.index(regime(s[j], acc))
        labels[j] = code
        phase = 0 if j < level else 2 if j >= descent else 1
        observations[j] = (fid, payload, s[j], acc, up[j], dwell, h[j], tilt[j], power[j], 0.,
                           not resolved[j], code, phase, t[j], height)
    # Each original adjacent interval contributes half its duration to either endpoint,
    # only while both points lie in the same accepted regime. No bridging gaps or turning intervals.
    dt = np.diff(t) * ((labels[:-1] >= 0) & (labels[:-1] == labels[1:]))
    observations[:-1, WT] += dt / 2
    observations[1:, WT] += dt / 2
    selected = observations[observations[:, WT] > 0]
    return selected, {**criteria, 'flight_id': fid, 'identical_duplicate_rows_removed': duplicate_count,
                      'upward_velocity_dot_altitude_change': sign_witness,
                      'vertical_velocity_altitude_rate_correlation': rho, 'vertical_direction_witness_rows': len(moving_z),
                      'retained_rows': len(selected), 'retained_s': float(selected[:, WT].sum())}


def cohort_data(split_ids, split, metadata, manifest, archive):
    import pandas as pd
    from . import external

    if len(split_ids) != len(set(split_ids)):
        raise ValueError(f'Duplicate {split} flight membership')
    matrices, diagnostics = [], []
    for fid in split_ids:
        spec = manifest.loc[fid]
        if not spec.included or spec['split'] != split or spec.published_role != 'fit':
            raise ValueError(f'Outside the registered development cohort: {fid}')
        with archive.open(f'flights/{fid}.csv') as record:
            samples = pd.read_csv(record, usecols=[*external.COLUMNS, 'position_x', 'position_y',
                                                   'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'])
        rows, info = flight_rows(fid, metadata.loc[fid], samples)
        matrices.append(rows)
        diagnostics.append(dict(split=split, **info))
    return np.concatenate(matrices), diagnostics


def residual_tables(model, data, split):
    import pandas as pd

    rows = []
    for observation in data:
        fid = int(observation[FID])
        prediction = predict(model, payload_kg=observation[MASS],
                             **dict(zip(FEATURES, observation[SPEED:TILT + 1], strict=True)),
                             _exclude_flight=fid if split == 'train_leave_one_flight_out' else None, _describe=False)
        error = np.nan if prediction['power_w'] is None else prediction['power_w'] - observation[POWER]
        rows.append((fid, float(observation[MASS]), REGIMES[int(observation[REG])], PHASES[int(observation[PHASE])],
                     observation[WT], prediction['reason'], error))
    frame = pd.DataFrame(rows, columns=['flight_id', 'payload_kg', 'regime', 'source_phase', 'weight_s', 'reason', 'error'])
    detail = []
    for (fid, payload, name, phase), g in frame.groupby(['flight_id', 'payload_kg', 'regime', 'source_phase']):
        eligible = g[g.error.notna()]
        w, e = eligible.weight_s.to_numpy(), eligible.error.to_numpy()
        span = w.sum()
        detail.append({'split': split, 'flight_id': fid, 'payload_kg': payload, 'regime': name, 'source_phase': phase,
                       'eligible_field_s': g.weight_s.sum(), 'supported_field_s': span, 'supported_points': len(w),
                       'bias_w': np.dot(w, e)/span if span else np.nan,
                       'mae_w': np.dot(w, abs(e))/span if span else np.nan,
                       'rmse_w': np.sqrt(np.dot(w, e*e)/span) if span else np.nan})
    results = []
    for tag in ('all', *REGIMES):
        region = frame if tag == 'all' else frame[frame.regime == tag]
        metrics = []
        for fid, g in region[region.error.notna()].groupby('flight_id'):
            w = g.weight_s.to_numpy() / g.weight_s.sum()
            e = g.error.to_numpy()
            metrics.append((np.dot(w, e), np.dot(w, abs(e)), np.dot(w, e**2)))
        mean = np.mean(metrics, axis=0) if metrics else [None]*3
        supported = region[region.error.notna()].weight_s.sum()
        total = region.weight_s.sum()
        results.append({'split': split, 'regime': tag, 'eligible_flights': region.flight_id.nunique(),
                        'flights_with_supported_points': len(metrics), 'eligible_field_s': total,
                        'supported_field_s': supported, 'local_coverage_pct_of_eligible_time': 100*supported/total if total else None,
                        'equal_flight_bias_w': mean[0], 'equal_flight_mae_w': mean[1],
                        'equal_flight_rmse_w': float(np.sqrt(mean[2])) if metrics else None})
    return detail, results


def build():
    import pandas as pd
    from . import external

    base = json.loads(FIT.read_text())
    sources = (external.INPUT/'parameters.csv', external.INPUT/'flights.zip', external.OUTPUT/'flight_manifest.csv')
    for path in sources:
        external.check_hash(path, base['input_and_code_sha256'][str(path.relative_to(ROOT))])
    metadata = pd.read_csv(sources[0]).set_index('flight')
    manifest = pd.read_csv(sources[2]).set_index('flight_id')
    train = base['cohorts']['train']['flight_ids']
    val = base['cohorts']['validation']['flight_ids']
    if (not metadata.index.is_unique or not manifest.index.is_unique or len(train) != 75 or len(val) != 35
            or set(train).intersection(val)):
        raise ValueError('Expected frozen disjoint development cohorts (75/35) and unique IDs')
    OUT.mkdir(parents=True, exist_ok=True)
    with ZipFile(sources[1]) as archive:
        data, diagnostics = cohort_data(train, 'train', metadata, manifest, archive)
        source_file = OUT/'field_support.npz'
        np.savez_compressed(source_file, observations=data)
        model = {
            'schema': 'CMU_LOCAL_NEAR_LEVEL_AIRCRAFT_PACK_POWER_V1', 'classification': 'LIMITED_EMPIRICAL_PROXY',
            'target': 'Signed measured pack electrical power (voltage × current) on accepted nearby field samples; positive discharge.',
            'source': 'Rodrigues et al., Carnegie Mellon University, https://doi.org/10.1184/R1/12683453 (CC BY 4.0)',
            'fit_ids': sorted(train), 'inputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (*sources, FIT, Path(__file__).resolve(), Path(external.__file__), Path(__file__).with_name('pack_drive_cmu_context.py').resolve())},
            'support_file': source_file.name, 'support_sha256': sha(source_file), 'columns': list(COLS),
            'regimes': list(REGIMES), 'source_phases': list(PHASES), 'features': list(FEATURES), 'policy': POLICY,
            'estimator': 'Same payload and regime; multiply the six compact (1−u²) factors evaluated at each measured operating point by its observed time weight. Normalize within each flight, require 0.15 equivalent kernel-seconds per flight, then average qualifying flight means equally. At least three distinct flight IDs are required. No calibration on validation, locked flights, PX4 traces or historical 110-flight model.',
            'field_alignment': 'Original strictly increasing timestamps, scalar horizontal speed, signed causal 1 s speed difference, simultaneous upward velocity (altitude-sign witness), normalized-quaternion tilt. AGL uses median altitude in first 5 s. Begin kinematics at first accepted airborne index. Include only actual AGL within max(3 m, 10% of commanded AGL) of planned plateau. Source phase is diagnostic only. Low dwell starts at first sampled entry with BOTH speed <0.6 m/s AND |up|<=0.8 m/s.',
            'turn_filter': 'WGS84 local EN GNSS heading between consecutive trailing 0.75 s legs with >=1 m displacement. Remove bends >20 degrees. Remove unresolved headings for speed >=2 m/s. Slower samples can have unresolved heading; return their locally weighted fraction.',
            'coverage_limits': 'Local actual AGL, tilt and dwell have additional observed-edge checks. Identical repeated rows are dropped, conflicting same-time rows or time reversal reject the source flight. Online repeated/out-of-order timestamps provide no new observation. Long or initially censored stops, aggressive acceleration, turns, ground/launch and landing away from the plateau are not estimated. Terrain slope, weather, component delivery and battery identity are unresolved. Field sample fit does not validate transfer to the simulator or identify clipped load.',
        }
        (OUT/'model.json').write_text(json.dumps(model, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        support = []
        for p in PAYLOADS:
            for g, name in enumerate(REGIMES):
                pts = data[(data[:, MASS] == p) & (data[:, REG] == g)]
                ids = np.unique(pts[:, FID]).astype(int).tolist()
                support.append({'payload_kg': p, 'regime': name, 'field_flights': len(ids),
                                'field_ids': ' '.join(map(str, ids)), 'field_points': len(pts),
                                'field_s': float(pts[:, WT].sum()),
                                'unresolved_heading_s': float(np.dot(pts[:, WT], pts[:, UNRES])),
                                'max_observed_slow_level_dwell_s': float(pts[:, DWELL].max()) if len(pts) else None})
        pd.DataFrame(support).to_csv(OUT/'support_by_payload_regime.csv', index=False, float_format='%.9g')
        print(f'Frozen train-only model: {len(data)} points, {len(np.unique(data[:, FID]))} flights, {source_file.stat().st_size:,} bytes', flush=True)
        # Validation signals are first opened here, after both model files are closed.
        val_data, val_diag = cohort_data(val, 'validation', metadata, manifest, archive)
    train_rows, train_summary = residual_tables(model, data, 'train_leave_one_flight_out')
    val_rows, val_summary = residual_tables(model, val_data, 'validation')
    pd.DataFrame(diagnostics + val_diag).to_csv(OUT/'source_coverage_by_flight.csv', index=False, float_format='%.9g')
    pd.DataFrame(train_rows + val_rows).to_csv(OUT/'residuals_by_flight_phase_regime.csv', index=False, float_format='%.9g')
    pd.DataFrame(train_summary + val_summary).to_csv(OUT/'residuals_by_regime.csv', index=False, float_format='%.9g')
    print('Development summaries written. Errors are predicted minus measured W, conditional on support.', flush=True)


if __name__ == '__main__':
    build()
