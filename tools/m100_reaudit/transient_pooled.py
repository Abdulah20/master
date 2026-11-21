"""Pooled near-level CMU pack-power regression. Build with python3 -B -m tools.m100_reaudit.transient_pooled."""

import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from . import transient_cmu as field


OUT = field.OUT
MODEL_PATH = OUT / 'pooled_model.json'
PHYSICS = field.ROOT / 'config/m100_reaudit/physics.json'
SCHEMA = 'CMU_POOLED_NEAR_LEVEL_AIRCRAFT_PACK_POWER_V1'
TERMS = ['intercept_0kg_W', 'intercept_0.25kg_W', 'intercept_0.5kg_W', 'horizontal_speed',
         'horizontal_speed_squared', 'signed_vertical_work', 'positive_horizontal_work',
         'braking_horizontal_work_magnitude', 'sec_tilt_minus_one']
FLAGS = {'speed_mps': 'HORIZONTAL_SPEED', 'along_accel_mps2': 'ACCELERATION',
         'vertical_up_mps': 'VERTICAL_SPEED', 'tilt_deg': 'TILT',
         'positive_horizontal_work': 'ACCELERATION_WORK', 'braking_horizontal_work_magnitude': 'BRAKING_WORK',
         'power_w': 'OUTPUT_POWER'}
G = 9.80665


def feature_rows(payload, speed, acceleration, up, tilt, base_mass):
    mass = base_mass + payload
    kinetic_work = mass * speed * acceleration
    return np.column_stack([*(payload == p for p in field.PAYLOADS), speed, speed**2, mass * G * up,
                            np.maximum(kinetic_work, 0.0), np.maximum(-kinetic_work, 0.0),
                            1.0 / np.cos(np.radians(tilt)) - 1.0])


def fit(data, base_mass):
    x = feature_rows(data[:, field.MASS], data[:, field.SPEED], data[:, field.ACC], data[:, field.UP],
                     data[:, field.TILT], base_mass)
    y = data[:, field.POWER]
    ids = np.unique(data[:, field.FID])
    weights = np.zeros(len(y))
    for fid in ids:
        in_flight = data[:, field.FID] == fid
        regimes = np.unique(data[in_flight, field.REG])
        for g in regimes:
            take = in_flight & (data[:, field.REG] == g)
            seconds = data[take, field.WT]
            weights[take] = seconds / (seconds.sum() * len(regimes) * len(ids))
    weighted_design = x * np.sqrt(weights)[:, None]
    scales = np.sqrt(np.sum(weighted_design**2, axis=0))
    if np.any(scales <= 0):
        raise ValueError('Empty training design term')
    coeff, _, rank, sv = np.linalg.lstsq(weighted_design / scales, np.sqrt(weights) * y, rcond=None)
    if rank != len(TERMS):
        raise ValueError('Pooled training design is rank deficient')
    model = {'schema': SCHEMA, 'scope': 'POOLED_FIELD_REGRESSION_MARGINAL_SUPPORT',
             'terms': TERMS, 'coefficients': dict(zip(TERMS, (coeff / scales).tolist(), strict=True)),
             'design_rank': int(rank), 'rms_scaled_condition_number': float(sv[0] / sv[-1]),
             'base_physical_mass_kg': base_mass, 'gravity_mps2': G,
             'fit_weighting': 'Equal total weight per contributing training flight; within flight equal weight per observed manoeuvre regime; within flight/regime weight by original adjacent-interval endpoint seconds.',
             'equation': 'P[W]=intercept(payload)+b1*s+b2*s²+b3*m*g*v_up+b4*max(m*s*a,0)+b5*max(-m*s*a,0)+b6*(sec(tilt)-1). m is instrumented aircraft physical mass including payload. Coefficients apply to net pack demand. Component shaft power and motor/ESC efficiency are not targets.',
             'application': 'Use reliable airborne kinematics and source timestamps. Scalar speed acceleration is the shared causal 1 s difference. For each trained payload, bound speed/rate/up/tilt, then both horizontal work features and positive output power to their observed training intervals. Flag altitude band gaps and slow, near-level dwell beyond the observed duration. Dwell and altitude describe transfer; neither contributes a regression term. Input bounds are marginal. They do not establish close joint field analogues.',
             'payload_support': {}}
    for p in field.PAYLOADS:
        take = data[:, field.MASS] == p
        group = data[take]
        values = {'speed_mps': group[:, field.SPEED], 'along_accel_mps2': group[:, field.ACC],
                  'vertical_up_mps': group[:, field.UP], 'tilt_deg': group[:, field.TILT],
                  'positive_horizontal_work': x[take, 6], 'braking_horizontal_work_magnitude': x[take, 7],
                  'power_w': group[:, field.POWER]}
        bounds = {k: [float(v.min()), float(v.max())] for k, v in values.items()}
        heights = {}
        for plan in np.unique(group[:, field.H_CMD]):
            agl = group[group[:, field.H_CMD] == plan, field.AGL]
            heights[f'{plan:g}'] = [float(agl.min()), float(agl.max())]
        slow = group[group[:, field.SPEED] < field.POLICY['low_speed_mps'], field.DWELL]
        model['payload_support'][f'{p:g}'] = {'ranges': bounds, 'training_flights': np.unique(group[:, field.FID]).astype(int).tolist(),
            'retained_training_points': len(group), 'retained_training_s': float(group[:, field.WT].sum()),
            'actual_agl_m_by_planned_altitude_m': heights,
            'max_observed_slow_level_dwell_s': float(slow.max()) if len(slow) else None}
    return model


def load_model(path=MODEL_PATH):
    model = json.loads(Path(path).read_text())
    if model['schema'] != SCHEMA or model['terms'] != TERMS:
        raise ValueError('Incompatible pooled power model')
    return model


def predict(model, *, payload_kg, speed_mps, along_accel_mps2, vertical_up_mps, low_speed_dwell_s,
            altitude_agl_m=None, tilt_deg=None):
    result = {'power_w': None, 'reason': 'INVALID_OR_MISSING_KINEMATICS', 'regime': None,
              'transfer_flags': [], 'support_basis': 'POOLED_FIELD_REGRESSION_MARGINAL_SUPPORT'}
    if payload_kg not in field.PAYLOADS:
        result['reason'] = 'UNSUPPORTED_PAYLOAD'
        return result
    if not all(map(field.finite, (speed_mps, along_accel_mps2, vertical_up_mps, altitude_agl_m, tilt_deg))):
        return result
    if speed_mps < 0 or not 0 <= tilt_deg < 90 or altitude_agl_m <= 0:
        return result
    if low_speed_dwell_s is not None and (not field.finite(low_speed_dwell_s) or low_speed_dwell_s < 0):
        return result
    result['regime'] = field.regime(speed_mps, along_accel_mps2)
    support = model['payload_support'][f'{payload_kg:g}']
    flags = result['transfer_flags']

    def bounded(value, key):
        low, high = support['ranges'][key]
        if value < low:
            flags.append(FLAGS[key] + '_CLAMPED_LOW')
            return low
        if value > high:
            flags.append(FLAGS[key] + '_CLAMPED_HIGH')
            return high
        return value

    s = bounded(speed_mps, 'speed_mps')
    a = bounded(along_accel_mps2, 'along_accel_mps2')
    up = bounded(vertical_up_mps, 'vertical_up_mps')
    theta = bounded(tilt_deg, 'tilt_deg')
    if speed_mps < field.POLICY['low_speed_mps'] and abs(vertical_up_mps) <= field.POLICY['max_abs_vertical_up_mps']:
        if low_speed_dwell_s is None:
            flags.append('UNKNOWN_LOW_SPEED_DWELL_TRANSFER')
        elif support['max_observed_slow_level_dwell_s'] is None or low_speed_dwell_s > support['max_observed_slow_level_dwell_s']:
            flags.append('LONG_LOW_SPEED_DWELL_TRANSFER')
    bands = list(support['actual_agl_m_by_planned_altitude_m'].values())
    if not any(lo <= altitude_agl_m <= hi for lo, hi in bands):
        inside = min(lo for lo, hi in bands) <= altitude_agl_m <= max(hi for lo, hi in bands)
        flags.append('ALTITUDE_BETWEEN_OBSERVED_PLATEAUS' if inside else 'ALTITUDE_OUTSIDE_TRAINING_RANGE')
    mass = model['base_physical_mass_kg'] + payload_kg
    work = mass * s * a
    positive = bounded(max(work, 0.0), 'positive_horizontal_work')
    braking = bounded(max(-work, 0.0), 'braking_horizontal_work_magnitude')
    coefficients = model['coefficients']
    p = (coefficients[f'intercept_{payload_kg:g}kg_W'] + coefficients['horizontal_speed'] * s
         + coefficients['horizontal_speed_squared'] * s*s + coefficients['signed_vertical_work'] * mass * model['gravity_mps2'] * up
         + coefficients['positive_horizontal_work'] * positive + coefficients['braking_horizontal_work_magnitude'] * braking
         + coefficients['sec_tilt_minus_one'] * (1.0/math.cos(math.radians(theta)) - 1.0))
    if not math.isfinite(p) or p <= 0:
        result['reason'] = 'NONPHYSICAL_REGRESSION_OUTPUT'
        return result
    result.update(power_w=float(bounded(p, 'power_w')), reason='POOLED_FIELD_ESTIMATE', unclipped_output_w=float(p))
    return result


def predict_kinematics(model, payload_kg, kinematics, altitude_agl_m, tilt_deg=None):
    if kinematics is None:
        return {'power_w': None, 'reason': 'INSUFFICIENT_CAUSAL_HISTORY', 'regime': None, 'transfer_flags': [],
                'support_basis': 'POOLED_FIELD_REGRESSION_MARGINAL_SUPPORT'}
    return predict(model, payload_kg=payload_kg, altitude_agl_m=altitude_agl_m, tilt_deg=tilt_deg,
                   **{name: kinematics.get(name) for name in field.FEATURES[:4]})


def describe_development(model, observations):
    import pandas as pd

    rows = []
    for point in observations:
        prediction = predict(model, payload_kg=point[field.MASS],
                             **dict(zip(field.FEATURES, point[field.SPEED:field.TILT + 1], strict=True)))
        rows.append({'flight_id': int(point[field.FID]), 'payload_kg': point[field.MASS],
                     'regime': field.REGIMES[int(point[field.REG])], 'weight_s': point[field.WT],
                     'measured_w': point[field.POWER], 'estimate_w': prediction['power_w'],
                     'transfer_flagged': bool(prediction['transfer_flags'])})
    frame = pd.DataFrame(rows)
    detailed, summary = [], []
    for scope in ('all_eligible', 'transient_regimes_only', *field.REGIMES):
        if scope == 'all_eligible':
            subset = frame
        elif scope == 'transient_regimes_only':
            subset = frame[frame.regime != 'nearly_steady_motion']
        else:
            subset = frame[frame.regime == scope]
        group_rows = []
        for fid, g in subset.groupby('flight_id'):
            selected = g[g.estimate_w.notna()]
            w = selected.weight_s.to_numpy()
            total = float(w.sum())
            observed = selected.measured_w.to_numpy()
            estimated = selected.estimate_w.to_numpy()
            e = estimated - observed
            row = {'scope': scope, 'flight_id': fid, 'payload_kg': g.payload_kg.iloc[0],
                   'eligible_samples': len(g), 'covered_samples': len(selected), 'eligible_s': g.weight_s.sum(),
                   'covered_s': total, 'flagged_covered_s': selected.loc[selected.transfer_flagged, 'weight_s'].sum(),
                   'point_mae_w': float(np.dot(w, abs(e)) / total) if total else None,
                   'point_bias_w': float(np.dot(w, e) / total) if total else None,
                   'point_rmse_w': float(np.sqrt(np.dot(w, e*e) / total)) if total else None,
                   'measured_on_covered_Wh': float(np.dot(w, observed) / 3600) if total else None,
                   'predicted_on_covered_Wh': float(np.dot(w, estimated) / 3600) if total else None,
                   'selected_window_energy_error_Wh': float(np.dot(w, e) / 3600) if total else None}
            group_rows.append(row)
        detailed.extend(group_rows)
        table = pd.DataFrame(group_rows)
        use = table[table.covered_s > 0]
        t = use.covered_s.to_numpy()
        weights = t / t.sum()
        summary.append({'scope': scope, 'development_flights_with_eligible_points': len(table),
                        'development_flights_with_predictions': len(use), 'eligible_s': table.eligible_s.sum(),
                        'covered_s': t.sum(), 'flagged_covered_s': use.flagged_covered_s.sum(),
                        'point_mae_w_time_pooled': float(np.dot(weights, use.point_mae_w)),
                        'point_bias_w_time_pooled': float(np.dot(weights, use.point_bias_w)),
                        'point_rmse_w_time_pooled': float(np.sqrt(np.dot(weights, use.point_rmse_w**2))),
                        'point_mae_w_equal_flight': use.point_mae_w.mean(), 'point_bias_w_equal_flight': use.point_bias_w.mean(),
                        'point_rmse_w_equal_flight': float(np.sqrt(np.mean(use.point_rmse_w**2))),
                        'mean_measured_selected_window_Wh': use.measured_on_covered_Wh.mean(),
                        'selected_window_energy_mae_Wh_equal_flight': use.selected_window_energy_error_Wh.abs().mean(),
                        'selected_window_energy_bias_Wh_equal_flight': use.selected_window_energy_error_Wh.mean(),
                        'selected_window_energy_rmse_Wh_equal_flight': float(np.sqrt(np.mean(use.selected_window_energy_error_Wh**2)))})
    return pd.DataFrame(detailed), pd.DataFrame(summary)


def build():
    import pandas as pd
    from . import external

    strict_path = OUT / 'model.json'
    strict = json.loads(strict_path.read_text())
    base = json.loads(field.FIT.read_text())
    source_file = OUT / strict['support_file']
    if field.sha(source_file) != strict['support_sha256'] or strict['columns'] != list(field.COLS):
        raise ValueError('Unexpected frozen field support table')
    train = sorted(base['cohorts']['train']['flight_ids'])
    validation = sorted(base['cohorts']['validation']['flight_ids'])
    with np.load(source_file, allow_pickle=False) as archive:
        data = archive['observations']
    contributing = np.unique(data[:, field.FID]).astype(int).tolist()
    if (train != strict['fit_ids'] or set(contributing) - set(train) or set(train) & set(validation)
            or len(train) != 75 or len(validation) != 35):
        raise ValueError('Unexpected training or development membership')
    source_paths = [external.INPUT/'parameters.csv', external.INPUT/'flights.zip', external.OUTPUT/'flight_manifest.csv']
    for p in source_paths:
        external.check_hash(p, base['input_and_code_sha256'][str(p.relative_to(field.ROOT))])
    mass = sum(json.loads(PHYSICS.read_text())['mass']['as_flown_components_kg'].values())
    model = fit(data, mass)  # Single prespecified weighted regression, using the frozen training support only.
    model.update(data_source=strict['source'], training_registered_ids=train, training_contributing_ids=contributing,
                 development_ids_for_descriptive_scoring=validation, retained_training_points=len(data),
                 retained_training_original_s=float(data[:, field.WT].sum()),
                 fit_provenance={str(p.resolve().relative_to(field.ROOT)): field.sha(p)
                                 for p in (strict_path, source_file, field.FIT, PHYSICS, Path(__file__), Path(field.__file__), *source_paths)},
                 development_protocol='After freezing the model, apply it once to existing 35 development recordings. Reuse the original timestamp and near-level/turn selection. Endpoint weights integrate only available adjacent observed intervals. Energy metrics first sum signed errors on that flight\'s covered selected pieces, then take absolute value and average equally over flights. This measures selected-window energy, not whole-flight energy. No holdout signal is accessed; no selection on development errors.')
    temporary = MODEL_PATH.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(model, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(MODEL_PATH)
    print(f'Frozen pooled model {field.sha(MODEL_PATH)} on {len(contributing)} contributing flights.', flush=True)

    metadata = pd.read_csv(source_paths[0]).set_index('flight')
    manifest = pd.read_csv(source_paths[2]).set_index('flight_id')
    if not metadata.index.is_unique or not manifest.index.is_unique:
        raise ValueError('Development metadata IDs must be unique')
    with ZipFile(source_paths[1]) as archive:
        val_data, _ = field.cohort_data(validation, 'validation', metadata, manifest, archive)
    detail, summary = describe_development(load_model(), val_data)
    detail.to_csv(OUT/'pooled_development_by_flight.csv', index=False, float_format='%.10g')
    summary.to_csv(OUT/'pooled_development.csv', index=False, float_format='%.10g')


if __name__ == '__main__':
    build()
