"""Declared simulator-only preflight M100 cruise-power transfer diagnostic.

The design intentionally has two columns. At a fixed reference sea-level
temperature of 15 C, estimate an intercept and slope on the simulator's
regression mass/density feature. Fit only the existing train split's calm
CRUISE windows at 15 C. Commands and labels from CMU never enter this fit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np

from .physics import Model


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/m100_reaudit'
SIMULATOR = RESULTS / 'gazebo_batch/training_research.csv'
FIELD = RESULTS / 'external/measured_cruise.csv'
FIELD_MANIFEST = RESULTS / 'external/flight_manifest.csv'
OUTPUT = RESULTS / 'transfer'
REFERENCE_TEMPERATURE_C = 15.0
LABEL = 'measurement_mean_power_W'
OBSERVED = 'observed_mean_power_W'


@dataclass(frozen=True)
class Commands:
    payload_kg: float
    speed_mps: float
    height_m: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.payload_kg, self.speed_mps, self.height_m)):
            raise ValueError('Nonfinite command')
        if self.payload_kg < 0 or self.speed_mps <= 0 or self.height_m < 0:
            raise ValueError('Cruise prediction needs nonnegative payload/height and a positive speed')


def feature(command, atmosphere, mass_baseline):
    rho = atmosphere(REFERENCE_TEMPERATURE_C, command.height_m).density_kg_m3
    return (mass_baseline + command.payload_kg) ** 1.5 / math.sqrt(rho)


@dataclass(frozen=True)
class Fit:
    intercept_W: float
    mass_density_slope_W: float
    design_rank: int

    def predict(self, x):
        return self.intercept_W + self.mass_density_slope_W * x


def fit_simulator(samples, atmosphere, mass_baseline):
    if len(samples) < 2:
        raise ValueError('Simulation training is empty or rank deficient')
    x = [feature(cmd, atmosphere, mass_baseline) for cmd, _ in samples]
    design = np.column_stack((np.ones(len(x)), x))
    labels = np.asarray([p for _, p in samples], dtype=float)
    if not np.all(np.isfinite(labels)) or np.any(labels <= 0):
        raise ValueError('Nonpositive or nonfinite simulator label')
    coefs, _, rank, _ = np.linalg.lstsq(design, labels, rcond=None)
    if rank != 2:
        raise ValueError('Rank-deficient simulator training design')
    return Fit(float(coefs[0]), float(coefs[1]), int(rank))


def errors(predictions):
    if not predictions:
        raise ValueError('Empty diagnostic stratum')
    differences = [est - obs for est, obs in predictions]
    return dict(flights=len(differences), mae_W=fmean(abs(v) for v in differences),
                rmse_W=math.sqrt(fmean(v*v for v in differences)), bias_W=fmean(differences))


def support(command, training):
    if command in training:
        return 'exact_payload_speed_altitude'
    if any((c.payload_kg, c.speed_mps) == (command.payload_kg, command.speed_mps) for c in training):
        return 'payload_speed_at_other_altitude'
    return 'payload_speed_pair_unseen'


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load_training(simulator):
    rows = read_rows(simulator)
    ids = [r['case_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Repeated simulator case')
    selected = []
    families = {}
    levels = {}
    train_size = 0
    for row in rows:
        group, split = row['split_group'], row['split']
        pair = float(row['payload_kg']), float(row['cruise_speed_mps'])
        if families.setdefault(group, split) != split or levels.setdefault(pair, split) != split:
            raise ValueError('Simulator family occurs in multiple splits')
        if split != 'train':
            continue
        train_size += 1
        if row['target_phase'] != 'CRUISE' or float(row['temperature_c']) != REFERENCE_TEMPERATURE_C or float(row['wind_speed_mps']) != 0:
            continue
        cmd = Commands(*pair, float(row['altitude_m']))
        selected.append((row['case_id'], cmd, float(row[LABEL])))
    commands = [cmd for _, cmd, _ in selected]
    if len(set(commands)) != len(commands):
        raise ValueError('Repeated simulator payload/speed/altitude in selected reference slice')
    return selected, len(rows), train_size


def load_field(measured, manifest):
    field_rows = [r for r in read_rows(measured) if r['split'] == 'test']
    manifest_rows = [r for r in read_rows(manifest) if r['split'] == 'test'
                     and r['published_role'] == 'holdout' and r['included'].lower() == 'true']
    eligible = {r['flight_id']: r for r in manifest_rows}
    if len(eligible) != 60 or len(manifest_rows) != 60 or len(field_rows) != 60 or {r['flight_id'] for r in field_rows} != set(eligible):
        raise ValueError('CMU comparison must contain each of the 60 existing quality-eligible test flights exactly once')
    observations = []
    for row in sorted(field_rows, key=lambda r: int(r['flight_id'])):
        membership = eligible[row['flight_id']]
        if any(float(row[k]) != float(membership[m]) for k, m in (
                ('payload_kg', 'payload_kg'), ('commanded_speed_mps', 'commanded_speed_mps'),
                ('commanded_altitude_agl_m', 'commanded_altitude_original_m'))):
            raise ValueError('Field command disagrees with source manifest')
        cmd = Commands(float(row['payload_kg']), float(row['commanded_speed_mps']), float(row['commanded_altitude_agl_m']))
        observation = float(row[OBSERVED])
        if not math.isfinite(observation) or observation <= 0:
            raise ValueError('Invalid field label')
        observations.append((row['flight_id'], cmd, observation))
    return observations


def run(simulator, measured, manifest, output):
    model = Model.load()
    training, simulator_count, train_count = load_training(simulator)
    mass = model.energy.published_regression_baseline_mass_kg
    train_commands = {cmd for _, cmd, _ in training}
    fit = fit_simulator([(cmd, p) for _, cmd, p in training], model.atmosphere, mass)

    # Freeze the fitted parameters before loading the field labels. No fit,
    # hyperparameter selection, or calibration on validation or test flights.
    observations = load_field(measured, manifest)
    predictions, groups = [], {}
    for fid, cmd, y in observations:
        x = feature(cmd, model.atmosphere, mass)
        estimate = fit.predict(x)
        coverage = support(cmd, train_commands)
        predictions.append(dict(flight_id=fid, payload_kg=cmd.payload_kg, commanded_speed_mps=cmd.speed_mps,
                                commanded_altitude_agl_m=cmd.height_m, training_support=coverage,
                                simulated_fit_prediction_W=estimate, measured_cruise_mean_W=y,
                                predicted_minus_measured_W=estimate-y))
        for axis, level in (('all', 'all'), ('payload_kg', f'{cmd.payload_kg:g}'),
                            ('commanded_speed_mps', f'{cmd.speed_mps:g}'),
                            ('commanded_altitude_agl_m', f'{cmd.height_m:g}'), ('training_support', coverage)):
            groups.setdefault((axis, level), []).append((estimate, y))
    by_group = []
    for axis in ('all', 'payload_kg', 'commanded_speed_mps', 'commanded_altitude_agl_m', 'training_support'):
        levels = [level for category, level in groups if category == axis]
        numeric = axis in {'payload_kg', 'commanded_speed_mps', 'commanded_altitude_agl_m'}
        for level in sorted(levels, key=float if numeric else str):
            by_group.append(dict(stratum=axis, value=level, **errors(groups[axis, level])))
    training_predictions = [(fit.predict(feature(c, model.atmosphere, mass)), p) for _, c, p in training]

    def dimensions(commands):
        return {field: sorted({getattr(c, field) for c in commands}) for field in ('payload_kg', 'speed_mps', 'height_m')}

    summary = dict(model='OLS with intercept on one prespecified M100 regression-mass/density feature',
                   target_simulator='Mean estimated electrical power over the stable outbound CRUISE measurement window',
                   target_field='Mean measured pack voltage times current over the whole recorded level-flight cruise phase',
                   reference_temperature_C=REFERENCE_TEMPERATURE_C,
                   air_state='Dry atmosphere, 15 C at ground datum 0 m MSL, height input = commanded AGL; applied to both domains.',
                   simulated_wind_mps=0, field_wind='unknown; no field wind, on-aircraft airflow or measured weather used',
                   regression_baseline_mass_kg=mass, fitted=asdict(fit),
                   slope_feature='(3.715 kg + payload_kg)^1.5 / sqrt(rho_15C_at_commanded_height_kg_m3)',
                   preflight_inputs_used=['payload_kg', 'commanded_altitude_agl_m'],
                   commanded_speed_role='selects cruise and diagnoses coverage; no independent fitted speed term',
                   simulator_accepted_primary_cases=simulator_count, simulator_existing_train_cases=train_count,
                   training_cruise_cases=len(training), training_distinct_commands=len(train_commands),
                   training_distinct_payload_speed_pairs=len({(c.payload_kg,c.speed_mps) for c in train_commands}),
                   training_case_ids=[cid for cid, _, _ in training],
                   training_coverage=dimensions(train_commands), field_test_coverage=dimensions([c for _,c,_ in observations]),
                   field_unique_commands=len({c for _,c,_ in observations}),
                   field_comparison=errors(groups['all','all']), fit_residuals_on_training_cases=errors(training_predictions),
                   field_sample_unit='one previously assigned test flight', weighting='one observation per simulator case / CMU flight',
                   source_sha256={p.name: digest(p) for p in (simulator, measured, manifest)},
                   implementation_sha256=digest(Path(__file__)),
                   model_inputs_sha256={p.name: digest(p) for p in (
                       Path(model.atmosphere_module.__file__),
                       model.runtime_root / 'ros2_ws/src/drone_nav/config' / model.settings['energy']['published_phase_config'],
                       ROOT / 'config/m100_reaudit/physics.json')},
                   simulator_scope='Accepted Gazebo / gz-physics-DART research-controller primary cases, one row per scenario.',
                   field_scope='CMU/Rodrigues instrumented M100 measured pack electrical power integrated at the actual electrical sample times.',
                   declared_policy='Only existing simulator TRAIN, T=15 C, wind=0, phase=CRUISE; no repeated cases. Field IDs fixed by the existing published holdout and quality filters.',
                   provenance_limit='The simulator is informed by CMU/Rodrigues data; this is a within-campaign domain-transfer diagnostic, not a fully independent experiment.')
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/'field_predictions.csv', predictions)
    write_rows(output/'field_groups.csv', by_group)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator', type=Path, default=SIMULATOR)
    parser.add_argument('--measured', type=Path, default=FIELD)
    parser.add_argument('--manifest', type=Path, default=FIELD_MANIFEST)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args=parser.parse_args()
    print(json.dumps(run(args.simulator,args.measured,args.manifest,args.out)['field_comparison'],indent=2))


if __name__ == '__main__':
    main()
