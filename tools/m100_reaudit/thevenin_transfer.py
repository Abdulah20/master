"""Identify a discharge-relative two-RC voltage proxy from whole CMU development flights."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .cmu_calibration import MANIFEST_SHA256, development_flights
from .external import SHA256

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/m100_reaudit/mission_150m/battery_fit.json'
HEADER = Path(__file__).with_name('pack_drive') / 'M100TheveninFit.hh'
TAUS = np.array([0.5, 1., 2., 4., 8., 16., 32., 64., 128.])


def observations(flight):
    t, current, volts = flight['samples'][['time', 'battery_current', 'battery_voltage']].to_numpy(float).T
    dt = np.diff(t)
    if not np.isfinite(np.r_[t, current, volts]).all() or np.any(dt <= 0) or np.max(dt) > 0.5000001:
        raise ValueError(f"Invalid time series in flight {flight['flight_id']}")
    # Positive discharge; zero polarization at the first measured timestamp is an assumption.
    filtered = np.empty((len(t), len(TAUS)))
    filtered[0] = 0.
    gain = -np.expm1(-dt[:, None] / TAUS)
    for j in range(len(dt)):
        filtered[j+1] = filtered[j] + gain[j] * (current[j] - filtered[j])
    q = np.r_[0., np.cumsum((current[:-1] + current[1:]) * dt / 7200.)]
    start, _, _, end = flight['bounds']
    take = slice(start + 1, end)
    i = current[take]
    h = 0.5 * dt[start+1:end-1] * ((i[:-1] > 0) & (i[1:] > 0))
    w = np.r_[0., h] + np.r_[h, 0.]
    keep = w > 0
    if not keep.any():
        raise ValueError('No positive time-weighted loads')
    x = np.column_stack((current, q, filtered))[take][keep]
    y = volts[take][keep]
    w = w[keep] / w.sum()
    return x, y, w


def fit():
    flights, _ = development_flights()
    samples = [(f['flight_id'], f['split'], *observations(f)) for f in flights]
    train = [s for s in samples if s[1] == 'train']
    means = [(w @ x, w @ y) for _, _, x, y, w in train]
    # Remove between-flight initial-voltage differences during identification.
    xx = np.zeros((2+len(TAUS), 2+len(TAUS)))
    xy = np.zeros(2+len(TAUS))
    for (_, _, x, y, w), (xm, ym) in zip(train, means):
        xx += (x-xm).T @ (w[:, None] * (x-xm))
        xy += (x-xm).T @ (w * (ym-y))
    candidates = []
    for j, k in itertools.combinations(range(len(TAUS)), 2):
        if TAUS[k] < 4 * TAUS[j]:
            continue
        ix = [0, 1, 2+j, 2+k]
        gram = xx[np.ix_(ix, ix)]
        if np.linalg.matrix_rank(gram) != len(ix):
            continue
        coef = np.linalg.solve(gram, xy[ix])
        if np.all(coef > 0):
            candidates.append((float(coef @ gram @ coef - 2 * coef @ xy[ix]), j, k, ix, coef))
    if not candidates:
        raise ValueError('No positive, full-rank two-branch transfer fit')
    _, j, k, ix, coef = min(candidates, key=lambda item: item[0])
    ref = float(np.mean([ym + xm[ix] @ coef for xm, ym in means]))
    pooled = np.concatenate([s[2][:, :2] for s in train])
    constraints = dict(current_A=[float(v) for v in (pooled[:, 0].min(), pooled[:, 0].max())],
                       discharged_Ah=[float(v) for v in (pooled[:, 1].min(), pooled[:, 1].max())])
    params = dict(reference_V=ref, discharge_slope_V_per_Ah=float(coef[1]), r0_ohm=float(coef[0]),
                  r1_ohm=float(coef[2]), tau1_s=float(TAUS[j]), r2_ohm=float(coef[3]), tau2_s=float(TAUS[k]))
    params.update(c1_F=params['tau1_s']/params['r1_ohm'], c2_F=params['tau2_s']/params['r2_ohm'])
    historical = json.loads((ROOT/'results/m100_reaudit/cmu_calibration/fit.json').read_text())['loaded_voltage']['coefficients']
    metrics = {}
    for split in ('train', 'validation'):
        errs = []
        for _, part, x, y, w in samples:
            if part == split:
                dyn = ref - x[:, ix] @ coef - y
                static = historical['a_V'] - x[:, 0]*historical['r_V_per_A'] - x[:, 1]*historical['k_V_per_Ah'] - y
                errs.append((w@abs(dyn), w@abs(static), w@dyn, w@dyn**2))
        mae, base_mae, bias, mse = np.mean(errs, axis=0)
        metrics[split] = dict(flights=len(errs), mae_V=mae, previous_static_mae_V=base_mae,
                              bias_V=bias, rmse_V=float(np.sqrt(mse)))
    profile_hash = hashlib.sha256(json.dumps([params, constraints], sort_keys=True).encode()).hexdigest()
    return dict(model_id='CMU_LOADED_THEVENIN_2RC_V1', parameter_sha256=profile_hash,
                target_pack='DJI TB47D nameplate, 4.5 Ah for nominal accounting only',
                classification='LIMITED_EMPIRICAL_PROXY', physical_tb47d_calibration=False,
                temperature_calibration=False, equilibrium_OCV_SOC_measured=False,
                source='https://doi.org/10.1184/R1/12683453', flight_archive_sha256=SHA256['flights.zip'],
                manifest_sha256=MANIFEST_SHA256,
                method='Time-weighted within-flight least squares with positive coefficients, equal flight weights. '
                       'Choose separated time constants on training residuals only; one fixed population reference voltage. '
                       'Effective source voltage is a function of discharged Ah since start, not measured open-circuit voltage.',
                tau_candidates_s=TAUS.tolist(), selected_fastest_at_search_boundary=bool(j == 0),
                initial_polarization_V=[0., 0.], parameters=params, domain=constraints, metrics=metrics,
                flight_ids={p:[i for i, s, *_ in samples if s==p] for p in metrics})


def write_header(profile, path):
    names = {'reference_V':'referenceV', 'discharge_slope_V_per_Ah':'slope', 'r0_ohm':'r0',
             'r1_ohm':'r1', 'r2_ohm':'r2', 'tau1_s':'tau1', 'tau2_s':'tau2'}
    lines = ['// Generated by tools.m100_reaudit.thevenin_transfer from measured CMU loads.', '#pragma once',
             'namespace m100::rcfit {', f'inline constexpr char hash[] = "{profile["parameter_sha256"]}";']
    for key, name in names.items():
        lines.append(f'inline constexpr double {name} = {profile["parameters"][key]:.17g};')
    lines.append('}')
    path.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--header', type=Path, default=HEADER)
    args = parser.parse_args()
    profile = fit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2, allow_nan=False)+'\n')
    write_header(profile, args.header)
    print(json.dumps(profile['metrics']))
