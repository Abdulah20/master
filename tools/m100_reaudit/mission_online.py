"""Validate a one-interval-ahead power model with predictors available at the interval start."""
import csv
import gzip
import math

import numpy as np

from .mission_campaign import save
from .mission_observations import table
from .mission_prediction import grouped_folds, ridge_fit, ridge_apply

PHASES=('TAKEOFF','CLIMB','STABILIZING','CRUISE','RETURN_BRAKING','HOVER','RETURNING','LANDING')
NAVS=('OFFBOARD','AUTO_TAKEOFF','AUTO_LAND','AUTO_LOITER')
NUMERIC=('payload_kg','commanded_speed_mps','commanded_altitude_agl_m','horizontal_speed_mps','vertical_speed_up_mps',
         'horizontal_tangential_accel_mps2','horizontal_accel_magnitude_mps2','vertical_accel_up_mps2',
         'estimated_height_since_takeoff_m','air_relative_horizontal_mps','ambient_temperature_C','nominal_soc_at_start_pct')


def inputs(row):
    raw=[row.get(k) for k in NUMERIC]
    if any(v is None or v=='' for v in raw) or str(row.get('causal_features_available')).lower()!='true':
        return None
    x=[float(v) for v in raw]
    x += [float(row['flight_phase']==s) for s in PHASES]
    x += [float(row['px4_mode']==s) for s in NAVS]
    return x if all(math.isfinite(v) for v in x) else None


def evaluate(path, flights, destination, *, observe_held_out=None):
    """Optionally expose the same held-out intervals used for the saved metrics."""
    fold_by_case=dict(zip([r['case_id'] for r in flights],grouped_folds(flights),strict=True))
    runs = {r['case_id']: r['run_id'] for r in flights}
    if len(runs) != len(flights):
        raise ValueError('Expected one selected flight per case')
    seen = set()
    data=[]
    omitted=0
    with gzip.open(path,'rt',newline='') as source:
        for row in csv.DictReader(source):
            key=row['case_id']
            if key not in fold_by_case:
                continue
            if row['run_id'] != runs[key]:
                raise ValueError(f'Intervals for {key} belong to a different flight')
            seen.add(key)
            x=inputs(row)
            if x is None:
                omitted+=1
                continue
            data.append((key,fold_by_case[key],row['flight_phase'],float(row['interval_s']),x,
                         float(row['future_interval_mean_power_W'])))
    missing = runs.keys() - seen
    if missing:
        raise ValueError(f'Missing interval records for {sorted(missing)}')
    if not data:
        raise ValueError('No intervals have complete causal features')
    ids=np.array([r[0] for r in data])
    folds=np.array([r[1] for r in data])
    phases=np.array([r[2] for r in data])
    dt=np.array([r[3] for r in data])
    x=np.array([r[4] for r in data])
    y=np.array([r[5] for r in data])
    eligible=np.isfinite(y)&(y>0)&(dt>0)&(dt<=2.0)
    omitted+=int((~eligible).sum())
    ids,folds,phases,dt,x,y=[a[eligible] for a in (ids,folds,phases,dt,x,y)]
    if len(np.unique(folds)) < 2:
        raise ValueError('At least two operating-condition folds must contain eligible intervals')
    predictions=np.empty_like(y)
    simple=np.empty_like(y)
    weights=dt.copy()
    for key in set(ids):
        mask=ids==key
        weights[mask]/=dt[mask].sum()
    for fold in np.unique(folds):
        train,test=folds!=fold,folds==fold
        fit=ridge_fit(x[train],y[train,None],weights[train],3.)
        predictions[test]=ridge_apply(x[test],fit).ravel()
        simple[test]=np.average(y[train],weights=weights[train])
        for phase in set(phases[test]):
            source=train&(phases==phase)
            if source.any():
                simple[test&(phases==phase)]=np.average(y[source],weights=weights[source])
    rows=[]
    for key in sorted(set(ids)):
        mask=ids==key
        row=dict(case_id=key,fold=int(folds[mask][0]),intervals=int(mask.sum()),seconds=float(dt[mask].sum()))
        for tag,yp in (('ridge',predictions),('phase_mean',simple)):
            e=yp[mask]-y[mask]
            row[tag+'_mae_W']=float(np.average(abs(e),weights=dt[mask]))
            row[tag+'_bias_W']=float(np.average(e,weights=dt[mask]))
        rows.append(row)
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite power prediction')
    table(destination/'held_out_by_flight.csv',rows)
    save(destination/'model.json',dict(features=[*NUMERIC,*['phase='+s for s in PHASES],*['nav='+s for s in NAVS]],
         fit=ridge_fit(x,y[:,None],weights,3.),target='following_interval_mean_pack_power_W',
         observations='Causal PX4 local estimates and commands at the left edge. Battery current and voltage are excluded.',
         phase_window_s='approximately one second; observed endpoints',field_validation=False))
    result=dict(eligible_intervals=len(y),excluded_intervals=omitted,flights=len(rows),
                input_flights=len(flights), flights_without_eligible_intervals=sorted(seen - set(ids)),
                method='Full flights assigned to the same grouped folds as the preflight task. Equal total fit weight per flight.',
                heldout_mae_W_equal_flight=float(np.mean([r['ridge_mae_W'] for r in rows])),
                phase_mean_reference_mae_W=float(np.mean([r['phase_mean_mae_W'] for r in rows])),
                heldout_bias_W_equal_flight=float(np.mean([r['ridge_bias_W'] for r in rows])))
    save(destination/'evaluation.json',result)
    if observe_held_out is not None:
        observe_held_out(ids, folds, phases, dt, y, predictions, simple)
    return result
