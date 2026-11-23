"""Preflight simulation-energy models and reproducible whole-condition validation."""
import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import physics
from .mission_campaign import OUT, save
from .mission_observations import table

ENERGY='estimated_airborne_energy_Wh'
TIME='airborne_duration_s'
FEATURES=('phase_energy_Wh','kinematic_time_s','climb_height_m','payload_kg','speed_mps','speed_squared',
          'outbound_ideal_s','return_ideal_s','hover_s','intermediate_stops','stops_times_speed',
          'payload_times_kinematic_time','along_wind_mps','abs_cross_wind_mps','wind_squared',
          'ambient_minus_15_C','nominal_discharge_fraction','hover_mission',
          'outbound_air_path_delta_m','return_air_path_delta_m',
          'outbound_crosswind_exposure_m','return_crosswind_exposure_m')


def number(row,name,default=0.):
    value=row.get(name)
    return default if value is None or value=='' else float(value)


@lru_cache(maxsize=1)
def published_model():
    return physics.Model.load(physics.PORTABLE_RUNTIME)


@lru_cache(maxsize=128)
def phase_powers(payload,temp,height):
    model=published_model()
    rho=model.atmosphere(temp,height).density_kg_m3
    return tuple(model.phase_power(phase,payload,rho) for phase in ('CLIMB','CRUISE','LANDING'))


def leg_wind_features(row, outbound_speed, return_speed, wind_east, wind_north):
    """Integrate planned airspeed change and crosswind on each outbound/return leg."""
    if outbound_speed == 0 or (wind_east == 0 and wind_north == 0):
        return (0., 0., 0., 0.)
    geometry = row.get('leg_conditions_json')
    if geometry:
        legs = json.loads(geometry) if isinstance(geometry, str) else geometry
        if len(legs) < 2:
            raise ValueError('The plan must include an outbound leg and a return leg')
        vectors = [(leg['end_enu_m'][0] - leg['start_enu_m'][0],
                    leg['end_enu_m'][1] - leg['start_enu_m'][1]) for leg in legs]
    else:
        # The historical eastbound protocol gives a survey length, not the actual
        # turnaround position. Its baseline uses that length for each heading.
        if row.get('mission_route') not in ('direct', 'eastbound_measurement_then_return'):
            raise ValueError('Planned leg geometry is required for this route')
        outbound = number(row, 'planned_outbound_m')
        vectors = [(outbound, 0.), (-number(row, 'planned_return_m', outbound), 0.)]
    air_path_delta = [0., 0.]
    crosswind_exposure = [0., 0.]
    for i, (east, north) in enumerate(vectors):
        length = math.hypot(east, north)
        if not math.isfinite(length) or length <= 0:
            raise ValueError('Invalid planned leg length')
        is_return = int(i == len(vectors) - 1)
        speed = return_speed if is_return else outbound_speed
        along = (wind_east * east + wind_north * north) / length
        cross = (wind_north * east - wind_east * north) / length
        seconds = length / speed
        air_path_delta[is_return] += seconds * (math.hypot(speed - along, cross) - speed)
        crosswind_exposure[is_return] += seconds * abs(cross)
    return air_path_delta[0], air_path_delta[1], crosswind_exposure[0], crosswind_exposure[1]


def features(c):
    """Only planned conditions. No outcome, realized speed, time, SOC, energy or battery samples."""
    payload, v, h, wind, deg, temp=[number(c,k) for k in
        ('payload_kg','cruise_speed_mps','altitude_m','wind_speed_mps','wind_direction_toward_deg_ENU','temperature_c')]
    out=number(c,'planned_outbound_m')
    ret=number(c,'planned_return_m',out)
    hold=number(c,'planned_hover_s')
    via=number(c,'intermediate_stops')
    soc=number(c,'initial_soc_percent',100.)
    up_w,cruise_w,down_w=phase_powers(payload,temp,h)
    # Simple reference schedule, fixed before learning. The real control loop includes holds,
    # capture, spool-up and acceleration. These residuals are learned using training flights only.
    up_s,down_s=h/2.,h/1.5
    out_s=out/v if v>0 else 0.
    return_speed=.9*min(4.,v)
    ret_s=ret/return_speed if v>0 else 0.
    total=up_s+down_s+out_s+ret_s+hold
    reference=(up_s*up_w + down_s*down_w + (out_s+ret_s+hold)*cruise_w)/3600
    angle=math.radians(deg)
    wind_east,wind_north=wind*math.cos(angle),wind*math.sin(angle)
    x=[reference,total,h,payload,v,v*v,out_s,ret_s,hold,via,via*v,payload*total,wind_east,
       abs(wind_north),wind*wind,temp-15.,(100.-soc)/100.,float(v==0),
       *leg_wind_features(c,v,return_speed,wind_east,wind_north)]
    if not np.isfinite(x).all() or total<=0 or reference<=0:
        raise ValueError('Nonphysical planning inputs')
    return x


def condition(c):
    # Keep all loads and all replicate seeds of an operating family on the same side of every split.
    fields=('cruise_speed_mps','altitude_m','wind_speed_mps','wind_direction_toward_deg_ENU','temperature_c',
            'initial_soc_percent','planned_outbound_m','planned_return_m','planned_hover_s')
    return json.dumps([c['mission_route'],*[number(c,k,100. if k=="initial_soc_percent" else 0.) for k in fields]],separators=(',',':'))


def grouped_folds(rows,n=5):
    groups=sorted({condition(r) for r in rows},key=lambda key:hashlib.sha256(key.encode()).digest())
    if len(groups)<2:
        raise ValueError('At least two independent operating conditions are required')
    lookup={key:i%min(n,len(groups)) for i,key in enumerate(groups)}
    return np.array([lookup[condition(row)] for row in rows])


def ridge_fit(x, targets, weights, penalty):
    """Weighted standardization and intercept use only rows in the training fold."""
    weights=np.asarray(weights,float)
    weights=weights/weights.sum()
    center=weights@x
    scale=np.sqrt(weights@((x-center)**2))
    scale[scale<1e-9]=1.
    xn=(x-center)/scale
    offset=weights@targets
    gram=xn.T@(weights[:,None]*xn)
    gram.flat[::len(gram)+1]+=penalty/len(x)
    slopes=np.linalg.solve(gram,xn.T@(weights[:,None]*(targets-offset)))
    return dict(center=center.tolist(),scale=scale.tolist(),offset=offset.tolist(),coefficients=slopes.tolist(),
                penalty=penalty)


def ridge_apply(x,fit):
    return np.asarray(fit['offset'])+((x-np.array(fit['center']))/np.array(fit['scale']))@np.array(fit['coefficients'])


def weights_by_case(rows):
    # Repeated native seeds of the same complete input command have equal combined influence.
    keys=[(condition(row),number(row,'payload_kg')) for row in rows]
    counts=defaultdict(int)
    for key in keys:
        counts[key]+=1
    return np.array([1./counts[key] for key in keys])


def scores(y,yp):
    err=np.asarray(yp)-np.asarray(y)
    return dict(n=len(err),mae=float(np.abs(err).mean()),bias=float(err.mean()),rmse=float(np.sqrt(np.mean(err**2))),
                p90_abs=float(np.quantile(abs(err),.9)),mare_pct=float(np.mean(abs(err)/np.asarray(y))*100))


def dataframe(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as src:
        return list(csv.DictReader(src))


def evaluate(rows):
    x=np.array([features(row) for row in rows])
    base=x[:,:2]
    y=np.array([[number(r,ENERGY),number(r,TIME)] for r in rows])
    folds=grouped_folds(rows)
    pred=np.empty_like(y)
    constant=np.empty_like(y)
    indices=np.arange(len(rows))
    for fold in np.unique(folds):
        train=indices[folds!=fold]
        test=indices[folds==fold]
        train_w=weights_by_case([rows[i] for i in train])
        fit=ridge_fit(x[train],(y-base)[train],train_w,1.)
        pred[test]=base[test]+ridge_apply(x[test],fit)
        constant[test]=np.average(y[train],axis=0,weights=train_w)
    checks=[]
    for i,c in enumerate(rows):
        checks.append(dict(case_id=c['case_id'],run_id=c['run_id'],fold=int(folds[i]),condition=condition(c),
            payload_kg=number(c,'payload_kg'),mission_route=c['mission_route'],measured_in_sim_energy_Wh=y[i,0],
            measured_in_sim_duration_s=y[i,1],predicted_energy_Wh=pred[i,0],predicted_duration_s=pred[i,1],
            phase_schedule_energy_Wh=base[i,0],training_mean_energy_Wh=constant[i,0]))
    metrics={}
    for tag,sel in [('all',indices)]+[(f'payload_{mass:g}',np.array([i for i,r in enumerate(rows) if number(r,'payload_kg')==mass]))
                                       for mass in sorted({number(r,'payload_kg') for r in rows})]:
        metrics[tag]={}
        for source,candidate in [('residual_ridge',pred),('phase_schedule',base),('train_mean',constant)]:
            metrics[tag][source]=dict(energy_Wh=scores(y[sel,0],candidate[sel,0]),duration_s=scores(y[sel,1],candidate[sel,1]))
    # Separate route-family transfer diagnostic. Unknown geometries never enter the constrained search.
    route_transfer={}
    for route in sorted({r['mission_route'] for r in rows}):
        train=np.array([i for i,r in enumerate(rows) if r['mission_route']!=route],dtype=int)
        test=np.array([i for i,r in enumerate(rows) if r['mission_route']==route],dtype=int)
        if len(train)>=3 and len(test):
            partial=ridge_fit(x[train],(y-base)[train],weights_by_case([rows[i] for i in train]),1.)
            estimate=base[test]+ridge_apply(x[test],partial)
            route_transfer[route]=scores(y[test,0],estimate[:,0])
    full=ridge_fit(x,y-base,weights_by_case(rows),1.)
    model=dict(schema='M100_SIM_PREFLIGHT_RIDGE_V2',features=FEATURES,target_units=['Wh','s'],fit=full,
               power_reference='Rodrigues phase coefficients; nominal scheduling rates 2 m/s up, 1.5 down, '
                               '0.9 times min(4, outbound speed) return; hover uses level-flight reference',
               wind_features='Each planned leg uses its heading, scheduled speed and the commanded ENU wind. '
               'Air-path change sums time times (airspeed minus ground speed), and crosswind exposure sums '
               'time times absolute crosswind. Outbound and return totals enter the fit separately.',
               split='Five deterministic operating-family folds. All payloads and seeds for each geometry, '
                    'speed, altitude, wind, ambient temperature and initial SOC stay together.',
               outputs=['airborne_energy_Wh','airborne_duration_s'],calibration='SIMULATOR_SURROGATE',
               validation_absolute_error_p90_Wh=metrics['all']['residual_ridge']['energy_Wh']['p90_abs'],
               validation_absolute_error_p90_s=metrics['all']['residual_ridge']['duration_s']['p90_abs'])
    return model,dict(overall_and_by_payload=metrics,leave_one_route_family_out_energy_Wh=route_transfer),checks


def predict(c, model):
    if tuple(model['features']) != FEATURES:
        raise ValueError('Preflight model feature schema differs from the active predictor')
    x=np.asarray([features(c)])
    y=x[:,:2]+ridge_apply(x,model['fit'])
    if not np.isfinite(y).all() or (y<=0).any():
        raise ValueError('Preflight surrogate returned a nonphysical prediction')
    return tuple(float(value) for value in y[0])


def save_evaluation(rows,destination):
    model,metrics,checks=evaluate(rows)
    save(destination/'model.json',model)
    save(destination/'evaluation.json',metrics)
    table(destination/'held_out_predictions.csv',checks)
    return model


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    save_evaluation(dataframe(args.input),args.output)
