"""Break down held-out simulator energy and next-interval pack power errors.

Published folds stay intact. A flight is counted once for preflight energy. For
power, interval durations weight errors within each flight and reporting slice,
then every contributing flight receives equal weight. RMSE is the square root
of the correspondingly weighted MSE. Bias is prediction minus simulator label.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .direct_airborne_evidence import digest
from .mission_campaign import OUT, read, save
from .mission_observations import table
from .mission_online import evaluate as evaluate_power
from .mission_prediction import ENERGY, condition, dataframe, grouped_folds


# The exploratory cohort has frozen preflight folds and no separately published
# online model. Its optional power rows are identified as a new exploratory CV.
COHORTS = {
    'historical': ('historical_preflight.csv', 'historical_intervals.csv.gz',
                   'historical_preflight_model', 'historical_power_model', {'historical_supervisor'}),
    'reference': ('mission_flights.csv', 'mission_intervals.csv.gz',
                  'preflight_reference', 'online_reference', {'factorial', 'route'}),
    'exploratory': ('mission_flights.csv', 'mission_intervals.csv.gz',
                    'preflight_exploratory', None, {'factorial', 'route', 'sensitivity'}),
}


def same(actual, expected, label):
    if not math.isclose(float(actual),float(expected),rel_tol=1e-10,abs_tol=1e-9):
        raise ValueError(f'{label}: computed {actual}, expected {expected}')


def digit(value):
    value=float(value)
    if not math.isfinite(value):
        raise ValueError('Commanded operating condition contains a nonfinite value')
    return format(value,'.12g')


def slice_labels(row):
    wind=digit(row['wind_speed_mps'])
    direction=digit(float(row['wind_direction_toward_deg_ENU'])%360) if float(wind)>0 else 'CALM'
    return {
        'all':'ALL',
        'payload_kg':digit(row['payload_kg']),
        'commanded_speed_mps':digit(row['cruise_speed_mps']),
        'commanded_altitude_agl_m':digit(row['altitude_m']),
        'wind_speed_mps':wind,
        'wind_toward_deg_ENU':direction,
        'wind_speed_and_direction':'CALM' if direction=='CALM' else f'{wind} m/s toward {direction} deg ENU',
        'mission_route':row['mission_route'],
        'ambient_temperature_C':digit(row['temperature_c']),
        'initial_nominal_soc_percent':digit(row['initial_soc_percent']),
    }


def stats(error, weights):
    return {
        'mae':float(np.average(np.abs(error),weights=weights)),
        'rmse':float(np.sqrt(np.average(np.square(error),weights=weights))),
        'bias':float(np.average(error,weights=weights)),
    }


def summarize(*, cohort, target, basis, ids, folds, observed, predictions, references, flights,
              seconds=None, phases=None):
    """One row per method/reference comparison within each marginal slice."""
    arrays=[np.asarray(a,dtype=float) for a in (observed,predictions,*references.values())]
    if any(a.shape!=(len(ids),) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Incomplete or nonfinite held-out predictions')
    selected=arrays[1]-arrays[0]
    residuals={name:value-arrays[0] for name,value in zip(references,arrays[2:],strict=True)}
    ids=np.asarray(ids)
    folds=np.asarray(folds)
    if seconds is None:
        if len(np.unique(ids))!=len(ids):
            raise ValueError('Duplicated airborne energy label')
    elif len(seconds)!=len(ids) or not np.isfinite(seconds).all() or (seconds<=0).any():
        raise ValueError('Invalid eligible airborne intervals')

    labels={cid:slice_labels(c) for cid,c in flights.items()}
    families={cid:condition(c) for cid,c in flights.items()}
    groups=defaultdict(list)
    for i,cid in enumerate(ids):
        for axis,value in labels[cid].items():
            groups[axis,value].append(i)
        if phases is not None:
            groups['flight_phase_at_interval_start',str(phases[i])].append(i)
    rows=[]
    for (axis,value),indices in sorted(groups.items()):
        index=np.asarray(indices,dtype=int)
        members,within=np.unique(ids[index],return_inverse=True)
        weights=None
        covered=None
        if seconds is not None:
            span=seconds[index]
            weights=span/np.bincount(within,weights=span)[within]
            covered=float(np.sum(span))
        tested=stats(selected[index],weights)
        for name,error in residuals.items():
            baseline=stats(error[index],weights)
            row=dict(cohort=cohort,target=target,unit='Wh' if seconds is None else 'W',validation_basis=basis,
                     slice_axis=axis,slice_value=value,n_flights=len(members),
                     n_command_families=len({families[cid] for cid in members}),n_folds=len(set(folds[index])),
                     n_intervals=len(index) if seconds is not None else None,interval_seconds=covered,
                     weighting='EQUAL_FLIGHT' if seconds is None else 'TIME_WITHIN_SLICE_AND_FLIGHT_THEN_EQUAL_FLIGHT',
                     model='preflight_residual_ridge' if seconds is None else 'next_interval_ridge',reference=name)
            for metric in ('mae','rmse','bias'):
                row['model_'+metric]=tested[metric]
                row['reference_'+metric]=baseline[metric]
                row[metric+'_model_minus_reference']=tested[metric]-baseline[metric]
            rows.append(row)
    return rows


def review_cohort(cohort,source,scratch):
    flight_name,series_name,preflight_name,online_name,allowed=COHORTS[cohort]
    flight_file=source/flight_name
    cohort_rows=[r for r in dataframe(flight_file) if r['group'] in allowed]
    flights={r['case_id']:r for r in cohort_rows}
    if not flights or len(flights)!=len(cohort_rows):
        raise ValueError(f'{cohort}: expected one selected flight per case')
    prediction_file=source/preflight_name/'held_out_predictions.csv'
    pre_eval=source/preflight_name/'evaluation.json'
    predictions=dataframe(prediction_file)
    ids=[r['case_id'] for r in predictions]
    if len(ids)!=len(set(ids)) or set(ids)!=flights.keys():
        raise ValueError(f'{cohort}: held-out flight identifiers differ from the registered cohort')
    ordered=[flights[cid] for cid in ids]
    folds=grouped_folds(ordered)
    fold_map=dict(zip(ids,folds,strict=True))
    for row,expected,fold in zip(predictions,ordered,folds,strict=True):
        if row['run_id']!=expected['run_id'] or row['condition']!=condition(expected) or int(row['fold'])!=fold:
            raise ValueError(f'{cohort}: held-out run, command family or fold differs for {row["case_id"]}')
        same(row['measured_in_sim_energy_Wh'],expected[ENERGY],'Airborne recorder target')

    def column(key):
        return np.array([float(r[key]) for r in predictions])

    energy_rows=summarize(cohort=cohort,target='full_airborne_energy',basis='SAVED_OUT_OF_FOLD_PREDICTIONS',
        ids=ids,folds=folds,observed=column('measured_in_sim_energy_Wh'),predictions=column('predicted_energy_Wh'),
        references={'phase_schedule':column('phase_schedule_energy_Wh'),'train_mean':column('training_mean_energy_Wh')},
        flights=flights)
    official=read(pre_eval)['overall_and_by_payload']['all']
    for row in energy_rows:
        if row['slice_axis']=='all':
            for metric in ('mae','rmse','bias'):
                same(row['model_'+metric],official['residual_ridge']['energy_Wh'][metric],f'Published energy {metric}')
                same(row['reference_'+metric],official[row['reference']]['energy_Wh'][metric],f'Published reference {metric}')

    power_rows=[]
    power_basis='REPRODUCED_PUBLISHED_OUT_OF_FOLD' if online_name else 'ADDITIONAL_EXPLORATORY_OUT_OF_FOLD'
    def collect_power(case_ids,folds,phases,dt,y,predicted,baseline):
        if any(fold_map[cid]!=fold for cid,fold in set(zip(case_ids,folds,strict=True))):
            raise ValueError('Interval model fold differs from the saved preflight fold')
        power_rows.extend(summarize(cohort=cohort,target='following_interval_mean_power',basis=power_basis,
            ids=case_ids,folds=folds,observed=y,predictions=predicted,references={'training_phase_mean':baseline},
            flights=flights,seconds=dt,phases=phases))

    series_file=source/series_name
    run=evaluate_power(series_file,ordered,scratch/cohort,observe_held_out=collect_power)
    overall=next(r for r in power_rows if r['slice_axis']=='all')
    validate={ 'model_mae':'heldout_mae_W_equal_flight', 'reference_mae':'phase_mean_reference_mae_W',
               'model_bias':'heldout_bias_W_equal_flight', 'n_intervals':'eligible_intervals','n_flights':'flights' }
    outputs=[run]
    files=[flight_file,prediction_file,pre_eval,series_file]
    if online_name:
        published=source/online_name/'evaluation.json'
        outputs.append(read(published))
        files.append(published)
    for evaluation in outputs:
        for measured,saved in validate.items():
            same(overall[measured],evaluation[saved],f'{cohort} power {measured}')
    phase_rows=[r for r in power_rows if r['slice_axis']=='flight_phase_at_interval_start']
    same(sum(r['n_intervals'] for r in phase_rows),overall['n_intervals'],'Phase interval count')
    same(sum(r['interval_seconds'] for r in phase_rows),overall['interval_seconds'],'Phase covered time')
    receipt=dict(flights=len(flights),command_families=len(set(condition(r) for r in ordered)),folds=len(set(folds)),
                 eligible_intervals=run['eligible_intervals'],excluded_intervals=run['excluded_intervals'],
                 power_validation_basis=power_basis, input_sha256={str(f.relative_to(source)):digest(f) for f in files},
                 reproduced_energy_mae_Wh=official['residual_ridge']['energy_Wh']['mae'],recomputed_power_mae_W=overall['model_mae'],
                 phase_interval_partition_verified=True)
    return energy_rows+power_rows,receipt


def build(cohorts,source,output):
    rows=[]
    details={}
    with TemporaryDirectory(prefix='m100-error-breakdown-') as temp:
        for cohort in cohorts:
            strata,details[cohort]=review_cohort(cohort,source,Path(temp))
            rows.extend(strata)
    report=output/'held_out_errors_by_condition.csv'
    table(report,rows)
    save(output/'verification.json',dict(
        cohorts=details,table_sha256=digest(report),table_rows=len(rows),labels='M100 simulator only',
        interpretation='Each slice is marginal; a row with one command family describes that setting. '
                       'No physical-flight prediction accuracy is inferred. Confirmation flights are excluded.',
        bias='prediction minus simulator label',differences='ridge metric minus named reference metric',
        power_weighting='Normalize interval duration within each contributing flight and slice, then weight flights equally. '
                        'RMSE is sqrt(equal-flight mean of time-weighted per-flight MSE). Phase is the command at interval start.',
        energy_weighting='One takeoff-to-landing observation per flight. A phase has no preflight whole-flight target.',
        command_family='Mission route, speed, altitude, wind vector command, temperature, initial nominal SOC and planned distances/hover time; '
                       'every seed and payload for that condition stays in the same fold.'))
    return details


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=['all',*COHORTS],default='all')
    parser.add_argument('--source',type=Path,default=OUT)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=build(list(COHORTS) if args.cohort=='all' else [args.cohort],args.source,args.output)
    for cohort,item in result.items():
        print(f"{cohort}: {item['flights']} flights, {item['command_families']} command families, "
              f"airborne energy MAE {item['reproduced_energy_mae_Wh']:.6f} Wh, interval power MAE {item['recomputed_power_mae_W']:.6f} W")
