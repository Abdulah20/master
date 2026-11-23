"""Rank only flown, fixed-destination simulator alternatives under explicit nominal mission constraints."""
import argparse
from collections import defaultdict
import math

import numpy as np

from .direct_airborne_evidence import digest
from .mission_campaign import OUT, make_case, read, save
from .mission_prediction import ENERGY, TIME, dataframe, number, predict


def options(rows,model,*,payload,wind=0.,direction=0.,ambient=15.,soc=100.,reserve=20.,max_duration=None,distance=150.):
    numbers=[payload,wind,direction,ambient,soc,reserve,distance]
    if not all(math.isfinite(x) for x in numbers) or payload<0 or wind<0 or not 0<=reserve<=soc<=100 or distance<=0:
        raise ValueError('Invalid payload, wind, distance, charge or reserve')
    if max_duration is not None and (not math.isfinite(max_duration) or max_duration<=0):
        raise ValueError('Maximum duration must be a positive number of seconds')
    groups=defaultdict(list)
    wanted=dict(payload_kg=payload,wind_speed_mps=wind,temperature_c=ambient,initial_soc_percent=soc,destination_east_m=distance)
    for row in rows:
        if row['group']=='confirmation' or any(not math.isclose(number(row,k),v,abs_tol=1e-8) for k,v in wanted.items()):
            continue
        if wind and abs(math.remainder(number(row,'wind_direction_toward_deg_ENU')-direction,360.))>1e-8:
            continue
        groups[(row['mission_route'],number(row,'cruise_speed_mps'),number(row,'altitude_m'))].append(row)
    ranked=[]
    for (route,v,h),repeats in groups.items():
        reference=repeats[0]
        predicted,duration=predict(reference,model)
        nameplate=number(reference,'nameplate_Wh')
        margin=nameplate*(soc-reserve)/100-predicted
        failed=[]
        if margin<0:
            failed.append('NOMINAL_LANDING_RESERVE')
        if max_duration is not None and duration>max_duration:
            failed.append('MISSION_DEADLINE')
        energies=[number(r,ENERGY) for r in repeats]
        ranked.append(dict(route=route,cruise_mps=v,height_agl_m=h,energy_Wh=predicted,duration_s=duration,
            payload_kg=payload,nominal_landing_soc_pct=soc-100*predicted/nameplate,nominal_reserve_pct=reserve,
            nominal_margin_Wh=margin,airborne_mean_power_W=predicted*3600/duration,
            conservative_margin_after_oof_p90_Wh=margin-model['validation_absolute_error_p90_Wh'],
            search_scope='EXACT_FLOWN_COMMANDS_AND_GEOMETRIES',simulation_replicates=len(repeats),
            discovery_mean_Wh=float(np.mean(energies)),discovery_range_Wh=max(energies)-min(energies),
            discovery_mean_duration_s=float(np.mean([number(r,TIME) for r in repeats])),
            feasible=not failed,rejection='|'.join(failed)))
    ranked.sort(key=lambda r:(not r['feasible'],r['energy_Wh'],r['duration_s'],r['route']))
    feasible=[r for r in ranked if r['feasible']]
    if feasible:
        threshold=2*model['validation_absolute_error_p90_Wh']
        winner=feasible[0]
        for choice in feasible:
            choice['within_combined_error_of_lowest']=choice['energy_Wh']-winner['energy_Wh']<=threshold
    return ranked


def make_confirmation(rows,model,model_path):
    discovery_file = OUT/'protocol.json'
    seeds = (2711,2712)
    if {c['seed'] for c in read(discovery_file)['cases']}.intersection(seeds):
        raise ValueError('A confirmation seed already belongs to the discovery protocol')
    selected = {}
    cases = []
    additional_flights = 0
    for weight in (0.,.25,.5):
        ranked = options(rows,model,payload=weight)
        feasible = [r for r in ranked if r['feasible']]
        winner = feasible[0] if feasible else None
        runner_up = feasible[1] if len(feasible)>1 else None
        reference = next((r for r in ranked if (r['route'],r['cruise_mps'],r['height_agl_m'])==('direct',8,50)),None)
        comparators = []
        if winner and reference and reference != winner:
            comparators.append(dict(roles=['reference'],choice=reference))
        if runner_up and (runner_up['within_combined_error_of_lowest'] or winner == reference):
            if runner_up == reference:
                comparators[0]['roles'].append('closest_runner_up')
            else:
                comparators.append(dict(roles=['closest_runner_up'],choice=runner_up))
                additional_flights += len(seeds)
        selected[str(weight)] = dict(winner=winner,reference=reference,closest_runner_up=runner_up,
            ranking=ranked,comparators=comparators,tie_band_Wh=2*model['validation_absolute_error_p90_Wh'])
        if winner:
            for choice in [winner, *(c['choice'] for c in comparators)]:
                for seed in seeds:
                    cases.append(make_case('confirmation',weight,int(choice['cruise_mps']),
                                           int(choice['height_agl_m']),choice['route'],seed))
    proposed=dict(parent_sha256=digest(discovery_file),model_sha256=digest(model_path),selection=selected,cases=cases,
                  reserve_basis='20 percent nominal TB47D nameplate',maximum_duration_s=None,
                  independent_seeds=list(seeds),additional_runner_up_flights=additional_flights,
                  verification='Paired seeds for the winner and the registered reference (direct, 8 m/s, 50 m). '
                  'Also compare the closest feasible runner-up when its predicted gap is within twice the '
                  'discovery out-of-fold absolute-error P90 or when the reference is the winner. '
                  'A shared reference/runner-up is flown once per seed. The tie band is a descriptive heuristic.')
    path=OUT/'confirmation_protocol.json'
    if path.exists() and read(path)!=proposed:
        raise ValueError('Previously frozen selection differs')
    if not path.exists():
        save(path,proposed)
    return proposed


def main():
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--payload',type=float,required=True)
    p.add_argument('--distance',type=float,default=150.)
    p.add_argument('--wind',type=float,default=0.)
    p.add_argument('--direction',type=float,default=0.)
    p.add_argument('--ambient',type=float,default=15.)
    p.add_argument('--soc',type=float,default=100.)
    p.add_argument('--reserve',type=float,default=20.)
    p.add_argument('--max-duration',type=float)
    c=p.parse_args()
    core=c.wind==0 and c.ambient==15 and c.soc==100
    model=read(OUT/('preflight_reference' if core else 'preflight_exploratory')/'model.json')
    ranked=options(dataframe(OUT/'mission_flights.csv'),model,**vars(c))
    feasible=[x for x in ranked if x['feasible']]
    status='FEASIBLE_TESTED_OPTIONS' if feasible else ('NO_FEASIBLE_TESTED_OPTION' if ranked else 'NO_MATCHING_FLOWN_CONDITION')
    print(json.dumps(dict(status=status,lowest_predicted_energy=feasible[0] if feasible else None,
                          candidates=ranked),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
