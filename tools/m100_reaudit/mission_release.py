"""Build verified M100 training tables and register or score independent recommendation flights."""
import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from .direct_airborne_evidence import confirm, digest
from .mission_audit import INTERVAL_CACHE
from .mission_campaign import OUT, read, save
from .mission_history import combine_intervals
from .mission_observations import table
from .mission_online import evaluate as power_evaluation
from .mission_prediction import ENERGY, TIME, dataframe, number, save_evaluation
from .mission_selector import make_confirmation


def inventory(planned):
    accepted=[]
    attempts=[]
    pending=[]
    for c in planned:
        cid=c['case_id']
        audits=[read(path) for path in sorted((OUT/'audits'/cid).glob('attempt_*.json'))]
        selected=[a for a in audits if a['accepted']]
        confirm(len(selected)<=1,'A scenario has more than one accepted attempt')
        for a in audits:
            attempts.append(dict(case_id=cid,attempt=a['attempt'],run_id=a.get('run_id'),
                status='ACCEPTED' if a['accepted'] else 'REJECTED',reason=a['reason'],archive_sha256=a['archive_sha256']))
        if selected:
            accepted.append(selected[0])
        else:
            pending.append(c)
            if not audits:
                attempts.append(dict(case_id=cid,attempt=None,run_id=None,status='PENDING',reason='',archive_sha256=None))
    return accepted,attempts,pending


def release_tables(selected,attempts):
    rows=[]
    files=[]
    for a in selected:
        m=a['metrics']
        file=INTERVAL_CACHE/a['case_id']/f"attempt_{a['attempt']}.csv"
        confirm(digest(file)==a['checks']['intervals_sha256'],'Verified time-series bytes changed')
        audit_file=Path('audits')/a['case_id']/f"attempt_{a['attempt']}.json"
        receipt=read(OUT/'acquisition'/a['case_id']/f"attempt_{a['attempt']}.json")
        rows.append(dict(**m,source_audit=str(audit_file),raw_archive=receipt['archive'],
                         raw_archive_sha256=a['archive_sha256']))
        files.append(file)
    table(OUT/'mission_flights.csv',rows)
    combine_intervals(files,OUT/'mission_intervals.csv.gz')
    table(OUT/'attempt_registry.csv',attempts)
    return rows


def factorial_contrasts(rows):
    factors=('payload_kg','cruise_speed_mps','altitude_m')
    data=[r for r in rows if r['group']=='factorial']
    outcomes=[]
    summaries=[]
    combos=defaultdict(list)
    for c in data:
        combos[tuple(c[k] for k in factors)].append(c)
    for comb,repeats in sorted(combos.items()):
        energies=[r[ENERGY] for r in repeats]
        summaries.append(dict(zip(factors,comb),n=len(repeats),mean_airborne_Wh=float(np.mean(energies)),
                              sd_between_repetitions_Wh=float(np.std(energies,ddof=1)) if len(repeats)>1 else None,
                              min_Wh=min(energies),max_Wh=max(energies),
                              mean_airborne_duration_s=float(np.mean([c[TIME] for c in repeats])),
                              mean_airborne_power_W=float(np.mean([c['airborne_mean_power_W'] for c in repeats]))))
    for factor in factors:
        others=[k for k in factors if k!=factor]
        grouped=defaultdict(dict)
        for c in data:
            key=tuple(c[k] for k in others)
            grouped[key].setdefault(c[factor],{})[c['seed']]=c
        for key,values in sorted(grouped.items()):
            levels=sorted(values)
            for first,last in zip(levels,levels[1:]):
                shared=sorted(values[first].keys()&values[last].keys())
                if not shared:
                    continue
                left=[values[first][s][ENERGY] for s in shared]
                right=[values[last][s][ENERGY] for s in shared]
                deltas=np.asarray(right)-np.asarray(left)
                outcomes.append(dict(factor=factor,**dict(zip(others,key)),from_level=first,to_level=last,
                    paired_seeds='|'.join(map(str,shared)),mean_difference_Wh=float(deltas.mean()),
                    difference_pct=100*float(np.mean(right)/np.mean(left)-1),min_paired_difference_Wh=min(deltas),
                    max_paired_difference_Wh=max(deltas)))
    table(OUT/'factorial_conditions.csv',summaries)
    if outcomes:
        table(OUT/'factorial_paired_effects.csv',outcomes)


def sensitivities(rows):
    ref={(r['mission_route'],r['seed']):r for r in rows if r['group'] in ('factorial','route') and
         (r['payload_kg'],r['cruise_speed_mps'],r['altitude_m'])==(.25,8,50)}
    contrasts=[]
    for c in rows:
        if c['group']!='sensitivity' or (c['mission_route'],c['seed']) not in ref:
            continue
        baseline=ref[c['mission_route'],c['seed']]
        contrasts.append(dict(case_id=c['case_id'],reference_case=baseline['case_id'],route=c['mission_route'],
            wind_mps=c['wind_speed_mps'],wind_toward_deg=c['wind_direction_toward_deg_ENU'],ambient_C=c['temperature_c'],
            nominal_initial_soc_percent=c['initial_soc_percent'],airborne_Wh=c[ENERGY],reference_Wh=baseline[ENERGY],
            difference_Wh=c[ENERGY]-baseline[ENERGY],difference_pct=100*(c[ENERGY]/baseline[ENERGY]-1),
            difference_duration_s=c[TIME]-baseline[TIME],minimum_supported_pack_voltage_V=c['min_supported_load_voltage_V'],
            reference_minimum_supported_voltage_V=baseline['min_supported_load_voltage_V'],
            model_coverage_exhausted=c['low_charge_domain_exhausted'],
            fallback_share_of_energy=c['recorded_manager_energy_Wh']/c[ENERGY],
            scope='PAIRED_SINGLE_SEED_SIMULATOR_SENSITIVITY'))
    if contrasts:
        table(OUT/'environment_and_soc.csv',contrasts)


def field_context(rows):
    field=dataframe(OUT/'field_holdout_by_flight.csv')
    groups=defaultdict(list)
    for r in rows:
        if r['group']=='factorial':
            groups[(r['payload_kg'],r['cruise_speed_mps'],r['altitude_m'])].append(r)
    reference=defaultdict(list)
    for f in field:
        reference[tuple(number(f,k) for k in ('payload_kg','commanded_speed_mps','commanded_altitude_agl_m'))].append(f)
    values=[]
    for k,sims in sorted(groups.items()):
        peers=reference.get(k,[])
        x=np.mean([r['airborne_mean_power_W'] for r in sims])
        y=np.mean([number(f,'field_mean_power_W') for f in peers]) if peers else None
        values.append(dict(payload_kg=k[0],speed_mps=k[1],height_agl_m=k[2],native_runs=len(sims),
            matched_command_field_flights=len(peers),field_ids='|'.join(f['flight_id'] for f in peers),
            mean_sim_airborne_W=float(x),mean_cmu_airborne_W=float(y) if peers else None,
            sim_minus_cmu_mean_W=float(x-y) if peers else None,
            mean_sim_airborne_s=float(np.mean([r[TIME] for r in sims])),
            mean_cmu_airborne_s=float(np.mean([number(f,'field_duration_s') for f in peers])) if peers else None,
            mean_sim_airborne_Wh=float(np.mean([r[ENERGY] for r in sims])),
            mean_cmu_airborne_Wh=float(np.mean([number(f,'field_energy_Wh') for f in peers])) if peers else None,
            identical_itinerary_established=False,scope='MATCHED_COMMANDS_ONLY' if peers else 'NO_SAME_COMMAND_FIELD_HOLDOUT'))
    table(OUT/'field_command_context.csv',values)


def confirmation_coverage(expected, observed):
    if expected and observed == expected:
        return 'COMPLETE'
    return 'PARTIAL' if observed else 'UNAVAILABLE'


def confirm_results(rows,protocol):
    seeds = protocol['independent_seeds']
    planned = {(r['payload_kg'],r['mission_route'],r['cruise_speed_mps'],r['altitude_m'],r['seed']):r['case_id']
               for r in protocol['cases']}
    completed = [r for r in rows if r['group']=='confirmation']
    observations = {r['case_id']:r for r in completed}
    confirm(len(planned)==len(protocol['cases']) and len(set(planned.values()))==len(planned),
            'Duplicate planned confirmation command or identifier')
    confirm(len(observations)==len(completed) and
            all(planned.get((r['payload_kg'],r['mission_route'],r['cruise_speed_mps'],r['altitude_m'],r['seed']))==r['case_id']
                for r in completed), 'Confirmation record does not match its unique registered command and seed')
    confirm(seeds and len(seeds)==len(set(seeds)) and all(key[-1] in seeds for key in planned),
            'Confirmation seeds disagree with the frozen protocol')
    results = []
    by_payload = {}
    deadline = protocol['maximum_duration_s']
    for mass,selection in protocol['selection'].items():
        payload = float(mass)
        winner = selection['winner']
        reference = selection['reference']
        alternatives = selection['comparators']
        if reference is None:
            relation = 'REFERENCE_UNAVAILABLE_IN_DISCOVERY'
        elif reference == winner:
            relation = 'REFERENCE_IS_WINNER'
        else:
            relation = 'DISTINCT_REFERENCE'
        groups = []
        winner_observed = []
        for seed in seeds:
            winner_id = (planned[payload,winner['route'],winner['cruise_mps'],winner['height_agl_m'],seed]
                         if winner else None)
            w = observations.get(winner_id)
            if w:
                winner_observed.append(seed)
            # Preserve a coverage row even when the required reference and the
            # winner coincide and no other feasible command exists.
            for alt in alternatives or [dict(roles=[],choice=None)]:
                choice = alt['choice']
                roles = '|'.join(alt['roles'])
                comp_id = (planned[payload,choice['route'],choice['cruise_mps'],choice['height_agl_m'],seed]
                           if choice else None)
                c = observations.get(comp_id)
                confirm(not choice or (winner_id and comp_id != winner_id), 'Cannot compare a flight with itself')
                status = 'FEASIBLE_PAIR'
                if not choice:
                    status = 'NO_DISTINCT_COMPARATOR' if winner else 'NO_FEASIBLE_DISCOVERY_OPTION'
                elif not w or not c:
                    missing = [role for role,obs in [('WINNER',w),('COMPARATOR',c)] if obs is None]
                    status = 'MISSING_'+'_AND_'.join(missing)
                elif (min(w['nominal_reserve_margin_Wh'],c['nominal_reserve_margin_Wh']) < 0 or
                      (deadline is not None and max(w[TIME],c[TIME]) > deadline)):
                    status = 'OBSERVED_NOMINAL_CONSTRAINT_VIOLATION'
                delta = c[ENERGY]-w[ENERGY] if w and c else None
                results.append(dict(payload_kg=payload,seed=seed,comparison_roles=roles,pair_status=status,
                    winner_case=winner_id,winner_run_id=w['run_id'] if w else None,
                    comparator_case=comp_id,comparator_run_id=c['run_id'] if c else None,
                    reference_case=comp_id if 'reference' in alt['roles'] else None,
                    selected_route=winner['route'] if winner else None,selected_speed_mps=winner['cruise_mps'] if winner else None,
                    selected_altitude_m=winner['height_agl_m'] if winner else None,
                    comparator_route=choice['route'] if choice else None,comparator_speed_mps=choice['cruise_mps'] if choice else None,
                    comparator_altitude_m=choice['height_agl_m'] if choice else None,
                    predicted_winner_Wh=winner['energy_Wh'] if winner else None,actual_winner_Wh=w[ENERGY] if w else None,
                    predicted_comparator_Wh=choice['energy_Wh'] if choice else None,actual_comparator_Wh=c[ENERGY] if c else None,
                    observed_difference_Wh=delta,actual_saved_Wh=delta if status=='FEASIBLE_PAIR' else None,
                    actual_saved_pct=100*delta/c[ENERGY] if status=='FEASIBLE_PAIR' else None,
                    actual_winner_duration_s=w[TIME] if w else None,
                    actual_nominal_landing_soc_pct=w['nominal_remaining_soc_percent'] if w else None,
                    actual_nominal_reserve_margin_Wh=w['nominal_reserve_margin_Wh'] if w else None,
                    comparator_duration_s=c[TIME] if c else None,comparator_nominal_landing_soc_pct=c['nominal_remaining_soc_percent'] if c else None,
                    comparator_nominal_reserve_margin_Wh=c['nominal_reserve_margin_Wh'] if c else None,
                    genuinely_distinct_alternatives=bool(winner_id and comp_id and winner_id!=comp_id)))
        for alt in alternatives:
            roles = '|'.join(alt['roles'])
            pair_rows = [r for r in results if r['payload_kg']==payload and r['comparison_roles']==roles]
            matched = [r for r in pair_rows if r['observed_difference_Wh'] is not None]
            feasible = [r for r in pair_rows if r['pair_status']=='FEASIBLE_PAIR']
            savings = [r['actual_saved_Wh'] for r in feasible]
            order = 'UNRESOLVED_MISSING_OR_INFEASIBLE_PAIRS'
            if len(savings)==len(seeds):
                if all(value>0 for value in savings):
                    order = 'WINNER_USED_LESS_ON_ALL_CONFIRMATION_SEEDS'
                elif all(value<0 for value in savings):
                    order = 'COMPARATOR_USED_LESS_ON_ALL_CONFIRMATION_SEEDS'
                else:
                    order = 'ORDER_VARIED_OR_INCLUDED_EQUAL_ENERGY'
            predicted_gap = alt['choice']['energy_Wh']-winner['energy_Wh']
            groups.append(dict(roles=alt['roles'],predicted_gap_Wh=predicted_gap,
                within_discovery_error_band=abs(predicted_gap)<=selection['tie_band_Wh'],
                descriptive_error_band_Wh=selection['tie_band_Wh'],
                mean_observed_difference_on_feasible_pairs_Wh=(sum(savings)/len(savings) if savings else None),
                min_observed_difference_on_feasible_pairs_Wh=min(savings,default=None),
                max_observed_difference_on_feasible_pairs_Wh=max(savings,default=None),
                expected_seeds=seeds,matched_seeds=[r['seed'] for r in matched],
                feasible_seeds=[r['seed'] for r in feasible],coverage=confirmation_coverage(len(seeds),len(matched)),
                observed_energy_order=order))
        by_payload[mass] = dict(reference_relation=relation,winner_observed_seeds=winner_observed,
            winner_coverage=confirmation_coverage(len(seeds),len(winner_observed)),comparisons=groups)
    expected = sum(r['genuinely_distinct_alternatives'] for r in results)
    observed = sum(r['observed_difference_Wh'] is not None for r in results)
    feasible = sum(r['pair_status']=='FEASIBLE_PAIR' for r in results)
    report = dict(expected_distinct_seed_pairs=expected,observed_distinct_seed_pairs=observed,
                  pairs_satisfying_nominal_constraints=feasible,coverage=confirmation_coverage(expected,observed),
                  by_payload=by_payload)
    table(OUT/'confirmed_savings.csv',results)
    save(OUT/'confirmation_coverage.json',report)
    return report

def build(stage):
    protocol=read(OUT/'protocol.json')
    confirmation_file=OUT/'confirmation_protocol.json'
    confirm_plan=read(confirmation_file) if confirmation_file.exists() else {}
    selected,attempts,pending=inventory(protocol['cases']+confirm_plan.get('cases',[]))
    if not selected:
        return
    if stage!='progress' and pending:
        # A scientifically unsupported flight remains explicitly excluded after its documented retries.
        untouched=[c['case_id'] for c in pending if not (OUT/'audits'/c['case_id']/'attempt_3.json').exists()]
        if untouched:
            raise ValueError(f'{len(untouched)} cases are still pending acquisition or audit')
    rows=release_tables(selected,attempts)
    if stage=='progress':
        return
    discovery=[r for r in rows if r['group']!='confirmation']
    reference=[r for r in discovery if r['group'] in ('factorial','route')]
    factorial_contrasts(rows)
    sensitivities(rows)
    field_context(rows)
    if stage=='freeze':
        model=save_evaluation(reference,OUT/'preflight_reference')
        save_evaluation(discovery,OUT/'preflight_exploratory')
        power_evaluation(OUT/'mission_intervals.csv.gz',reference,OUT/'online_reference')
        make_confirmation(reference,model,OUT/'preflight_reference/model.json')
        return
    confirm(stage=='final' and confirm_plan and digest(OUT/'preflight_reference/model.json')==confirm_plan['model_sha256'],
            'Confirmation requires a frozen model and recommendation register')
    checks=confirm_results(rows,confirm_plan)
    counts={k:sum(r['group']==k for r in rows) for k in ('factorial','route','sensitivity','confirmation')}
    save(OUT/'verification.json',dict(registered=len(protocol['cases'])+len(confirm_plan['cases']),accepted=len(rows),
        attempts=sum(r['status']!='PENDING' for r in attempts),excluded_after_retries=[c['case_id'] for c in pending],
        counts=counts,paired_confirmation_rows=checks['observed_distinct_seed_pairs'],
        confirmation_coverage=checks['coverage'],eligible_paired_confirmation_rows=checks['pairs_satisfying_nominal_constraints'],
        expected_paired_confirmation_rows=checks['expected_distinct_seed_pairs'],new_preflight_csv_sha256=digest(OUT/'mission_flights.csv'),
        new_series_sha256=digest(OUT/'mission_intervals.csv.gz'),model_sha256=confirm_plan['model_sha256'],
        positive_uncovered_s=sum(r['energy_uncovered_s']>0 for r in rows),
        maximum_pack_equation_residual_V=max(r['native_2rc_equation_max_residual_V'] or 0 for r in rows),
        maximum_supported_power_residual_W=max(r['pack_power_balance_max_residual_W'] or 0 for r in rows),
        full_energy_observation='Direct airborne recorder target. Same PX4 takeoff and landing edges. Zero uncovered time.',
        independent_confirmation_scope='Separate Gazebo seeds; paired coverage and actual constraint checks are in confirmation_coverage.json.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['progress','freeze','final'])
    build(p.parse_args().stage)
