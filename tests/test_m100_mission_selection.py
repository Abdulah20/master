"""Fixed-destination features and independent paired confirmation without launching the simulator."""
import csv
import json
import math

import numpy as np
import pytest

from tools.m100_reaudit import mission_prediction as prediction
from tools.m100_reaudit import mission_release as release
from tools.m100_reaudit import mission_selector as selector
from tools.m100_reaudit.mission_campaign import save


def planned(route, wind, direction):
    points = [(0, 0), (150, 0), (0, 0)]
    if route == 'north_dogleg':
        points.insert(1, (75, 37.5))
    legs = [dict(start_enu_m=a, end_enu_m=b) for a, b in zip(points, points[1:])]
    length = sum(math.dist(leg['start_enu_m'],leg['end_enu_m']) for leg in legs[:-1])
    return dict(payload_kg=.25, cruise_speed_mps=8, altitude_m=50, temperature_c=15,
                wind_speed_mps=wind,wind_direction_toward_deg_ENU=direction,initial_soc_percent=100,
                planned_outbound_m=length,planned_return_m=150,mission_route=route,
                intermediate_stops=len(legs)-2,leg_conditions_json=json.dumps(legs))


def test_leg_wind_features_use_preflight_geometry_and_reverse_the_return(monkeypatch):
    monkeypatch.setattr(prediction,'phase_powers',lambda *args:(500.,450.,420.))
    wind_keys=('outbound_air_path_delta_m','return_air_path_delta_m',
               'outbound_crosswind_exposure_m','return_crosswind_exposure_m')

    def wind_terms(row):
        terms=dict(zip(prediction.FEATURES,prediction.features(row),strict=True))
        return [terms[key] for key in wind_keys]

    direct=planned('direct',3,0)
    dogleg=planned('north_dogleg',3,0)
    assert wind_terms(direct) == pytest.approx([-3*150/8,3*150/3.6,0.,0.])
    assert wind_terms(dogleg)[0] != pytest.approx(wind_terms(direct)[0])
    assert wind_terms(dogleg)[1] == pytest.approx(wind_terms(direct)[1])
    assert wind_terms(dogleg)[2] > 0
    assert wind_terms(planned('north_dogleg',0,90)) == [0.]*4
    assert wind_terms(planned('direct',3,180))[:2] == pytest.approx([3*150/8,-3*150/3.6])
    # Post-flight observations cannot enter a preflight ranking.
    changed=dict(dogleg,estimated_airborne_energy_Wh=500,airborne_duration_s=9999,
                 nominal_remaining_soc_percent=-10,airborne_horizontal_distance_m=999)
    assert prediction.features(dogleg) == prediction.features(changed)
    legacy=dict(direct,mission_route='eastbound_measurement_then_return',leg_conditions_json='',planned_return_m=None)
    assert wind_terms(legacy)==pytest.approx(wind_terms(direct))


def test_command_folds_do_not_split_loads_or_repeated_seeds():
    rows=[dict(planned(r,0,0),payload_kg=p,seed=s) for r in ('direct','north_dogleg')
          for p in (0,.25,.5) for s in (1701,1702,1703)]
    folds=prediction.grouped_folds(rows)
    assert set(folds[:9]).isdisjoint(folds[9:])
    np.testing.assert_array_equal(folds[:9],np.repeat(folds[0],9))
    np.testing.assert_array_equal(folds[9:],np.repeat(folds[9],9))


def test_reserve_duration_and_missing_condition_remain_explicit(monkeypatch):
    monkeypatch.setattr(selector,'predict',lambda row,model:(20.,100.))
    row=dict(planned('direct',3,0),group='factorial',seed=1701,destination_east_m=150,
             nameplate_Wh=100,estimated_airborne_energy_Wh=19.,airborne_duration_s=99.)
    model=dict(validation_absolute_error_p90_Wh=1.)
    condition=dict(payload=.25,wind=3,direction=360.)
    assert selector.options([row],model,**condition,reserve=80.,max_duration=100.)[0]['feasible']
    result=selector.options([row],model,**condition,reserve=81.,max_duration=99.)[0]
    assert not result['feasible']
    assert set(result['rejection'].split('|'))=={'NOMINAL_LANDING_RESERVE','MISSION_DEADLINE'}
    assert selector.options([row],model,**condition,distance=151.)==[]


def choice(v,h,energy,near=True):
    return dict(route='direct',cruise_mps=v,height_agl_m=h,energy_Wh=energy,feasible=True,
                within_combined_error_of_lowest=near)


def register(tmp_path,monkeypatch,ranking):
    monkeypatch.setattr(selector,'OUT',tmp_path)
    monkeypatch.setattr(release,'OUT',tmp_path)
    save(tmp_path/'protocol.json',{'cases':[{'seed':s} for s in (1701,1702,1703)]})
    model={'validation_absolute_error_p90_Wh':.5}
    file=tmp_path/'model.json'
    save(file,model)
    monkeypatch.setattr(selector,'options',lambda rows,model,payload:ranking)
    return selector.make_confirmation([],model,file)


def observed(plan):
    result=[]
    for c in plan['cases']:
        result.append(dict(**c,run_id=c['case_id']+'_run',estimated_airborne_energy_Wh=15.+c['altitude_m']/10-c['cruise_speed_mps']/10,
                           airborne_duration_s=120,nominal_reserve_margin_Wh=45.,nominal_remaining_soc_percent=65.))
    return result


def paired_rows(tmp_path):
    with (tmp_path/'confirmed_savings.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def test_close_runner_up_is_paired_with_winner_on_new_seeds(tmp_path,monkeypatch):
    plan=register(tmp_path,monkeypatch,[choice(10,25,15),choice(8,25,15.5),choice(8,50,22,False)])
    assert len(plan['cases'])==18 and plan['additional_runner_up_flights']==6
    assert len({c['case_id'] for c in plan['cases']})==18
    assert {c['seed'] for c in plan['cases']}=={2711,2712}
    report=release.confirm_results(observed(plan),plan)
    assert report['coverage']=='COMPLETE'
    assert report['observed_distinct_seed_pairs']==report['pairs_satisfying_nominal_constraints']==12
    assert all(r['winner_case']!=r['comparator_case'] for r in paired_rows(tmp_path))
    assert {r['comparison_roles'] for r in paired_rows(tmp_path)}=={'reference','closest_runner_up'}
    assert all(r['observed_energy_order']=='WINNER_USED_LESS_ON_ALL_CONFIRMATION_SEEDS'
               for r in report['by_payload']['0.25']['comparisons'])


def test_reference_winner_has_no_artificial_zero_saving(tmp_path,monkeypatch):
    plan=register(tmp_path,monkeypatch,[choice(8,50,20),choice(8,25,24,False)])
    assert len(plan['cases'])==12
    report=release.confirm_results(observed(plan),plan)
    assert report['by_payload']['0.0']['reference_relation']=='REFERENCE_IS_WINNER'
    assert {r['comparison_roles'] for r in paired_rows(tmp_path)}=={'closest_runner_up'}
    assert all(r['genuinely_distinct_alternatives']=='True' for r in paired_rows(tmp_path))
    assert all(r['reference_case']=='' and r['winner_case']!=r['comparator_case'] for r in paired_rows(tmp_path))


def test_partial_absent_and_infeasible_confirmation_are_distinguished(tmp_path,monkeypatch):
    # The reference is also the nearest challenger. It is flown just once per seed.
    plan=register(tmp_path,monkeypatch,[choice(10,25,20),choice(8,50,20.2)])
    assert len(plan['cases'])==12 and plan['additional_runner_up_flights']==0
    assert plan['selection']['0.25']['comparators'][0]['roles']==['reference','closest_runner_up']
    rows=observed(plan)
    rows=[r for r in rows if not (r['payload_kg']==0 and r['altitude_m']==50 and r['seed']==2712)
          and r['payload_kg']!=.5]
    for r in rows:
        if r['payload_kg']==.25 and r['seed']==2712 and r['altitude_m']==25:
            r['nominal_reserve_margin_Wh']=-1.
    report=release.confirm_results(rows,plan)
    assert report['expected_distinct_seed_pairs']==6 and report['observed_distinct_seed_pairs']==3
    assert report['pairs_satisfying_nominal_constraints']==2
    assert report['coverage']=='PARTIAL'
    first=lambda mass:report['by_payload'][mass]['comparisons'][0]
    assert first('0.0')['coverage']=='PARTIAL' and first('0.5')['coverage']=='UNAVAILABLE'
    assert first('0.25')['coverage']=='COMPLETE' and first('0.25')['feasible_seeds']==[2711]
    assert first('0.25')['observed_energy_order']=='UNRESOLVED_MISSING_OR_INFEASIBLE_PAIRS'
    partial=[r for r in paired_rows(tmp_path) if r['pair_status']!='FEASIBLE_PAIR']
    assert all(r['actual_saved_Wh']=='' for r in partial)
    assert any(r['observed_difference_Wh']!='' for r in partial)


def test_infeasible_reference_without_alternative_has_no_verified_pair(tmp_path,monkeypatch):
    plan=register(tmp_path,monkeypatch,[dict(choice(8,50,90),feasible=False)])
    assert plan['cases']==[]
    report=release.confirm_results([],plan)
    assert report['coverage']=='UNAVAILABLE' and report['observed_distinct_seed_pairs']==0
    assert all(r['pair_status']=='NO_FEASIBLE_DISCOVERY_OPTION' for r in paired_rows(tmp_path))


def test_confirmation_rejects_seed_mismatch_even_when_the_record_id_matches(tmp_path,monkeypatch):
    plan=register(tmp_path,monkeypatch,[choice(10,25,20),choice(8,50,21)])
    rows=observed(plan)
    rows[0]['seed']=1701
    with pytest.raises(ValueError,match='command and seed'):
        release.confirm_results(rows,plan)
