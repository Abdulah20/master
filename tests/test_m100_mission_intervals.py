import csv
import gzip
import math

import numpy as np
import pytest

from tools.m100_reaudit import mission_observations as obs
from tools.m100_reaudit import mission_online
from tools.m100_reaudit import mission_release as release


CONDITIONS = dict(payload_kg=0.25, cruise_speed_mps=8, altitude_m=50, wind_speed_mps=0,
                  wind_direction_toward_deg_ENU=0, temperature_c=15, initial_soc_percent=80,
                  nameplate_Wh=99.9, mission_route='direct')


def ledger(start, stamp):
    elapsed = stamp - start
    energy = elapsed * 0.125
    sources = {key: {'energy_Wh': energy if key == 'recorded_manager' else 0.0} for key in obs.POWER_SOURCES}
    return dict(run_id='flight', status='AIRBORNE', start_s=start, covered_s=elapsed, uncovered_s=0.0,
                sources=sources, estimated_airborne_energy_Wh=energy)


def packet(start, stamp, **native):
    return dict(time_s=stamp, acknowledged_request=dict(run_id='flight', phase='CLIMB', time_s=stamp),
                native=dict(time_s=stamp, airborne_budget=ledger(start, stamp), **native))


def topics():
    # The fourth state is published after the 10-s prediction point despite its
    # estimator sample preceding 10 s. Its large velocity must stay invisible.
    published = np.array([9_100_000, 9_490_000, 9_980_000, 10_040_000, 10_490_000, 10_980_000, 11_980_000])
    sampled = np.array([9_090_000, 9_460_000, 9_920_000, 9_990_000, 10_480_000, 10_970_000, 11_960_000])
    valid = np.ones(len(sampled), dtype=bool)
    resets = np.zeros(len(sampled), dtype=int)
    position = dict(timestamp=published, timestamp_sample=sampled, z_valid=valid.copy(),
                    v_xy_valid=valid.copy(), v_z_valid=valid.copy(), z_reset_counter=resets.copy(),
                    vxy_reset_counter=resets.copy(), vz_reset_counter=resets.copy(),
                    vx=np.array([0., 2., 12., 999., 4., 3., 4.]), vy=np.zeros(len(sampled)), vz=np.zeros(len(sampled)),
                    z=np.array([0., 0., 0., -2., -3., -4., -8.]))
    return dict(vehicle_local_position=position,
                vehicle_status=dict(timestamp=np.array([9_900_000, 10_150_000]), nav_state=np.array([14, 17])))


def test_boundary_rounded_in_binary_claims_each_integer_second_only_once():
    start_us, end_us = 11_784_000, 28_084_000
    start, end = start_us / 1e6, end_us / 1e6
    stamps = [start_us + n * 1_000_000 for n in range(1, 17)]
    stamps.extend([25_783_999, 25_836_000])
    packets = [packet(start, us / 1e6) for us in sorted(stamps)]
    assert 25.784 - start < 14  # Real example from the supervisor archive.
    points, sources = obs.supported_points(packets, 'flight', start, end, ledger(start, end))
    assert points == [start, *[(start_us + n*1_000_000)/1e6 for n in range(1, 17)], end]
    assert math.fsum(sources[-1].values()) == pytest.approx(.125*(end-start))


def test_causal_position_uses_publication_cutoff_and_elapsed_estimator_samples():
    start, end = 10.0, 11.604
    final = dict(ledger(start, end), end_s=end)
    records = [
        packet(start, start, status='DISCHARGE_DOMAIN_EXHAUSTED_CAP_HELD', current_a=None,
               polarization_slow_v=.51, polarization_load_history_complete=False,
               loaded_charge_coverage_complete=False, q_domain_exhausted=True,
               discharged_ah=.6, unaccounted_load_time_s=2.),
        packet(start, 10.04, current_a=999.),
        packet(start, 11.0),  # Older record without a pack-coverage field.
    ]
    rows = obs.online_series('case', 'flight', records, topics(), final, CONDITIONS)
    first, second = rows
    assert first['px4_mode'] == 'OFFBOARD'
    assert first['horizontal_speed_mps'] == 12
    assert first['horizontal_tangential_accel_mps2'] == pytest.approx((12-2)/(.92-.46))
    assert first['slow_polarization_V'] == .51
    assert first['starting_polarization_history_complete'] is False
    assert first['starting_loaded_charge_history_complete'] is False
    assert first['starting_charge_domain_exhausted'] is True
    assert first['starting_pack_current_available'] is False and first['starting_pack_current_A'] is None
    assert first['starting_accounted_discharge_Ah'] == .6 and first['starting_unaccounted_load_s'] == 2
    assert first['causal_features_available'] is True and mission_online.inputs(first) is not None
    assert second['causal_features_available'] is True
    assert second['starting_polarization_history_complete'] is None
    assert second['starting_charge_domain_exhausted'] is None and second['starting_pack_current_available'] is None
    assert first['nominal_soc_at_start_pct'] == 80
    assert second['nominal_soc_at_start_pct'] == pytest.approx(80 - 100*.125/99.9)
    assert sum(row['future_interval_energy_Wh'] for row in rows) == pytest.approx(final['estimated_airborne_energy_Wh'])


@pytest.mark.parametrize('invalid', ['sample_too_old', 'z_flag', 'z_reset', 'velocity_reset', 'nonfinite_velocity'])
def test_invalid_px4_history_excludes_power_features_and_retains_energy(invalid):
    state = topics()
    pos = state['vehicle_local_position']
    if invalid == 'sample_too_old':
        pos['timestamp_sample'][2] = 9_600_000
    elif invalid == 'z_flag':
        pos['z_valid'][2] = False
    elif invalid == 'z_reset':
        pos['z_reset_counter'][5] = 1
    elif invalid == 'velocity_reset':
        pos['vxy_reset_counter'][2] = 1
    else:
        pos['vx'][2] = math.nan
    start, end = 10.0, 11.604
    final = dict(ledger(start, end), end_s=end)
    rows = obs.online_series('case', 'flight', [packet(start, 10.), packet(start, 11.)], state, final, CONDITIONS)
    row = rows[1 if invalid == 'z_reset' else 0]
    assert row['causal_features_available'] is False and mission_online.inputs(row) is None
    assert row['future_interval_energy_Wh'] > 0


def test_power_training_rejects_intervals_belonging_to_another_attempt(tmp_path, monkeypatch):
    inputs = tmp_path / 'intervals.csv.gz'
    with gzip.open(inputs, 'wt', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('case_id','run_id'))
        writer.writerow(('case_a','rejected_attempt'))
    flights = [dict(case_id='case_a', run_id='accepted_attempt'), dict(case_id='case_b', run_id='other')]
    monkeypatch.setattr(mission_online, 'grouped_folds', lambda rows: [0, 1])
    with pytest.raises(ValueError, match='different flight'):
        mission_online.evaluate(inputs, flights, tmp_path)


@pytest.mark.parametrize('fault', ['none','gap','wrong_run','wrong_power','missing_last_interval'])
def test_published_interval_join_and_energy_closure(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(release,'OUT',tmp_path)
    row=dict(case_id='case_a',run_id='flight',estimated_airborne_energy_Wh=.25,source_audit='audit.json')
    obs.table(tmp_path/'mission_flights.csv',[row])
    release.save(tmp_path/'audit.json',dict(case_id='case_a',run_id='flight',accepted=True,
        checks=dict(px4_takeoff_us=10_000_000,px4_landing_us=12_000_000,recorder_energy_Wh=.25)))
    slices=[dict(case_id='case_a',run_id='flight',interval_start_sim_s=10+i,interval_end_sim_s=11+i,
                 interval_s=1,cumulative_airborne_Wh_at_start=i*.125,cumulative_airborne_Wh_at_end=(i+1)*.125,
                 future_interval_energy_Wh=.125,future_interval_mean_power_W=450,causal_features_available=True)
            for i in range(2)]
    if fault=='gap':
        slices[1]['interval_start_sim_s']+=.1
    elif fault=='wrong_run':
        slices[1]['run_id']='prior_attempt'
    elif fault=='wrong_power':
        slices[1]['future_interval_mean_power_W']=400
    elif fault=='missing_last_interval':
        slices.pop()
    with gzip.open(tmp_path/'mission_intervals.csv.gz','wt',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(slices[0]))
        writer.writeheader()
        writer.writerows(slices)
    if fault=='none':
        counts=release.verify_published_intervals([row])
        assert counts['interval_rows']==counts['interval_rows_with_causal_features']==2
    else:
        with pytest.raises(ValueError,match='Published|unselected'):
            release.verify_published_intervals([row])


def test_online_prediction_uses_only_features_available_at_the_left_edge():
    names=mission_online.FEATURE_NAMES
    slopes=[[1.] if k=='horizontal_speed_mps' else [0.] for k in names]
    model=dict(features=names,target='following_interval_mean_pack_power_W',
               fit=dict(center=[0.]*len(names),scale=[1.]*len(names),coefficients=slopes,offset=[450.]))
    row={key:1. for key in mission_online.NUMERIC}
    row.update(horizontal_speed_mps=8.,causal_features_available=True,px4_mode='OFFBOARD',flight_phase='CRUISE')
    assert mission_online.predict(row,model)==458.
    assert mission_online.predict(dict(row,starting_pack_voltage_V=25.,future_interval_mean_power_W=999.),model)==458.
    assert mission_online.predict(dict(row,px4_mode=None),model) is None
    assert mission_online.predict(dict(row,causal_features_available=False),model) is None
    with pytest.raises(ValueError,match='Unsupported online'):
        mission_online.predict(row,dict(model,features=names[:-1]))
