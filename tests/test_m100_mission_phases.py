import csv
import json
import math

import numpy as np
import pytest

from tools.m100_reaudit import mission_phases as phases


def budget(at):
    elapsed = min(max(at-1., 0.), .6)
    first = min(elapsed, .35)
    seconds = (first, elapsed-first, 0., 0.)
    sources = {s: dict(seconds=t, energy_Wh=t*p/3600)
               for s, t, p in zip(phases.POWER_SOURCES, seconds, (400, 700, 0, 0), strict=True)}
    return dict(run_id='flight', status='COMPLETE' if at >= 1.6 else 'AIRBORNE',
                boundary_source=phases.BOUNDARY_SOURCE, start_s=1., end_s=1.6 if at >= 1.6 else None,
                uncovered_s=0., estimated_airborne_energy_Wh=sum(x['energy_Wh'] for x in sources.values()),
                sources=sources)


def packet(at):
    demand = at-.01
    return dict(run_id='flight', epoch=1, time_s=at, demand_fresh=True,
                native=dict(time_s=at, demand_fresh=True, model='drone', demand_time_s=demand, airborne_budget=budget(at),
                            effective_demand_source='native_pack_model' if at <= 1.35 else 'cmu_local',
                            command_limited=False, voltage_v=20., current_a=21. if at <= 1.35 else 32.5),
                acknowledged_request=dict(time_s=demand, model='drone', producer='m100_experiment_manager',
                                          run_id='flight', phase='STABILIZING' if demand < 1.285 else 'CRUISE'))


def example():
    packets = [packet(ms/1000) for ms in range(940, 1741, 100)]
    final = budget(1.6)
    times = list(range(900_000, 1_710_000, 10_000))
    manager = phases.Observations(times, [('STABILIZING' if t < 1_285_000 else 'CRUISE', t >= 1_120_000)
                                          for t in times])
    status = dict(timestamp=[890_000, 1_100_000, 1_300_000, 1_520_000, 1_720_000],
                  nav_state=[17,17,17,14,14], nav_state_timestamp=[800_000]*3+[1_455_000]*2)
    ground = dict(timestamp_sample=np.arange(900_000, 1_710_000, 200_000),
                  y=np.array([0,2,2,0,0]), x=np.array([0,0,2,2,0]), z=np.arange(5)*-1.,
                  xy_reset_counter=np.zeros(5), z_reset_counter=np.zeros(5))
    flight = dict(case_id='example', run_id='flight', attempt=1, source_audit='audit', raw_archive='archive', raw_archive_sha256='hash',
                  airborne_horizontal_distance_m=6., airborne_duration_s=.6,
                  estimated_airborne_energy_Wh=(400*.35+700*.25)/3600)
    for source, entry in final['sources'].items():
        flight[source+'_energy_Wh'] = entry['energy_Wh']
        flight[source+'_coverage_s'] = entry['seconds']
    return flight, packets, final, manager, ground, status


def derive(flight, packets, final, manager, ground, status):
    native, _ = phases.native_observations(packets, 'flight', final)
    return phases.segments(flight, native, manager, ground, status)


def test_irregular_edges_mixed_sources_and_turning_path_close_without_prorating_changes():
    rows = derive(*example())
    assert rows[0]['start_sim_s'] == 1. and rows[-1]['end_sim_s'] == 1.6
    assert all(l['end_sim_s'] == r['start_sim_s'] for l, r in zip(rows, rows[1:]))
    assert rows[0]['start_basis'] == 'PX4_TAKEOFF_DETECTOR' and rows[-1]['end_basis'] == 'PX4_LANDING_DETECTOR'
    assert math.fsum(r['horizontal_distance_m'] for r in rows) == pytest.approx(6.)
    assert (math.fsum(r['net_east_m'] for r in rows), math.fsum(r['net_north_m'] for r in rows)) == pytest.approx((-1., 1.))
    assert rows[0]['start_height_from_takeoff_m'] == 0. and rows[-1]['end_height_from_takeoff_m'] == pytest.approx(3.)
    assert any(r['px4_mode'] == 'AUTO_TAKEOFF|OFFBOARD' and r['flight_phase'] == 'CRUISE' for r in rows)
    transition, = [r for r in rows if r['manager_phase_change_bracket_present']]
    assert transition['phase_unresolved'] is True and transition['phase_unattributed_Wh'] == transition['estimated_energy_Wh']
    assert json.loads(transition['manager_phase_change_brackets_json']) == [
        dict(last_old_sim_s=1.28, first_new_sim_s=1.29, from_phase='STABILIZING', to_phase='CRUISE')]
    assert transition['start_sim_s'] <= 1.28 < 1.29 <= transition['end_sim_s']
    assert not any(r['phase_unresolved'] and r['route_direction'] != 'UNRESOLVED' for r in rows)
    assert any(r['manager_measurement_gate'] == 'UNRESOLVED' and r['flight_phase'] == 'STABILIZING' for r in rows)
    for r in rows:
        assert math.fsum(r[s+'_s'] for s in phases.POWER_SOURCES) == pytest.approx(r['duration_s'])
        assert math.fsum(r[s+'_Wh'] for s in phases.POWER_SOURCES) == pytest.approx(r['estimated_energy_Wh'])
        assert r['estimated_energy_Wh'] == pytest.approx(r['mean_power_W']*r['duration_s']/3600)


@pytest.mark.parametrize('change', ['missing', 'stale', 'foreign', 'wrong_echo', 'future'])
def test_invalid_ack_leaves_phase_unassigned_and_keeps_full_native_energy(change):
    flight, packets, final, *other = example()
    p = next(p for p in packets if p['time_s'] == 1.44)
    if change == 'missing':
        p['acknowledged_request'] = None
    elif change == 'stale':
        p['native']['demand_fresh'] = False
    elif change == 'foreign':
        p['acknowledged_request']['run_id'] = 'another'
    elif change == 'wrong_echo':
        p['native']['demand_time_s'] += .01
    else:
        p['native']['demand_time_s'] = p['acknowledged_request']['time_s'] = p['time_s']+.001
    rows = derive(flight, packets, final, *other)
    unresolved = [r for r in rows if r['phase_attribution_method'] == 'MISSING_OR_INVALID_ACKNOWLEDGEMENT']
    assert unresolved and all(r['phase_unresolved'] for r in unresolved)
    assert math.fsum(r['duration_s'] for r in unresolved) == pytest.approx(.2)
    assert math.fsum(r['estimated_energy_Wh'] for r in rows) == pytest.approx(flight['estimated_airborne_energy_Wh'])


def test_vi_independent_trapezoids_use_only_same_unmixed_spans_and_keep_empty_estimate_missing():
    flight, packets, final, manager, ground, status = example()
    native, vi = phases.native_observations(packets, 'flight', final)
    rows = phases.segments(flight, native, manager, ground, status)
    check = phases.flight_checks(flight, rows, vi)
    assert check['vi_eligible_intervals'] == 4 and check['vi_eligible_s'] == pytest.approx(.4)
    assert check['vi_coverage_fraction'] == pytest.approx(2/3)
    assert check['published_VI_trapezoids_Wh'] == pytest.approx((420*.3+650*.1)/3600)
    assert check['same_intervals_native_Wh'] == pytest.approx((400*.3+700*.1)/3600)
    assert check['published_minus_native_Wh'] == pytest.approx(1/3600)
    for p in packets:
        p['native']['current_a'] = None
    _, unsupported = phases.native_observations(packets, 'flight', final)
    check = phases.flight_checks(flight, rows, unsupported)
    assert check['vi_eligible_intervals'] == 0 and check['vi_eligible_s'] == 0 and check['vi_coverage_fraction'] == 0
    assert check['published_VI_trapezoids_Wh'] is None and check['same_intervals_native_Wh'] is None
    assert check['published_minus_native_Wh'] is None


def test_missing_native_publications_retain_energy_with_unknown_phase_and_visible_gap():
    flight, packets, final, *other = example()
    packets = [p for p in packets if not 1.1 < p['time_s'] < 1.4]
    rows = derive(flight, packets, final, *other)
    missing, = [r for r in rows if r['phase_attribution_method'] == 'NATIVE_PUBLICATION_GAP']
    assert missing['duration_s'] == pytest.approx(.4)
    assert missing['phase_unattributed_Wh'] > 0 and missing['energy_uncovered_s'] == 0
    assert missing['max_counter_interval_s'] == .4


def test_published_waiting_zero_after_stamped_takeoff_is_not_an_airborne_counter_observation():
    flight, packets, final, manager, ground, status = example()
    waiting = {**budget(1.), 'status':'WAITING_FOR_TAKEOFF', 'start_s':None, 'end_s':None}
    packets[1]['native']['airborne_budget'] = waiting
    native, vi = phases.native_observations(packets, 'flight', final)
    assert native[0][:2] == [1_000_000, 1_140_000]
    rows = phases.segments(flight, native, manager, ground, status)
    assert rows[0]['start_sim_s'] == 1. and sum(r['duration_s'] for r in rows) == pytest.approx(.6)
    assert math.fsum(r['estimated_energy_Wh'] for r in rows) == pytest.approx(flight['estimated_airborne_energy_Wh'])
    packets[4]['native']['airborne_budget'] = waiting
    with pytest.raises(ValueError, match='Incomplete native budget'):
        phases.native_observations(packets, 'flight', final)


@pytest.mark.parametrize('fault', ['counter_reversed', 'terminal_mismatch', 'uncovered', 'ground_gap', 'published_total'])
def test_rejects_inconsistent_sources(fault):
    values = list(example())
    flight, packets, final, manager, ground, status = values
    if fault == 'counter_reversed':
        packets[4]['native']['airborne_budget']['sources']['native_pack_model']['energy_Wh'] = -1.
    elif fault == 'terminal_mismatch':
        final['sources']['recorded_manager']['energy_Wh'] += .2
    elif fault == 'uncovered':
        final['uncovered_s'] = .01
    elif fault == 'ground_gap':
        ground['timestamp_sample'][3:] += 400_000
    else:
        flight['airborne_horizontal_distance_m'] = 5.9
    with pytest.raises(ValueError):
        derive(*values)


def test_px4_onset_half_open_intervals_and_unknown_status_coverage():
    status = dict(timestamp=[1_010_000,1_200_000,2_000_000,4_000_000],
                  nav_state_timestamp=[900_000,900_000,1_700_000,1_700_000], nav_state=[17,17,14,14])
    nav = phases.navigation(status)
    assert phases.nav_context(nav, 1_400_000, 1_700_000)[0] == 'AUTO_TAKEOFF'
    assert phases.nav_context(nav, 1_700_000, 1_800_000)[0] == 'OFFBOARD'
    assert phases.nav_context(nav, 2_100_000, 2_200_000) == ('UNKNOWN', 'PX4_STATUS_GAP')
    land = dict(timestamp=[20,30,44,50], landed=[1,0,0,1])
    assert phases.px4_gate(land) == (30,50)
    with pytest.raises(ValueError, match='complete PX4 airborne'):
        phases.px4_gate({**land, 'landed':[1,0,1,0]})


def test_rejected_attempt_cannot_supply_segments(tmp_path):
    path = tmp_path/'audit.json'
    path.write_text(json.dumps({'accepted': False}))
    with pytest.raises(ValueError, match='Unaccepted flight'):
        phases.analyze_flight(dict(source_audit=path.name), tmp_path, tmp_path)


@pytest.mark.parametrize('cid', ['factorial_p000_v6_h25_direct_w0d0_t15_q100_s1701',
                                'route_p000_v6_h25_north_dogleg_w0d0_t15_q100_s1701'])
def test_recorded_150m_flight_reconciles_and_separates_both_uses_of_braking(cid):
    table = phases.OUT/'mission_flights.csv'
    if not table.is_file():
        pytest.skip('Local 150 m release is unavailable')
    with table.open() as file:
        flight = next((r for r in csv.DictReader(file) if r['case_id'] == cid), None)
    if flight is None or not (phases.ARCHIVE/flight['raw_archive']).is_file():
        pytest.skip('Accepted flight archive is unavailable')
    rows, check = phases.analyze_flight(flight, phases.OUT, phases.ARCHIVE)
    assert rows and all(a['end_sim_s'] == b['start_sim_s'] for a,b in zip(rows, rows[1:]))
    assert {r['route_direction'] for r in rows if r['flight_phase'] == 'RETURN_BRAKING'} >= {
        'OUTBOUND_WAYPOINT_SEQUENCE', 'BRAKING_BEFORE_HOMEWARD'}
    assert any(r['phase_unresolved'] and r['manager_phase_change_bracket_present'] for r in rows)
    assert sum(r['duration_s'] for r in rows if r['flight_phase'] == 'RETURNING') > 0
    assert 0 < check['vi_eligible_s'] < float(flight['airborne_duration_s'])
    assert check['published_minus_native_Wh'] == pytest.approx(check['published_VI_trapezoids_Wh']-check['same_intervals_native_Wh'])
    for column, metric in [('duration_s','airborne_duration_s'),('estimated_energy_Wh','estimated_airborne_energy_Wh'),
                           ('horizontal_distance_m','airborne_horizontal_distance_m')]:
        assert math.fsum(r[column] for r in rows) == pytest.approx(float(flight[metric]), abs=1e-6, rel=0)
