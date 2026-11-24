"""Protect the two energy definitions in the supervisor repeat export."""
import csv
import gzip

import pytest

from tools.m100_reaudit.supervisor_2rc import measurement_trace
from tools.m100_reaudit.supervisor_2rc_compare import compare


def write_trace(path, points):
    with gzip.open(path, 'wt', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('sim_time_s', 'measuring', 'phase', 'power_w'))
        writer.writerow((0, 0, 'CLIMB', 1000))
        writer.writerows((t, 1, 'CRUISE', power) for t, power in points)
        writer.writerow((2, 0, 'RETURNING', 1000))


def test_same_measurement_gate_excludes_climb_and_return(tmp_path):
    trace = tmp_path / 'flight.trace.csv.gz'
    write_trace(trace, [(1, 480), (1.25, 480), (1.5, 480)])
    check = measurement_trace(trace, dict(case_id='test', cruise_speed_mps=10), 480*.5/3600, .5)
    assert check['complete']
    assert check['duration_s']==.5
    assert check['integrated_mean_power_W']==pytest.approx(480)


def test_known_subtotal_does_not_verify_an_uncovered_segment(tmp_path):
    trace = tmp_path / 'flight.trace.csv.gz'
    write_trace(trace, [(1, 480), (1.25, 480), (1.9, 480)])
    check = measurement_trace(trace, dict(case_id='test', cruise_speed_mps=10), 480*.25/3600, .5)
    assert not check['complete']
    assert check['invalid_intervals']==1


def test_export_never_uses_original_measurement_as_the_full_flight_baseline():
    case = dict(case_id='case_001', payload_kg=0, cruise_speed_mps=10, wind_speed_mps=0,
                wind_direction_toward_deg_ENU=0, altitude_m=100, temperature_c=15, initial_soc_percent=100,
                measurement_distance_m=500, hover_duration_s=60, seed=1701, physics_step_s=.004)
    previous = dict(run_id='previous', estimated_airborne_energy_Wh='48', measurement_energy_Wh='7')
    original = dict(run_id='original', measurement_energy_Wh='6', data_quality='PASS')
    doc = dict(accepted=True, verified_airborne=True, verified_segment=True, attempt=1,
               metrics=dict(estimated_airborne_energy_Wh=46, measurement_energy_Wh=8, nameplate_Wh=99.9))
    row = compare(case, ['experiment_1_payload'], previous, original, {'complete':True}, doc, {}, 1)
    assert row['delta_airborne_energy_Wh']==-2
    assert row['delta_measurement_energy_Wh']==1
    assert row['delta_measurement_vs_original_Wh']==2
    doc.update(verified_segment=False, accepted=False)
    limited = compare(case, ['experiment_1_payload'], previous, original, {'complete':True}, doc, {}, 1)
    assert limited['estimated_airborne_energy_Wh']==46
    assert limited['verification_status']=='AIRBORNE_ONLY'
    assert limited['delta_measurement_energy_Wh'] is None
