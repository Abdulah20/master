import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/m100_reaudit'))
from replay import Nav, Observed, closest_nav, density_slope, integrate_interval, measurement, paired_rows, phase_budgets


def test_join_is_local_and_clock_tolerant():
    nav = [Nav(.99, 'CLIMB', (), (), None), Nav(1.02, 'CRUISE', (), (), None)]
    chosen, skew = closest_nav([.99,1.02], nav, 1.018)
    assert chosen.phase == 'CRUISE'
    assert skew == pytest.approx(.002)
    assert closest_nav([.99,1.02], nav, .58)[0] is None


def test_density_corrected_estimate_keeps_other_transient_terms_and_wind():
    params = {'mass_over_density_feature':72.2, 'mass_feature_times_horizontal_speed_squared':-.156}
    mass, wind, ground = 3.715, 6., 10.
    air = abs(ground - wind)
    def full(rho):
        f = mass**1.5 / math.sqrt(rho)
        motion = 72.2*f - .156*f*ground**2 + .036*ground**3 + 16.3
        delta = -.156*f*(air**2-ground**2) + .036*(air**3-ground**3)
        return motion + delta
    k = density_slope('RETURN_BRAKING', params, ground, air, True, {})
    predicted = full(1.2) + k * mass**1.5 * (1/math.sqrt(1.23) - 1/math.sqrt(1.2))
    assert predicted == pytest.approx(full(1.23), abs=1e-10)
    assert density_slope('CRUISE', params, ground, air, False, {'CRUISE':68.9}) == 68.9


def point(t, phase, p, x, measured=True):
    return dict(t_s=t, phase=phase, power_archived_W=p, power_replay_W=p,
                x_enu_m=x, y_enu_m=0., z_agl_m=10., interval_status='INTEGRATED',
                archive_measurement_counter_Wh=t, in_measurement=measured, ground_speed_mps=10., air_speed_mps=12.)


def test_duration_energy_split_at_boundary_and_gap():
    a, b = point(1,'CLIMB',100.,0.),point(1.2,'CRUISE',300.,2.)
    c = point(2.1,'CRUISE',900.,9.)
    e, b['interval_status'] = integrate_interval(a,b,.5)
    assert e == pytest.approx(40 / 3600)
    e, c['interval_status'] = integrate_interval(b,c,.5)
    assert (e,c['interval_status']) == (0.,'TIME_GAP')
    budget = phase_budgets([a,b,c], 'example', {'CRUISE':1})
    assert budget[0]['replay_quadrature_Wh'] == pytest.approx(10/3600)
    assert budget[1]['replay_quadrature_Wh'] == pytest.approx(30/3600)
    assert budget[1]['integrated_duration_s'] == pytest.approx(.1)
    assert sum(p['path_observed_m'] for p in budget) == pytest.approx(2.)


def test_incomplete_replay_excludes_total_and_preserves_partial_energy():
    rows = [point(1.,'CRUISE',300.,2.),point(1.1,'CRUISE',360.,3.),point(2.,'CRUISE',400.,12.)]
    rows[-1]['interval_status'] = 'TIME_GAP'
    m = measurement(rows, {'cruise_speed_mps':10,'altitude_m':10})
    assert m['measurement_replay_energy_Wh'] is None
    assert m['measurement_clock_complete'] is False
    assert m['measurement_partial_integral_Wh'] == pytest.approx(33/3600)


def test_manager_phase_is_authoritative_at_exact_energy_timestamp():
    case = dict(cruise_speed_mps=10, temperature_c=15, wind_speed_mps=0)
    coefficients = {'mass_feature_times_horizontal_speed_squared': -.1, 'mass_over_density_feature': 70.}
    meta = dict(atmosphere={'case': {'density_kg_m3': 1.2}}, energy_summary={'published_regression_mass_kg': 3.715},
                transient_power_model={'coefficients': coefficients}, config={'wind_vector_enu_mps': [0, 0, 0]})
    model = SimpleNamespace(atmosphere=lambda t, z: SimpleNamespace(density_kg_m3=1.2))
    e = Observed(arrival=11., t=100., power=700., speed=10., airspeed=10., voltage=None, current=None,
                 measured=.3, measured_x=5., applied_wind_w=None, roll=1., pitch=2., energy_model='PUBLISHED')
    n = Nav(11.01, 'RETURN_BRAKING', (5., 0., 10.), (10., 0., 0.), 0.)
    row, = paired_rows([e], [n], case, meta, model, {'CRUISE': 68.9}, {100.: 'CRUISE'})
    assert row['paired_nav_phase'] == 'RETURN_BRAKING'
    assert row['phase'] == 'CRUISE' and row['in_measurement'] == 1
    assert row['phase_source'] == 'SYNCHRONOUS_CSV'
    orphan, = paired_rows([e], [], case, meta, model, {}, {100.: 'CRUISE'})
    assert orphan['phase'] == 'CRUISE' and orphan['power_replay_W'] is None


def test_measurement_uses_duration_weighted_earth_and_air_speeds_and_signed_displacement():
    points = [point(0., 'CRUISE', 360., 3.), point(.1, 'CRUISE', 360., 1.), point(.5, 'CRUISE', 360., 0.)]
    for row, v in zip(points, (2., 4., 10.)):
        row['ground_speed_mps'] = v
    result = measurement(points, {'cruise_speed_mps': 10., 'altitude_m': 10.})
    assert result['measurement_achieved_mean_ground_speed_mps'] == pytest.approx(6.2)
    assert result['measurement_achieved_mean_air_speed_mps'] == pytest.approx(12.)
    assert result['measurement_achieved_mean_altitude_m'] == pytest.approx(10.)
    assert result['measurement_horizontal_path_m'] == pytest.approx(3.)
    assert result['measurement_outbound_displacement_m'] == pytest.approx(-3.)
