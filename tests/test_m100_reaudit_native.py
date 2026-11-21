from types import SimpleNamespace

import pytest

from tools.m100_reaudit.native_campaign import case_plan
from tools.m100_reaudit.native_metrics import assert_same, verify


def native_fixture(hover_s=.4):
    model = SimpleNamespace(
        atmosphere=lambda t,z: SimpleNamespace(density_kg_m3=1.2),
        assess=lambda *args: SimpleNamespace(stock_reasons=('research_mass',),evidence_flags=('limited',)),
        budget=lambda wh: SimpleNamespace(nominal_remaining_wh=99.9-wh,status='PARTIAL'))
    receipt = dict(case_id='example',run_id='one',cruise_speed_mps=0.,altitude_m=3.,payload_kg=0.,
        temperature_c=15.,wind_speed_mps=0.,wind_direction_toward_deg_ENU=0.,measurement_distance_m=0.,
        hover_duration_s=hover_s,observed_physics_step_s=.004,outcome='COMPLETE',landed=True,armed=False)
    samples = [dict(sim_time_s=t,phase='HOVER',measuring=True,x_m=0.,y_m=0.,z_m=3.,vx_mps=0.,vy_mps=0.,
               counter_wh=e,power_w=p,voltage_v=20.,current_a=p/20.,rho_kg_m3=1.2)
               for t,e,p in [(4.,0.,360.),(4.1,.015,720.),(4.4,.075,720.)]]
    provenance=dict(source_files={'m100_experiment_manager':{'sha256':'fixture'}},
                    energy_summary=dict(phase_energy_by_phase_wh={'HOVER':.075},integrated_supported_phase_energy_wh=.075,
                    is_full_flight_energy=False,measurement_window_distance_m=None,measurement_window_energy_wh=.075))
    return samples,receipt,provenance,model


def test_raw_endpoint_quadrature_uses_irregular_flight_timestamps():
    samples,receipt,provenance,model=native_fixture()
    row,phases=verify(samples,receipt,provenance,model)
    assert row['data_verdict']=='NUMERICALLY_ACCEPTED'
    assert row['aligned_mean_power_w']==pytest.approx(675.)
    assert row['aligned_energy_wh']==pytest.approx(.075)
    assert phases[0]['supported_wh']==pytest.approx(.075)
    assert row['aligned_wh_per_km'] is None
    provenance['energy_summary']['integrated_supported_phase_energy_wh']=.1
    row,_=verify(samples,receipt,provenance,model)
    assert 'PHASE_OR_TOTAL_COUNTER_RESIDUAL' in row['reason']


def test_completed_receipt_with_truncated_hover_is_rejected():
    row,_=verify(*native_fixture(12))
    assert row['outcome']=='COMPLETE' and row['reason']=='HOVER_DURATION_NOT_REACHED'


def test_native_scenario_design_preserves_count_and_seed_identity():
    plan=case_plan('originals')
    assert len(plan)==len({r['case_id'] for r in plan})==69
    fine=[r for r in case_plan('pilots') if r['case_id'].startswith('seed_')]
    assert len(fine)==4 and {r['seed'] for r in fine}=={1701,1702}


def test_archive_verifier_refuses_changed_values():
    with pytest.raises(ValueError,match='mean_W'):
        assert_same(dict(run_id='id',mean_W='300'),dict(mean_W=301.))
