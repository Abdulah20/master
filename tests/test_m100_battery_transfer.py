import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tools.m100_reaudit.thevenin_transfer import HEADER, OUTPUT, TAUS, observations, write_header
from tools.m100_reaudit.pack_drive.m100_airborne_training import BOUNDARY_SOURCE, CALIBRATION_ID, POWER_SOURCES, training_row


def test_current_step_is_causal_and_two_states_relax_at_distinct_rates():
    current = np.array([1.,1.,11.,11.,11.,2.,2.,2.,2.])
    t = np.arange(len(current)) * .25
    flight = dict(flight_id=1, bounds=(0,0,0,len(t)), samples=pd.DataFrame(
        dict(time=t,battery_current=current,battery_voltage=np.full(len(t),22.))))
    x,_,weights=observations(flight)
    for tau in (.5,8.):
        state=x[:,2+list(TAUS).index(tau)]
        assert state[0]==pytest.approx(1-np.exp(-.25/tau))
        assert state[1]==pytest.approx(1-np.exp(-.5/tau))
        # The newly issued 11 A step first enters the filter at the next timestamp.
        assert state[2]==pytest.approx(11+(state[1]-11)*np.exp(-.25/tau))
    assert x[2,2+list(TAUS).index(.5)]>x[2,2+list(TAUS).index(8.)]
    assert x[1,1]==pytest.approx((.25*1 + .25*6)/3600)
    assert weights.sum()==pytest.approx(1)


def test_deployed_constants_follow_registered_flight_fit(tmp_path):
    fit=json.loads(OUTPUT.read_bytes())
    assert set(fit['flight_ids']['train']).isdisjoint(fit['flight_ids']['validation'])
    assert fit['physical_tb47d_calibration'] is False and fit['equilibrium_OCV_SOC_measured'] is False
    fingerprint=hashlib.sha256(json.dumps([fit['parameters'],fit['domain']],sort_keys=True).encode()).hexdigest()
    assert fingerprint==fit['parameter_sha256']
    header=tmp_path/'fit.hh'
    write_header(fit,header)
    assert header.read_bytes()==HEADER.read_bytes()
    params=fit['parameters']
    for i in (1,2):
        assert params[f'r{i}_ohm']*params[f'c{i}_F']==pytest.approx(params[f'tau{i}_s'])


def test_early_discharge_configuration_marks_transfer_separately_from_native_energy():
    run='m100_example'
    config=dict(payload_kg=.25,cruise_speed_mps=8,altitude_m=50,temperature_c=15,wind_speed_mps=0,
                wind_direction_deg=0,route_length_m=150,initial_nominal_soc_percent=80)
    budget=dict(run_id=run,calibration_id=CALIBRATION_ID,boundary_source=BOUNDARY_SOURCE,status='COMPLETE',
                start_s=20,end_s=24,duration_s=4,covered_s=4,uncovered_s=0,estimated_airborne_energy_Wh=.5,
                battery_dynamics_model='CMU_LOADED_THEVENIN_2RC_V1',initial_nominal_soc_percent=80,
                sources={k:{'seconds':4 if k=='native_pack_model' else 0,
                            'energy_Wh':.5 if k=='native_pack_model' else 0} for k in POWER_SOURCES})
    result=training_row(config,run,'COMPLETE',budget,recording_start_s=19,epoch_changed=False)
    assert result['estimated_airborne_energy_Wh']==.5
    assert result['initial_nominal_soc_percent']==80 and result['training_eligible'] is False
    assert result['label_source']=='simulation_transfer_outside_CMU_reference'
