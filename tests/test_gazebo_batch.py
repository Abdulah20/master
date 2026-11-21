from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.m100_reaudit import gazebo_batch as gb


def test_sdf_retains_four_rotors_inertias_joints_and_research_payload(tmp_path):
    assets=gb.DEFAULT_PX4/'Tools/simulation/gz/models'
    dest=tmp_path/'world.sdf'
    hashes=gb.make_world(assets,.5,.008,dest)
    root=ET.parse(dest).getroot()
    aircraft=root.find(".//model[@name='research_m100']")
    motors=[m for m in aircraft.findall('plugin') if 'MotorModel' in m.get('name','')]
    assert len(motors)==4
    assert len(aircraft.findall('joint'))==5
    assert len(aircraft.findall('.//collision'))>5
    assert sum(float(v.text) for v in aircraft.findall('link/inertial/mass'))==pytest.approx(4.18)
    assert {p.findtext('maxRotVelocity') for p in motors}=={'916.2978572970229'}
    assert len(hashes)==2 and root.findtext('world/physics/max_step_size')=='0.008'


def test_design_preserves_original_and_includes_matched_controls():
    rows=gb.scenarios(gb.research_study());model=gb.Model.load()
    assert len({r['case_id'] for r in rows})==181
    assert sum(r['cohort']=='original' for r in rows)==69
    assert sum(gb.assess(model,c).runtime_eligible for c in rows)==176
    assert not any(not gb.assess(model,c).stock_reasons for c in rows)
    fixed=[r for r in rows[69:] if r['design']=='matched_temperature_c' and r['payload_kg']==.25 and r['cruise_speed_mps']==6
           and r['measurement_distance_m']==125 and r['wind_speed_mps']==0 and r['altitude_m']==50]
    assert set(r['temperature_c'] for r in fixed)=={0,15,25,35}


def test_integral_includes_boundary_endpoints_and_preserves_gaps():
    f=pd.DataFrame(dict(time_s=[0,.1,.1,.2,.3,.4,.4,.45],phase=['GROUND','GROUND','CRUISE','CRUISE','CRUISE','CRUISE','LANDING','LANDING'],
                        power_W=[np.nan,np.nan,100,300,np.nan,10,100,80]))
    c,valid,dt=gb.integrate(f)
    assert c[-1]==pytest.approx((.1*200+.05*90)/3600)
    assert np.flatnonzero(valid).tolist()==[2,6]
    bad=f.copy();bad.loc[2,'time_s']=.105
    with pytest.raises(ValueError,match='unpaired'): gb.integrate(bad)


def test_saved_case_agrees_with_independent_trapezoids():
    folder=gb.OUTPUT
    intervals=pd.read_csv(folder/'phases.csv')
    case='case_001'
    truth=pd.read_csv(folder/'traces'/f'{case}.csv.gz')
    de,dt=gb.independent_check(truth,intervals[intervals.run_id==case].to_dict('records'))
    assert de<gb.BUDGET_ATOL_WH and dt<gb.DURATION_ATOL_S
    assert set(truth.power_model[truth.phase=='CRUISE'])=={'RODRIGUES_PATTERNS_M100_PHASE_POWER_V1'}
    assert truth[['voltage_V','current_A']].isna().all().all()
    assert np.allclose(truth.air_e_mps,truth.ve_mps-truth.wind_e_mps)


def test_family_splits_and_status_counts_preserved():
    summary=gb.OUTPUT/'cases.csv'; frame=pd.read_csv(summary)
    base=frame[frame.step_s==gb.BASE_STEP]
    assert len(base)==181 and base.case_id.is_unique
    assert (base.engine_result=='not_launched').sum()==5
    data=pd.read_csv(gb.OUTPUT/'training_research.csv')
    assert set(data.case_id)<=set(base[base.status=='passed'].case_id)
    assert data.groupby('split_group').split.nunique().max()==1
    assert set(data.split)=={'train','validation','test'}
    assert pd.read_csv(gb.OUTPUT/'training_stock.csv').empty
    assert not {'run_id','nominal_remaining_Wh','measurement_Wh','observed_time_s'}&set(data.columns)
    assert data[data.target_phase=='HOVER'].route_for_cruise_m.isna().all()
    planned = pd.read_csv(gb.OUTPUT/'scenario_plan.csv').set_index('case_id')
    for row in data.itertuples():
        assert (row.split_group, row.split) == tuple(planned.loc[row.case_id, ['split_group', 'split']])


def test_long_hover_completes_landing_but_exhausts_the_nameplate():
    cases = pd.read_csv(gb.OUTPUT / 'cases.csv').set_index('run_id')
    boundary = cases.loc['exp_107']
    assert boundary.engine_result == 'completed'
    assert boundary.status == 'failed'
    assert boundary.failure_reasons == 'nameplate_exhausted_before_mission_end'
    assert boundary.nominal_budget_status == 'NAMEPLATE_EXHAUSTED'
    assert boundary.nominal_remaining_Wh < 0
    trace = pd.read_csv(gb.OUTPUT / 'traces/exp_107.csv.gz')
    assert trace.phase.iat[-1] == 'LANDED'
    assert trace.loc[trace.phase == 'LANDING', 'time_s'].max() > trace.loc[trace.phase == 'HOVER', 'time_s'].max()
    first = trace.loc[trace.battery_state == 'NAMEPLATE_EXHAUSTED', 'time_s'].iat[0]
    assert first == pytest.approx(boundary.first_sample_nameplate_exhaustion_s)


def test_delivered_tables_and_compressed_traces_match_recorded_digests():
    manifest = json.loads((gb.OUTPUT / 'study.json').read_text())
    for name, digest in manifest['output_table_sha256'].items():
        assert gb.sha(gb.OUTPUT / name) == digest
    cases = pd.read_csv(gb.OUTPUT / 'cases.csv').dropna(subset=['sha256'])
    for row in cases.itertuples():
        assert gb.sha(gb.OUTPUT / 'traces' / f'{row.run_id}.csv.gz') == row.sha256
    repeats = pd.read_csv(gb.OUTPUT / 'repeatability.csv')
    for row in repeats.itertuples():
        assert gb.sha(gb.OUTPUT / 'traces' / f'repeat_{row.case_id}.csv.gz') == row.repeat_trace_sha256
