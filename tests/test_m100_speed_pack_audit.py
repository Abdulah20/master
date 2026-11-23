"""Regression checks for native causal attribution, asymmetric gates and a changing source ledger."""
import copy

import numpy as np
import pytest
import yaml

from tools.m100_reaudit import speed_pack_audit as audit


FIT=yaml.safe_load(audit.CONFIG.read_text())
CASE=dict(mode="on", altitude_m=50, cruise_speed_mps=6, payload_kg=.25, temperature_c=15,
          wind_speed_mps=0, measurement_distance_m=200)


def packet(case=CASE, phase="CRUISE", active=True):
    proposed=audit.expected_delta(case,FIT) if phase=="CRUISE" and active else None
    applied=proposed if case["mode"]=="on" else None
    demand=500.+(applied or 0.)
    return dict(time_s=1.02, demand_fresh=True, native=dict(time_s=1.02,demand_time_s=1.,
                demand_fresh=True,model="m",effective_demand_source="native_pack_model",
                effective_demand_power_w=demand,demand_power_w=demand),
                acknowledged_request=dict(time_s=1.,model="m",phase=phase,run_id="run",producer="m100_experiment_manager",
                measurement_active=active,payload_kg=case["payload_kg"], commanded_speed_mps=case["cruise_speed_mps"],
                commanded_altitude_agl_m=case["altitude_m"],temperature_c=case["temperature_c"],wind_speed_mps=case["wind_speed_mps"],
                stable_cruise_speed_enabled=case["mode"]=="on",stable_cruise_speed_eligible_adjustment_w=proposed,
                stable_cruise_speed_adjustment_w=applied,stable_cruise_speed_model_id=audit.SPEED_ID if applied is not None else None,
                phase_base_power_w=500.,phase_power_w=demand,power_w=demand))


@pytest.mark.parametrize("change",[
    {"cruise_speed_mps":v} for v in [0,5,10,15,20]
]+[{"wind_speed_mps":3},{"payload_kg":.75},{"altitude_m":200},{"temperature_c":25},{"measurement_distance_m":100}])
def test_gate_on_unsupported_conditions(change):
    case={**CASE,**change}
    assert audit.expected_delta(case,FIT) is None
    links, stats=audit.check_wires([packet(case)],case,"run",None)
    assert stats["native_corrected_count"]==0 and links[0]["delta"]==0.


def test_acknowledged_phase_recovery_off_control_return_and_double_count():
    expected=audit.expected_delta(CASE,FIT)
    direct=packet()
    recovery=copy.deepcopy(direct)
    recovery["acknowledged_request"]["power_w"]=recovery["native"]["demand_power_w"]=None
    links,stats=audit.check_wires([direct,recovery,packet(phase="RETURNING")],CASE,"run",expected)
    assert stats["direct_request_corrected_count"]==stats["native_phase_recovery_corrected_count"]==1
    assert [l["used"] for l in links]==[True,True,False]
    off={**CASE,"mode":"off"}
    links,_=audit.check_wires([packet(off)],off,"run",expected)
    assert not links[0]["used"] and links[0]["base"]==500.
    double=copy.deepcopy(direct)
    double["acknowledged_request"]["phase_power_w"]+=expected
    with pytest.raises(ValueError,match="double-counted"):
        audit.check_wires([double],CASE,"run",expected)
    false_phase=copy.deepcopy(direct)
    false_phase["acknowledged_request"]["phase"]="STABILIZING"
    with pytest.raises(ValueError,match="outside calibrated gate"):
        audit.check_wires([false_phase],CASE,"run",expected)


def test_fallback_time_cannot_be_counted_as_native_vi():
    records=[]
    for time, source, counters in ((0., "native_pack_model", (0.,0.)), (.1,"native_pack_model",(.1,0.)),
                                    (.25,"native_pack_model",(.2,.05))):
        values={key:dict(seconds=0.) for key in audit.POWER_SOURCES}
        values["native_pack_model"]["seconds"],values["recorded_manager"]["seconds"]=counters
        records.append(dict(native=dict(airborne_budget=dict(sources=values),effective_demand_source=source,
                                         command_limited=False,voltage_v=20.,current_a=22.)))
    links=[dict(base=450.,delta=-10.,used=True)]*3
    tracks=dict(vel=np.array([4.,4.,4.]),alt=np.array([50.,50.,50.]),est=np.array([4.,4.,4.]))
    mask=np.ones(2,dtype=bool)
    row=audit.interval_stats(records,links,np.array([0.,.1,.25]),tracks,mask,mask,"stable")
    assert row["stable_duration_s"]==pytest.approx(.1)
    assert row["stable_recorded_manager_time_fraction"]==pytest.approx(.2)
    assert row["stable_pack_W"]==440.
    assert row["stable_applied_time_fraction"]==1.
