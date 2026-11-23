"""Score frozen battery and CMU-route aircraft predictions once on whole published holdout flights."""
import hashlib
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd

from . import cmu_calibration as calibration, cmu_flight_calibration as aircraft, external
from .direct_airborne_evidence import confirm, digest, json_file as read, write_json as save
from .thevenin_transfer import observations, TAUS

OUT=calibration.ROOT/'results/m100_reaudit/mission_150m'
FROZEN_RC='33c782361854cecd99562ae43cd1166e3cefd6f7f8cb95d2603a49520309b669'
FROZEN_AIRCRAFT='d3413efef49a278a10f0dc404c6b34e5c6d07e42dd698ad21aae0ea00824377f'


def score():
    for name in ('flights.zip','parameters.csv','README.txt'):
        external.check_hash(external.INPUT/name,external.SHA256[name])
    external.check_hash(calibration.MANIFEST,calibration.MANIFEST_SHA256)
    external.check_hash(external.CONTEXT,external.SHA256['published_context.csv'])
    old_file=calibration.DESTINATION/'fit.json'
    external.check_hash(old_file,FROZEN_AIRCRAFT)
    old=read(old_file)
    rc=read(OUT/'battery_fit.json')
    confirm(rc['parameter_sha256']==FROZEN_RC==hashlib.sha256(json.dumps([rc['parameters'],rc['domain']],
            sort_keys=True).encode()).hexdigest(),'Frozen voltage parameters differ')
    manifest=pd.read_csv(calibration.MANIFEST).set_index('flight_id')
    metadata=pd.read_csv(external.INPUT/'parameters.csv').set_index('flight')
    context=pd.read_csv(external.CONTEXT).set_index('flight_id')
    members=manifest[manifest.included & manifest.split.eq('test')]
    selected=set(members.index)
    confirm(len(selected)==60 and selected.isdisjoint(rc['flight_ids']['train']+rc['flight_ids']['validation']),
            'Frozen holdout membership differs')
    st=old['loaded_voltage']['coefficients']
    p=rc['parameters']
    ix=[0,1,2+list(TAUS).index(p['tau1_s']),2+list(TAUS).index(p['tau2_s'])]
    coef=[p[k] for k in ('r0_ohm','discharge_slope_V_per_Ah','r1_ohm','r2_ohm')]
    rows=[]
    with ZipFile(external.INPUT/'flights.zip') as z:
        for fid in sorted(selected):
            meta=metadata.loc[fid]
            confirm(context.loc[fid,'published_role']=='holdout','Non-holdout flight selected')
            with z.open(f'flights/{fid}.csv') as src:
                samples=pd.read_csv(src,usecols=external.COLUMNS)
            confirm(not external.quality_flags(samples),'Changed flight quality')
            bounds=external.phase_boundaries(samples.position_z.to_numpy())
            x,y,w=observations(dict(flight_id=fid,samples=samples,bounds=bounds))
            dynamic=p['reference_V']-x[:,ix]@coef-y
            historical=st['a_V']-st['r_V_per_A']*x[:,0]-st['k_V_per_Ah']*x[:,1]-y
            ilo,ihi=rc['domain']['current_A']
            qlo,qhi=rc['domain']['discharged_Ah']
            supported=(x[:,0]>=ilo)&(x[:,0]<=ihi)&(x[:,1]>=qlo)&(x[:,1]<=qhi)
            payload,speed,height=float(meta.payload)/1000,float(meta.speed),float(meta.altitude)
            power=calibration.reference_powers(old['published_power_reference'],payload_kg=payload,
                                                commanded_altitude_agl_m=height)
            predicted=aircraft.predict(old['aircraft'],base_phase_power_W=power,commanded_speed_mps=speed,
                                       commanded_altitude_agl_m=height)
            scopes,_,_=external.measured_scopes(samples,meta,context.loc[fid])
            measured=scopes['airborne']
            e=predicted[aircraft.REFERENCE]['airborne']['energy_Wh']
            pred_w=predicted[aircraft.CALIBRATED]['airborne']['mean_power_W']
            rows.append(dict(flight_id=int(fid),route=meta.route,payload_kg=payload,
                commanded_speed_mps=speed,commanded_altitude_agl_m=height,
                field_energy_Wh=measured['observed_energy_Wh'],field_mean_power_W=measured['observed_mean_power_W'],
                field_duration_s=measured['covered_duration_s'],cmu_route_predicted_Wh=e,cmu_route_predicted_mean_power_W=pred_w,
                cmu_route_energy_residual_Wh=e-measured['observed_energy_Wh'],
                cmu_route_mean_power_residual_W=pred_w-measured['observed_mean_power_W'],
                voltage_mae_2rc_V=float(w@abs(dynamic)),voltage_mae_static_V=float(w@abs(historical)),
                voltage_2rc_bias_V=float(w@dynamic),in_domain_fraction_time=float(w[supported].sum()),
                in_domain_voltage_mae_2rc_V=float(np.average(abs(dynamic[supported]),weights=w[supported])) if supported.any() else None,
                in_domain_voltage_mae_static_V=float(np.average(abs(historical[supported]),weights=w[supported])) if supported.any() else None))
    frame=pd.DataFrame(rows)
    external.write_table(frame,OUT/'field_holdout_by_flight.csv')
    train_days=set(manifest.loc[rc['flight_ids']['train'],'date'])
    test_days=set(members.date)
    save(OUT/'field_holdout.json',dict(flights=len(rows),source='https://doi.org/10.1184/R1/12683453',
         voltage_input='Measured past and present field current. No field flight offsets were fitted.',
         field_scope='Aircraft energy and power refer to the CMU route protocol. No identical 150 m itinerary established.',
         campaign_independence='Held-out flights of the same CMU experiment. Battery identity and absolute SOC unresolved.',
         test_days=len(test_days),days_shared_with_fitting=len(train_days & test_days),
         flights_with_lower_2rc_mae=int((frame.voltage_mae_2rc_V<frame.voltage_mae_static_V).sum()),
         frozen_parameter_sha256=FROZEN_RC,frozen_previous_aircraft_fit_sha256=FROZEN_AIRCRAFT,retuned=False,
         verification_table_sha256=digest(OUT/'field_holdout_by_flight.csv'),
         voltage_mae_2rc_V=frame.voltage_mae_2rc_V.mean(),voltage_mae_static_V=frame.voltage_mae_static_V.mean(),
         in_domain_voltage_mae_2rc_V=frame.in_domain_voltage_mae_2rc_V.mean(),in_domain_voltage_mae_static_V=frame.in_domain_voltage_mae_static_V.mean(),
         mean_in_domain_fraction_time=frame.in_domain_fraction_time.mean(),voltage_bias_2rc_V=frame.voltage_2rc_bias_V.mean(),
         cmu_route_energy_mae_Wh=frame.cmu_route_energy_residual_Wh.abs().mean(),
         cmu_route_mean_power_mae_W=frame.cmu_route_mean_power_residual_W.abs().mean()))
    print('Scored',len(rows),'whole holdout flights')


if __name__=='__main__':
    score()
