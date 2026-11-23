"""Reconcile timestamped airborne energy with causal navigation features in original flight records."""
import bisect
import csv
import math
from pathlib import Path

import numpy as np
from pyulog import ULog

from .direct_airborne_evidence import confirm
from .pack_drive.m100_airborne_training import POWER_SOURCES

TOPICS = ['vehicle_status','vehicle_local_position','vehicle_local_position_groundtruth']
NAV = {4:'AUTO_LOITER', 14:'OFFBOARD', 17:'AUTO_TAKEOFF', 18:'AUTO_LAND'}
US_PER_S = 1_000_000


def flight_topics(ulog):
    original = ULog(str(ulog), TOPICS)
    return {topic:original.get_dataset(topic).data for topic in TOPICS}


def trajectory(topics, start, end, waypoints, height, horizontal_tol=2, vertical_tol=2):
    data = topics['vehicle_local_position_groundtruth']
    t = np.asarray(data['timestamp_sample'], dtype=np.int64)/1e6
    lo = max(0,np.searchsorted(t,start,'right')-1)
    hi = np.searchsorted(t,end,'left')+1
    t = t[lo:hi]
    confirm(len(t)>1 and t[0]<=start<end<=t[-1] and (np.diff(t)>0).all() and np.diff(t).max()<=.5,
            'Ground-truth trajectory has gaps or does not bracket the flight')
    for key in ('xy_reset_counter','z_reset_counter'):
        confirm(len(set(data[key][lo:hi]))==1, 'Ground-truth coordinates changed reference')
    clipped = np.r_[start,t[(t>start)&(t<end)],end]
    e,n,z = (np.interp(clipped,t,data[key][lo:hi]) for key in ('y','x','z'))
    e,n,h = e-e[0], n-n[0], z[0]-z
    confirm(np.isfinite(np.r_[e,n,h]).all(), 'Non-finite ground-truth trajectory')
    checkpoints = []
    index=0
    for east,north in waypoints:
        dist = np.hypot(e[index:]-east,n[index:]-north)
        near = np.flatnonzero((dist<=horizontal_tol) & (abs(h[index:]-height)<=vertical_tol))
        confirm(near.size>0, f'Waypoint {east}, {north} not reached at the required altitude')
        # Among one continuous encounter select the closest observed position.
        breaks=np.flatnonzero(np.diff(near)>1)
        encounter=near[:int(breaks[0])+1] if breaks.size else near
        index += int(encounter[np.argmin(dist[encounter])])
        checkpoints.append(dict(time_s=float(clipped[index]), east_m=float(e[index]),north_m=float(n[index]),
                                height_agl_m=float(h[index]), error_m=float(math.hypot(e[index]-east,n[index]-north))))
        index+=1
    landing_error=math.hypot(e[-1],n[-1])
    confirm(not waypoints or landing_error<=horizontal_tol, 'Returned landing location outside tolerance')
    return dict(groundtruth_airborne_distance_m=float(np.hypot(np.diff(e),np.diff(n)).sum()),
                landing_error_m=landing_error, max_agl_m=float(h.max()), waypoints=checkpoints)


def supported_points(records, run, start, end, final_budget):
    """Keep original native published energy at each knot. The two detector edges are exact."""
    start_us = round(start * US_PER_S)
    end_us = round(end * US_PER_S)
    times=[start]
    zeros={key:0. for key in POWER_SOURCES}
    ledgers=[zeros]
    next_window=1
    for packet in records:
        n=packet['native']
        b=n['airborne_budget']
        t=n['time_s']
        stamp_us = round(t * US_PER_S)
        window = (stamp_us - start_us) // US_PER_S
        if b['run_id']!=run or b['status']!='AIRBORNE' or window<next_window or end_us-stamp_us<=150_000:
            continue
        confirm(abs(b['start_s']-start)<1e-7 and b['uncovered_s']<1e-8
                and abs(b['covered_s']-(t-start))<1e-7, 'Partial published airborne interval')
        times.append(t)
        ledgers.append({k:b['sources'][k]['energy_Wh'] for k in POWER_SOURCES})
        next_window=window+1
    times.append(end)
    ledgers.append({key:final_budget['sources'][key]['energy_Wh'] for key in POWER_SOURCES})
    return times,ledgers


def online_series(case_id, run, records, topics, budget, conditions):
    start,end=budget['start_s'],budget['end_s']
    times,ledgers=supported_points(records,run,start,end,budget)
    packets=[r['time_s'] for r in records]
    state=topics['vehicle_status']
    pos=topics['vehicle_local_position']
    position_published_us = np.asarray(pos['timestamp'], dtype=np.int64)
    position_sample_us = np.asarray(pos['timestamp_sample'], dtype=np.int64)
    mode_published_us = np.asarray(state['timestamp'], dtype=np.int64)
    for stamps in (position_published_us, mode_published_us):
        confirm(len(stamps)>0 and (np.diff(stamps)>=0).all(), 'PX4 feature clock reversed')
    # Publication time governs availability. Estimator sample time governs the
    # physical age and separation of the velocity observations.
    valid_stamp = (position_sample_us > 0) & (position_sample_us <= position_published_us)
    finite_velocity = np.isfinite(pos['vx']) & np.isfinite(pos['vy']) & np.isfinite(pos['vz'])
    valid_velocity = valid_stamp & pos['v_xy_valid'] & pos['v_z_valid'] & finite_velocity
    valid_height = valid_stamp & pos['z_valid'] & np.isfinite(pos['z'])
    weight=conditions['payload_kg']
    angle=math.radians(conditions['wind_direction_toward_deg_ENU'])
    wind_e=conditions['wind_speed_mps']*math.cos(angle)
    wind_n=conditions['wind_speed_mps']*math.sin(angle)
    charge=conditions['initial_soc_percent']
    nameplate=conditions['nameplate_Wh']
    start_us = round(start * US_PER_S)
    origin = int(np.searchsorted(position_published_us, start_us, 'right')) - 1
    origin_valid = origin >= 0 and valid_height[origin] and start_us - position_sample_us[origin] <= 250_000
    rows=[]
    for k,(left,right) in enumerate(zip(times,times[1:])):
        dt=right-left
        prev,total=(math.fsum(ledgers[i].values()) for i in (k,k+1))
        delta=total-prev
        confirm(dt>0 and delta>0 and all(ledgers[k+1][s]>=ledgers[k][s]-1e-9 for s in POWER_SOURCES),
                'Nonpositive or nonmonotonic energy interval')
        left_us = round(left * US_PER_S)
        past_us = left_us - 500_000
        ip = int(np.searchsorted(position_published_us, left_us, 'right')) - 1
        old = int(np.searchsorted(position_published_us, past_us, 'right')) - 1
        nav = int(np.searchsorted(mode_published_us, left_us, 'right')) - 1
        iw=bisect.bisect_right(packets,left)-1
        record=records[iw] if iw>=0 and left-packets[iw]<=.30 else {}
        native=record.get('native',{})
        ack=record.get('acknowledged_request') or {}
        sample_fresh = ip >= 0 and left_us - position_sample_us[ip] <= 250_000
        fresh = sample_fresh and valid_velocity[ip]
        history=(fresh and old>=0 and valid_velocity[old] and past_us-position_sample_us[old]<=250_000
                 and position_sample_us[old]<position_sample_us[ip] and pos['vxy_reset_counter'][old]==pos['vxy_reset_counter'][ip]
                 and pos['vz_reset_counter'][old]==pos['vz_reset_counter'][ip])
        height = None
        if origin_valid and sample_fresh and valid_height[ip] and pos['z_reset_counter'][origin] == pos['z_reset_counter'][ip]:
            height = float(pos['z'][origin]) - float(pos['z'][ip])
        mode=int(state['nav_state'][nav]) if nav>=0 and left_us-mode_published_us[nav]<=1_500_000 else None
        phase=ack.get('phase') if ack.get('run_id')==run and 0<=left-ack['time_s']<=.5 else None
        speed=math.hypot(pos['vx'][ip],pos['vy'][ip]) if fresh else None
        vup=-float(pos['vz'][ip]) if fresh else None
        ground_course_e=ground_course_n=None
        if fresh and speed>.3:
            ground_course_e=float(pos['vy'][ip])/speed
            ground_course_n=float(pos['vx'][ip])/speed
        acceleration=vertical_acceleration=horizontal_accel_norm=None
        if history:
            span=(position_sample_us[ip]-position_sample_us[old])/US_PER_S
            acceleration=(speed-math.hypot(pos['vx'][old],pos['vy'][old]))/span
            vertical_acceleration=(vup+float(pos['vz'][old]))/span
            horizontal_accel_norm=math.hypot(float(pos['vx'][ip])-float(pos['vx'][old]),
                                              float(pos['vy'][ip])-float(pos['vy'][old]))/span
        row=dict(case_id=case_id, run_id=run, interval_start_sim_s=left, interval_end_sim_s=right,
                 time_since_takeoff_s=left-start, interval_s=dt, payload_kg=weight, base_uav_mass_kg=3.68,
                 total_mass_kg=3.68+weight, commanded_speed_mps=conditions['cruise_speed_mps'],
                 commanded_altitude_agl_m=conditions['altitude_m'], wind_speed_mps=conditions['wind_speed_mps'],
                 wind_toward_deg=conditions['wind_direction_toward_deg_ENU'], ambient_temperature_C=conditions['temperature_c'],
                 nominal_soc_at_takeoff_pct=charge, route_id=conditions['mission_route'], flight_phase=phase,
                 px4_nav_state=mode, px4_mode=NAV.get(mode,str(mode) if mode is not None else None),
                 horizontal_speed_mps=speed, vertical_speed_up_mps=vup,
                 estimated_height_since_takeoff_m=height,
                 air_relative_horizontal_mps=(math.hypot(float(pos['vy'][ip])-wind_e,float(pos['vx'][ip])-wind_n)
                                              if fresh else None),
                 horizontal_tangential_accel_mps2=acceleration, horizontal_accel_magnitude_mps2=horizontal_accel_norm,
                 vertical_accel_up_mps2=vertical_acceleration,
                 wind_along_ground_track_mps=None if ground_course_e is None else wind_e*ground_course_e+wind_n*ground_course_n,
                 wind_left_of_ground_track_mps=None if ground_course_e is None else wind_n*ground_course_e-wind_e*ground_course_n,
                 nominal_soc_at_start_pct=charge-100*prev/nameplate, cumulative_airborne_Wh_at_start=prev,
                 starting_pack_voltage_V=native.get('voltage_v'),starting_pack_current_A=native.get('current_a'),
                 starting_pack_current_available=(native['current_a'] is not None if 'current_a' in native else None),
                 starting_pack_status=native.get('status'),
                 starting_polarization_history_complete=native.get('polarization_load_history_complete'),
                 starting_loaded_charge_history_complete=native.get('loaded_charge_coverage_complete'),
                 starting_charge_domain_exhausted=native.get('q_domain_exhausted'),
                 starting_accounted_discharge_Ah=native.get('discharged_ah'),
                 starting_unaccounted_load_s=native.get('unaccounted_load_time_s'),
                 fast_polarization_V=native.get('polarization_fast_v'),slow_polarization_V=native.get('polarization_slow_v'),
                 future_interval_energy_Wh=delta, future_interval_mean_power_W=3600*delta/dt,
                 cumulative_airborne_Wh_at_end=total, causal_features_available=bool(history and phase and mode is not None
                                                                                    and height is not None),
                 native_sample_within_0p30s=bool(record))
        for source in POWER_SOURCES:
            row[source+'_Wh_in_interval']=ledgers[k+1][source]-ledgers[k][source]
        rows.append(row)
    confirm(math.isclose(math.fsum(r['future_interval_energy_Wh'] for r in rows),
                         budget['estimated_airborne_energy_Wh'],abs_tol=1e-9),'Intervals do not recover the recorder target')
    return rows


def table(path, rows):
    path=Path(path)
    if not rows:
        raise ValueError(f'No rows for {path.name}')
    cols=list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix('.part')
    with temporary.open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=cols,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
