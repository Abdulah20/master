"""Read the three native flight records and export motion metrics and compact trajectories.

Positions and velocities are PX4 estimates in local ENU, relative to the manager's
captured home. The measured leg is eastward. The controller gates on east velocity,
north velocity and height, and stops after the change in east position reaches the
commanded distance. The subsequent braking leg belongs to the recorded mission.

The trace samples the latest LOCAL_POSITION_NED report at each Gazebo joint callback.
Time-weighted statistics hold that report until the following callback, using the
actual Gazebo stamps and only consecutive valid pairs in the measurement window.
The compact CSV keeps every new PX4 report, flag/phase transition and the endpoints.
Attitude columns use the preceding ULog vehicle_attitude quaternion at each position
report's PX4 boot time, provided its age is no greater than 0.1 s.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK = ROOT / 'tmp/m100_reaudit_pack_three'
DEFAULT_OUT = ROOT / 'results/m100_reaudit/pack_drive_three'
KINEMATICS = ['x_m', 'y_m', 'z_m', 'vx_mps', 'vy_mps', 'vz_mps']
FLAGS = ['phase', 'measuring', 'fresh', 'armed', 'landed']
LIMITS = ROOT / 'config/m100_reaudit/acceptance.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def final_transition(times, values, final):
    """First stamp of the uninterrupted final state, following its opposite."""
    values = np.asarray(values)
    if not len(values) or values[-1] != final:
        return None
    other = np.flatnonzero(values != final)
    return float(np.asarray(times)[other[-1]+1]) if len(other) else None


def link_attitude(frame, directory):
    """Join closed flight ULog attitudes without future interpolation or extrapolation."""
    for axis in ('roll_deg', 'pitch_deg', 'yaw_deg'):
        frame[axis] = np.nan
    logs = sorted((directory / 'state/rootfs/log').glob('*/*.ulg'))
    if len(logs) != 1:
        return {'ulog_status': f'expected_one_found_{len(logs)}'}
    try:
        from pyulog import ULog
    except ImportError:
        return {'ulog_status': 'pyulog_not_installed'}

    path = logs[0]
    try:
        ulog = ULog(str(path), ['vehicle_attitude', 'vehicle_status', 'vehicle_land_detected'])
    except (OSError, EOFError, ValueError, IndexError) as error:
        return {'ulog_status': f'unreadable_{type(error).__name__}', 'ulog_sha256': digest(path)}
    topics = {d.name: d.data for d in ulog.data_list if d.multi_id == 0}
    result = {'ulog_status': 'read', 'ulog_name': path.name, 'ulog_sha256': digest(path),
              'ulog_dropout_records': len(ulog.dropouts)}
    # The logger may stop with the disarm command. These refer to the last written
    # message of EACH named topic, which need not equal the later manager snapshot.
    for topic, variable, field, final in [('vehicle_status', 'arming_state', 'arming_state', 1),
                                          ('vehicle_land_detected', 'landed', 'landed', 1)]:
        data = topics.get(topic, {})
        if variable in data and len(data[variable]):
            times = data['timestamp'].astype('float64') * 1e-6
            result['ulog_last_' + field] = int(data[variable][-1])
            result['ulog_last_' + field + '_time_s'] = times[-1]
            result['ulog_final_' + field + '_transition_px4_s'] = final_transition(times, data[variable], final)
    att = topics.get('vehicle_attitude')
    if att is None or not len(att['timestamp']):
        return {**result, 'ulog_status': 'attitude_missing'}
    stamps = att['timestamp_sample'].astype('float64') * 1e-6
    if np.any(np.diff(stamps) < 0):
        return {**result, 'ulog_status': 'attitude_clock_reversed'}
    q = np.column_stack([att[f'q[{i}]'] for i in range(4)]).astype('float64')
    norms = np.linalg.norm(q, axis=1)
    usable = np.isfinite(norms) & (norms > 0.)
    q[usable] /= norms[usable, None]
    q[~usable] = np.nan
    w, x, y, z = q.T
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    time = frame.px4_time_s.to_numpy()
    interval = np.flatnonzero((stamps >= frame.px4_time_s.min()) & (stamps <= frame.px4_time_s.max()) & usable)
    if len(interval):
        result['ulog_attitude_samples_in_recording'] = len(interval)
        for name, angles in [('roll', roll), ('pitch', pitch)]:
            peak = interval[np.argmax(np.abs(angles[interval]))]
            result['ulog_mission_' + name + '_at_abs_peak_deg'] = np.degrees(angles[peak])
            result['ulog_mission_' + name + '_at_abs_peak_px4_s'] = stamps[peak]
    index = np.searchsorted(stamps, time + 1e-9, side='right') - 1
    age = time - stamps[np.maximum(index, 0)]
    good = (index >= 0) & np.isfinite(time) & (age >= -1e-8) & (age <= .1)
    for axis, value in [('roll_deg', roll), ('pitch_deg', pitch), ('yaw_deg', yaw)]:
        frame.loc[good, axis] = np.degrees(value[index[good]])
    return result


def export_trajectory(frame, path):
    flags = frame[FLAGS].ne(frame[FLAGS].shift()).any(axis=1)
    keep = flags | frame.px4_time_s.ne(frame.px4_time_s.shift())
    keep.iloc[[0, -1]] = True
    columns = ['sim_time_s', 'elapsed_s', 'px4_time_s', *FLAGS, *KINEMATICS,
               'horizontal_speed_mps', 'roll_deg', 'pitch_deg', 'yaw_deg']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as target, gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as stream:
            frame.loc[keep, columns].to_csv(stream, index=False, float_format='%.9g')
    return int(keep.sum())


def motion(case, directory, output, limits):
    cid = case['case_id']
    receipt_path = directory / f'{cid}.json'
    row = {key: case[key] for key in ('case_id', 'payload_kg', 'cruise_speed_mps', 'altitude_m',
                                       'measurement_distance_m', 'seed')}
    if not receipt_path.is_file():
        return {**row, 'motion_status': 'closed_attempt_without_terminal_receipt'}
    receipt = json.loads(receipt_path.read_text())
    row.update(run_id=receipt.get('run_id'), outcome=receipt.get('outcome'), reason=receipt.get('reason'),
               receipt_state=receipt.get('terminal_state'), receipt_landed=receipt.get('landed'),
               receipt_armed=receipt.get('armed'), receipt_sha256=digest(receipt_path),
               observed_seed=receipt.get('observed_seed'), observed_step_s=receipt.get('observed_physics_step_s'))
    if not row['run_id']:
        return {**row, 'motion_status': 'no_recorded_run'}
    stem = directory / row['run_id']
    prov_path = stem.with_suffix('.provenance.json')
    provenance = json.loads(prov_path.read_text()) if prov_path.is_file() else {}
    config = provenance.get('config', {})
    checked_config = {'altitude_m': 'altitude_m', 'cruise_speed_mps': 'cruise_speed_mps',
                      'measurement_distance_m': 'route_length_m', 'temperature_c': 'temperature_c',
                      'wind_speed_mps': 'wind_speed_mps', 'wind_direction_toward_deg_ENU': 'wind_direction_deg',
                      'hover_duration_s': 'hover_duration_s', 'payload_kg': 'payload_kg'}
    row['applied_config_matches_plan'] = all(config.get(key) == case[field] for field, key in checked_config.items())
    gate_distance = provenance.get('energy_summary', {}).get('measurement_window_distance_m')
    recording_end = provenance.get('recording_end', {})
    row.update(manager_gate_distance_m=gate_distance, recording_end_outcome=recording_end.get('outcome'),
               recording_end_state=recording_end.get('flight_state'),
               recording_end_landed=recording_end.get('landed'), recording_end_armed=recording_end.get('armed'))
    trace = stem.with_suffix('.trace.csv.gz')
    if not trace.is_file():
        return {**row, 'motion_status': 'trace_missing'}
    try:
        f = pd.read_csv(trace)
    except (OSError, EOFError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        return {**row, 'motion_status': f'trace_unreadable_{type(error).__name__}', 'trace_sha256': digest(trace)}
    if f.empty:
        return {**row, 'motion_status': 'trace_empty'}
    row.update(link_attitude(f, directory))
    f['horizontal_speed_mps'] = np.hypot(f.vx_mps, f.vy_mps)
    f['elapsed_s'] = f.sim_time_s - f.sim_time_s.iloc[0]
    compact = output / 'data' / f'{cid}.trajectory.csv.gz'
    row.update(trace_sha256=digest(trace), trajectory_rows=export_trajectory(f, compact),
               trajectory_sha256=digest(compact))
    dt = f.sim_time_s.diff()
    good_clock = dt.gt(0) & dt.le(limits['max_integrated_gap_s'])
    new_position = f.px4_time_s.notna() & f.px4_time_s.ne(f.px4_time_s.shift())
    since_update = f.sim_time_s - f.sim_time_s.where(new_position).ffill()
    position_ok = (np.isfinite(f[KINEMATICS]).all(axis=1) & f.fresh.eq(1) & np.isfinite(f.px4_time_s)
                   & since_update.between(0, limits['max_integrated_gap_s'])
                   & f.local_position_age_wall_s.between(0, limits['max_integrated_gap_s']))
    valid_pair = good_clock & position_ok & position_ok.shift(fill_value=False)
    in_cruise = f.phase.eq('CRUISE') & f.measuring.eq(1)
    cruise_pair = valid_pair & in_cruise & in_cruise.shift(fill_value=False)
    progress = f.x_m.diff()
    sideways = f.y_m.diff()
    finite = f[position_ok]
    gate = f[in_cruise & position_ok]
    px4_change = f.px4_time_s.diff()
    last = f.iloc[-1]
    speed = case['cruise_speed_mps']
    altitude = case['altitude_m']
    row.update(samples=len(f), span_s=f.elapsed_s.iloc[-1],
               phase_sequence='>'.join(f.phase[f.phase.ne(f.phase.shift())]),
               px4_position_reports=f.px4_time_s.nunique(), median_callback_step_s=dt[dt.gt(0)].median(),
               median_px4_position_step_s=px4_change[px4_change.gt(0)].median(),
               sim_nonincreasing=int(dt.le(0).sum()), sim_largest_gap_s=dt.max(),
               sim_gaps_above_limit=int(dt.gt(limits['max_integrated_gap_s']).sum()),
               px4_reversals=int(px4_change.lt(0).sum()), px4_largest_step_s=px4_change.max(),
               px4_gaps_above_limit=int(px4_change.gt(limits['max_integrated_gap_s']).sum()),
               px4_max_hold_sim_s=since_update.max(), px4_max_local_position_age_wall_s=f.local_position_age_wall_s.max(),
               invalid_or_stale_motion_samples=int((~position_ok).sum()),
               observed_path_horizontal_m=np.hypot(progress[valid_pair], sideways[valid_pair]).sum(),
               max_height_m=finite.z_m.max(), max_height_overshoot_m=max(0., finite.z_m.max()-altitude) if len(finite) else None,
               max_ground_speed_mps=finite.horizontal_speed_mps.max(), max_east_from_home_m=finite.x_m.max(),
               max_height_phase=finite.loc[finite.z_m.idxmax(), 'phase'] if len(finite) else None,
               max_ground_speed_phase=finite.loc[finite.horizontal_speed_mps.idxmax(), 'phase'] if len(finite) else None,
               max_abs_north_from_home_m=finite.y_m.abs().max(),
               max_abs_roll_at_position_reports_deg=finite.roll_deg.abs().max(),
               max_abs_pitch_at_position_reports_deg=finite.pitch_deg.abs().max(),
               end_phase=last.phase, end_trace_landed=int(last.landed), end_trace_armed=int(last.armed),
               end_height_m=last.z_m, end_distance_to_home_m=np.hypot(last.x_m, last.y_m),
               end_sim_time_s=last.sim_time_s, end_px4_position_time_s=last.px4_time_s,
               trace_final_landed_transition_sim_s=final_transition(f.sim_time_s, f.landed, 1),
               trace_final_disarm_transition_sim_s=final_transition(f.sim_time_s, f.armed, 0),
               end_local_position_age_wall_s=last.local_position_age_wall_s,
               measurement_windows=int((in_cruise & ~in_cruise.shift(fill_value=False)).sum()),
               measurement_missing_or_stale_rows=int((in_cruise & ~position_ok).sum()))
    # Point-to-point progress uses every valid consecutive observation. Repeated
    # positions between PX4 updates contribute zero and create no synthetic motion.
    if not len(gate) or not cruise_pair.any():
        return {**row, 'motion_status': 'no_usable_measured_cruise'}
    duration = dt[cruise_pair].sum()
    displacement = gate.x_m.iloc[-1] - gate.x_m.iloc[0]
    allowance = limits['native_route_shortfall_tolerance_m'] + speed * limits['native_endpoint_allowance_s']
    tol = max(limits['measurement_velocity_floor_tolerance_mps'], limits['measurement_velocity_relative_tolerance'] * speed)
    within = f.vx_mps.sub(speed).abs().le(tol) & f.vy_mps.abs().le(tol) & f.z_m.sub(altitude).abs().le(limits['measurement_altitude_tolerance_m'])

    def weighted(values):
        previous = values.shift()
        mask = cruise_pair & np.isfinite(previous)
        return np.average(previous[mask], weights=dt[mask]) if mask.any() else np.nan

    avg = weighted(f.horizontal_speed_mps)
    row.update(motion_status='measured_cruise', cruise_first_sim_s=gate.sim_time_s.iloc[0],
               cruise_last_sim_s=gate.sim_time_s.iloc[-1], cruise_span_s=gate.sim_time_s.iloc[-1]-gate.sim_time_s.iloc[0],
               cruise_covered_s=duration, cruise_east_start_m=gate.x_m.iloc[0], cruise_east_end_m=gate.x_m.iloc[-1],
               cruise_east_displacement_m=displacement, observed_minus_manager_gate_m=(displacement - gate_distance) if gate_distance is not None else None,
               gate_endpoint_agreement_limit_m=allowance, gate_endpoint_agrees=(abs(displacement - gate_distance) <= allowance) if gate_distance is not None else None,
               cruise_east_reverse_travel_m=progress[cruise_pair].clip(upper=0).abs().sum(),
               beyond_cruise_end_east_m=finite.x_m.max()-gate.x_m.iloc[-1],
               cruise_max_abs_north_from_home_m=gate.y_m.abs().max(),
               cruise_max_cross_track_from_gate_start_m=(gate.y_m-gate.y_m.iloc[0]).abs().max(),
               cruise_mean_ground_speed_mps=avg, cruise_sd_ground_speed_mps=np.sqrt(weighted((f.horizontal_speed_mps-avg)**2)),
               cruise_min_ground_speed_mps=gate.horizontal_speed_mps.min(), cruise_max_ground_speed_mps=gate.horizontal_speed_mps.max(),
               cruise_max_east_velocity_error_mps=(gate.vx_mps-speed).abs().max(), cruise_max_abs_north_velocity_mps=gate.vy_mps.abs().max(),
               cruise_mean_height_m=weighted(f.z_m), cruise_min_height_m=gate.z_m.min(), cruise_max_height_m=gate.z_m.max(),
               cruise_max_height_error_m=(gate.z_m-altitude).abs().max(), cruise_max_abs_vertical_speed_mps=gate.vz_mps.abs().max(),
               cruise_inside_speed_lateral_height_time_fraction=weighted(within.astype(float)),
               cruise_roll_min_deg=gate.roll_deg.min(), cruise_roll_max_deg=gate.roll_deg.max(),
               cruise_pitch_min_deg=gate.pitch_deg.min(), cruise_pitch_max_deg=gate.pitch_deg.max(),
               cruise_attitude_coverage_fraction=weighted(f.pitch_deg.notna().astype(float)))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    cases = json.loads((args.out / 'protocol.json').read_text())['cases']
    limits = json.loads(LIMITS.read_text())
    pending = [c['case_id'] for c in cases if not (args.work/c['case_id']/'attempt_1'/'stopped_utc').is_file()]
    if pending:
        raise SystemExit('Awaiting closed acquisition directories: ' + ', '.join(pending))
    rows = [motion(c, args.work/c['case_id']/'attempt_1', args.out, limits) for c in cases]
    pd.DataFrame(rows).to_csv(args.out / 'motion.csv', index=False, float_format='%.12g')


if __name__ == '__main__':
    main()
