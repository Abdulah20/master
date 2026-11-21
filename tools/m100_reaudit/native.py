"""Acquire sample-synchronous traces from the real ROS 2/PX4/Gazebo experiment manager."""
import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.environ.get('M100_RUNTIME', '/home/abdulah/Master/Drone/project'))
FIELDS = ('sim_time_s', 'px4_time_s', 'phase', 'measuring', 'fresh', 'armed', 'landed',
          'x_m', 'y_m', 'z_m', 'vx_mps', 'vy_mps', 'vz_mps', 'wind_east_mps',
          'wind_north_mps', 'air_relative_horizontal_mps', 'mass_kg', 'rho_kg_m3', 'pressure_pa',
          'power_w', 'voltage_v', 'current_a', 'counter_wh', 'soc_percent', 'bms_state',
          'bms_level', 'model_id', 'wind_delta_w', 'local_position_age_wall_s')


def manager(run_dir, partition):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from drone_nav import m100_experiment_manager as backend
    M100ExperimentManager = backend.M100ExperimentManager
    if os.environ.get("M100_AUDIT_ISOLATED"):
        backend.LOG_ROOT = run_dir / "state/px4_text_logs"
        backend.TEMPERATURE_MODEL_ROOT = run_dir / "state/temperature_models"

    class TraceManager(M100ExperimentManager):
        trace = None

        def _campaign_sample_observer(self, model, mav, enu, measuring):
            if self.trace is None:
                self.trace = gzip.open(self._csv_path.with_suffix('.trace.csv.gz'), 'xt', newline='', compresslevel=1)
                self.trace_writer = csv.writer(self.trace)
                self.trace_writer.writerow(FIELDS)
            wind = self._config.wind_vector_enu_mps
            air = self._power_air_state
            vx, vy = enu['vx'], enu['vy']
            airspeed = None if vx is None or vy is None else math.hypot(vx-wind[0], vy-wind[1])
            self.trace_writer.writerow((model.timestamp_s, mav.get('boot_time_s'), model.phase,
                int(measuring), int(mav['fresh']), int(mav['armed']), int(mav['landed']),
                *(enu[k] for k in ('x', 'y', 'z', 'vx', 'vy', 'vz')), wind[0], wind[1], airspeed,
                model.physical_total_mass_kg, model.air_density_kg_m3, air.pressure_pa if air else None,
                model.operational_power_w, model.battery_voltage_v, model.battery_current_a,
                model.cumulative_energy_wh, model.soc_percent, model.bms_state, model.bms_safety_level,
                model.model_id, model.wind_power_adjustment_w, None if mav.get('last_local_position_monotonic') is None else time.monotonic()-mav['last_local_position_monotonic']))

        def _stop_recording(self, *, outcome='MANUAL_STOP'):
            with self._lock:
                path = super()._stop_recording(outcome=outcome)
                if self.trace:
                    self.trace.close()
                    self.trace = None
                return path

    run_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=['--ros-args', '-p', 'auto_start_stack:=false', '-p', f'gz_partition:={partition}',
                     '-p', f'csv_root:={run_dir}'])
    node = TraceManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def collect(plan, run_dir):
    import rclpy
    sys.path.insert(0, str(RUNTIME / 'analysis/m100_high_speed_stability'))
    from run_one_stability_case import StabilityCaseRunner
    from physics_settings import physics_request

    class Guard(StabilityCaseRunner):
        def _record(self, stream, payload):
            pass  # The manager writes directly from its model callback.

    rclpy.init()
    runner = Guard(Path(os.devnull))
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        for case in json.loads(plan.read_text()):
            receipt = run_dir / (case['case_id'] + '.json')
            if receipt.exists():
                continue
            old = set(run_dir.glob('m100_*.csv'))
            result = dict(case)
            try:
                runner.wait_for_status()
                runner.wait_configurable()
                # COMPLETE is published before the worker releases its active-run flag.
                # Consume the subsequent idle status before submitting the next APPLY.
                settled_by = time.monotonic() + 15
                while (runner.status.get('experiment') or {}).get('auto_running'):
                    if time.monotonic() > settled_by:
                        raise RuntimeError('completed run still holds the experiment worker')
                    rclpy.spin_once(runner, timeout_sec=.1)
                seq = runner.status['experiment']['apply_seq']
                cmd = 'APPLY {temperature_c} {wind_speed_mps} {wind_direction_toward_deg_ENU} {payload_kg} {cruise_speed_mps} {altitude_m} {measurement_distance_m} {hover_duration_s} 0.9 0.04 0.45 1.3 0.06 0.35'.format(**case)
                runner.publish_reliably(runner.apply_pub, cmd)
                runner.wait_apply_ack(seq)
                seed = os.environ.get('M100_GAZEBO_SEED')
                if seed is not None:
                    observed_seed = False
                    for proc in Path('/proc').glob('[0-9]*/cmdline'):
                        try:
                            argv = proc.read_bytes().replace(b'\x00', b' ').decode(errors='replace')
                        except OSError:
                            continue
                        if ('gz sim' in argv and ' -s ' in argv and f'--seed {seed} ' in argv
                                and 'm100_experiment_park.sdf' in argv):
                            observed_seed = True
                            break
                    if not observed_seed:
                        raise RuntimeError('Gazebo server did not expose requested noise seed')
                    result['observed_seed'] = int(seed)
                step = case.get('physics_step_s', 0.004)
                ack = subprocess.run(['gz', 'service', '-s','/world/m100_experiment_park/set_physics',
                    '--reqtype','gz.msgs.Physics','--reptype','gz.msgs.Boolean','--timeout','10000',
                    '--req',physics_request(step)], capture_output=True, text=True, timeout=15, check=True)
                if 'data: true' not in ack.stdout:
                    raise RuntimeError('Gazebo rejected physics update')
                runner.clock_times.clear()
                deadline = time.monotonic() + 10
                while len(runner.clock_times) < 64 and time.monotonic() < deadline:
                    rclpy.spin_once(runner, timeout_sec=.1)
                stamps = list(runner.clock_times)
                observed = min((b-a for a,b in zip(stamps, stamps[1:]) if b>a), default=0) * 1e-9
                result['observed_physics_step_s'] = observed
                if abs(step-observed) > 1e-9:
                    raise RuntimeError('physics step unconfirmed')
                runner.publish_reliably(runner.command_pub, 'RUNEXPERIMENT')
                result['outcome'], result['reason'] = runner.guarded_wait_for_completion(case['cruise_speed_mps'], case['altitude_m'])
            except Exception as exc:
                result.update(outcome='RUNNER_ERROR', reason=str(exc))
                if (runner.status or {}).get('armed'):
                    runner.publish_reliably(runner.command_pub,'ABORTEXPERIMENT')
            status = runner.status or {}
            result.update(landed=status.get('landed'), armed=status.get('armed'), terminal_state=status.get('state'))
            new = set(run_dir.glob('m100_*.csv'))-old
            if len(new) == 1:
                result['run_id'] = next(iter(new)).stem
            tmp = receipt.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            tmp.rename(receipt)
            print(result['case_id'], result['outcome'], result['reason'], flush=True)
            if result['outcome'] in ('RUNNER_ERROR','ERROR') and not new:
                break
    finally:
        runner.close()
        runner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['manager','collect'])
    p.add_argument('run_dir', type=Path)
    p.add_argument('--plan', type=Path)
    a = p.parse_args()
    if a.mode == 'manager':
        manager(a.run_dir.resolve(), os.environ['GZ_PARTITION'])
    else:
        collect(a.plan, a.run_dir.resolve())
