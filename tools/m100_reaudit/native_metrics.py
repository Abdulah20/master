"""Verify callback-level power and kinematics independently of PX4 manager counters."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path
import shutil
import tempfile

from .physics import Model

ROOT = Path(__file__).resolve().parents[2]
LIMITS = json.loads((ROOT / 'config/m100_reaudit/acceptance.json').read_text())


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def table(path, records):
    if not records:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in records for k in r)), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def read_trace(path):
    opener = lzma.open if path.suffix == '.xz' else gzip.open
    with opener(path, 'rt', newline='') as f:
        for row in csv.DictReader(f):
            for k in row.keys() - {'phase','model_id','bms_state','bms_level'}:
                row[k] = float(row[k]) if row[k] else None
                if row[k] is not None and not math.isfinite(row[k]):
                    raise ValueError(f'non-finite {k}')
            yield row


def verify(samples, receipt, provenance, model):
    case_id = receipt['case_id']
    cmd_speed = receipt['cruise_speed_mps']
    first = previous = measure_first = measure_last = None
    terminal = provenance['energy_summary']
    per_phase = defaultdict(float)
    spans, measure_intervals = [], []
    counts = Counter()
    active = prev_active = False
    vi_error = rho_error = height_error = speed_error = biggest_gap = 0.
    unsupported_time = integrated_time = path_h = 0.
    speeds, heights = [], []
    endpoint = None
    dt_limit = LIMITS['max_integrated_gap_s']
    for row in samples:
        counts['observations'] += 1
        t, power = row['sim_time_s'], row['power_w']
        if t is None:
            raise ValueError(f'{case_id} missing flight clock')
        if first is None:
            first = row
        phase = row['phase']
        changed = previous is None or previous['phase'] != phase
        if changed:
            middle = t if previous is None else (previous['sim_time_s']+t)/2
            if spans:
                spans[-1]['end_s'] = middle
            spans.append(dict(case_id=case_id, run_id=receipt['run_id'], phase=phase, start_s=middle, end_s=t,
                              observed_duration_s=0., supported_duration_s=0., supported_wh=0.,
                              horizontal_path_m=0., absolute_vertical_path_m=0.))
        spans[-1]['end_s'] = t
        if power is not None:
            counts['powered'] += 1
            if row['voltage_v'] is not None and row['current_a'] is not None:
                counts['vi_available'] += 1
                vi_error = max(vi_error, abs(power-row['voltage_v']*row['current_a']))
            else:
                counts['vi_missing_powered'] += 1
            if row['z_m'] is not None and row['rho_kg_m3'] is not None:
                expected = model.atmosphere(receipt['temperature_c'],max(0.,row['z_m'])).density_kg_m3
                rho_error = max(rho_error, abs(row['rho_kg_m3']-expected))
        active = bool(row['measuring'])
        if active and not prev_active:
            counts['measurement_windows'] += 1
        if active:
            if measure_first is None:
                measure_first = row
            measure_last = row
            if row['vx_mps'] is not None and row['vy_mps'] is not None and row['z_m'] is not None:
                v = math.hypot(row['vx_mps'],row['vy_mps'])
                speeds.append(v); heights.append(row['z_m'])
                speed_error = max(speed_error,abs(v-cmd_speed))
                height_error = max(height_error,abs(row['z_m']-receipt['altitude_m']))
            else:
                counts['missing_tracking_in_window'] += 1
        if previous is not None:
            dt = t-previous['sim_time_s']
            biggest_gap = max(biggest_gap,dt)
            if dt <= 0:
                counts['nonincreasing_intervals'] += 1
            elif dt > dt_limit:
                counts['excluded_gap_intervals'] += 1
            shares = (spans[-2],spans[-1]) if changed else (spans[-1],spans[-1])
            if dt > 0:
                for s in shares:
                    s['observed_duration_s'] += 0.5*dt
            good = 0 < dt <= dt_limit
            p0 = previous['power_w']
            if good and all(row[k] is not None and previous[k] is not None for k in ('x_m','y_m','z_m')):
                dxy=math.hypot(row['x_m']-previous['x_m'],row['y_m']-previous['y_m'])
                dz=abs(row['z_m']-previous['z_m'])
                for s in shares:
                    s['horizontal_path_m']+=0.5*dxy; s['absolute_vertical_path_m']+=0.5*dz
                path_h+=dxy
            if good and p0 is not None and power is not None:
                for s, p in zip(shares,(p0,power)):
                    wh = dt*p/7200
                    s['supported_duration_s'] += dt/2
                    s['supported_wh'] += wh
                    per_phase[s['phase']] += wh
                integrated_time += dt
                if active and prev_active:
                    # Independent endpoint products, summed separately from the native accumulator.
                    measure_intervals.append((dt,(p0+power)*0.5*dt/3600))
            elif dt > 0:
                unsupported_time += dt
                if active and prev_active:
                    counts['unsupported_measurement_intervals'] += 1
        previous, prev_active, endpoint = row, active, row
    if not spans or not endpoint:
        raise ValueError(f'empty trace {case_id}')
    phases_total = math.fsum(s['supported_wh'] for s in spans)
    delta_counter = endpoint['counter_wh']-first['counter_wh']
    tol = LIMITS['native_energy_abs_tolerance_wh']
    phase_errors = [abs(per_phase[k] - terminal['phase_energy_by_phase_wh'].get(k,0.))
                    for k in set(terminal['phase_energy_by_phase_wh']) | set(per_phase)]
    reasons=[]
    if abs(phases_total-delta_counter)>tol or abs(phases_total-terminal['integrated_supported_phase_energy_wh'])>tol or max(phase_errors,default=0.)>tol:
        reasons.append('PHASE_OR_TOTAL_COUNTER_RESIDUAL')
    if vi_error>LIMITS['native_voltage_current_abs_tolerance_w']:
        reasons.append('VI_RESIDUAL')
    if counts['nonincreasing_intervals'] or counts['excluded_gap_intervals']:
        reasons.append('BROKEN_CLOCK_OR_TELEMETRY_GAP')
    if counts['measurement_windows'] != 1 or counts['unsupported_measurement_intervals'] or not measure_intervals:
        reasons.append('INCOMPLETE_MEASUREMENT_POWER')
    track_ok = bool(speeds) and counts['missing_tracking_in_window']==0 and speed_error<=max(LIMITS['measurement_velocity_floor_tolerance_mps'],LIMITS['measurement_velocity_relative_tolerance']*cmd_speed) and height_error<=LIMITS['measurement_altitude_tolerance_m']
    if not track_ok:
        reasons.append('TRACKING_TOLERANCE')
    if receipt['outcome']!='COMPLETE' or not receipt['landed'] or receipt['armed']:
        reasons.append('INCOMPLETE_FLIGHT')
    amounts = dict(aligned_start_s=None,aligned_end_s=None,aligned_span_s=None,aligned_energy_wh=None,
                   aligned_mean_power_w=None,aligned_counter_delta_wh=None,measurement_integral_error_wh=None,
                   aligned_outbound_distance_m=None,aligned_wh_per_km=None)
    if measure_first is not None and measure_last is not None and measure_intervals:
        wh=math.fsum(i[1] for i in measure_intervals)
        span=measure_last['sim_time_s']-measure_first['sim_time_s']
        counter=measure_last['counter_wh']-measure_first['counter_wh']
        measured_distance = ((measure_last['x_m']-measure_first['x_m']) if cmd_speed and measure_last['x_m'] is not None and measure_first['x_m'] is not None else None)
        amounts.update(aligned_start_s=measure_first['sim_time_s'],aligned_end_s=measure_last['sim_time_s'],
                       aligned_span_s=span,aligned_energy_wh=wh,aligned_mean_power_w=wh*3600/span if span>0 and not counts['unsupported_measurement_intervals'] else None,
                       aligned_counter_delta_wh=counter,measurement_integral_error_wh=wh-counter,
                       aligned_outbound_distance_m=measured_distance,
                       aligned_wh_per_km=1000*wh/measured_distance if measured_distance and measured_distance>0 else None)
        if abs(wh-counter)>tol: reasons.append('MEASUREMENT_COUNTER_RESIDUAL')
        if not cmd_speed and abs(span-receipt['hover_duration_s']) > LIMITS['native_hover_duration_tolerance_s']:
            reasons.append('HOVER_DURATION_NOT_REACHED' if span < receipt['hover_duration_s'] else 'HOVER_DURATION_OVERRUN')
        if cmd_speed:
            gate_distance = terminal['measurement_window_distance_m']
            if gate_distance is None or gate_distance < receipt['measurement_distance_m']-LIMITS['native_route_shortfall_tolerance_m']:
                reasons.append('MEASUREMENT_ROUTE_SHORTFALL')
            # The gate snapshots and callback samples can straddle a 20 Hz position report.
            max_endpoint_difference = LIMITS['native_route_shortfall_tolerance_m'] + cmd_speed * LIMITS['native_endpoint_allowance_s']
            if measured_distance is None or gate_distance is None or abs(measured_distance-gate_distance)>max_endpoint_difference:
                reasons.append('GATE_AND_OBSERVER_DISTANCE_DISAGREE')
    a=model.assess(receipt['payload_kg'],cmd_speed,receipt['wind_speed_mps'],receipt['altitude_m'],receipt['temperature_c'],
                   receipt['measurement_distance_m'],receipt['hover_duration_s'],receipt['wind_direction_toward_deg_ENU'])
    budget=model.budget(phases_total)
    if budget.status in ('NAMEPLATE_EXHAUSTED','BELOW_PLANNED_RESERVE'): reasons.append(budget.status)
    return (dict(case_id=case_id,run_id=receipt['run_id'],simulator='PX4_OFFBOARD_GAZEBO_8.11',seed=receipt.get('observed_seed'),
                 manager_sha256=provenance['source_files']['m100_experiment_manager']['sha256'],
                 **{k:receipt[k] for k in ('payload_kg','cruise_speed_mps','temperature_c','altitude_m','measurement_distance_m','hover_duration_s','wind_speed_mps','wind_direction_toward_deg_ENU','observed_physics_step_s','outcome')},
                 data_verdict='REJECTED' if reasons else 'NUMERICALLY_ACCEPTED',reason='|'.join(reasons), stock_exclusions='|'.join(a.stock_reasons),
                 evidence_flags='|'.join(a.evidence_flags),
                 source_observations=counts['observations'],recorded_start_s=first['sim_time_s'],recorded_end_s=endpoint['sim_time_s'],
                 powered_observations=counts['powered'],paired_vi=counts['vi_available'],powered_without_vi=counts['vi_missing_powered'],max_abs_vi_w=vi_error,
                 max_sample_interval_s=biggest_gap,unsupported_or_gap_s=unsupported_time,integrated_duration_s=integrated_time,
                 summed_phase_wh=phases_total,runtime_supported_wh=terminal['integrated_supported_phase_energy_wh'],
                 phase_sum_residual_wh=phases_total-terminal['integrated_supported_phase_energy_wh'],max_abs_individual_phase_error_wh=max(phase_errors,default=0.),
                 full_flight_energy_claim=terminal['is_full_flight_energy'],max_abs_density_error_kg_m3=rho_error,
                 gate_distance_m=terminal['measurement_window_distance_m'],gate_energy_wh=terminal['measurement_window_energy_wh'],
                 max_speed_error_mps=speed_error,max_height_error_m=height_error,recorded_horizontal_path_m=path_h,
                 achieved_speed_min_mps=min(speeds) if speeds else None,achieved_speed_max_mps=max(speeds) if speeds else None,
                 achieved_height_min_m=min(heights) if heights else None,achieved_height_max_m=max(heights) if heights else None,
                 nominal_remaining_wh=budget.nominal_remaining_wh, budget_status=budget.status, **amounts),spans)


def archive_pack_drive(directory, out, cid, rid, provenance):
    """Preserve the native rotor and pack samples recorded alongside new flight CSVs."""
    if "native_pack_drive" not in provenance:
        return {}
    receipt = provenance["native_pack_drive"]["trace"]
    name = f"{rid}.pack_drive.jsonl"
    if receipt["filename"] != name:
        raise ValueError(f"{rid}: pack trace filename differs from recording receipt")
    archived = out / "pack_drive" / f"{cid}-{rid}.jsonl.gz"
    archived.parent.mkdir(exist_ok=True)
    checksum = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=archived.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with (directory / name).open("rb") as source, temporary_path.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=7, mtime=0) as encoded:
                while chunk := source.read(1024 * 1024):
                    checksum.update(chunk)
                    encoded.write(chunk)
        if checksum.hexdigest() != receipt["sha256"]:
            raise ValueError(f"{rid}: pack trace SHA-256 differs from recording receipt")
        temporary_path.replace(archived)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"pack_drive_path": str(archived.relative_to(out)), "pack_drive_jsonl_sha256": receipt["sha256"]}


def package(run_dirs, out):
    model=Model.load(); out.mkdir(parents=True,exist_ok=True)
    (out/'traces').mkdir(exist_ok=True); (out/'provenance').mkdir(exist_ok=True)
    (out/'telemetry_1Hz').mkdir(exist_ok=True)
    records=[]; segments=[]; attempts=[]; sources=[]
    for directory in run_dirs:
        for path in sorted(directory.glob('*.json')):
            info=json.loads(path.read_text()); cid=info.get('case_id')
            if cid is None or 'outcome' not in info: continue
            rid=info.get('run_id'); attempts.append(dict(source_directory=directory.name,**info))
            if not rid:
                if info['outcome'] == 'COMPLETE':
                    raise ValueError(f'completed case {cid} has no flight ID')
                continue
            trace=directory/(rid+'.trace.csv.gz'); provenance=directory/(rid+'.provenance.json')
            if not provenance.is_file() or not trace.is_file():
                raise ValueError(f'flight {rid} is missing trace or provenance in {directory}')
            packed=out/'traces'/f'{cid}-{rid}.csv.xz'
            if not packed.exists():
                tmp=packed.with_suffix('.tmp')
                with gzip.open(trace,'rb') as src, lzma.open(tmp,'wb',preset=4) as dst:
                    shutil.copyfileobj(src,dst)
                tmp.replace(packed)
            with gzip.open(trace, 'rb') as original, lzma.open(packed, 'rb') as stored:
                plain_sha = hashlib.file_digest(original, 'sha256').hexdigest()
                if hashlib.file_digest(stored, 'sha256').hexdigest() != plain_sha:
                    raise ValueError(f'lossless trace check failed for {rid}')
            doc=json.loads(provenance.read_text()); source_csv=directory/(rid+'.csv')
            if digest(source_csv)!=doc['csv']['sha256']: raise ValueError('native CSV SHA mismatch')
            csv_archive=out/'telemetry_1Hz'/f'{cid}-{rid}.csv.gz'
            csv_archive.write_bytes(gzip.compress(source_csv.read_bytes(),compresslevel=7,mtime=0))
            receipt=out/'provenance'/f'{cid}-{rid}.json'
            receipt.write_text(json.dumps(doc,sort_keys=True,indent=2)+'\n')
            row, spans=verify(read_trace(packed),info,doc,model)
            records.append(row); segments.extend(spans)
            sources.append(dict(case_id=cid,run_id=rid,trace_path=str(packed.relative_to(out)),
                                trace_xz_sha256=digest(packed),trace_csv_sha256=plain_sha,raw_input_gz_sha256=digest(trace),
                                provenance_path=str(receipt.relative_to(out)),provenance_sha256=digest(receipt),
                                telemetry_path=str(csv_archive.relative_to(out)),original_csv_sha256=doc['csv']['sha256'],
                                **archive_pack_drive(directory, out, cid, rid, doc)))
    if not records or len({(r['case_id'],r['run_id']) for r in records}) != len(records):
        raise ValueError('expected unique native flight records')
    table(out/'runs.csv',records); table(out/'phases.csv',segments); table(out/'attempts.csv',attempts); table(out/'sources.csv',sources)
    return records


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def restore_acquisition(row):
    receipt = dict(row)
    for k in ('payload_kg', 'cruise_speed_mps', 'wind_speed_mps', 'wind_direction_toward_deg_ENU',
              'altitude_m', 'temperature_c', 'measurement_distance_m', 'hover_duration_s', 'observed_physics_step_s'):
        receipt[k] = float(row[k])
    for k in ('armed','landed'):
        if row[k] not in ('True','False'):
            raise ValueError(f'invalid terminal state {k}={row[k]}')
        receipt[k] = row[k] == 'True'
    receipt['observed_seed'] = int(row['observed_seed']) if row.get('observed_seed') else None
    return receipt


def assert_same(expected, measured):
    for k, val in measured.items():
        reported = expected[k]
        if isinstance(val, (float, int)) and not isinstance(val, bool):
            ok = bool(reported) and math.isclose(float(reported), val, rel_tol=1e-12, abs_tol=1e-12)
        else:
            ok = reported == ('' if val is None else str(val))
        if not ok:
            raise ValueError(f"{expected.get('run_id')} {k}: stored {reported!r}, recomputed {val!r}")


def verify_saved(out):
    """Recompute every published run and chronological phase from the compressed callback records."""
    model = Model.load()
    key = lambda r: (r['case_id'],r['run_id'])
    attempts = {key(r): restore_acquisition(r) for r in read_rows(out/'attempts.csv') if r['run_id']}
    runs = {key(r): r for r in read_rows(out/'runs.csv')}
    phases = defaultdict(list)
    for r in read_rows(out/'phases.csv'):
        phases[key(r)].append(r)
    sources = read_rows(out/'sources.csv')
    if {key(r) for r in sources} != runs.keys() or runs.keys() != attempts.keys() or len(sources) != len(runs):
        raise ValueError('native receipt, trace and results inventories disagree')
    for source in sources:
        k = key(source)
        trace = out/source['trace_path']
        provenance = out/source['provenance_path']
        if digest(trace) != source['trace_xz_sha256'] or digest(provenance) != source['provenance_sha256']:
            raise ValueError(f'{k} native artifact digest differs')
        with gzip.open(out/source['telemetry_path'], 'rb') as raw:
            if hashlib.file_digest(raw,'sha256').hexdigest() != source['original_csv_sha256']:
                raise ValueError(f'{k} original CSV hash differs')
        doc = json.loads(provenance.read_text())
        if "native_pack_drive" in doc:
            expected = doc["native_pack_drive"]["trace"]["sha256"]
            with gzip.open(out / source["pack_drive_path"], "rb") as raw:
                if hashlib.file_digest(raw, "sha256").hexdigest() != expected or source["pack_drive_jsonl_sha256"] != expected:
                    raise ValueError(f"{k} pack drive recording hash differs")
        record, spans = verify(read_trace(trace), attempts[k], doc, model)
        assert_same(runs[k],record)
        for old, new in zip(phases[k],spans,strict=True):
            assert_same(old,new)
    return len(sources)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dirs', nargs='*', type=Path)
    p.add_argument('--out', type=Path, default=ROOT/'results/m100_reaudit/native')
    p.add_argument('--verify-saved', action='store_true')
    a = p.parse_args()
    if a.verify_saved and not a.run_dirs:
        print(f'{verify_saved(a.out)} native traces verified')
    elif a.run_dirs and not a.verify_saved:
        for r in package(a.run_dirs,a.out):
            print(r['case_id'],r['data_verdict'],r['reason'],r['measurement_integral_error_wh'])
    else:
        p.error('supply acquisition directories or --verify-saved')
