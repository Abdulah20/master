"""Export the original supervisor cohort from its selected recorder bytes and PX4 logs."""
import csv
import gzip
import json
from pathlib import Path
import zipfile

from .direct_airborne_evidence import confirm, digest
from .mission_campaign import OUT, ROOT, read, save
from .mission_observations import online_series, flight_topics, table
from .supervisor_receipts import verify_export

HIST = OUT.parent/'supervisor_rerun'
RAW = ROOT/'tmp/m100_reaudit_supervisor_rerun'
CACHE = ROOT/'tmp/m100_closure/historical_intervals'


def combine_intervals(files, path):
    import shutil

    tmp=path.with_suffix('.part')
    header=None
    with tmp.open('wb') as packed:
        with gzip.GzipFile(filename='', mode='wb',fileobj=packed,compresslevel=6,mtime=0) as stream:
            for file in files:
                with file.open('rb') as src:
                    first=src.readline()
                    if header is None:
                        header=first
                        stream.write(header)
                    confirm(first==header,'Interval schema changes within a release')
                    shutil.copyfileobj(src,stream)
    tmp.replace(path)


def main():
    csv_in=HIST/'M100_69_experiments_rerun.csv'
    with (HIST/'recorder_files.zip').open('rb') as original, zipfile.ZipFile(original) as z:
        verify_export(z,HIST)
    with csv_in.open(encoding='utf-8-sig',newline='') as f:
        cases=list(csv.DictReader(f))
    observations=[]
    files=[]
    for case in cases:
        cid=case['case_id']
        run=case['run_id']
        raw=RAW/cid/('attempt_'+case['attempt'])
        audit=read(HIST/case['audit_file'])
        info=read(raw/f'{run}.provenance.json')
        fit=info['airborne_training']
        confirm(case['estimated_airborne_energy_Wh'] and float(case['estimated_airborne_energy_Wh'])==fit['estimated_airborne_energy_Wh']
                and digest(raw/fit['relative_path'])==case['direct_row_sha256']==fit['sha256']
                and float(case['uncovered_s'])==0,'Historical airborne row differs')
        nominal=99.9
        energy=float(case['estimated_airborne_energy_Wh'])
        duration=float(case['airborne_duration_s'])
        distance=float(case['airborne_horizontal_distance_m'])
        c={k:float(case[k]) for k in ('payload_kg','cruise_speed_mps','altitude_m','wind_speed_mps','wind_direction_toward_deg_ENU','temperature_c')}
        hover=c['cruise_speed_mps']==0
        c.update(case_id=cid,run_id=run,group='historical_supervisor',initial_soc_percent=100.,nameplate_Wh=nominal,
                 mission_route='timed_hover' if hover else 'eastbound_measurement_then_return')
        ulog=raw/audit['sources']['original_ulog']
        wire=raw/f'{run}.pack_drive.jsonl'
        caches=CACHE/cid
        series=caches/'intervals.csv'
        tag=caches/'source.json'
        provenance=dict(ulog_sha256=audit['sources']['ulog_sha256'],wire_sha256=info['native_pack_drive']['trace']['sha256'],
                        recorder_sha256=fit['sha256'],extractor_sha256=digest(Path(__file__).with_name('mission_observations.py')))
        if not (tag.exists() and series.exists() and read(tag)==dict(**provenance,intervals_sha256=digest(series))):
            confirm(digest(ulog)==provenance['ulog_sha256'] and digest(wire)==provenance['wire_sha256'],
                    'Historical ULog or wire digest differs')
            with wire.open() as source:
                packets=[json.loads(line) for line in source]
            rows=online_series(cid,run,packets,flight_topics(ulog),fit['native_budget'],c)
            table(series,rows)
            save(tag,dict(**provenance,intervals_sha256=digest(series)))
        power=energy*3600/duration
        observations.append(dict(**c,base_uav_mass_kg=3.68,total_mass_kg=info['config']['total_mass_kg'],
            initial_nominal_soc_percent=100,planned_outbound_m=float(case['planned_outbound_measurement_m'] or 0),
            planned_hover_s=float(case['planned_hover_duration_s'] or 0),planned_return_m=None,intermediate_stops=0,
            planned_xy_accel_limit_mps2=None,route_definition='TIMED_HOVER' if hover else 'MEASUREMENT_AFTER_ACCELERATION',
            estimated_airborne_energy_Wh=energy,airborne_mean_power_W=power,airborne_duration_s=duration,
            airborne_horizontal_distance_m=distance,nominal_remaining_energy_Wh=nominal-energy,
            nominal_remaining_soc_percent=100*(1-energy/nominal),nominal_reserve_percent=20,
            nominal_reserve_margin_Wh=.8*nominal-energy,nominal_endurance_to_reserve_min=60*.8*nominal/power,
            nominal_horizontal_range_to_reserve_km=distance*.001*.8*nominal/energy,
            original_training_eligible=fit['training_eligible'],flight_energy_source=case['label_source'],
            battery_dynamics_model='CMU_STATIC_LOADED_V1',energy_uncovered_s=0,recorded_manager_energy_Wh=float(case['airborne_manager_fallback_Wh']),
            recorded_manager_coverage_s=float(case['airborne_manager_fallback_s']),training_row_sha256=fit['sha256'],
            source_audit='../supervisor_rerun/'+case['audit_file']))
        files.append(series)
    table(OUT/'historical_preflight.csv',observations)
    combine_intervals(files,OUT/'historical_intervals.csv.gz')
    save(OUT/'historical_provenance.json',dict(selected_flights=len(cases), input_sha256=digest(csv_in),
         recorder_bundle_sha256=digest(HIST/'recorder_files.zip'),preflight_sha256=digest(OUT/'historical_preflight.csv'),
         intervals_sha256=digest(OUT/'historical_intervals.csv.gz'),
         feature_extractor_sha256=digest(Path(__file__).with_name('mission_observations.py')),
         simulator_revision='prior static loaded pack',
         field_data=False, underlying_energy_labels_unchanged=True))
    print(len(cases),'original flights exported')


if __name__=='__main__':
    main()
