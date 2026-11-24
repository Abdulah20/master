"""Publish case-matched supervisor energies from the 2RC acquisitions and both historical releases."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, median
import zipfile

from . import supervisor_2rc as repeat
from .direct_airborne_evidence import TARGET, confirm, digest
from .mission_campaign import read, save
from .supervisor_receipts import verify_export
from .supervisor_release import SEGMENT, write_bytes, write_table


CSV = 'M100_69_experiments_Thevenin_2RC.csv'
HISTORICAL = repeat.ROOT / 'tmp/m100_reaudit_supervisor_rerun'


def numeric(row, field):
    value = row.get(field)
    if value is None or value=='':
        return None
    number = float(value)
    confirm(math.isfinite(number), f'Invalid {field}')
    return number


def difference(new, old):
    if new is None or old is None:
        return None, None
    return new-old, 100*(new-old)/old if old else None


def historical_windows(protocol):
    path = repeat.WORK / 'historical_measurement_check.json'
    previous = repeat.rows(repeat.REFERENCE)
    confirm(digest(repeat.REFERENCE)==protocol['previous_csv_sha256'] and
            digest(repeat.ORIGINAL)==protocol['original_csv_sha256'], 'Historical release bytes changed')
    if path.is_file():
        saved = read(path)
        confirm(saved['previous_csv_sha256']==protocol['previous_csv_sha256'], 'Cached historical study differs')
        return saved['windows']
    with zipfile.ZipFile(repeat.OLD / 'recorder_files.zip') as bundle:
        verify_export(bundle, repeat.OLD)
        by_case = {}
        cases = {case['case_id']:case for case in protocol['cases']}
        limit = repeat.gap_limit()
        for row in previous:
            cid, run = row['case_id'], row['run_id']
            prefix = Path(cid, f"attempt_{row['attempt']}")
            provenance = json.loads(bundle.read(str(prefix / f'{run}.provenance.json')))
            config = provenance['config']
            confirm(config['route_definition']=='ONE_WAY_PLUS_X_ENU_MEASUREMENT_DISTANCE' and
                    config['route_length_m']==cases[cid]['measurement_distance_m'] and
                    config['hover_duration_s']==cases[cid]['hover_duration_s'], 'Historical measurement protocol differs')
            confirm(provenance['energy_summary']['measurement_window_energy_wh']==numeric(row, SEGMENT)
                    and provenance['energy_summary']['measurement_frozen'], 'Historical segment differs from recorder')
            path_to_trace = HISTORICAL / prefix / f'{run}.trace.csv.gz'
            if path_to_trace.is_file():
                by_case[cid] = repeat.measurement_trace(path_to_trace, cases[cid], numeric(row, SEGMENT), limit)
            else:
                by_case[cid] = dict(complete=False, reason='HISTORICAL_HIGH_RATE_TRACE_UNAVAILABLE')
    save(path, dict(previous_csv_sha256=protocol['previous_csv_sha256'], windows=by_case))
    return by_case


def acquired(case):
    cid = case['case_id']
    audited = []
    for path in sorted((repeat.OUT / 'audits' / cid).glob('attempt_*.json')):
        doc = read(path)
        receipt = read(repeat.OUT / 'acquisition' / cid / path.name)
        confirm(doc['case_id']==receipt['case_id']==cid and doc['attempt']==receipt['attempt']
                and doc['archive_sha256']==receipt['archive_sha256']
                and receipt['protocol_sha256']==digest(repeat.PROTOCOL), 'Attempt receipt and audit differ')
        audited.append((doc, receipt))
    accepted = [(doc, receipt) for doc, receipt in audited if doc['accepted']]
    confirm(len(accepted)<=1, 'More than one accepted attempt for a unique setting')
    full_only = [(doc, receipt) for doc, receipt in audited if doc['verified_airborne']]
    selected = (accepted or full_only or audited[-1:] or [({}, {})])[0]
    return selected, audited


def compare(case, groups, previous, original, prior_window, doc, receipt, attempts):
    speed = case['cruise_speed_mps']
    flew = doc.get('verified_airborne', False)
    metrics = doc.get('metrics', {})
    check = doc.get('measurement_trace', {})
    measured = flew and doc.get('verified_segment', False)
    new_full = numeric(metrics, TARGET) if flew else None
    new_segment = numeric(metrics, SEGMENT) if measured else None
    old_full, old_segment = numeric(previous, TARGET), numeric(previous, SEGMENT)
    first_segment = numeric(original, SEGMENT)
    delta_full, pct_full = difference(new_full, old_full)
    delta_old, pct_old = difference(new_segment, old_segment if prior_window['complete'] else None)
    delta_original, pct_original = difference(new_segment, first_segment if original['data_quality'].startswith('PASS') else None)
    cid = case['case_id']
    wind = case['wind_speed_mps']
    return dict(case_id=cid, study_groups='|'.join(groups), **{k:case[k] for k in repeat.CASE_FIELDS},
        wind_relative_to_outbound={0:'tailwind',90:'crosswind',180:'headwind'}[case['wind_direction_toward_deg_ENU']] if wind else 'calm',
        measurement_phase='CRUISE' if speed else 'HOVER', planned_outbound_measurement_m=case['measurement_distance_m'] if speed else None,
        planned_hover_duration_s=None if speed else case['hover_duration_s'],
        estimated_airborne_energy_Wh=new_full, previous_estimated_airborne_energy_Wh=old_full,
        delta_airborne_energy_Wh=delta_full, delta_airborne_energy_pct=pct_full,
        measurement_energy_Wh=new_segment, previous_measurement_energy_Wh=old_segment, original_measurement_energy_Wh=first_segment,
        delta_measurement_energy_Wh=delta_old, delta_measurement_energy_pct=pct_old,
        delta_measurement_vs_original_Wh=delta_original, delta_measurement_vs_original_pct=pct_original,
        airborne_duration_s=metrics.get('airborne_duration_s') if flew else None,
        previous_airborne_duration_s=numeric(previous,'airborne_duration_s'),
        airborne_horizontal_distance_m=metrics.get('airborne_horizontal_distance_m') if flew else None,
        previous_airborne_horizontal_distance_m=numeric(previous,'airborne_horizontal_distance_m'),
        measurement_window_duration_s=check.get('duration_s') if measured else None,
        previous_measurement_window_duration_s=prior_window.get('duration_s'),
        measurement_distance_m=metrics.get('measurement_distance_m') if measured else None,
        previous_measurement_distance_m=numeric(previous,'measurement_distance_m'),
        measurement_mean_power_W=check.get('integrated_mean_power_W') if measured else None,
        previous_measurement_mean_power_W=prior_window.get('integrated_mean_power_W'),
        airborne_mean_power_W=metrics.get('airborne_mean_power_W') if flew else None,
        previous_airborne_mean_power_W=numeric(previous,'airborne_mean_power_W'),
        nominal_remaining_energy_Wh=(metrics['nameplate_Wh']-new_full) if flew else None,
        nominal_remaining_soc_percent=metrics.get('nominal_energy_remaining_percent_after_airborne') if flew else None,
        energy_uncovered_s=metrics.get('uncovered_s'),
        airborne_manager_fallback_Wh=metrics.get('airborne_manager_fallback_Wh') if flew else None,
        airborne_manager_fallback_s=metrics.get('airborne_manager_fallback_s') if flew else None,
        previous_airborne_manager_fallback_Wh=numeric(previous,'airborne_manager_fallback_Wh'),
        fit_discharge_limit_reached=metrics.get('charge_fit_exhaustion_observed'),
        min_supported_load_voltage_V=metrics.get('min_supported_load_voltage_V'),
        max_supported_current_A=metrics.get('max_supported_current_A'),
        motor_command_limited_publications=metrics.get('motor_command_limited_publications'),
        battery_dynamics_model=metrics.get('battery_dynamics_model'), initial_nominal_soc_percent=case['initial_soc_percent'],
        wind_electrical_calibration=metrics.get('wind_model_classification'),
        wind_power_runtime_applied=metrics.get('legacy_wind_power_runtime_applied'),
        cruise_correction_applied=metrics.get('cruise_correction_applied'),
        training_eligible=bool(flew and metrics.get('training_eligible')), label_source=metrics.get('label_source'),
        recorder_status=metrics.get('training_status'), measurement_comparison_verified=bool(measured and prior_window['complete']),
        verification_status='VERIFIED' if doc.get('accepted') else 'AIRBORNE_ONLY' if flew else 'NO_VERIFIED_NEW_TARGET',
        rejection_reason=doc.get('reason'), run_id=metrics.get('run_id'), previous_run_id=previous['run_id'],
        original_run_id=original['run_id'], attempt=doc.get('attempt'), attempts_reviewed=attempts,
        seed=case['seed'], physics_step_s=case['physics_step_s'], raw_archive=receipt.get('archive'),
        raw_archive_sha256=receipt.get('archive_sha256'), direct_recorder_sha256=metrics.get('direct_row_sha256') if flew else None)


def changes(rows, delta_Wh, delta_pct):
    values = [r[delta_Wh] for r in rows if r[delta_Wh] is not None]
    pct = [r[delta_pct] for r in rows if r[delta_pct] is not None]
    if not values:
        return dict(pairs=0)
    return dict(pairs=len(values), mean_Wh=mean(values), median_Wh=median(values), mean_abs_Wh=mean(map(abs,values)),
        min_Wh=min(values), max_Wh=max(values), mean_pct=mean(pct), median_pct=median(pct),
        min_pct=min(pct), max_pct=max(pct), more=sum(x>0 for x in values), less=sum(x<0 for x in values),
        maximum_abs_case=max((r for r in rows if r[delta_Wh] is not None),key=lambda r:abs(r[delta_Wh]))['case_id'])


def describe(rows):
    return dict(full_vs_previous=changes(rows,'delta_airborne_energy_Wh','delta_airborne_energy_pct'),
                measurement_vs_previous=changes(rows,'delta_measurement_energy_Wh','delta_measurement_energy_pct'),
                measurement_vs_original=changes(rows,'delta_measurement_vs_original_Wh','delta_measurement_vs_original_pct'))


def full_energy_components(rows):
    """Reconcile a measured change in E with the corresponding average power and duration."""
    time_changes = []
    power_changes = []
    for row in rows:
        if row[TARGET] is None:
            continue
        t0, t1 = row['previous_airborne_duration_s'], row['airborne_duration_s']
        p0, p1 = row['previous_airborne_mean_power_W'], row['airborne_mean_power_W']
        time_change = .5*(p1+p0)*(t1-t0)/3600
        power_change = .5*(t1+t0)*(p1-p0)/3600
        confirm(math.isclose(time_change+power_change, row['delta_airborne_energy_Wh'], abs_tol=1e-10),
                'Power and duration do not reproduce the change in full-flight energy')
        time_changes.append(abs(time_change))
        power_changes.append(abs(power_change))
    return dict(mean_abs_duration_component_Wh=mean(time_changes) if time_changes else None,
                mean_abs_power_component_Wh=mean(power_changes) if power_changes else None)


def render(data, summary):
    lines=['# إعادة تجارب المشرفة باستخدام Thévenin 2RC','',
           f"سُجّلت طاقة الإقلاع إلى الهبوط في **{summary['verified_airborne']} من {len(data)} حالة**. "
           f"تطابق مقطع القياس في **{summary['verified_segment_pairs']} حالة** مع التجارب السابقة المسجّلة بالنموذج الساكن. "
           f"نفّذ المحاكي {summary['attempts']} محاولة. "
           f"[ملف النتائج والمقارنة]({CSV}) · [ملخص التحقق](verification.json)",'',
           'أُعيدت الحالات بإعدادات الحمولة والسرعة والارتفاع والحرارة والرياح نفسها، وبذرة Gazebo وخطوة الفيزياء نفسيهما. '
           'استُخدمت نسخة إعدادات بدء PX4 التي تطابق بصمة الحملة السابقة. '
           'مقطع الطيران الأفقي هو ذهاب 500 م بعد استقرار السرعة، ومقطع التحويم 60 ثانية. '
           'تُقاس طاقة الرحلة الكاملة بين إشارتَي الإقلاع والهبوط في PX4. '
           'عُرض زمن كل مقطع والمسافة المحققة لتفسير التغيّر بين التشغيلات. [بروتوكول التشغيل](protocol.json)','',
           '## الفرق بين التشغيلين','',
           '| المقارنة | الأزواج | متوسط الفرق (Wh) | متوسط الفرق المطلق (Wh) | متوسط الفرق النسبي |',
           '|---|---:|---:|---:|---:|']
    for name,label in (('measurement_vs_previous','مقطع القياس مقابل إعادة التشغيل السابقة'),
                       ('measurement_vs_original','مقطع القياس مقابل نتائج المشرفة الأولى'),
                       ('full_vs_previous','الإقلاع إلى الهبوط مقابل إعادة التشغيل السابقة')):
        stats=summary['overall'][name]
        if stats['pairs']:
            lines.append(f"| {label} | {stats['pairs']} | {stats['mean_Wh']:+.3f} | {stats['mean_abs_Wh']:.3f} | {stats['mean_pct']:+.2f}% |")
    lines+=['','حُسب كل فرق بطرح الطاقة السابقة من الطاقة الجديدة. النسبة المعروضة متوسط النسب لكل حالة. '
            'تستخدم المقارنة مع الملف الأول طاقة مقطع القياس المنشورة فيه. يتوفر هدف الإقلاع إلى الهبوط في إعادة التشغيل السابقة، '
            'وكُتب في الحالتين مباشرة عند الهبوط.','']
    parts=summary['full_energy_components']
    if parts['mean_abs_duration_component_Wh'] is not None:
        lines+=['عند تفكيك فرق طاقة الرحلة حسابيًا، بلغ متوسط المقدار المرتبط بتغيّر المدة '
                f"{parts['mean_abs_duration_component_Wh']:.3f} Wh. وبلغ المقدار المرتبط بتغيّر متوسط القدرة "
                f"{parts['mean_abs_power_component_Wh']:.3f} Wh. قد يتعاكس الاتجاهان في الرحلة نفسها. "
                'تقارن هذه الأرقام إصدارين شُغّلا بالأوامر نفسها، وتشمل الفروق في مدّة المراحل وتتبع المسار.','']
    lines+=['## السيناريوهات A وB وC','',
            '| الحالة | طاقة المقطع سابقًا ← الآن (Wh) | طاقة الرحلة سابقًا ← الآن (Wh) | الفرق في طاقة الرحلة (Wh) |',
            '|---|---:|---:|---:|']
    def fmt(value, sign=False):
        return '—' if value is None else f'{value:+.3f}' if sign else f'{value:.3f}'
    for row in data:
        group=next((g for g in row['study_groups'].split('|') if g.startswith('scenario_')), None)
        if group:
            lines.append(f"| {group[-1]} | {fmt(row['previous_measurement_energy_Wh'])} ← {fmt(row[SEGMENT])} | "
                         f"{fmt(row['previous_estimated_airborne_energy_Wh'])} ← {fmt(row[TARGET])} | {fmt(row['delta_airborne_energy_Wh'],True)} |")
    lines+=['','## حدود الاستنتاج','',
            'يستخدم مقطع القياس تعريف القدرة التشغيلية نفسه في مدير الرحلة. '
            'تحققنا من قيمه بإعادة تكامل عينات القدرة بحدود نافذة القياس لكل تشغيل. '
            'يؤثر 2RC في جهد الحزمة والتيار وسقف الدفع، وقد يغيّر مسار الطائرة وزمن المراحل. '
            'لم تُفعّل معايرة سرعة الطيران المستقر لأن أوامر السرعة هنا خارج سرعات تفعيلها (4 و6 و8 م/ث).','',
            f"وصل نموذج الحزمة إلى حد ملاءمة الشحنة في {summary['fit_limit_reached']} رحلة مقبولة. "
            'يبيّن عمود `airborne_manager_fallback_Wh` حصة الطاقة التي قدّرها مدير الرحلة عندما تعذّر حساب الجهد والتيار ضمن المجال المدعوم. '
            'صُنّف تأثير الرياح في القدرة الكهربائية `C_NOT_CALIBRATED`. '
            'تحتاج معايرة بطارية TB47D وخسائر ESC 620D إلى قياسات مستقلة.','']
    if summary['rejected_attempt_reasons']:
        lines+=['## المحاولات المستبعدة','', '| السبب المسجّل | العدد |','|---|---:|',
                *[f"| {reason.replace('|', '، ')} | {n} |" for reason,n in summary['rejected_attempt_reasons'].items()],'']
    write_bytes(repeat.OUT / 'comparison_ar.md', ('\n'.join(lines).rstrip()+'\n').encode())


def publish(protocol):
    prior_windows = historical_windows(protocol)
    previous = {r['case_id']:r for r in repeat.rows(repeat.REFERENCE)}
    original = {r['run_id']:r for r in repeat.rows(repeat.ORIGINAL)}
    data, attempts = [], []
    for case in protocol['cases']:
        cid = case['case_id']
        last = previous[cid]
        (doc, receipt), candidates = acquired(case)
        attempts.extend(item for item,_ in candidates)
        groups = [name for name, members in protocol['groups'].items() if cid in members]
        data.append(compare(case, groups, last, original[last['historical_run_id']], prior_windows[cid], doc, receipt, len(candidates)))
    considered = {a['case_id'] for a in attempts}
    confirm(len(considered)==len(data), 'One or more cases have not completed their first historical repeat')
    confirm(all(r['attempts_reviewed']==protocol['max_attempts'] or r['verification_status']=='VERIFIED' for r in data),
            'An incomplete case is still eligible for a replacement acquisition')
    counts = Counter(r['verification_status'] for r in data)
    verify = dict(registered=len(data), attempts=len(attempts), verified_airborne=sum(r[TARGET] is not None for r in data),
        verified_segment_pairs=sum(r['measurement_comparison_verified'] for r in data),
        original_measurement_pairs=sum(r['delta_measurement_vs_original_Wh'] is not None for r in data),
        status_counts=dict(counts), rejected_attempt_reasons=dict(Counter(a['reason'] for a in attempts if not a['accepted'])),
        historical_full_flights_verified=len(previous), historical_high_rate_windows_verified=sum(w['complete'] for w in prior_windows.values()),
        fit_limit_reached=sum(bool(r['fit_discharge_limit_reached']) for r in data if r[TARGET] is not None),
        protocol_sha256=digest(repeat.PROTOCOL), baseline_csv_sha256=protocol['previous_csv_sha256'],
        original_csv_sha256=protocol['original_csv_sha256'], overall=describe(data), full_energy_components=full_energy_components(data),
        by_segment={phase:describe([r for r in data if r['measurement_phase']==phase]) for phase in ('CRUISE','HOVER')},
        by_wind={key:describe([r for r in data if (r['wind_speed_mps']>0)==present]) for key,present in [('calm',False),('wind',True)]})
    file = repeat.OUT / CSV
    write_table(file, data)
    verify['comparison_csv_sha256']=digest(file)
    save(repeat.OUT / 'verification.json',verify)
    render(data,verify)
    print('Published', len(data), 'cases and', verify['verified_segment_pairs'], 'measurement pairs', flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical-only', action='store_true')
    args = p.parse_args()
    protocol = read(repeat.PROTOCOL)
    if args.historical_only:
        checks = historical_windows(protocol)
        print(sum(w['complete'] for w in checks.values()), 'of', len(checks), 'previous measurement windows verified')
    else:
        publish(protocol)
