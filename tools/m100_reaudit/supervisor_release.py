"""Publish the supervisor flight matrix from audited native acquisition attempts."""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
import time


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/m100_reaudit/supervisor_rerun'
ORIGINAL = Path(__file__).resolve().parent / 'runtime_snapshot/original_study'
MAIN_CSV = 'M100_69_experiments_rerun.csv'
COMPARISONS = 'supervisor_comparisons.csv'
SEGMENT = 'measurement_energy_Wh'
AIRBORNE = 'estimated_airborne_energy_Wh'
WIND_SLOTS = ('headwind', 'tailwind', 'crosswind')
PLAN_FIELDS = ('payload_kg', 'cruise_speed_mps', 'altitude_m', 'temperature_c', 'wind_speed_mps',
               'wind_direction_toward_deg_ENU')
MEASURES = (
    SEGMENT, AIRBORNE, 'supported_phase_mission_energy_Wh', 'sampled_mean_power_W',
    'airborne_mean_power_W', 'airborne_duration_s', 'airborne_horizontal_distance_m',
    'airborne_distance_source', 'airborne_covered_s', 'uncovered_s',
    'airborne_known_subtotal_Wh', 'airborne_manager_fallback_s', 'airborne_manager_fallback_Wh', 'native_VI_coverage_s',
    'charge_fit_exhaustion_observed', 'first_charge_fit_exhaustion_s', 'last_accounted_discharge_Ah',
    'measurement_distance_m', 'sampled_measurement_span_s', 'energy_per_distance_Wh_per_km',
    'sampled_mean_ground_speed_mps', 'sampled_mean_altitude_m',
    'ideal_steady_endurance_min', 'ideal_steady_range_km',
    'nominal_energy_remaining_percent', 'nominal_energy_remaining_percent_after_airborne',
)


def read_json(path):
    return json.loads(path.read_bytes())


def write_bytes(path, content):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def write_table(path, rows):
    data = io.StringIO(newline='')
    writer = csv.DictWriter(data, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    write_bytes(path, data.getvalue().encode('utf-8-sig'))


def complete(metrics):
    return metrics['mission_outcome'] == 'COMPLETE' and metrics[AIRBORNE] is not None


def select_attempts(out, cases):
    attempts = defaultdict(list)
    for path in sorted((out / 'audits').glob('*/attempt_*.json')):
        doc = read_json(path)
        cid, i, m = doc['case_id'], doc['attempt'], doc['metrics']
        if cid not in cases or path.relative_to(out / 'audits') != Path(cid, f'attempt_{i}.json'):
            raise ValueError(f'Unexpected audit identity in {path}')
        energy = m[AIRBORNE]
        if energy is not None and (not math.isfinite(energy) or energy <= 0 or m['uncovered_s'] != 0):
            raise ValueError(f'Unqualified full-flight energy in {path}')
        if complete(m):
            audit, sources = doc['audit'], doc['sources']
            if not (m.get('run_id') and m['run_id'] == audit['run_id']
                    and m.get('direct_row_sha256') and m['direct_row_sha256'] == audit['direct_row_sha256'] == sources['training.csv']
                    and energy == audit[AIRBORNE] and m['airborne_duration_s'] == audit['duration_s']
                    and audit['training_row_matches_recorder_and_terminal_native_budget'] and audit['ulog_and_native_edges_match']):
                raise ValueError(f'Direct recorder row or takeoff-to-landing audit disagrees in {path}')
        attempts[cid].append((i, m, str(path.relative_to(out))))
    chosen = {}
    for cid, takes in attempts.items():
        takes.sort(key=lambda t: t[0])
        chosen[cid] = next((t for t in takes if complete(t[1])), takes[-1])
    return chosen, attempts


def make_row(case, groups, take, count, old):
    number, metrics, audit_path = take
    finished = complete(metrics)
    wind = case['wind_speed_mps']
    orientation = {180: 'headwind', 0: 'tailwind', 90: 'crosswind'}[case['wind_direction_toward_deg_ENU']] if wind else 'calm'
    row = {
        'case_id': case['case_id'], 'study_groups': '|'.join(groups), 'run_id': metrics.get('run_id'),
        'attempt': number, 'attempts_reviewed': count, 'mission_outcome': metrics['mission_outcome'],
        'training_status': metrics['training_status'], 'label_source': metrics.get('label_source'),
        'training_eligible': metrics.get('training_eligible'), **{k: case[k] for k in PLAN_FIELDS},
        'wind_relative_to_outbound': orientation,
        'wind_model_classification': metrics.get('wind_model_classification'),
        'wind_vector_application_verified': metrics.get('wind_vector_application_verified'),
        'manager_wind_power_runtime_applied': metrics.get('legacy_wind_power_runtime_applied'),
        'planned_outbound_measurement_m': case['measurement_distance_m'] if case['cruise_speed_mps'] else None,
        'planned_hover_duration_s': None if case['cruise_speed_mps'] else case['hover_duration_s'],
        **{k: metrics.get(k) for k in MEASURES},
        'data_quality': metrics.get('data_quality'), 'energy_evidence_status': metrics.get('energy_evidence_status'),
        'rejection_reason': metrics.get('rejection_reason'),
        'nominal_airborne_energy_margin_Wh': metrics['nameplate_Wh'] - metrics[AIRBORNE] if finished else None,
        'historical_run_id': old['run_id'],
        'historical_measurement_energy_Wh': old[SEGMENT], 'delta_measurement_energy_Wh': None,
        'delta_measurement_energy_pct': None, 'observed_unfinished_mission_airborne_Wh': None,
        'direct_row_sha256': metrics.get('direct_row_sha256'), 'audit_file': audit_path,
    }
    if not finished:
        row['observed_unfinished_mission_airborne_Wh'] = row[AIRBORNE]
        row[AIRBORNE] = None
    if (metrics['mission_outcome'] == 'COMPLETE' and metrics.get('measurement_frozen')
            and (row['data_quality'] or '').startswith('PASS') and row[SEGMENT] is not None):
        delta = row[SEGMENT] - old[SEGMENT]
        row['delta_measurement_energy_Wh'] = delta
        row['delta_measurement_energy_pct'] = delta * 100 / old[SEGMENT] if old[SEGMENT] > 0 else None
    return row


def logical_table(groups, rows):
    table = []
    for group, ids in groups.items():
        for i, cid in enumerate(ids):
            row = rows[cid]
            wind_slot = WIND_SLOTS[i % 3] if group == 'experiment_3_payload_wind' else None
            if wind_slot and row['wind_speed_mps'] and wind_slot != row['wind_relative_to_outbound']:
                raise ValueError(f'Unexpected wind comparison slot {i} in {group}')
            table.append({'experiment': group, 'comparison_slot': i + 1, 'assigned_wind_direction': wind_slot,
                          **{k: v for k, v in row.items() if k != 'study_groups'}})
    return table


def format_number(number, places=3):
    return '—' if number is None else f'{number:.{places}f}'


def draw_energy(out, rows, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    def numbers(selected):
        return [float('nan') if r[AIRBORNE] is None else r[AIRBORNE] for r in selected]

    fig = plt.figure(figsize=(12.5, 7.2), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=[1.12, 1])
    payload = fig.add_subplot(grid[0, 0])
    exp1 = [rows[k] for k in groups['experiment_1_payload']]
    mass = [r['payload_kg'] for r in exp1]
    payload.plot(mass, numbers(exp1), 'o-', lw=1.8, color='#006e70')
    payload.set(title='Payload at 10 m/s', xlabel='Added payload (kg)', ylabel='Flight energy (Wh)')
    payload.set_xticks(mass, [f'{v:g}' for v in mass])
    payload.margins(x=.08)
    speed = fig.add_subplot(grid[0, 1:])
    exp2 = [rows[k] for k in groups['experiment_2_payload_velocity']]
    speeds = sorted({r['cruise_speed_mps'] for r in exp2})
    matrix = np.full((len(mass), len(speeds)), np.nan)
    for r in exp2:
        if r[AIRBORNE] is not None:
            matrix[mass.index(r['payload_kg']), speeds.index(r['cruise_speed_mps'])] = r[AIRBORNE]
    cmap = plt.get_cmap('viridis').with_extremes(bad='#e8e9e9')
    heat = speed.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap=cmap)
    speed.set_title('Payload and commanded ground speed')
    speed.set_xlabel('Speed (m/s) for 500 m outbound, or 60 s hover')
    speed.set_xticks(range(len(speeds)), ['Hover' if v == 0 else f'{v:g}' for v in speeds])
    speed.set_ylabel('Added payload (kg)')
    speed.set_yticks(range(len(mass)), [f'{p:g}' for p in mass])
    speed.axvline(.5, color='white', lw=2)
    fig.colorbar(heat, ax=speed, label='Flight energy (Wh)', shrink=.8)
    for (i, j), val in np.ndenumerate(matrix):
        if math.isfinite(val):
            speed.text(j, i, f'{val:.1f}', ha='center', va='center',
                       color='white' if heat.norm(val) < .52 else '#122115', fontsize=8)
    exp3 = [rows[k] for k in groups['experiment_3_payload_wind']]
    wind_panels = []
    for i, p in enumerate(sorted({r['payload_kg'] for r in exp3})):
        ax = fig.add_subplot(grid[1, i], sharey=wind_panels[0] if wind_panels else None)
        wind_panels.append(ax)
        for direction, marker, color in zip(WIND_SLOTS, ('o', '^', 's'), ('#a64c23', '#006e70', '#72559a')):
            samples = {r['wind_speed_mps']: r for r in exp3 if r['payload_kg'] == p
                       and r['wind_relative_to_outbound'] in ('calm', direction)}
            values = sorted(samples)
            ax.plot(values, numbers([samples[k] for k in values]), marker=marker, color=color, label=direction.title())
        ax.set_title(f'Wind at {p:g} kg payload')
        ax.set_xlabel('Commanded wind (m/s)')
        ax.set_xticks(sorted({r['wind_speed_mps'] for r in exp3}))
        if not i:
            ax.set_ylabel('Flight energy (Wh)')
        else:
            ax.tick_params(labelleft=False)
    wind_panels[0].legend(frameon=False, fontsize=8)
    for ax in (payload, *wind_panels):
        ax.grid(axis='y', lw=.65, color='#dce0e2')
        ax.set_axisbelow(True)
        ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle('Matrice 100 estimated energy from takeoff to landing', fontsize=16, fontweight='bold')
    with NamedTemporaryFile(dir=out, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        fig.savefig(tmp, format='png', dpi=155, facecolor='white')
        tmp.chmod(0o644)
        tmp.replace(out / 'study_energy.png')
    finally:
        plt.close(fig)
        tmp.unlink(missing_ok=True)


def report(rows, attempts):
    full = [r for r in rows.values() if r[AIRBORNE] is not None]
    counts = Counter(r['training_status'] for r in rows.values())
    lines = [
        '# نتائج إعادة تجارب المشرفة على Matrice 100',
        '',
        f'تغطي الحملة {len(rows)} إعدادًا فريدًا و84 موضع مقارنة. جرت {sum(map(len, attempts.values()))} محاولة. '
        f'سُجّلت طاقة الإقلاع إلى الهبوط لعدد **{len(full)} من {len(rows)}** مهمة مكتملة. '
        f'استوفى **{sum(bool(r["training_eligible"]) for r in full)}** صفًا شروط جدول التدريب ضمن مجال CMU. '
        'تتضمن الملفات أيضًا النتائج العددية المكتملة خارج مجال CMU مع وسم مصدرها.',
        '',
        '## تجربة الحمولة',
        '',
        'البروتوكول عند 10 م/ث و100 م و15°م ورياح مأمورة صفرية ومقطع ذهاب مأمور طوله 500 م.',
        '',
        '| الحمولة (كغ) | طاقة المقطع (Wh) | فرق طاقة المقطع عن السابق (Wh) | طاقة الإقلاع إلى الهبوط (Wh) | زمن الطيران (ث) |',
        '|---:|---:|---:|---:|---:|',
    ]
    for r in rows.values():
        if 'experiment_1_payload' in r['study_groups'].split('|'):
            lines.append(f"| {r['payload_kg']:g} | {format_number(r[SEGMENT])} | {format_number(r['delta_measurement_energy_Wh'])} | "
                         f"{format_number(r[AIRBORNE])} | {format_number(r['airborne_duration_s'])} |")
    speed_rows = {(r['payload_kg'], r['cruise_speed_mps']): r[AIRBORNE] for r in rows.values()
                  if 'experiment_2_payload_velocity' in r['study_groups'].split('|')}
    speeds = sorted({speed for _, speed in speed_rows})
    lines += ['', '## تجربة الحمولة والسرعة', '',
              'تبيّن الخلايا طاقة المهمة من الإقلاع إلى الهبوط بوحدة Wh. في عمود التحويم تقضي الطائرة '
              '60 ثانية عند نقطة القياس. تقطع الطائرة في أعمدة السرعة الأخرى مقطع ذهاب مأمورًا طوله 500 م قبل العودة.',
              '', '| الحمولة (كغ) | ' + ' | '.join('تحويم' if s == 0 else f'{s:g} م/ث' for s in speeds) + ' |',
              '|' + '---:|' * (len(speeds) + 1)]
    for payload in sorted({p for p, _ in speed_rows}):
        lines.append(f'| {payload:g} | ' + ' | '.join(format_number(speed_rows[payload, s]) for s in speeds) + ' |')
    wind_rows = {(r['payload_kg'], r['wind_relative_to_outbound'], r['wind_speed_mps']): r[AIRBORNE]
                 for r in rows.values() if 'experiment_3_payload_wind' in r['study_groups'].split('|')}
    winds = sorted({wind for _, _, wind in wind_rows})
    directions = {'headwind': 'أمامي', 'tailwind': 'خلفي', 'crosswind': 'جانبي'}
    lines += ['', '## تجربة الحمولة والرياح', '',
              'طاقة المهمة الكاملة بوحدة Wh حسب سرعة الرياح المأمورة واتجاهها بالنسبة إلى مقطع الذهاب. '
              'عند عكس المسار في العودة تصبح الرياح الأمامية خلفية والعكس، وتصل الرياح الجانبية من الجانب المقابل. '
              'عند سرعة رياح صفرية تتشارك الاتجاهات الثلاثة الرحلة المرجعية نفسها لكل حمولة.', '',
              'تؤكد استجابة خدمة Gazebo تطبيق متجه الرياح المسجّل في `wind_environment_application`. '
              'لجميع سرعات الرياح غير الصفرية، بما فيها 3 م/ث، يمثل السحب الهوائي وتغيّر القدرة الكهربائية '
              'نقلًا نظريًا مصنّفًا `C_NOT_CALIBRATED` في `wind_simulation_model`. يُطبّق المحاكي قوة سحب على الهيكل، '
              'ويستخدم مدير الرحلة مقدار السرعة بالنسبة للهواء لحساب تغيّر القدرة بالنموذج الكهربائي. '
              'يوثّق `manager_wind_power_runtime_applied` تفعيل هذه المعالجة لكل حالة. '
              'يقارن الجدول استجابة هذا النموذج للرياح المأمورة، ويتطلب تقدير دقته قياسات رياح محيطة '
              'واستهلاك كهربائي متزامنة في الطيران الحقيقي.',
              '', '| الحمولة (كغ) | الاتجاه | ' + ' | '.join(f'{w:g} م/ث' for w in winds) + ' |',
              '|---:|---|' + '---:|' * len(winds)]
    for payload in sorted({p for p, _, _ in wind_rows}):
        for direction in WIND_SLOTS:
            values = (wind_rows[payload, direction if w else 'calm', w] for w in winds)
            lines.append(f'| {payload:g} | {directions[direction]} | ' + ' | '.join(map(format_number, values)) + ' |')
    calm, head, tail = (rows[c] for c in ('case_001', 'case_037', 'case_038'))
    if all(r[AIRBORNE] is not None and r[SEGMENT] is not None for r in (calm, head, tail)):
        lines += ['', f'دون حمولة بلغت طاقة مقطع الذهاب {calm[SEGMENT]:.2f} Wh في الهدوء، و{head[SEGMENT]:.2f} Wh '
                  f'مع رياح أمامية بسرعة 9 م/ث، و{tail[SEGMENT]:.2f} Wh مع رياح خلفية بنفس السرعة. '
                  f'بلغت طاقة المهمة كاملة في الحالات الثلاث بالترتيب {calm[AIRBORNE]:.2f} و{head[AIRBORNE]:.2f} '
                  f'و{tail[AIRBORNE]:.2f} Wh. يصف الترتيب استجابة منحنى القدرة النظري على المسار المحاكى.']
    observed_motion = [r for r in full if r['sampled_mean_ground_speed_mps'] is not None and r['sampled_mean_altitude_m'] is not None]
    if observed_motion:
        max_speed = max(observed_motion, key=lambda r: abs(r['sampled_mean_ground_speed_mps'] - r['cruise_speed_mps']))
        max_altitude = max(observed_motion, key=lambda r: abs(r['sampled_mean_altitude_m'] - r['altitude_m']))
        lines.extend(['', '## السرعة والارتفاع أثناء مقطع القياس', '',
                      f'توافرت متوسطات السرعة الأرضية والارتفاع الموزونة بالزمن لعدد {len(observed_motion)} رحلة. '
                      f'كان أكبر فرق مطلق لمتوسط السرعة عن الأمر في `{max_speed["case_id"]}` '
                      f'بتسجيل {max_speed["sampled_mean_ground_speed_mps"]:.2f} م/ث لأمر {max_speed["cruise_speed_mps"]:g} م/ث. '
                      f'وكان أكبر فرق لمتوسط الارتفاع في `{max_altitude["case_id"]}` '
                      f'بتسجيل {max_altitude["sampled_mean_altitude_m"]:.2f} م لأمر {max_altitude["altitude_m"]:g} م. '
                      'تغطي هذه المتوسطات التحويم أو مقطع الذهاب فقط.'])
    lines.extend([
        '', '## السيناريوهات A وB وC', '',
        'اتجاه الرياح أمامي أثناء مقطع الذهاب كما في بروتوكول الحملة السابقة. '
        'الزمن الفعلي هنا هو المدة بين الإقلاع والهبوط. التحمل والمدى المثاليان إسقاطان عند استهلاك المقطع '
        'وباستخدام الطاقة الاسمية 99.9 واط·ساعة. النسبة في العمود الأخير تعبّر عن رصيد طاقة اسمي بعد الرحلة '
        'على أساس بدء كل حالة برصيد 100% من هذه القيمة وبحد أدنى صفر. '
        'تحتاج حالة الشحن الكهروكيميائية الحقيقية وزمن الفصل إلى قياس مستقل لبطارية TB47D.',
        '', '| الحالة | طاقة الإقلاع إلى الهبوط (Wh) | زمن الطيران (دقيقة) | المسار الأفقي الفعلي في المحاكي (م) | التحمل المثالي (دقيقة) | المدى المثالي (كم) | المتبقي الاسمي بعد الرحلة (%) |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for r in rows.values():
        scenario = next((s for s in r['study_groups'].split('|') if s.startswith('scenario_')), None)
        if scenario:
            duration = r['airborne_duration_s']
            vals = (r[AIRBORNE], duration / 60 if duration is not None else None, r['airborne_horizontal_distance_m'], r['ideal_steady_endurance_min'],
                    r['ideal_steady_range_km'], r['nominal_energy_remaining_percent_after_airborne'])
            lines.append('| ' + scenario[-1] + ' | ' + ' | '.join(format_number(v) for v in vals) + ' |')
    vi_fraction = [100 * r['native_VI_coverage_s'] / r['airborne_duration_s']
                   for r in full if r['native_VI_coverage_s'] is not None]
    manager_fraction = [100 * r['airborne_manager_fallback_s'] / r['airborne_duration_s']
                        for r in full if r['airborne_manager_fallback_s'] is not None]
    if vi_fraction and manager_fraction:
        lines.extend([
            '', '## تغطية الجهد والتيار', '',
            f'في المهام ذات طاقة كاملة بلغ وسيط نسبة زمن عينات الجهد والتيار الصالحة للفحص '
            f'**{median(vi_fraction):.1f}%** من المدة الجوية، وتراوحت النسبة بين '
            f'**{min(vi_fraction):.1f}%** و**{max(vi_fraction):.1f}%**. يعتمد هذا الفحص على الفترات المتصلة '
            'بين الرسائل المنشورة التي استوفت شروط الجهد والتيار وسرعة المحركات.',
            '',
            f'بلغ وسيط حصة زمن تقدير القدرة الاحتياطي لمدير الرحلة **{median(manager_fraction):.1f}%**، '
            f'وتراوحت بين **{min(manager_fraction):.1f}%** و**{max(manager_fraction):.1f}%**. '
            f'اعتمدت **{sum(f >= 10 for f in manager_fraction)} من {len(manager_fraction)}** رحلة على هذا التقدير '
            'في عُشر المدة الجوية أو أكثر. يحتفظ تقدير الطاقة بقيم قدرة الطائرة الكلية المؤهلة '
            'عند غياب زوج جهد وتيار مدعوم. يتطلب تحديد تيار الخلايا وشحنها عندئذ قياسات ميدانية. '
            'تعكس الفروق عن الحملة السابقة تغيّر المسار الفعلي والمدة والتقديرات معًا.',
        ])
    observed = [r['charge_fit_exhaustion_observed'] for r in full if r['charge_fit_exhaustion_observed'] is not None]
    if observed:
        lines += ['', f'رصد سجل الحزمة بلوغ حد مجال الشحنة المسحوبة للملاءمة في **{sum(observed)} من {len(observed)}** '
                  'رحلة مكتملة التغطية. يعبّر هذا الحد عن أقصى مجال المعايرة المتاح للجهد تحت الحمل.']
    lines.extend([
        '', '## قراءة الملفات', '',
        f'- [النتائج لكل إعداد فريد]({MAIN_CSV}) فيها عمود `estimated_airborne_energy_Wh` لهدف الرحلة الكاملة الذي كتبه المسجّل. '
        'تبقى الخانة فارغة عند غياب هدف موثّق للمهمة المكتملة. يحدد `uncovered_s` زمن النقص، ويعطي '
        '`airborne_known_subtotal_Wh` مجموع الفترات المغطاة فقط.',
        f'- [التجارب بترتيب المشرفة]({COMPARISONS}) تضم 6 مقارنات للحمولة، و30 للحمولة والسرعة، و45 للحمولة والرياح، '
        'والسيناريوهات الثلاثة. يتكرر معرّف الرحلة عند إعادة استخدامها في أكثر من موضع. في الرياح الصفرية تشير '
        'خانات الاتجاهات الثلاثة إلى رحلة واحدة.',
        '- تقارن أعمدة `delta_measurement_energy` طاقة مقطع الذهاب ذي الـ500 م أو التحويم ذي الـ60 ث بالمقطع المناظر في نوفمبر 2026. '
        'في التحويم يظل المدى المبني على مسافة فارغًا. يُقاس زمن المقارنة المنشور على العينات المتاحة لنفس المقطع.',
        '- استعملت الرحلات التمهيدية الثلاث سرعة 6 م/ث وارتفاع 25 م وذهابًا بطول 100 م. '
        'تخص المقارنة مع رحلات CMU المقاسة ذلك البروتوكول. '
        '[أوامر الرحلات التمهيدية وقياساتها المرجعية](../supervisor_readiness/verified_pilot_results.csv)',
        '- يحتفظ `supported_phase_mission_energy_Wh` بتكامل القدرات التي يدعمها مدير الرحلة وفق تعريف الملف السابق. '
        'النسبة `nominal_energy_remaining_percent` مشتقة من هذا العمود. النسبة '
        '`nominal_energy_remaining_percent_after_airborne` تستخدم هدف الرحلة الكاملة. يحفظ '
        '`nominal_airborne_energy_margin_Wh` الفرق الموقّع بين 99.9 واط·ساعة وهدف الرحلة، وتوضح القيمة السالبة مقدار التجاوز.',
        '- تسجل `native_VI_coverage_s` زمن فحص عينات الجهد والتيار المتاحة، ويُظهر '
        '`airborne_manager_fallback_s` الجزء المغطى بتقدير مدير الرحلة. جميع قيم الجهد والتيار والطاقة في هذه الحملة محاكاة.',
        '- تبقى آثار الرياح غير الصفرية والحمولات فوق 0.5 كغ وسرعات 15 و20 م/ث ضمن التجارب الاستكشافية. '
        'يطبّق التحويم مهمة مختلفة عن الطيران لمسافة ثابتة، لذلك يظهر عند سرعة صفر كحالة مستقلة.',
        '- تمثل كل حالة فريدة مسارًا محاكى واحدًا مختارًا وفق البروتوكول. يعتمد تقدير تفاوت الطاقة بين '
        'التكرارات أو خطأ التنبؤ في الطيران الحقيقي على بيانات إضافية مستقلة. عند تقسيم هذه النتائج للتدريب '
        'والتقييم، تُجمع جميع مواضع المقارنة ذات معرّف الرحلة الواحد في المجموعة نفسها.',
        '', '## حالة صفوف المسجّل', '',
    ])
    lines.extend(f'- `{k}` — {v}' for k, v in sorted(counts.items()))
    lines += ['', '[الرسم المقارن لطاقة الرحلات](study_energy.png) · [شروط التشغيل](protocol.json) · '
              '[مراجعة مجال المعايرة](../supervisor_readiness/report_ar.md)', '']
    return '\n'.join(lines)


def publish_locked(out, final):
    protocol = read_json(out / 'protocol.json')
    old = read_json(ORIGINAL / 'protocol.json')
    if hashlib.sha256((ORIGINAL / 'protocol.json').read_bytes()).hexdigest() != protocol['previous_protocol_sha256']:
        raise ValueError('Original supervisor protocol hash differs')
    archived = (ORIGINAL / 'results.json').read_bytes()
    previous_export = read_json(ORIGINAL / 'M100_69_experiments_summary.csv.export.json')
    if hashlib.sha256(archived).hexdigest() != previous_export['results_sha256']:
        raise ValueError('Archived comparison results differ from their original export register')
    hist = json.loads(archived)
    cases = {c['case_id']: c for c in protocol['cases']}
    groups = protocol['groups']
    if list(cases) != [c['case_id'] for c in old['cases']] or groups != old['groups'] or len(cases) != 69:
        raise ValueError('Supervisor study plan differs')
    chosen, attempts = select_attempts(out, cases)
    rows = {}
    for c in old['cases']:
        cid = c['case_id']
        if any(value != cases[cid].get(k) for k, value in c.items() if k != 'energy_evidence_status'):
            raise ValueError(f'Unmatched original conditions for {cid}')
        if cid in chosen:
            rows[cid] = make_row(c, [g for g, ids in groups.items() if cid in ids], chosen[cid],
                                 len(attempts[cid]), hist[cid]['metrics'])
    if not rows:
        return 0, 0
    full = sum(r[AIRBORNE] is not None for r in rows.values())
    if not final:
        write_table(out / 'progress.csv', list(rows.values()))
        return len(rows), full
    if set(rows) != set(cases):
        raise ValueError(f'Final release requires a terminal attempt in each of the 69 cases, found {len(rows)}')
    logical = logical_table(groups, rows)
    observed_labels = [r['direct_row_sha256'] for r in rows.values() if r['direct_row_sha256']]
    if len(logical) != 84 or len(set(observed_labels)) != len(observed_labels):
        raise ValueError('Repeated source recorder row or missing comparison slot')
    write_table(out / MAIN_CSV, list(rows.values()))
    write_table(out / COMPARISONS, logical)
    draw_energy(out, rows, groups)
    write_bytes(out / 'report_ar.md', report(rows, attempts).encode('utf-8'))
    return len(rows), full


def publish(out, final):
    directory_fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        return publish_locked(out, final)
    finally:
        os.close(directory_fd)


def follow(out):
    last = None
    while True:
        signature = sorted((p, p.stat().st_mtime_ns) for p in (out / 'audits').glob('*/attempt_*.json'))
        if signature != last:
            n, full = publish(out, final=False)
            print('Audited cases and full planned-flight labels:', n, full, flush=True)
            if full == 69:
                publish(out, final=True)
                return
            last = signature
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--final', action='store_true', help='Publish the complete case matrix and report')
    mode.add_argument('--follow', '--watch', action='store_true',
                      help='Refresh progress after each new audit and release when all 69 planned flights have labels')
    args = parser.parse_args()
    if args.follow:
        follow(args.out)
    else:
        print('Audited cases and full planned-flight labels:', *publish(args.out, args.final))
