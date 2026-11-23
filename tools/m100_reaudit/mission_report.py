"""Render the completed 150 m native flight release and its Arabic research summary."""
from collections import Counter

import numpy as np

from .direct_airborne_evidence import digest
from .mission_campaign import OUT, read
from .mission_prediction import dataframe, number

ROUTES={'direct':'مباشر','north_dogleg':'عبر نقطة شمالية'}


def close_alternatives(protocol):
    coverage=read(OUT/'confirmation_coverage.json')['by_payload']
    order={
        'WINNER_USED_LESS_ON_ALL_CONFIRMATION_SEEDS':'استهلك الاختيار المتوقع طاقة أقل في المرتين',
        'COMPARATOR_USED_LESS_ON_ALL_CONFIRMATION_SEEDS':'استهلك البديل المقارب طاقة أقل في المرتين',
        'ORDER_VARIED_OR_INCLUDED_EQUAL_ENERGY':'اختلف الترتيب أو تساوت الطاقة بين التكرارين',
        'UNRESOLVED_MISSING_OR_INFEASIBLE_PAIRS':'التغطية المتاحة غير كافية لحسم ترتيب البديلين',
    }
    rows=[]
    for mass,series in sorted(coverage.items(),key=lambda item:float(item[0])):
        for group in series['comparisons']:
            if 'closest_runner_up' not in group['roles']:
                continue
            other=protocol['selection'][mass]['closest_runner_up']
            gap=group['mean_observed_difference_on_feasible_pairs_Wh']
            delta=f'{gap:+.3f} Wh' if gap is not None else 'غير متاح'
            rows.append(f"| {float(mass):g} كغ | {ROUTES[other['route']]}، {other['cruise_mps']:g} م/ث، {other['height_agl_m']:g} م | "
                        f"{delta} | {order[group['observed_energy_order']]} |")
    if not rows:
        return []
    return ['', 'مقارنة أقرب بديل إلى الاختيار المتوقع. الفرق الموجب يعني أن البديل استهلك طاقة أكثر.','',
            '| الحمولة | البديل المقارب | متوسط فرق الطاقة الفعلي | ترتيب التكرارين المستقلين |','|---|---|---:|---|',*rows]


def charts(conditions,sensitivity,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'font.size':10,'axes.labelcolor':'#182d3a','text.color':'#182d3a','axes.edgecolor':'#82919a'})
    fig,axes=plt.subplots(1,3,figsize=(10.5,4.4),sharey=True,layout='constrained')
    colors=('#216d80','#a45622','#524c85')
    for i,(ax,mass) in enumerate(zip(axes,(0,.25,.5))):
        for color,height in zip(colors,(25,50,100)):
            part=sorted((r for r in conditions if number(r,'payload_kg')==mass and number(r,'altitude_m')==height),
                        key=lambda r:number(r,'cruise_speed_mps'))
            if not part:
                continue
            means=np.array([number(r,'mean_airborne_Wh') for r in part])
            lo=means-np.array([number(r,'min_Wh') for r in part])
            hi=np.array([number(r,'max_Wh') for r in part])-means
            ax.errorbar([number(r,'cruise_speed_mps') for r in part],means,yerr=[lo,hi],
                        label=f'{height} m AGL',marker='o',markersize=4.2,lw=1.8,capsize=4,color=color)
        ax.set_xticks([6,8,10])
        ax.set_title(f'{mass:g} kg payload',loc='left',fontweight='bold')
        ax.set_xlabel('Outbound speed command (m/s)')
        ax.grid(axis='y',color='#e1e7e9')
        if i==0:
            ax.set_ylabel('Mean takeoff-to-landing energy (Wh)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
    fig.suptitle('Direct flight to 150 m and return in calm air',fontsize=15,fontweight='bold')
    fig.savefig(out/'payload_speed_height.png',dpi=180)
    plt.close(fig)

    if not sensitivity:
        return
    fig,axes=plt.subplots(1,3,figsize=(11,4.5),sharex=True,layout='constrained',gridspec_kw={'width_ratios':[1.5,1,1]})
    categories=(('wind',[(3,180),(3,0),(3,90),(6,180),(6,0),(6,90)]),
                ('ambient',[10,25]),('charge',[90,80]))
    for ax,(kind,settings) in zip(axes,categories):
        labels=[]
        for j,setting in enumerate(settings):
            if kind=='wind':
                wind,deg=setting
                selected=[r for r in sensitivity if number(r,'wind_mps')==wind and number(r,'wind_toward_deg')==deg]
                labels.append(f'{wind} m/s '+{0:'east',90:'north',180:'west'}[deg])
            elif kind=='ambient':
                selected=[r for r in sensitivity if number(r,'ambient_C')==setting]
                labels.append(f'{setting} °C')
            else:
                selected=[r for r in sensitivity if number(r,'nominal_initial_soc_percent')==setting]
                labels.append(f'{setting}%')
            for i,(route,color,marker) in enumerate((('direct','#216d80','o'),('north_dogleg','#a45622','s'))):
                obs=[r for r in selected if r['route']==route]
                if obs:
                    ax.plot(number(obs[0],'difference_Wh'),j+(i-.5)*.18,marker,color=color,
                            label=('Direct' if i==0 else 'Via waypoint') if j==0 else None)
        ax.axvline(0,color='#82919a',lw=1)
        ax.set_yticks(range(len(labels)),labels)
        ax.invert_yaxis()
        ax.grid(axis='x',color='#e1e7e9')
        ax.set_xlabel('Change in energy (Wh)')
        ax.set_title({'wind':'Wind toward','ambient':'Air temperature','charge':'Initial nominal SOC'}[kind],loc='left')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=2,frameon=False)
    fig.suptitle('Matched single-seed changes from calm air, 15 °C, 100% charge',fontsize=14,fontweight='bold')
    fig.savefig(out/'environment_sensitivity.png',dpi=180)
    plt.close(fig)


def write():
    verify=read(OUT/'verification.json')
    reference=read(OUT/'preflight_reference/evaluation.json')['overall_and_by_payload']['all']
    explore=read(OUT/'preflight_exploratory/evaluation.json')['overall_and_by_payload']['all']
    online=read(OUT/'online_reference/evaluation.json')
    field=read(OUT/'field_holdout.json')
    flights=dataframe(OUT/'mission_flights.csv')
    conds=dataframe(OUT/'factorial_conditions.csv')
    env=dataframe(OUT/'environment_and_soc.csv') if (OUT/'environment_and_soc.csv').exists() else []
    paired=dataframe(OUT/'confirmed_savings.csv')
    protocol=read(OUT/'confirmation_protocol.json')
    attempts=dataframe(OUT/'attempt_registry.csv')
    matches=dataframe(OUT/'field_command_context.csv')
    charts(conds,env,OUT)
    counts=verify['counts']
    intervals=verify['published_interval_checks']
    reasons=Counter(a['reason'] for a in attempts if a['status']=='REJECTED')
    rate=np.array([number(c,'recorded_manager_energy_Wh')/number(c,'estimated_airborne_energy_Wh') for c in flights])*100
    exhausted=sum(r['low_charge_domain_exhausted']=='True' for r in flights)
    supported=[r for r in flights if number(r,'native_vi_publications')>0]
    lines=['# نتائج مهمة Matrice 100 إلى وجهة على بُعد 150 م ثم العودة','',
           f"اجتازت **{verify['accepted']} رحلة محاكاة** التحقق من طاقة الإقلاع إلى الهبوط والوصول إلى الوجهة والعودة. "
           f"سجّلت الحملة {verify['attempts']} محاولة لعدد {verify['registered']} حالة وبذرة مسجّلة. "
           f"من الرحلات المقبولة، {counts['factorial']} لتجربة الحمولة والسرعة والارتفاع، و{counts['route']} لبديل المسار، "
           f"و{counts['sensitivity']} للرياح وحرارة الهواء وشحن البداية، و{counts['confirmation']} للتحقق المنفصل من الاختيارات. "
           f"تضم الجداول {intervals['interval_rows']:,} فترة زمنية، توفرت مدخلات الملاحة السابقة للتنبؤ في {intervals['interval_rows_with_causal_features']:,} منها. "
           "تحتفظ الجداول بطاقة الرحلة التي كتبها المسجّل عند الهبوط. [ملف الرحلات](mission_flights.csv) · [فترات القدرة والطاقة](mission_intervals.csv.gz) · [سجل المحاولات](attempt_registry.csv)",'',
           '## اختيار السرعة والارتفاع والمسار','',
           'استُخدمت وجهة تبعد 150 م شرقًا عن نقطة الإقلاع والعودة إلى النقطة نفسها بالحمولة نفسها. استُخدم احتياطي اسمي قابل للتغيير قدره 20%، ولم يحدد للمهمة موعد وصول. جُرّبت سرعات ذهاب 6 و8 و10 م/ث وارتفاعات 25 و50 و100 م. يتبع المسار البديل نقطة وسطية، فيبلغ طوله المخطط مع العودة 317.71 م مقابل 300 م للمباشر. بقيت سرعة العودة محدودة بأربعة م/ث في كلا المسارين.','',
           '| الحمولة | اختيار المتنبئ | الطاقة المتوقعة | الطاقة المرصودة للتحقق | زمن المهمة المرصود | شحن الهبوط الاسمي | الوفر عن المباشر عند 8 م/ث و50 م |',
           '|---|---|---:|---:|---:|---:|---:|']
    for mass,selection in sorted(protocol['selection'].items(),key=lambda item:float(item[0])):
        win=selection['winner']
        if not win:
            lines.append(f'| {mass} كغ | لم يتوفر بديل يستوفي القيود | — | — | — | — | — |')
            continue
        trials=[r for r in flights if r['group']=='confirmation' and number(r,'payload_kg')==float(mass)
                and r['mission_route']==win['route'] and number(r,'altitude_m')==win['height_agl_m']
                and number(r,'cruise_speed_mps')==win['cruise_mps']]
        def mean_str(field,unit):
            return f"{np.mean([number(c,field) for c in trials]):.2f} {unit}" if trials else 'غير متاح'
        pairs=[r for r in paired if number(r,'payload_kg')==float(mass) and 'reference' in r['comparison_roles']
               and r['pair_status']=='FEASIBLE_PAIR']
        saving=(f"{np.mean([number(r,'actual_saved_Wh') for r in pairs]):.2f} Wh "
                f"({np.mean([number(r,'actual_saved_pct') for r in pairs]):.1f}%)" if pairs else 'لا توجد مقارنة مستقلة مع المرجع')
        lines.append(f"| {float(mass):g} كغ | {ROUTES[win['route']]}، {win['cruise_mps']:g} م/ث، {win['height_agl_m']:g} م | "
                     f"{win['energy_Wh']:.2f} Wh | {mean_str('estimated_airborne_energy_Wh','Wh')} | "
                     f"{mean_str('airborne_duration_s','ث')} | {mean_str('nominal_remaining_soc_percent','%')} | {saving} |")
    lines+=close_alternatives(protocol)
    lines+=['',f"جُمّدت التوقعات قبل رحلات البذرتين 2711 و2712. توفّر {verify['eligible_paired_confirmation_rows']} زوجًا يستوفي الاحتياطي من {verify['expected_paired_confirmation_rows']} زوجًا مبرمجًا. "
            f"كان المئين التسعون للخطأ المطلق في طاقة الظروف المرجعية {reference['residual_ridge']['energy_Wh']['p90_abs']:.3f} Wh. "
            "يحمل كل بديل قريب من الأول إشارة تقارب. تحقق الوفورات على محاكاة المهمة المحددة، وتبقى أجزاء فضاء السرعة والارتفاع والمسارات خارج الشبكة غير مختبرة. "
            "[التوقع المجمد](confirmation_protocol.json) · [القياسات المزدوجة](confirmed_savings.csv) · [حالة كل مقارنة](confirmation_coverage.json)",'',
            '![متوسط طاقة الرحلة في الحالات المضبوطة ومجال تكراراتها](payload_speed_height.png)','',
            '## التحقق من البطارية والبيانات','',
            f"طابق وسم كل رحلة مقبولة سجلَّ Gazebo النهائي وحدّي كاشف الهبوط المستقل في PX4. بلغ أقصى خطأ عددي لفحص معادلة الجهد {verify['maximum_pack_equation_residual_V']:.2g} V، "
            f"ولحاصل الجهد في التيار {verify['maximum_supported_power_residual_W']:.2g} W، على العينات التي يدعمها نموذج الحزمة. "
            f"وصلت الشحنة المسحوبة إلى حد الملاءمة في {exhausted} من {len(flights)} رحلة. "
            f"تراوحت مساهمة تقدير القدرة البديل من {rate.min():.1f}% إلى {rate.max():.1f}% من طاقة الرحلة. "
            "تظهر أعلام التغطية في كل سجل. [التحقق والعدادات](verification.json) · [قاموس الأعمدة](column_dictionary.csv)",'',
            f"على {field['flights']} رحلة ميدانية محجوزة من CMU، انخفض متوسط خطأ الجهد عند إدخال التيار المقاس من {field['voltage_mae_static_V']:.3f} إلى {field['voltage_mae_2rc_V']:.3f} V. "
            f"توجد حالات متساوية في الحمولة والسرعة والارتفاع لـ{sum(number(r,'matched_command_field_flights')>0 for r in matches)} من {len(matches)} تركيبة في هذه الحملة. "
            "يعرض جدولها قدرات ومدد الطرفين. يظل طول المسار وتقسيم مراحله مختلفين، فلا تستعمل فروق الطاقة الكلية كنسبة خطأ ميداني لمهمة 150 م. "
            "[مقارنة الأوامر المتطابقة](field_command_context.csv) · [اختبار CMU](field_holdout.json)",'',
            '## جاهزية تدريب نموذج الطاقة','',
            f"بلغ متوسط الخطأ المطلق على عائلات أوامر محجوبة في الرحلات المرجعية الجديدة **{reference['residual_ridge']['energy_Wh']['mae']:.3f} Wh**، "
            f"مقابل {reference['phase_schedule']['energy_Wh']['mae']:.3f} Wh للمرجع الزمني ذي قدرة المراحل المنشورة و{reference['train_mean']['energy_Wh']['mae']:.3f} Wh لمرجع متوسط التدريب. "
            f"عند ضم الحالات الاستكشافية بلغ الخطأ {explore['residual_ridge']['energy_Wh']['mae']:.3f} Wh. "
            f"وفي التنبؤ أثناء الطيران بلغ خطأ القدرة للفترة التالية {online['heldout_mae_W_equal_flight']:.2f} W على {online['flights']} رحلة مرجعية. "
            "تصلح هذه النتائج لتدريب نموذج يتنبأ بمخرجات المحاكاة ضمن الشروط المسجلة. تبقى معايرة سعة TB47D الفعلية ومنحنى الجهد المفتوح، وحرارة الخلايا، وعزم المحركات وخسائر ESC 620D والتحقق الميداني لنفس المهمة مفتوحة. "
            "[تقسيم البيانات وطريقة الاستعمال](../../../docs/m100_reaudit/mission_training_ar.md) · [أخطاء الطاقة](preflight_reference/evaluation.json) · [أخطاء القدرة](online_reference/evaluation.json)",'',
            '## حساسية الرياح وحرارة الهواء وشحن البداية','',
            "تُغيّر المقارنة متغيرًا واحدًا عند 0.25 كغ و8 م/ث و50 م. استُخدمت بذرة واحدة لكل زوج. قيمة موجبة تعني استهلاكًا أكبر من مرجع المسار نفسه. أثّرت حرارة الجو في كثافة الهواء، بينما بقيت معاملات البطارية الحرارية غير مقاسة. "
            "[جدول التأثيرات](environment_and_soc.csv) · [الفروق المقترنة للحمولة والسرعة والارتفاع](factorial_paired_effects.csv)",'']
    if env:
        lines+=['![فروق الطاقة عن المرجع عند تغيير الرياح والحرارة وشحن البداية](environment_sensitivity.png)','']
        low=next((r for r in env if r['route']=='direct' and number(r,'nominal_initial_soc_percent')==80),None)
        if low and low['minimum_supported_pack_voltage_V'] and low['reference_minimum_supported_voltage_V']:
            lines+=[f"في المسار المباشر عند شحن ابتدائي اسمي 80%، بلغ أدنى جهد مدعوم {number(low,'minimum_supported_pack_voltage_V'):.2f} V "
                    f"مقابل {number(low,'reference_minimum_supported_voltage_V'):.2f} V عند 100%. "
                    f"بلغت طاقة المهمة {number(low,'airborne_Wh'):.2f} Wh مقابل {number(low,'reference_Wh'):.2f} Wh للمرجع، "
                    f"وغطّى تقدير قدرة الطائرة البديل {100*number(low,'fallback_share_of_energy'):.1f}% من طاقة رحلة 80%. "
                    "هذه مقارنة لبذرة واحدة ضمن نموذج البطارية المنقول من CMU.",'']
    if supported:
        minimum=min(number(r,'min_supported_load_voltage_V') for r in supported)
        peak=max(number(r,'max_supported_current_A') for r in supported)
        thrust=min(number(r,'min_supported_available_thrust_fraction') for r in supported)
        clipped=sum(number(r,'motor_command_limited_publications')>0 for r in flights)
        lines+=['## أثر الجهد في التيار والدفع داخل المحاكي','',
                f"بلغ أدنى جهد مدعوم تحت الحمل {minimum:.2f} V، وأعلى تيار مدعوم {peak:.2f} A عبر الرحلات. "
                f"وصل سقف الدفع المتاح المحسوب من مربع نسبة سرعة المحرك إلى {100*thrust:.1f}% من السقف الاسمي. "
                f"في {clipped} من {len(flights)} رحلة احتوت الرسائل المنشورة على أمر محرك مقصوص عند السقف. "
                "يتعلق هذا السقف بأقصى دفع يسمح به النموذج في تلك اللحظة. يحلّ نموذج البطارية الجهد والتيار معًا حتى يساوي حاصل ضربهما القدرة المطلوبة، "
                "ويبقى فقد التوصيل في ESC 620D وعزم كل محرك خارج القياس المتاح. [مقاييس الحزمة والدفع لكل رحلة](mission_flights.csv)",'']
    if (OUT/'mission_phase_checks.csv').is_file():
        phases=dataframe(OUT/'mission_phase_checks.csv')
        duration=sum(number(r,'duration_s') for r in phases)
        vi_fraction=sum(number(r,'vi_eligible_s') for r in phases)/duration
        unresolved=sum(number(r,'phase_unattributed_s') for r in phases)/duration
        residual=max(abs(number(r,'energy_minus_flight_Wh')) for r in phases)
        vi_diff=[abs(number(r,'published_minus_native_Wh')) for r in phases if r['published_minus_native_Wh']!='']
        lines+=['## الطاقة والمسافة في مراحل الرحلة','',
                f"جُمعت طاقات المراحل في {len(phases)} رحلة. بلغ أكبر فرق عن طاقة الإقلاع إلى الهبوط {residual:.2g} Wh. "
                f"تعذّر تحديد اسم المرحلة في {unresolved*100:.2f}% من الزمن الجوي، وتظل طاقته مدرجة في المجموع. "
                f"أتاحت سجلات الجهد والتيار تكاملًا مستقلًا لطاقة {vi_fraction*100:.1f}% من الزمن الجوي على فترات متجانسة المصدر. "
                + (f"بلغ أكبر فرق مطلق بين تكامل القراءات المنشورة وعداد خطوة الفيزياء للفترات نفسها {max(vi_diff):.4f} Wh. " if vi_diff else '')
                + "[طاقة المقاطع وأزمنتها ومسافاتها](mission_phases.csv) · [التحقق لكل رحلة](mission_phase_checks.csv)",'']
    if (OUT/'error_breakdown/held_out_errors_by_condition.csv').is_file():
        lines+=['يعرض جدول أخطاء التنبؤ متوسط الخطأ المطلق والجذر التربيعي لمتوسط مربع الخطأ والانحياز. يذكر عدد الرحلات وعائلات الأوامر الداخلة في كل شريحة من الحمولة والسرعة والارتفاع والرياح والمسار، وكذلك مرحلة الرحلة في تنبؤ القدرة. [أخطاء التنبؤ بحسب ظروف التشغيل](error_breakdown/held_out_errors_by_condition.csv)','']
    if (OUT/'postflight_verification.json').is_file():
        post=read(OUT/'postflight_verification.json')
        trials=post['flights']
        lines+=['## تحقق الإصدار المثبّت بعد الهبوط','',
                f'أُعيدت رحلة مباشرة بحمولة صفر وسرعة ذهاب 6 م/ث وارتفاع 25 م بالبذرة نفسها عند {len(trials)} خطوات زمنية. '
                'يسجّل الإصدار المثبّت تعافي جهد الفرعين عندما تؤكد إشارات المحركات والبطارية وPX4 توقف الحمل بعد فصل المحركات. '
                'خُصصت هذه الرحلات للتحقق الهندسي.','',
                '| خطوة الحساب | طاقة الإقلاع إلى الهبوط | زمن الطيران | الراحة المؤكدة في نافذة التحقق | تغير الشحنة أثناء الراحة |',
                '|---|---:|---:|---:|---:|']
        for obs in trials:
            lines.append(f"| {1000*obs['step_s']:g} ملي ثانية | {obs['energy_Wh']:.3f} Wh | {obs['airborne_s']:.2f} ث | "
                         f"{obs['confirmed_rest_s']:.2f} ث | {obs['max_rest_charge_variation_Ah']:.2g} Ah |")
        for change in post['consecutive_step_changes']:
            lines+=['',f"عند الانتقال من {1000*change['from_step_s']:g} إلى {1000*change['to_step_s']:g} ملي ثانية "
                    f"تغيرت طاقة الرحلة {change['delta_energy_Wh']:+.3f} Wh ({change['delta_energy_percent']:+.2f}%)، "
                    f"وزمنها {change['delta_time_s']:+.2f} ث، ومسافتها الأفقية الفعلية {change['delta_horizontal_distance_m']:+.2f} م، "
                    f"ومتوسط القدرة {change['delta_mean_power_W']:+.2f} W."]
        lines+=['',"تشمل المقارنة تغير التتبع والتوقيت في منظومة التحكم. وهي تخص هذه الحالة والبذرة. "
                f"كان أقصى فرق عن الاضمحلال الأسي المتوقع للفرعين {max(o['max_rest_branch_error_V'] for o in trials):.2g} V. "
                "[سجل التحقق من البطارية وحجم الخطوة](postflight_verification.json) · [تجارب إعداد الحملة](engineering_setup.json)",'']
    if reasons:
        lines+=['رُفضت محاولات قبل إعداد جدول التدريب للأسباب الآتية. يحفظ السجل هوية كل رحلة أصلية أو بديلة، وتعاد الحالات بإعداداتها وبذورها الأصلية. [سجل المحاولات](attempt_registry.csv) · [إعداد المحاولات البديلة](retry_policy.json)','',
                *[f'- `{reason}` في {count} محاولة.' for reason,count in reasons.items()],'']
    (OUT/'report_ar.md').write_text('\n'.join(lines).rstrip()+'\n')
    manifest=OUT/'SHA256SUMS'
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p!=manifest and not p.name.endswith(('.part','.tmp'))]
    manifest.write_text(''.join(f'{digest(p)}  {p.relative_to(OUT)}\n' for p in files))


if __name__=='__main__':
    write()
