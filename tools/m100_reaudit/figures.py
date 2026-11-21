"""Draw matched Gazebo cases and flight-held-out CMU comparisons from delivered tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results/m100_reaudit"
MODEL = "M100_RESEARCH_GAZEBO_DART_LEE_V1"
P, BASE = "measurement_mean_power_W", "phase_baseline_only_mean_W"
CENTER = {
    "payload_kg": 0.25, "cruise_speed_mps": 6, "wind_speed_mps": 0,
    "wind_direction_toward_deg_ENU": 0, "altitude_m": 50, "temperature_c": 15,
    "measurement_distance_m": 125, "hover_duration_s": 20,
}
INK, GRAY, BORDER, GRID = "#162f3c", "#536674", "#b3c0c5", "#e1e7e9"
TEAL, ORANGE, PURPLE = "#007b76", "#b34f1e", "#71579b"
BUDGET_GROUPS = {
    "Climb": ("CLIMB",),
    "Transitions": ("STABILIZING", "ACCELERATING", "RETURN_BRAKING"),
    "Outbound": ("CRUISE",), "Return": ("RETURNING",), "Landing": ("LANDING",),
}


def plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.3,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.labelcolor": INK, "axes.edgecolor": BORDER, "axes.linewidth": 0.7,
        "xtick.color": GRAY, "ytick.color": GRAY, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "legend.fontsize": 8.0,
        "savefig.dpi": 300, "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def panel(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", pad=9, color=INK)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def matched(cases, **replace):
    """Exactly one accepted primary-step flight for every complete parameter tuple."""
    selected = cases[cases.step_s.eq(0.008) & cases.cohort.eq("expanded")
                     & cases.status.eq("passed") & cases.model_version.eq(MODEL)]
    for field, value in (CENTER | replace).items():
        selected = selected.loc[selected[field].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one completed matched flight for {CENTER | replace}, found {len(selected)}")
    return selected.iloc[0]


def sweeps(cases, field, levels, **shared):
    return pd.DataFrame([matched(cases, **{field: level}, **shared) for level in levels])


def branch_plot(ax, data, x):
    ax.plot(data[x], data[BASE], color=GRAY, linestyle="--", linewidth=1.4,
            marker="o", markerfacecolor="white", markersize=6.5, label="Published phase only")
    ax.plot(data[x], data[P], color=TEAL, linewidth=1.7, marker=".", markersize=6,
            label="Applied model")
    ax.legend(loc="best", handlelength=2.1)
    ax.set_xticks(data[x])


def sim_figure(cases, phases, destination):
    speed = sweeps(cases, "cruise_speed_mps", (4, 6, 8, 10, 12, 17, 20))
    payload = sweeps(cases, "payload_kg", (0, .25, .5, .75, 1, 1.25))
    temperature = sweeps(cases, "temperature_c", (0, 15, 25, 35))
    fig = plt.figure(figsize=(12.5, 10.35))
    grid = fig.add_gridspec(3, 2, left=.082, right=.975, top=.884, bottom=.074,
                            height_ratios=(1, 1, .83), wspace=.24, hspace=.66)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    fig.text(.082, .972, "Electrical power and phase energy in Gazebo", size=21,
             weight="bold", ha="left", va="top")
    fig.text(.082, .937, "Instrumented Matrice 100  ·  50 m AGL  ·  125 m eastbound segment  ·  8 ms physics step",
             color=GRAY, size=11, va="top")
    for ax, data, field, title, xlabel in zip(
        (axes[0], axes[1], axes[3]), (speed, payload, temperature),
        ("cruise_speed_mps", "payload_kg", "temperature_c"),
        ("Outbound speed", "External payload", "Ambient temperature"),
        ("Commanded ground speed (m/s)", "Payload (kg)", "Air temperature at datum (°C)"), strict=True,
    ):
        panel(ax, title, xlabel, "Mean power (W)")
        branch_plot(ax, data, field)

    wind_ax = axes[2]
    panel(wind_ax, "Steady wind of 6 m/s", "Commanded ground speed (m/s)", "Mean power (W)")
    wind_rows = {}
    for angle, label, color, marker in ((0, "Tailwind", TEAL, "o"), (90, "Crosswind", PURPLE, "s"),
                                        (180, "Headwind", ORANGE, "^")):
        series = sweeps(cases, "cruise_speed_mps", (4, 6, 8, 10, 12),
                        wind_speed_mps=6, wind_direction_toward_deg_ENU=angle)
        wind_rows[label] = series.loc[series.cruise_speed_mps.eq(6)].iloc[0]
        wind_ax.plot(series.cruise_speed_mps, series[P], color=color, marker=marker,
                     linewidth=1.55, markersize=4.0, label=label)
    wind_ax.set_xticks([4, 6, 8, 10, 12])
    wind_ax.legend(ncol=3, loc="upper left", handlelength=1.3, columnspacing=.9)
    temp_ax = axes[3]
    actual = float(temperature.loc[temperature.temperature_c.eq(15), P].iloc[0])
    temp_ax.annotate("Stable-speed adjustment", xy=(15, actual), xytext=(.57, .12),
                     textcoords="axes fraction", ha="left", va="center", size=8.3, color=TEAL,
                     arrowprops=dict(arrowstyle="-", color=TEAL, lw=.8, shrinkB=3))

    chart = fig.add_subplot(grid[2, :])
    selected = {"Calm": matched(cases), **wind_rows}
    summary = []
    for name, row in selected.items():
        part = phases[phases.run_id.eq(row.run_id)].set_index("phase")
        if not part.index.is_unique:
            raise ValueError(f"Repeated phase name in {row.run_id}")
        energy = part.loc[part.energy_Wh.notna(), "energy_Wh"]
        recognized = {phase for group in BUDGET_GROUPS.values() for phase in group}
        if set(energy.index) != recognized:
            raise ValueError(f"{row.run_id} has missing or additional supported phases: {set(energy.index) ^ recognized}")
        totals = {k:float(energy.loc[list(group)].sum()) for k, group in BUDGET_GROUPS.items()}
        if not np.isclose(sum(totals.values()), row.mission_supported_Wh, rtol=0, atol=1e-7):
            raise ValueError(f"Phase budget does not close in {row.run_id}")
        summary.append({"name":name, "run_id":row.run_id, "mean_W":row[P], "total_Wh":sum(totals.values()), **totals})

    frame = pd.DataFrame(summary).set_index("name")
    y = np.arange(len(frame))
    bottom = np.zeros(len(frame))
    for column, color in zip(BUDGET_GROUPS, ("#5486a6", "#d8d0b8", "#007b76", "#ab8abb", "#da9b68"), strict=True):
        heights = frame[column].to_numpy()
        chart.barh(y, heights, left=bottom, color=color, edgecolor="white", linewidth=.65,
                   height=.70, label=column)
        bottom += heights
    chart.set_title("Energy by flight phase at 6 m/s ground speed", loc="left", pad=9)
    chart.set(xlabel="Electrical energy over supported phases (Wh)", yticks=y, xlim=(0, max(bottom)*1.105))
    chart.set_yticklabels(frame.index)
    chart.invert_yaxis()
    chart.set_axisbelow(True)
    chart.grid(axis="x", color=GRID, linewidth=.65)
    for yy, total in zip(y, bottom, strict=True):
        chart.text(total+.12, yy, f"{total:.2f}", va="center", fontsize=8.2)
    chart.legend(ncol=5, bbox_to_anchor=(1,1.022), loc="lower right", borderaxespad=0,
                 handlelength=1.2, columnspacing=1.1)
    fig.savefig(destination)
    plt.close(fig)
    return speed, temperature, frame


def phase_figure(observations, metrics, destination):
    fields = ("observed_mean_power_W", "estimated_mean_power_W")
    phases = ("takeoff", "cruise", "landing")
    cohort = observations[observations.split.eq("test") & observations.model.eq("published_phase")
                          & observations.scope.isin(phases)].copy()
    if cohort.empty or set(cohort.published_role) != {"holdout"} or cohort.duplicated(["flight_id","scope"]).any():
        raise ValueError("Held-out phase identity is missing or duplicated")
    if not cohort.groupby("flight_id").size().eq(len(phases)).all() or not np.isfinite(cohort[list(fields)]).all().all():
        raise ValueError("Every plotted flight requires three complete measured and predicted phases")
    n = cohort.flight_id.nunique()
    lower = 20 * np.floor((cohort[list(fields)].min().min() - 8)/20)
    upper = 20 * np.ceil((cohort[list(fields)].max().max() + 8)/20)
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.9))
    fig.subplots_adjust(left=.069, right=.984, top=.748, bottom=.19, wspace=.21)
    fig.text(.069, .966, "Measured and predicted M100 phase power", va="top",
             weight="bold", size=20.5)
    fig.text(.069, .875, f"{n} held-out CMU flights  ·  One electrical-power mean per flight and phase",
             color=GRAY, size=11, va="top")
    phase_metrics = {}
    marks = ((0, "#18796f", "o"), (.25, "#497ba7", "^"), (.5, "#b55327", "s"))
    for idx, (ax, name) in enumerate(zip(axes, phases, strict=True)):
        sub = cohort[cohort.scope.eq(name)]
        if set(sub.payload_kg) != {p for p, _, _ in marks}:
            raise ValueError(f"Unexpected holdout payloads for {name}")
        ax.plot([lower, upper], [lower, upper], color="#87959c", ls="--", lw=1, zorder=1)
        for mass, color, marker in marks:
            group = sub[sub.payload_kg.eq(mass)]
            ax.scatter(group[fields[0]],group[fields[1]], s=21, marker=marker, c=color,
                       edgecolors="white", linewidths=.38, alpha=.86, label=f"{mass:g} kg", zorder=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="Measured mean (W)")
        if idx == 0:
            ax.set_ylabel("Predicted mean (W)")
        ax.set_xticks(np.arange(np.ceil(lower/50)*50, upper+.1, 100))
        ax.set_yticks(ax.get_xticks())
        ax.grid(color=GRID, linewidth=.65)
        ax.set_axisbelow(True)
        ax.set_title(name.capitalize(), loc="left", pad=8)
        residual = sub[fields[1]]-sub[fields[0]]
        mae, bias = float(residual.abs().mean()), float(residual.mean())
        match = metrics[(metrics.split=="test")&(metrics.scope==name)&(metrics.model=="published_phase")
                        &(metrics.stratum=="all")&(metrics.value=="all")]
        if len(match) != 1 or not np.allclose([mae,bias], match.iloc[0][["MAE_W","bias_W"]].to_numpy(float), atol=1e-7, rtol=0):
            raise ValueError(f"{name} residuals disagree with grouped_errors.csv")
        phase_metrics[name] = mae
        ax.text(.055,.954,f"MAE {mae:.1f} W\nBias {bias:+.1f} W", transform=ax.transAxes, size=8.3,
                va="top", color=GRAY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="External payload", title_fontsize=9, fontsize=9,
               ncol=3, loc="upper right", bbox_to_anchor=(.984,.995), handletextpad=.25, columnspacing=.9)
    fig.text(.069,.042,"Dashed line shows equal power. Bias is predicted minus measured.", size=9, color=GRAY)
    fig.savefig(destination)
    plt.close(fig)
    return n, phase_metrics


def run(root):
    root = Path(root)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(root / "gazebo_batch/cases.csv")
    phases = pd.read_csv(root / "gazebo_batch/phases.csv")
    observations = pd.read_csv(root / "external/phase_comparisons.csv")
    metrics = pd.read_csv(root / "external/grouped_errors.csv")
    plot_style()
    speed, temperature, budget = sim_figure(cases, phases, out/"matched_simulation.png")
    count, errors = phase_figure(observations, metrics, out/"measured_holdout.png")
    reference = temperature.loc[temperature.temperature_c.eq(15)].iloc[0]
    high = speed.loc[speed.cruise_speed_mps.eq(20)].iloc[0]
    text = f"""# شرح الأشكال

**الشكل 1. القدرة الكهربائية وطاقة المراحل في Gazebo.** تمثل كل نقطة متوسط مقطع الذهاب لرحلة ناجحة عند خطوة فيزياء 8 ms، وارتفاع 50 m AGL ومسافة 125 m باتجاه الشرق. أساس المقارنة حمولة 0.25 kg، وسرعة أرضية 6 m/s، وحرارة 15 °C ورياح صفرية. يغيّر كل رسم العامل المحدد على محوره، ويستعمل رسم الرياح ثلاث زوايا ثابتة عند 6 m/s. يشير اتجاه الرياح إلى الجهة التي يتجه إليها الهواء. ينطبق منحنى المعادلة المنشورة عند الكثافة الفعلية للمقطع. عند 15 °C تنخفض قدرة النموذج التشغيلي للمقطع المرجعي من {reference[BASE]:.2f} إلى {reference[P]:.2f} W بسبب تفعيل تصحيح السرعة المستقرة. عند 8 m/s تستغرق المسافة المطلوبة أقل من 20 s فتُغلق بوابة التصحيح. عند 20 m/s تساوي القدرة {high[P]:.2f} W، وتقع تلك السرعة خارج مستويات دراسة CMU للقدرة. تصف درجات الحرارة المختلفة حساسية الهواء، واستجابة الرياح افتراض محاكاة غير معاير. [الشكل](matched_simulation.png) · [جدول الحالات](../gazebo_batch/cases.csv)

تجمع الأشرطة السفلى طاقة الصعود والذهاب والرجوع والهبوط، وتشمل «الانتقالات» الاستقرار والتسارع والكبح عند بدء العودة. تشمل الطاقة فترات المراحل ذات تقدير القدرة. عند الرياح الخلفية ترتفع قدرة الذهاب من {budget.loc['Calm','mean_W']:.2f} إلى {budget.loc['Tailwind','mean_W']:.2f} W بينما تتغير طاقة المراحل الممثلة من {budget.loc['Calm','total_Wh']:.2f} إلى {budget.loc['Tailwind','total_Wh']:.2f} Wh. تُقرأ هذه المقارنة ضمن حركة متحكّم Gazebo ونموذج الرياح المستخدم. [تفصيل المراحل](../gazebo_batch/phases.csv)

**الشكل 2. القدرة المقيسة والمتوقعة في رحلات CMU المحجوزة للاختبار.** تمثل كل علامة رحلة واحدة لكل مرحلة من {count} رحلة لها قياسات بطارية فعلية. المحور الأفقي متوسط القدرة المقيسة من الجهد والتيار موزوناً بزمن العينة، والعمودي متوسط تقدير المعادلة المنشورة خلال النافذة نفسها. اللون وشكل العلامة يحددان الحمولة، والخط المتقطع يمثل تطابق القيمتين. بلغ متوسط الخطأ المطلق {errors['takeoff']:.1f} W في الإقلاع و{errors['cruise']:.1f} W في الطيران الأفقي و{errors['landing']:.1f} W في الهبوط. الانحياز يساوي التوقع ناقص القياس. تنتمي الرحلات المحجوزة إلى حملة الطائرة البحثية التي استُخلص منها النموذج. [الشكل](measured_holdout.png) · [قياسات وتوقعات كل مرحلة](../external/phase_comparisons.csv) · [التحقق الخارجي](../../../docs/m100_reaudit/external_ar.md)
"""
    (out/"captions.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_RESULTS)
    run(parser.parse_args().root)
