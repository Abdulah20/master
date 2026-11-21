"""Compare the timed simulator cruise windows with the selected CMU GNSS windows."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parents[2] / "results/m100_reaudit/pack_drive_three"
SIM, REAL, GATE = "#185778", "#c15e28", "#d9ece6"


def window(case, motion, native, context):
    t = native.time_s.to_numpy()
    dt = np.diff(t)
    start, end = motion.cruise_first_sim_s, motion.cruise_last_sim_s
    valid = (native.electrical_pair_qualified.eq(True) & native.demand_phase.eq("CRUISE")
             & native.time_s.between(start, end)).to_numpy(dtype=bool)
    covered = native.previous_interval_covered.eq(True).to_numpy(dtype=bool)
    edges = valid[:-1] & valid[1:] & covered[1:] & (dt > 0) & (dt <= 0.5)
    durations = dt[edges]
    if durations.size == 0:
        raise ValueError(f"No qualified electrical interval in {case['case_id']}")
    seconds = durations.sum()
    def mean(x):
        values = np.asarray(x, dtype=float)
        return np.average((values[:-1][edges] + values[1:][edges]) / 2, weights=durations)
    p = native.voltage_v * native.current_a
    w, v, i = mean(p), mean(native.voltage_v), mean(native.current_a)
    return dict(case_id=case["case_id"], run_id=motion.run_id, payload_kg=case["payload_kg"],
                outcome=motion.outcome, manager_gate_distance_m=motion.manager_gate_distance_m,
                measured_window_east_displacement_m=motion.cruise_east_displacement_m,
                ground_speed_mps=motion.cruise_mean_ground_speed_mps,
                cruise_height_min_m=motion.cruise_min_height_m, cruise_height_max_m=motion.cruise_max_height_m,
                cruise_duration_s=motion.cruise_span_s, qualified_electrical_duration_s=seconds,
                simulated_mean_W=w, simulated_mean_V=v, simulated_mean_A=i,
                simulated_covered_cruise_Wh=w*seconds/3600,
                simulated_within_cruise_sd_W=np.sqrt(mean((p-w)**2)),
                simulated_cruise_delta_V=native.voltage_v[valid].iloc[-1]-native.voltage_v[valid].iloc[0],
                simulated_cruise_delta_A=native.current_a[valid].iloc[-1]-native.current_a[valid].iloc[0],
                cmu_windows=int(context.flights_with_window),
                cmu_flight_equal_mean_W=context.flight_mean_mean_power_W,
                cmu_flight_equal_mean_V=context.flight_mean_mean_voltage_V,
                cmu_flight_equal_mean_A=context.flight_mean_mean_current_A,
                cmu_flight_equal_mean_within_window_sd_W=context.flight_mean_within_flight_sd_W,
                descriptive_sim_minus_cmu_W=w-context.flight_mean_mean_power_W,
                descriptive_sim_minus_cmu_percent=100*(w/context.flight_mean_mean_power_W-1))


def compare(out):
    cases = json.loads((out/"protocol.json").read_text())["cases"]
    motion = pd.read_csv(out/"motion.csv").set_index("case_id")
    electrical = pd.read_csv(out/"electrical.csv").set_index("case_id")
    cmu = pd.read_csv(out/"cmu_context.csv")
    cmu = cmu.loc[cmu.scope.eq("first_contiguous_steady_straight_100m_crossing")].set_index("payload_kg")
    measurements = pd.read_csv(out/"data/cmu_windows.csv.gz")
    fig, axes = plt.subplots(3, len(cases), figsize=(10.8, 7.2), sharey="row", sharex="row")
    summaries = []
    for col, case in enumerate(cases):
        cid, payload = case["case_id"], case["payload_kg"]
        m = motion.loc[cid]
        native = pd.read_csv(out/f"data/{cid}.pack.csv.gz")
        trajectory = pd.read_csv(out/f"data/{cid}.trajectory.csv.gz")
        result = window(case, m, native, cmu.loc[payload])
        energy = electrical.loc[cid]
        result.update(motion_recording_duration_s=m.span_s,
                      native_recording_duration_s=energy.observed_span_s,
                      covered_mission_duration_s=energy.covered_s,
                      covered_fraction_of_native_recording=energy.covered_s/energy.observed_span_s,
                      covered_mission_Wh=energy.covered_electrical_Wh)
        summaries.append(result)
        zero = trajectory.sim_time_s.iloc[0]
        start, end = m.cruise_first_sim_s, m.cruise_last_sim_s
        axes[0, col].set_title(f"{payload:g} kg payload", fontsize=12, fontweight="bold")
        axes[0, col].plot(trajectory.sim_time_s-zero, trajectory.z_m, color=SIM, linewidth=1.55)
        axes[1, col].plot(native.time_s-zero, native.voltage_v.where(native.electrical_pair_qualified.eq(True)),
                          color=SIM, linewidth=1.5)
        for ax in axes[:2, col]:
            ax.axvspan(start-zero, end-zero, color=GATE, zorder=0)
        for _, real in measurements[measurements.payload_kg.eq(payload)].groupby("flight_id"):
            axes[2, col].plot(real.elapsed_s, real.measured_power_w, color=REAL, linewidth=.95, alpha=.8)
        qualified = native.electrical_pair_qualified.eq(True) & native.demand_phase.eq("CRUISE")
        power = (native.voltage_v*native.current_a).where(qualified)
        gate = native.time_s.between(start, end)
        axes[2, col].plot(native.time_s[gate]-start, power[gate], color=SIM, linewidth=2.1)
        axes[1, col].set_xlabel("Recording time (s)")
        axes[2, col].set_xlabel("Time in cruise window (s)")
    axes[0, 0].set_ylabel("Altitude AGL (m)")
    axes[1, 0].set_ylabel("Pack voltage (V)")
    axes[2, 0].set_ylabel("Cruise pack power (W)")
    for ax in axes.ravel():
        ax.grid(axis="y", color="#dde2e4", lw=.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=.015)
    fig.suptitle("Matrice 100  |  Flight profiles and measured cruise power", fontsize=16, x=.071, ha="left", y=.979)
    fig.legend(handles=[Line2D([0],[0],color=SIM,lw=2,label="PX4 / Gazebo"),
                        Line2D([0],[0],color=REAL,lw=1.3,label="CMU measured flights"),
                        Patch(facecolor=GATE,label="100 m simulated cruise window")],
               loc="upper left", bbox_to_anchor=(.064,.946), frameon=False, ncol=3, fontsize=9.8)
    fig.subplots_adjust(top=.84,bottom=.08,left=.082,right=.98,hspace=.57,wspace=.15)
    fig.savefig(out/"flight_profiles.png",dpi=160,facecolor="white")
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(out/"summary.csv",index=False,float_format="%.12g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUTPUT)
    compare(parser.parse_args().out)
