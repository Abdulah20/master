"""Plot flight-level mean absolute errors and paired flight-date uncertainty."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[2] / "results/m100_reaudit/cruise_power_decision"


def make_figure(out=OUT):
    with (out/"metrics.csv").open(newline="") as stream:
        means = {(r["cohort"],r["model"]):float(r["MAE_W"]) for r in csv.DictReader(stream)
                 if r["grouping"]=="all" and r["weighting"]=="flight"}
    with (out/"paired_changes.csv").open(newline="") as stream:
        changes = {r["reference"]:r for r in csv.DictReader(stream) if r["cohort"]=="evaluation"
                   and r["grouping"]=="all" and r["weighting"]=="flight" and r["model"]=="selected"}
    fig, (left,right) = plt.subplots(1,2,figsize=(10.3,4.5),width_ratios=(1.3,1),layout="constrained")
    x = np.arange(3)
    cohorts = ("evaluation","earlier_five","other_evaluation")
    for shift,model,label,color in ((-.24,"baseline","Base","#56616a"),
                                    (0,"legacy","Previous","#bb7436"),(.24,"selected","New","#077e79")):
        values = [means[cohort,model] for cohort in cohorts]
        bars = left.bar(x+shift,values,.22,label=label,color=color)
        left.bar_label(bars,fmt="%.1f",padding=2,fontsize=8)
    left.set(xticks=x,xticklabels=["All 33","Matched 5","Other 28"],ylabel="Mean absolute error (W)",
             ylim=(0,23),title="CMU stable-cruise power")
    left.legend(frameon=False,ncols=3,loc="upper center",bbox_to_anchor=(.5,-.09))
    right.axvline(0,color="#626b72",linestyle="--",linewidth=1)
    for y,key in ((1,"baseline"),(0,"legacy")):
        r=changes[key]
        d,lo,hi=(float(r[k]) for k in ("MAE_change_W","date_bootstrap_95_low_W","date_bootstrap_95_high_W"))
        right.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt="o",color="#077e79",capsize=5,lw=2)
        right.text(d,y+.15,f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]",ha="center",fontsize=9)
    right.set(yticks=[1,0],yticklabels=["New − base","New − previous"],ylim=(-.5,1.55),xlim=(-8,3),
              xlabel="Change in mean absolute error (W)",title="Paired differences on 33 flights")
    for ax in (left,right):
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(length=3)
    fig.savefig(out/"field_power_comparison.png",dpi=180,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    make_figure()
