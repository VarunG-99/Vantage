from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_subset, SENSORS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="data/raw")
    ap.add_argument("--output-dir",default="reports/figures")
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    summaries={}
    datasets={}
    for sub in ["FD001","FD002","FD003","FD004"]:
        tr,te,rul=load_subset(args.data_dir,sub)
        datasets[sub]=(tr,te,rul)
        summaries[sub]={
            "train_rows":len(tr),"test_rows":len(te),
            "train_engines":tr.unit.nunique(),"test_engines":te.unit.nunique(),
            "train_median_life":tr.groupby("unit").cycle.max().median(),
            "train_min_life":tr.groupby("unit").cycle.max().min(),
            "train_max_life":tr.groupby("unit").cycle.max().max(),
            "test_rul_min":rul.min(),"test_rul_max":rul.max()
        }
    pd.DataFrame(summaries).T.to_csv(out.parent/"dataset_profile.csv")
    # Overview
    prof=pd.DataFrame(summaries).T
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    prof[["train_engines","test_engines"]].plot(kind="bar",ax=axs[0])
    axs[0].set_title("Engine counts"); axs[0].set_ylabel("Engines")
    prof[["train_rows","test_rows"]].plot(kind="bar",ax=axs[1])
    axs[1].set_title("Time-series records"); axs[1].set_ylabel("Rows")
    plt.tight_layout(); fig.savefig(out/"dataset_overview.png",dpi=180); plt.close(fig)
    # Life lengths
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for ax,(sub,(tr,_,_)) in zip(axs.ravel(),datasets.items()):
        ax.hist(tr.groupby("unit").cycle.max(),bins=18,edgecolor="white")
        ax.set_title(f"{sub}: trajectory lengths"); ax.set_xlabel("Cycles"); ax.set_ylabel("Engines")
    plt.tight_layout(); fig.savefig(out/"trajectory_lengths.png",dpi=180); plt.close(fig)
    # Correlation
    fig,ax=plt.subplots(figsize=(10,8))
    sns.heatmap(datasets["FD001"][0][SENSORS].corr(),cmap="vlag",center=0,square=True,ax=ax)
    ax.set_title("FD001 sensor correlation matrix")
    plt.tight_layout(); fig.savefig(out/"fd001_sensor_correlation.png",dpi=180); plt.close(fig)
    # Example traces
    fig,axs=plt.subplots(2,2,figsize=(12,7))
    for ax,(sub,(tr,_,_)) in zip(axs.ravel(),datasets.items()):
        lens=tr.groupby("unit").cycle.max()
        uid=int((lens-lens.median()).abs().idxmin())
        g=tr[tr.unit==uid]
        chosen=g[SENSORS].var().sort_values(ascending=False).head(3).index
        for sn in chosen: ax.plot(g.cycle,g[sn],label=sn,lw=1)
        ax.set_title(f"{sub}: engine {uid}"); ax.set_xlabel("Cycle"); ax.legend(fontsize=7)
    plt.tight_layout(); fig.savefig(out/"sample_sensor_trajectories.png",dpi=180); plt.close(fig)
    print(pd.DataFrame(summaries).T.to_string())

if __name__=="__main__": main()
