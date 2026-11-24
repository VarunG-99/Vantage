from pathlib import Path
import argparse, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reports",default="reports")
    args=ap.parse_args()
    root=Path(args.reports); out=root/"figures"; out.mkdir(parents=True,exist_ok=True)
    # Per-subset predictions
    for sub in ["FD001","FD002","FD003","FD004"]:
        p=root/f"{sub}_xgb_test_predictions.csv"
        if p.exists():
            d=pd.read_csv(p)
            fig,ax=plt.subplots(figsize=(5.5,5))
            ax.scatter(d.true_RUL_raw,d.predicted_RUL,alpha=.7,s=22)
            m=max(d.true_RUL_raw.max(),d.predicted_RUL.max())+5
            ax.plot([0,m],[0,m],"--",color="black",lw=1)
            ax.set(xlim=(0,m),ylim=(0,m),xlabel="True RUL (cycles)",
                   ylabel="Predicted RUL (cycles)",title=f"{sub}: XGBoost test predictions")
            fig.tight_layout(); fig.savefig(out/f"{sub.lower()}_xgb_predicted_vs_true.png",dpi=180); plt.close(fig)
        fi=root/f"{sub}_xgb_feature_importance.csv"
        if fi.exists():
            d=pd.read_csv(fi,index_col=0).squeeze("columns").sort_values(ascending=False).head(15).sort_values()
            fig,ax=plt.subplots(figsize=(7,5))
            d.plot(kind="barh",ax=ax)
            ax.set_title(f"{sub}: XGBoost top 15 feature importances")
            ax.set_xlabel("Feature importance")
            fig.tight_layout(); fig.savefig(out/f"{sub.lower()}_xgb_feature_importance.png",dpi=180); plt.close(fig)
    print("Figures saved to",out)

if __name__=="__main__": main()
