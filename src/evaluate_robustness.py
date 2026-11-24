from pathlib import Path
import argparse, json
import numpy as np, pandas as pd, joblib
from data_utils import load_subset,label_train,label_test,make_features,get_final_rows,evaluate,SENSORS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="data/raw")
    ap.add_argument("--model-dir",default="models")
    ap.add_argument("--output",default="reports/robustness_results.json")
    ap.add_argument("--subsets",nargs="+",default=["FD001","FD004"])
    ap.add_argument("--seeds",type=int,default=5)
    args=ap.parse_args()
    results={}
    for sub in args.subsets:
        tr0,te0,rul=load_subset(args.data_dir,sub)
        tr=label_train(tr0,125); te=label_test(te0,rul)
        model=joblib.load(Path(args.model_dir)/f"xgboost_{sub}.joblib")
        fte=make_features(te,30)
        features=[c for c in fte.columns if c!="unit"]
        idx=get_final_rows(te); y=te.loc[idx,"RUL"].to_numpy()
        results[sub]={}
        for level in [0,.05,.10,.20]:
            vals=[]
            for seed in range(args.seeds):
                corrupted=te.copy()
                if level:
                    rng=np.random.default_rng(1000+seed)
                    scale=tr[SENSORS].std().to_numpy()
                    noise=rng.normal(0,1,(len(corrupted),len(SENSORS)))*scale*level
                    corrupted.loc[:,SENSORS]=corrupted[SENSORS].to_numpy()+noise
                fc=make_features(corrupted,30)
                pred=np.maximum(model.predict(fc.loc[idx,features]),0)
                vals.append(evaluate(y,pred))
            results[sub][str(level)]={
                metric:{"mean":float(np.mean([v[metric] for v in vals])),
                        "std":float(np.std([v[metric] for v in vals],ddof=1)) if len(vals)>1 else 0.0}
                for metric in ["MAE","RMSE","R2"]
            }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
