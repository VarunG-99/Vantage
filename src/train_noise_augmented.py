from pathlib import Path
import argparse, json, joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from data_utils import load_subset,label_train,label_test,make_features,get_final_rows,evaluate,SENSORS

def make_model():
    return XGBRegressor(n_estimators=180,max_depth=5,learning_rate=.05,
        subsample=.85,colsample_bytree=.85,reg_lambda=2,
        objective="reg:squarederror",n_jobs=4,random_state=42,tree_method="hist")

def perturb(frame, train_std, level, seed):
    out=frame.copy()
    if level:
        rng=np.random.default_rng(seed)
        noise=rng.normal(0,1,(len(out),len(SENSORS)))*train_std*level
        out.loc[:,SENSORS]=out[SENSORS].to_numpy()+noise
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="data/raw")
    ap.add_argument("--model-dir",default="models")
    ap.add_argument("--output-dir",default="reports")
    ap.add_argument("--subsets",nargs="+",default=["FD001","FD004"])
    ap.add_argument("--noise-std",type=float,default=.10)
    ap.add_argument("--seeds",type=int,default=5)
    args=ap.parse_args()
    model_dir=Path(args.model_dir); model_dir.mkdir(parents=True,exist_ok=True)
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    summary={}
    for sub in args.subsets:
        train_raw,test_raw,rul=load_subset(args.data_dir,sub)
        train=label_train(train_raw,125); test=label_test(test_raw,rul)
        split=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42)
        it,iv=next(split.split(train,train.RUL,groups=train.unit))
        tr=train.iloc[it].copy(); va=train.iloc[iv].copy()
        ftr=make_features(tr); fva=make_features(va)
        cols=[c for c in ftr.columns if c!="unit"]

        # Baseline model trained only on training engines.
        baseline=make_model()
        baseline.fit(ftr[cols],tr.RUL.to_numpy())

        # Noise augmentation: one perturbed copy of each training trajectory.
        std=tr[SENSORS].std().to_numpy()
        aug_raw=perturb(tr,std,args.noise_std,321)
        faug=make_features(aug_raw)
        fcomb=pd.concat([ftr,faug],ignore_index=True)
        ycomb=np.r_[tr.RUL.to_numpy(),aug_raw.RUL.to_numpy()]
        augmented=make_model()
        augmented.fit(fcomb[cols],ycomb)

        validation={}
        for name,model in [("baseline",baseline),("noise_augmented",augmented)]:
            validation[name]={}
            for level in [0,.05,.10,.20]:
                vals=[]
                for seed in range(args.seeds):
                    corrupted=perturb(va,std,level,1000+seed)
                    fc=make_features(corrupted)
                    pred=np.maximum(model.predict(fc[cols]),0)
                    vals.append(evaluate(va.RUL.to_numpy(),pred))
                validation[name][str(level)]={
                    metric:{"mean":float(np.mean([v[metric] for v in vals])),
                            "std":float(np.std([v[metric] for v in vals],ddof=1)) if len(vals)>1 else 0.0}
                    for metric in ["MAE","RMSE","R2"]
                }

        # Refit on all official training engines; hold test data for final evaluation.
        f_all=make_features(train)
        aug_all=perturb(train,train[SENSORS].std().to_numpy(),args.noise_std,321)
        f_aug_all=make_features(aug_all)
        f_comb_all=pd.concat([f_all,f_aug_all],ignore_index=True)
        y_comb_all=np.r_[train.RUL.to_numpy(),aug_all.RUL.to_numpy()]
        augmented.fit(f_comb_all[cols],y_comb_all)
        joblib.dump(augmented,model_dir/f"xgboost_noiseaug_{sub}.joblib")

        test_idx=get_final_rows(test); ytest=test.loc[test_idx,"RUL"].to_numpy()
        test_scores={}
        for level in [0,.05,.10,.20]:
            vals=[]
            for seed in range(args.seeds):
                corrupted=perturb(test,train[SENSORS].std().to_numpy(),level,1000+seed)
                fc=make_features(corrupted)
                pred=np.maximum(augmented.predict(fc.loc[test_idx,cols]),0)
                vals.append(evaluate(ytest,pred))
            test_scores[str(level)]={
                metric:{"mean":float(np.mean([v[metric] for v in vals])),
                        "std":float(np.std([v[metric] for v in vals],ddof=1)) if len(vals)>1 else 0.0}
                for metric in ["MAE","RMSE","R2"]
            }
        summary[sub]={"validation":validation,"test_noise_augmented_model":test_scores}
        print(sub,json.dumps(summary[sub],indent=2),flush=True)
    (outdir/"noise_augmentation_results.json").write_text(json.dumps(summary,indent=2))
    print("Saved",outdir/"noise_augmentation_results.json")

if __name__=="__main__": main()
