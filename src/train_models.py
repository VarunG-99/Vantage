from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from data_utils import load_subset, label_train, label_test, make_features, get_final_rows, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--output-dir", default="reports")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--subset", choices=["FD001","FD002","FD003","FD004","ALL"], default="ALL")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--rul-cap", type=int, default=125)
    args = ap.parse_args()

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    modeldir=Path(args.model_dir); modeldir.mkdir(parents=True,exist_ok=True)
    subsets=["FD001","FD002","FD003","FD004"] if args.subset=="ALL" else [args.subset]
    rows=[]
    for subset in subsets:
        print(f"Training {subset}", flush=True)
        train_raw,test_raw,rul=load_subset(args.data_dir,subset)
        train=label_train(train_raw,args.rul_cap)
        test=label_test(test_raw,rul)
        ftrain=make_features(train,args.window)
        ftest=make_features(test,args.window)
        feature_cols=[c for c in ftrain.columns if c!="unit"]

        # Split by engine, never by individual overlapping rows/windows.
        splitter=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=42)
        tr_idx,va_idx=next(splitter.split(ftrain[feature_cols],train.RUL,groups=train.unit))
        Xtr=ftrain.iloc[tr_idx][feature_cols]
        ytr=train.iloc[tr_idx].RUL.to_numpy()
        Xva=ftrain.iloc[va_idx][feature_cols]
        yva=train.iloc[va_idx].RUL.to_numpy()

        final_idx=get_final_rows(test)
        Xtest=ftest.loc[final_idx,feature_cols]
        ytest=test.loc[final_idx,"RUL"].to_numpy()
        ytest_capped=np.minimum(ytest,args.rul_cap)

        models={
            "Ridge":make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
            "XGBoost":XGBRegressor(
                n_estimators=180,max_depth=5,learning_rate=0.05,
                subsample=0.85,colsample_bytree=0.85,reg_lambda=2.0,
                objective="reg:squarederror",n_jobs=4,random_state=42,
                tree_method="hist"
            )
        }
        for name,model in models.items():
            model.fit(Xtr,ytr)
            val_pred=np.maximum(model.predict(Xva),0)
            val_metrics=evaluate(yva,val_pred)
            # Refit final model on all official training trajectories.
            model.fit(ftrain[feature_cols],train.RUL.to_numpy())
            test_pred=np.maximum(model.predict(Xtest),0)
            test_metrics=evaluate(ytest,test_pred)
            capped_metrics=evaluate(ytest_capped,test_pred)
            joblib.dump(model,modeldir/f"{name.lower()}_{subset}.joblib")
            rows.append({
                "Subset":subset,"Model":name,
                "Validation_MAE":val_metrics["MAE"],
                "Validation_RMSE":val_metrics["RMSE"],
                "Validation_R2":val_metrics["R2"],
                "Test_MAE_raw_RUL":test_metrics["MAE"],
                "Test_RMSE_raw_RUL":test_metrics["RMSE"],
                "Test_R2_raw_RUL":test_metrics["R2"],
                "Test_NASA_score_raw_RUL":test_metrics["NASA_score"],
                "Test_MAE_capped_RUL":capped_metrics["MAE"],
                "Test_RMSE_capped_RUL":capped_metrics["RMSE"],
                "Test_R2_capped_RUL":capped_metrics["R2"],
                "Train_engines":int(train.unit.nunique()),
                "Test_engines":int(test.unit.nunique()),
                "Train_rows":int(len(train)),
                "Feature_count":len(feature_cols)
            })
            print(subset,name,"validation",val_metrics,"test raw",test_metrics,
                  "test capped",capped_metrics,flush=True)
            if name=="XGBoost":
                fi=pd.Series(model.feature_importances_,index=feature_cols).sort_values(ascending=False)
                fi.rename("importance").to_csv(outdir/f"{subset}_xgb_feature_importance.csv")
                pred_df=pd.DataFrame({
                    "unit":np.sort(test.unit.unique()),
                    "true_RUL_raw":ytest,
                    "true_RUL_capped":ytest_capped,
                    "predicted_RUL":test_pred
                })
                pred_df.to_csv(outdir/f"{subset}_xgb_test_predictions.csv",index=False)

    result=pd.DataFrame(rows)
    result.to_csv(outdir/"benchmark_results.csv",index=False)
    print("\nSaved",outdir/"benchmark_results.csv")

if __name__=="__main__":
    main()
