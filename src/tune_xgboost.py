
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor

from data_utils import (
    load_subset,
    label_train,
    label_test,
    make_features,
    get_final_rows,
    evaluate,
)


def make_baseline():
    return XGBRegressor(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=42,
        tree_method="hist",
    )


def main():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(root / "data" / "raw"))
    parser.add_argument("--model-dir", default=str(root / "models"))
    parser.add_argument("--output-dir", default=str(root / "reports"))
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["FD001", "FD002", "FD003", "FD004"],
    )
    parser.add_argument("--iterations", type=int, default=16)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search space for randomized hyperparameter tuning.
    param_dist = {
        "n_estimators": [120, 180, 240, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.07, 0.10],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.75, 0.85, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
        "min_child_weight": [1, 3, 5],
    }

    all_results = []
    all_best_params = {}

    for subset in args.subsets:
        print(f"\n{'=' * 55}")
        print(f"TUNING {subset}")
        print("=" * 55)

        train_raw, test_raw, rul = load_subset(args.data_dir, subset)

        train = label_train(train_raw, cap=125)
        test = label_test(test_raw, rul)

        ftrain = make_features(train, window=30)
        ftest = make_features(test, window=30)

        feature_cols = [c for c in ftrain.columns if c != "unit"]

        X = ftrain[feature_cols]
        y = train["RUL"].to_numpy()
        groups = train["unit"].to_numpy()

        # All folds are separated by engine.
        cv = GroupKFold(n_splits=3)

        # Original baseline, evaluated with the same CV folds.
        baseline = make_baseline()

        baseline_scores = cross_val_score(
            baseline,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )

        baseline_cv_mae = float(-baseline_scores.mean())

        print(f"Baseline 3-fold CV MAE: {baseline_cv_mae:.4f}")

        # Randomized search using training engines only.
        search = RandomizedSearchCV(
            estimator=make_baseline(),
            param_distributions=param_dist,
            n_iter=args.iterations,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=42,
            n_jobs=1,
            verbose=1,
            refit=True,
            return_train_score=True,
        )

        search.fit(X, y, groups=groups)

        tuned_cv_mae = float(-search.best_score_)

        print(f"Tuned 3-fold CV MAE: {tuned_cv_mae:.4f}")
        print("Best parameters:")
        print(json.dumps(search.best_params_, indent=4))

        # Official test set is used only after parameter selection.
        final_model = XGBRegressor(
            **search.best_params_,
            objective="reg:squarederror",
            n_jobs=4,
            random_state=42,
            tree_method="hist",
        )

        final_model.fit(X, y)

        test_indices = get_final_rows(test)
        X_test = ftest.loc[test_indices, feature_cols]

        y_test = test.loc[test_indices, "RUL"].to_numpy()
        y_test_capped = np.minimum(y_test, 125)

        tuned_predictions = np.maximum(final_model.predict(X_test), 0)

        tuned_test_metrics = evaluate(y_test, tuned_predictions)
        tuned_capped_metrics = evaluate(y_test_capped, tuned_predictions)

        # Baseline official test evaluation for direct comparison.
        baseline_final = make_baseline()
        baseline_final.fit(X, y)

        baseline_predictions = np.maximum(
            baseline_final.predict(X_test), 0
        )

        baseline_test_metrics = evaluate(y_test, baseline_predictions)

        model_path = model_dir / f"xgboost_tuned_{subset}.joblib"
        joblib.dump(final_model, model_path)

        all_best_params[subset] = search.best_params_

        all_results.append({
            "Subset": subset,
            "Baseline_CV_MAE": baseline_cv_mae,
            "Tuned_CV_MAE": tuned_cv_mae,
            "CV_MAE_Improvement": baseline_cv_mae - tuned_cv_mae,
            "Baseline_Test_MAE": baseline_test_metrics["MAE"],
            "Tuned_Test_MAE": tuned_test_metrics["MAE"],
            "Tuned_Test_RMSE": tuned_test_metrics["RMSE"],
            "Tuned_Test_R2": tuned_test_metrics["R2"],
            "Tuned_Test_NASA_Score": tuned_test_metrics["NASA_score"],
            "Tuned_Test_Capped_MAE": tuned_capped_metrics["MAE"],
            "Model_File": model_path.name,
        })

        print("\nOfficial test comparison:")
        print(f"Baseline MAE: {baseline_test_metrics['MAE']:.4f}")
        print(f"Tuned MAE:    {tuned_test_metrics['MAE']:.4f}")

        print(f"\nSaved tuned model: {model_path}")

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        output_dir / "tuning_results.csv",
        index=False,
    )

    (output_dir / "tuning_best_params.json").write_text(
        json.dumps(all_best_params, indent=4)
    )

    print("\nTUNING SUMMARY")
    print(results_df.to_string(index=False))

    print("\nSaved:")
    print(output_dir / "tuning_results.csv")
    print(output_dir / "tuning_best_params.json")


if __name__ == "__main__":
    main()
