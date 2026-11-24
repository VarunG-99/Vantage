# VANTAGE — Aircraft Engine Remaining Useful Life Prediction

VANTAGE is an end-to-end machine learning project for predicting the Remaining Useful Life (RUL) of aircraft turbofan engines using NASA's C-MAPSS dataset.

The project includes exploratory data analysis, preprocessing, feature engineering, baseline regression models, hyperparameter tuning, robustness experiments, a Streamlit dashboard and a FastAPI prediction service.

## 1. Project structure

```text
VANTAGE/
│
├── app.py
├── api.py
├── requirements.txt
├── README.md
├── README_DEPLOYMENT.md
│
├── data/
│   └── raw/
│
├── models/
│   ├── ridge_FD001.joblib
│   ├── xgboost_FD001.joblib
│   ├── xgboost_tuned_FD001.joblib
│   └── ...
│
├── reports/
│   ├── tuning_results.csv
│   ├── tuning_best_params.json
│   └── figures/
│
├── src/
│   ├── data_utils.py
│   ├── run_eda.py
│   ├── train_models.py
│   ├── tune_xgboost.py
│   ├── train_noise_augmented.py
│   ├── evaluate_robustness.py
│   └── make_result_figures.py
│
└── tests/
    ├── test_api.py
    └── test_data_utils.py
```

## 2. Dataset

VANTAGE uses NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset.

It contains four subsets:

| Subset | Operating conditions | Fault modes |
|---|---:|---:|
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

Each observation contains 26 columns:
- Engine unit identifier
- Operating cycle
- Three operational settings
- 21 sensor measurements

The dataset contains training trajectories, test trajectories and corresponding final-cycle RUL labels.

## 3. Environment setup

Recommended Python versions: 3.10–3.12.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Data preprocessing and feature engineering

Implemented in `src/data_utils.py`.

The preprocessing pipeline includes:

1. Loading the four C-MAPSS subsets.
2. Assigning canonical column names.
3. Sorting observations by engine and cycle.
4. Constructing training RUL labels.
5. Reconstructing test RUL labels using the supplied final-cycle labels.
6. Generating causal rolling sensor features.

Training RUL is calculated as:

\[
RUL = \text{Maximum engine cycle} - \text{Current cycle}
\]

Training targets are capped at 125 cycles.

For each of the 21 sensors, three features are generated:
- Current sensor measurement
- Trailing 30-cycle rolling mean
- Trailing 30-cycle rolling standard deviation

The model receives 67 features:

\[
4 + 21 + 21 + 21 = 67
\]

These consist of cycle, three operational settings, and the three groups of sensor features. The engine identifier is excluded from model inputs.

The rolling calculations use only the current and preceding observations, avoiding future-cycle information.

## 5. Exploratory data analysis

Implemented in `src/run_eda.py`.

Run:

```bash
python src/run_eda.py --data-dir data/raw --output-dir reports/figures
```

The EDA workflow profiles the dataset and generates visualizations to examine sensor behaviour, operating conditions and degradation patterns.

## 6. Baseline model training

Implemented in `src/train_models.py`.

Run:

```bash
python src/train_models.py --data-dir data/raw --output-dir reports --model-dir models --subset ALL
```

Two regression approaches are trained:

**Ridge Regression**
- StandardScaler preprocessing
- Regularization parameter alpha = 10

**XGBoost**
- 180 estimators
- Maximum depth = 5
- Learning rate = 0.05
- Subsample = 0.85
- Column sampling = 0.85
- L2 regularization = 2.0

Validation uses an 80:20 engine-wise split with random state 42. All cycles belonging to an engine remain in the same partition.

Final models are refitted using all official training engines.

## 7. Hyperparameter tuning

Implemented in `src/tune_xgboost.py`.

XGBoost tuning uses:

- GroupKFold with three folds.
- RandomizedSearchCV.
- 16 parameter combinations by default.
- Engine IDs as grouping variables.
- Negative mean absolute error as the search objective.

Run:

```bash
python src/tune_xgboost.py --data-dir data/raw --model-dir models --output-dir reports --iterations 16
```

The search explores:
- Number of estimators
- Maximum tree depth
- Learning rate
- Subsample ratio
- Column sampling ratio
- L2 regularization
- Minimum child weight

The official test set is evaluated after parameter selection and is not used by the cross-validation search.

Generated outputs:

```text
models/
    xgboost_tuned_FD001.joblib
    xgboost_tuned_FD002.joblib
    xgboost_tuned_FD003.joblib
    xgboost_tuned_FD004.joblib

reports/
    tuning_results.csv
    tuning_best_params.json
```

Model selection based on the completed baseline-versus-tuned comparison:

| Subset | Selected model |
|---|---|
| FD001 | Baseline XGBoost |
| FD002 | Tuned XGBoost |
| FD003 | Tuned XGBoost |
| FD004 | Tuned XGBoost |

Tuning improvements vary by subset. A lower cross-validation error does not guarantee improved official test performance.

## 8. Noise augmentation and robustness

Implemented in:
- `src/train_noise_augmented.py`
- `src/evaluate_robustness.py`

Noise augmentation experiments are performed on FD001 and FD004.

Example:

```bash
python src/train_noise_augmented.py --data-dir data/raw --model-dir models --output-dir reports --subsets FD001 FD004 --noise-std 0.10 --seeds 5
```

The robustness evaluation uses multiple random seeds and Gaussian sensor noise at different magnitudes.

Run:

```bash
python src/evaluate_robustness.py --data-dir data/raw --model-dir models --output reports/robustness_results.json --subsets FD001 FD004 --seeds 5
```

These experiments investigate model sensitivity to synthetic sensor corruption. Their findings should not be interpreted as proof of robustness to real aircraft sensor faults.

## 9. Evaluation metrics

The project uses:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- NASA asymmetric scoring function

MAE and RMSE measure prediction error in cycles. The NASA score penalizes overestimation more heavily than underestimation.

Both raw-RUL and cap-aligned evaluation results are retained where applicable. The evaluation convention should always be stated when reporting results.

## 10. Streamlit dashboard

The interactive dashboard is implemented in `app.py`.

Run locally:

```bash
streamlit run app.py
```

Dashboard capabilities include:
- Selecting FD001–FD004.
- Selecting available trained models.
- Exploring recorded test engines.
- Uploading engine trajectory CSV files.
- Viewing sensor measurements.
- Comparing model performance.
- Inspecting prediction errors.
- Exploring stored robustness results.

The dashboard loads existing trained models and does not retrain them during startup.

See `README_DEPLOYMENT.md` for deployment and CSV input instructions.

## 11. FastAPI prediction service

The REST API is implemented in `api.py`.

Start the service:

```bash
python -m uvicorn api:app --reload
```

Available endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Model-file availability |
| `/models` | GET | Available models and recommendations |
| `/predict` | POST | Predict RUL from engine trajectories |
| `/docs` | GET | Interactive Swagger documentation |

The prediction endpoint accepts a dataset subset, model selection and sensor records.

Example request:

```json
{
  "subset": "FD001",
  "model": "recommended",
  "records": [
    {
      "unit": 1,
      "cycle": 1,
      "setting_1": 0.0023,
      "setting_2": 0.0002,
      "setting_3": 100.0,
      "sensor_1": 518.67
    }
  ]
}
```

The example above is illustrative only; every record must contain all 26 canonical input columns.

Predictions are clipped to the 0–125 cycle training target range for display. Raw model predictions are also returned.

## 12. Automated testing

The project uses pytest and FastAPI TestClient.

Run:

```bash
python -m pytest tests -v
```

The automated suite covers:
- Data schema and preprocessing.
- Training and test RUL calculations.
- Rolling feature generation.
- Engine-wise feature isolation.
- Final-cycle selection.
- Evaluation metrics.
- API endpoints.
- Valid and invalid prediction requests.
- Multiple-engine processing.
- RUL clipping.
- Missing-model error handling.

The current test suite contains 22 passing tests.

Most API prediction tests use a mock model to isolate API behaviour. Real-model integration was also manually verified using an actual FD001 test-engine trajectory.

## 13. Important limitations

- C-MAPSS is a simulated turbofan dataset, not real operational aircraft telemetry.
- The system assumes the input belongs to the selected FD001–FD004 operating-condition and fault regime.
- Noise robustness experiments use synthetic Gaussian corruption.
- Robustness experiments are focused on FD001 and FD004.
- Predictive uncertainty and external N-CMAPSS validation are not implemented.
- Predictions are research estimates, not certified aircraft maintenance decisions.

## 14. Project objective

VANTAGE demonstrates an end-to-end machine learning workflow for aircraft engine health monitoring, combining data analysis, feature engineering, regression modelling, model selection, robustness evaluation, interactive visualization and REST API deployment.