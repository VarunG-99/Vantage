# VANTAGE — Deployment Guide

This document describes how to run the VANTAGE Streamlit dashboard and FastAPI prediction service locally and how to deploy the dashboard.

## 1. Requirements

- Python 3.10–3.12 recommended.
- Git.
- Project dependencies from `requirements.txt`.
- Trained model artifacts in `models/`.
- NASA C-MAPSS files in `data/raw/`.

## 2. Local installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run the Streamlit dashboard

From the project root:

```bash
streamlit run app.py
```

Streamlit will display the local dashboard URL in the terminal.

The dashboard loads existing model artifacts and recorded C-MAPSS data. It does not retrain models during startup.

## 4. Run the FastAPI service

From the project root:

```bash
python -m uvicorn api:app --reload
```

The local API address is:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Checks expected model-file availability |
| `/models` | GET | Lists available models and recommended selections |
| `/predict` | POST | Accepts sensor records and returns RUL predictions |

### Health check

Send a GET request to:

```text
http://127.0.0.1:8000/health
```

A healthy response indicates that all expected baseline and tuned model files are present. It does not independently validate every model's prediction behaviour.

### Model listing

Send a GET request to:

```text
http://127.0.0.1:8000/models
```

The API returns available model choices and the recommended model for each subset.

### Prediction request

Send a POST request to:

```text
http://127.0.0.1:8000/predict
```

Example:

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
      "sensor_1": 518.67,
      "sensor_2": 641.82,
      "sensor_3": 1589.70,
      "sensor_4": 1400.60,
      "sensor_5": 14.62,
      "sensor_6": 21.61,
      "sensor_7": 554.36,
      "sensor_8": 2388.06,
      "sensor_9": 9046.19,
      "sensor_10": 1.30,
      "sensor_11": 47.47,
      "sensor_12": 521.66,
      "sensor_13": 2388.02,
      "sensor_14": 8138.62,
      "sensor_15": 8.4195,
      "sensor_16": 0.03,
      "sensor_17": 392,
      "sensor_18": 2388,
      "sensor_19": 100.0,
      "sensor_20": 39.06,
      "sensor_21": 23.419
    }
  ]
}
```

Each record must contain all 26 canonical fields. Multiple cycles for an engine should be supplied to provide rolling sensor history.

The API returns the selected model, processed engine count, raw predictions and displayed predictions.

The displayed RUL is clipped to the 0–125 cycle training target range.

### Model selection

Use one of:

- `recommended`
- `baseline`
- `tuned`

Recommended selection:

| Subset | Model |
|---|---|
| FD001 | Baseline XGBoost |
| FD002 | Tuned XGBoost |
| FD003 | Tuned XGBoost |
| FD004 | Tuned XGBoost |

## 5. CSV upload format for Streamlit

The dashboard accepts engine trajectories in CSV format.

Canonical header:

```text
unit,cycle,setting_1,setting_2,setting_3,sensor_1,...,sensor_21
```

A single-engine file may omit `unit`, in which case the dashboard assigns unit ID 1.

Headerless 25- or 26-column NASA-style files are also accepted.

Important input requirements:

- All sensor and setting values must be numeric.
- Include the complete available trajectory, not just the last cycle.
- Keep engine IDs consistent across their respective records.
- The dashboard handles row sorting.

The 30-cycle rolling features require historical observations.

## 6. Streamlit Community Cloud deployment

1. Create a GitHub repository.
2. Push the verified project files.
3. Open Streamlit Community Cloud.
4. Select **Create app**.
5. Connect the GitHub repository and select the appropriate branch.
6. Set the application entry point to `app.py`.
7. Deploy.

Ensure that the required model artifacts, data files and dependencies are available to the deployed application.

The repository should not contain:
- Virtual environments.
- API keys or credentials.
- Personal files.
- Unnecessary temporary files.

Check GitHub's current file and repository limits before including large datasets and model artifacts.

The Streamlit deployment instructions apply to the dashboard. The FastAPI service requires a separate hosting environment if it is to be publicly accessible.

## 7. Reproducibility and testing

Install dependencies:

```bash
pip install -r requirements.txt
```

Run automated tests:

```bash
python -m pytest tests -v
```

Run the model training, tuning and robustness scripts using the commands in `README.md`.

The dashboard and API load saved model artifacts; they do not retrain models during startup.

## 8. Important limitations

VANTAGE is an academic research demonstrator. It is not certified for aircraft maintenance or operational safety decisions.

Real aircraft telemetry may differ from C-MAPSS in sensor calibration, measurement units, sampling frequency, operating conditions and degradation behaviour.

The selected FD subset must reasonably match the input operating-condition and fault regime.

Uploaded engine trajectories do not include ground-truth RUL labels, so their predictions cannot be directly evaluated by the dashboard.