
from pathlib import Path
from functools import lru_cache
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_utils import COLUMNS, make_features


MODEL_DIR = ROOT / "models"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# Model selection based on the completed validation and test comparison.
RECOMMENDED_MODELS = {
    "FD001": "baseline",
    "FD002": "tuned",
    "FD003": "tuned",
    "FD004": "tuned",
}

MODEL_FILES = {
    "baseline": "xgboost_{subset}.joblib",
    "tuned": "xgboost_tuned_{subset}.joblib",
}

app = FastAPI(
    title="VANTAGE Aircraft Engine RUL API",
    description=(
        "REST API for predicting aircraft engine Remaining Useful Life "
        "using NASA C-MAPSS trained XGBoost models."
    ),
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    subset: Literal["FD001", "FD002", "FD003", "FD004"]
    records: list[dict[str, float]] = Field(min_length=1)
    model: Literal["recommended", "baseline", "tuned"] = "recommended"


@lru_cache(maxsize=8)
def load_model(path: str):
    return joblib.load(path)


def resolve_model(subset: str, choice: str):
    selected = (
        RECOMMENDED_MODELS[subset]
        if choice == "recommended"
        else choice
    )

    filename = MODEL_FILES[selected].format(subset=subset)
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {filename}")

    return load_model(str(path)), selected, filename


@app.get("/")
def root():
    return {
        "project": "VANTAGE",
        "message": "Aircraft Engine RUL Prediction API",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    model_status = {}

    for subset in SUBSETS:
        model_status[subset] = {}

        for choice, pattern in MODEL_FILES.items():
            filename = pattern.format(subset=subset)
            model_status[subset][choice] = (
                (MODEL_DIR / filename).exists()
            )

    all_available = all(
        all(models.values())
        for models in model_status.values()
    )

    return {
        "status": "healthy" if all_available else "degraded",
        "models": model_status,
    }


@app.get("/models")
def list_models():
    result = {}

    for subset in SUBSETS:
        result[subset] = {
            "recommended": RECOMMENDED_MODELS[subset],
            "available": [],
        }

        for choice, pattern in MODEL_FILES.items():
            filename = pattern.format(subset=subset)

            if (MODEL_DIR / filename).exists():
                result[subset]["available"].append(choice)

    return result


@app.post("/predict")
def predict(request: PredictionRequest):
    # Check that every record has the complete training schema.
    expected = set(COLUMNS)

    for index, record in enumerate(request.records):
        supplied = set(record.keys())

        missing = expected - supplied
        extra = supplied - expected

        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={
                    "record": index,
                    "missing_columns": sorted(missing),
                    "unexpected_columns": sorted(extra),
                },
            )

    try:
        frame = pd.DataFrame(request.records, columns=COLUMNS)

        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Input contains NaN or infinite values.")

        if (frame["unit"] <= 0).any():
            raise ValueError("Engine unit IDs must be positive.")

        if (frame["cycle"] <= 0).any():
            raise ValueError("Cycle values must be positive.")

        if frame.duplicated(["unit", "cycle"]).any():
            raise ValueError("Duplicate unit-cycle combinations found.")

        frame = frame.sort_values(
            ["unit", "cycle"]
        ).reset_index(drop=True)

        model, selected_model, filename = resolve_model(
            request.subset,
            request.model,
        )

        # Reuse the exact training feature-engineering pipeline.
        features = make_features(frame, window=30)

        expected_features = list(
            getattr(model, "feature_names_in_", [])
        )

        if expected_features:
            features = features[expected_features]

        # Predict the latest observed cycle for each engine.
        final_indices = (
            frame.groupby("unit")["cycle"].idxmax().to_numpy()
        )

        final_features = features.loc[final_indices]

        raw_predictions = np.asarray(
            model.predict(final_features),
            dtype=float,
        )

        if not np.isfinite(raw_predictions).all():
            raise ValueError("Model produced a non-finite prediction.")

        predictions = np.clip(raw_predictions, 0.0, 125.0)

        results = []

        for index, row_index in enumerate(final_indices):
            results.append({
                "unit": int(frame.loc[row_index, "unit"]),
                "last_cycle": int(frame.loc[row_index, "cycle"]),
                "predicted_rul_raw": float(raw_predictions[index]),
                "predicted_rul": float(predictions[index]),
            })

        return {
            "project": "VANTAGE",
            "subset": request.subset,
            "model_requested": request.model,
            "model_used": selected_model,
            "model_file": filename,
            "feature_count": len(expected_features) or features.shape[1],
            "engines_processed": len(results),
            "predictions": results,
            "note": (
                "Displayed RUL is clipped to the 0–125 cycle "
                "training target range. Predictions are research "
                "estimates, not certified maintenance decisions."
            ),
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    except HTTPException:
        raise

    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
