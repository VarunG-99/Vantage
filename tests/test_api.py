
import numpy as np
import pytest

from fastapi.testclient import TestClient

import api
from src.data_utils import COLUMNS, SENSORS


client = TestClient(api.app)


class MockModel:
    # Actual model inputs exclude the engine identifier.
    feature_names_in_ = np.array([
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
        *[f"sensor_{i}_last" for i in range(1, 22)],
        *[f"sensor_{i}_rollmean" for i in range(1, 22)],
        *[f"sensor_{i}_rollstd" for i in range(1, 22)],
    ])

    def predict(self, features):
        assert list(features.columns) == list(self.feature_names_in_)
        return np.array([50.0] * len(features))


def make_record(unit=1, cycle=1):
    record = {
        "unit": unit,
        "cycle": cycle,
        "setting_1": 0.0023,
        "setting_2": 0.0002,
        "setting_3": 100.0,
    }

    for sensor in SENSORS:
        record[sensor] = 100.0

    return record


def make_payload(records=None, subset="FD001", model="recommended"):
    return {
        "subset": subset,
        "model": model,
        "records": records if records is not None else [make_record()],
    }


@pytest.fixture
def mock_model(monkeypatch):
    def fake_resolve_model(subset, choice):
        selected = (
            api.RECOMMENDED_MODELS[subset]
            if choice == "recommended"
            else choice
        )

        return (
            MockModel(),
            selected,
            f"mock_{subset}.joblib",
        )

    monkeypatch.setattr(api, "resolve_model", fake_resolve_model)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == "VANTAGE"


def test_models_endpoint():
    response = client.get("/models")

    assert response.status_code == 200

    result = response.json()

    assert set(result.keys()) == {
        "FD001", "FD002", "FD003", "FD004"
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()


def test_valid_prediction(mock_model):
    response = client.post(
        "/predict",
        json=make_payload()
    )

    assert response.status_code == 200

    result = response.json()

    assert result["project"] == "VANTAGE"
    assert result["model_used"] == "baseline"
    assert result["engines_processed"] == 1
    assert result["feature_count"] == 67
    assert result["predictions"][0]["predicted_rul"] == 50.0


def test_multiple_engines(mock_model):
    records = [
        make_record(unit=1, cycle=1),
        make_record(unit=1, cycle=2),
        make_record(unit=2, cycle=1),
        make_record(unit=2, cycle=2),
    ]

    response = client.post(
        "/predict",
        json=make_payload(records)
    )

    assert response.status_code == 200
    assert response.json()["engines_processed"] == 2


def test_missing_sensor_column():
    record = make_record()
    del record["sensor_1"]

    response = client.post(
        "/predict",
        json=make_payload([record])
    )

    assert response.status_code == 422


def test_unexpected_column():
    record = make_record()
    record["extra_sensor"] = 123.0

    response = client.post(
        "/predict",
        json=make_payload([record])
    )

    assert response.status_code == 422


def test_invalid_subset():
    response = client.post(
        "/predict",
        json=make_payload(subset="FD005")
    )

    assert response.status_code == 422


def test_empty_records():
    response = client.post(
        "/predict",
        json=make_payload(records=[])
    )

    assert response.status_code == 422


def test_duplicate_engine_cycle(mock_model):
    records = [
        make_record(unit=1, cycle=1),
        make_record(unit=1, cycle=1),
    ]

    response = client.post(
        "/predict",
        json=make_payload(records)
    )

    assert response.status_code == 422


def test_non_positive_engine_id(mock_model):
    response = client.post(
        "/predict",
        json=make_payload([make_record(unit=0)])
    )

    assert response.status_code == 422


def test_rul_clipping(mock_model, monkeypatch):
    class HighPredictionModel(MockModel):
        def predict(self, features):
            return np.array([150.0] * len(features))

    monkeypatch.setattr(
        api,
        "resolve_model",
        lambda subset, choice: (
            HighPredictionModel(),
            "baseline",
            "mock.joblib",
        )
    )

    response = client.post(
        "/predict",
        json=make_payload()
    )

    assert response.status_code == 200
    assert response.json()["predictions"][0]["predicted_rul"] == 125.0


def test_missing_model_returns_503(monkeypatch):
    def missing_model(subset, choice):
        raise FileNotFoundError("Test model missing")

    monkeypatch.setattr(api, "resolve_model", missing_model)

    response = client.post(
        "/predict",
        json=make_payload()
    )

    assert response.status_code == 503
