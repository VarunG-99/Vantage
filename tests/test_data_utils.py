
import numpy as np
import pandas as pd
import pytest

from src.data_utils import (
    COLUMNS,
    SENSORS,
    label_train,
    label_test,
    make_features,
    get_final_rows,
    nasa_score,
    evaluate,
)


def sample_data():
    rows = []

    for unit in [1, 2]:
        for cycle in [1, 2, 3]:
            row = {
                "unit": unit,
                "cycle": cycle,
                "setting_1": 0.1,
                "setting_2": 0.2,
                "setting_3": 100.0,
            }

            for sensor in SENSORS:
                row[sensor] = float(cycle * 10 + unit)

            rows.append(row)

    return pd.DataFrame(rows, columns=COLUMNS)


def test_schema():
    assert len(COLUMNS) == 26
    assert len(SENSORS) == 21


def test_label_train():
    result = label_train(sample_data(), cap=2)

    assert "RUL" in result.columns
    assert "RUL_raw" in result.columns
    assert result["RUL"].max() <= 2

    engine_1 = result[result.unit == 1]
    assert engine_1["RUL_raw"].tolist() == [2, 1, 0]


def test_label_test():
    result = label_test(
        sample_data(),
        np.array([10, 20])
    )

    engine_1 = result[result.unit == 1]
    engine_2 = result[result.unit == 2]

    assert engine_1["RUL"].tolist() == [12, 11, 10]
    assert engine_2["RUL"].tolist() == [22, 21, 20]


def test_label_test_rejects_wrong_rul_count():
    with pytest.raises(ValueError):
        label_test(sample_data(), np.array([10]))


def test_feature_generation():
    features = make_features(sample_data(), window=3)

    assert len(features) == 6
    assert features.shape[1] == 68
    assert "sensor_1_last" in features.columns
    assert "sensor_1_rollmean" in features.columns
    assert "sensor_1_rollstd" in features.columns


def test_rolling_mean_is_grouped_by_engine():
    features = make_features(sample_data(), window=3)

    # First cycle of engine 1
    first = features[
        (features.unit == 1) & (features.cycle == 1)
    ].iloc[0]

    # First cycle of engine 2
    second = features[
        (features.unit == 2) & (features.cycle == 1)
    ].iloc[0]

    assert first["sensor_1_rollmean"] == 11.0
    assert second["sensor_1_rollmean"] == 12.0


def test_final_rows():
    data = sample_data()
    indices = get_final_rows(data)

    final = data.loc[indices]

    assert set(final.unit) == {1, 2}
    assert all(final.cycle == 3)


def test_nasa_score():
    score = nasa_score(
        np.array([100, 100]),
        np.array([100, 110])
    )

    assert score > 0


def test_evaluate_metrics():
    result = evaluate(
        np.array([10, 20, 30]),
        np.array([12, 18, 30])
    )

    assert set(result.keys()) == {
        "MAE", "RMSE", "R2", "NASA_score"
    }

    assert result["MAE"] == pytest.approx(4 / 3)
    assert result["RMSE"] > 0
    assert np.isfinite(result["NASA_score"])
