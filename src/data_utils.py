from pathlib import Path
import numpy as np
import pandas as pd

INDEX = ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
SETTINGS = ["setting_1", "setting_2", "setting_3"]
COLUMNS = INDEX + SENSORS

def load_subset(data_dir, subset):
    data_dir = Path(data_dir)
    def load(name):
        df = pd.read_csv(data_dir / name, sep=r"\s+", header=None)
        if df.shape[1] < 26:
            raise ValueError(f"{name}: expected at least 26 columns; found {df.shape[1]}")
        df = df.iloc[:, :26].copy()
        df.columns = COLUMNS
        return df
    train = load(f"train_{subset}.txt")
    test = load(f"test_{subset}.txt")
    rul = np.loadtxt(data_dir / f"RUL_{subset}.txt").reshape(-1)
    if test.unit.nunique() != len(rul):
        raise ValueError(f"{subset}: test engines and RUL labels do not match")
    train = train.sort_values(["unit", "cycle"]).reset_index(drop=True)
    test = test.sort_values(["unit", "cycle"]).reset_index(drop=True)
    return train, test, rul

def label_train(train, cap=125):
    out = train.copy()
    last = out.groupby("unit").cycle.transform("max")
    raw = last - out.cycle
    out["RUL_raw"] = raw
    out["RUL"] = np.minimum(raw, cap) if cap is not None else raw
    return out

def label_test(test, rul_vector):
    out = test.copy()
    units = np.sort(out.unit.unique())
    if len(units) != len(rul_vector):
        raise ValueError("Test unit count does not match supplied RUL vector")
    final_rul = dict(zip(units, np.asarray(rul_vector, dtype=float)))
    last = out.groupby("unit").cycle.transform("max")
    out["RUL"] = out.unit.map(final_rul) + (last - out.cycle)
    return out

def make_features(frame, window=30):
    """Causal rolling features; only current and preceding cycles are used."""
    frame = frame.sort_values(["unit", "cycle"]).copy()
    sensors = SENSORS
    roll_mean = frame.groupby("unit", sort=False)[sensors].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    roll_std = frame.groupby("unit", sort=False)[sensors].transform(
        lambda x: x.rolling(window, min_periods=2).std().fillna(0.0)
    )
    last = frame[sensors].copy()
    last.columns = [f"{c}_last" for c in sensors]
    roll_mean.columns = [f"{c}_rollmean" for c in sensors]
    roll_std.columns = [f"{c}_rollstd" for c in sensors]
    features = pd.concat([
        frame[["unit", "cycle"] + SETTINGS].reset_index(drop=True),
        last.reset_index(drop=True),
        roll_mean.reset_index(drop=True),
        roll_std.reset_index(drop=True)
    ], axis=1)
    return features

def get_final_rows(frame):
    return frame.groupby("unit").cycle.idxmax().to_numpy()

def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred) - np.asarray(y_true)
    terms = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(terms))

def evaluate(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "NASA_score": nasa_score(y_true, y_pred)
    }
