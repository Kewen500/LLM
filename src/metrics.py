from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": round(mae(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 4),
    }


def metrics_frame(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(results, orient="index").reset_index(names="model")

