from __future__ import annotations

import pandas as pd


def detect_anomalies(
    data: pd.DataFrame,
    date_col: str,
    target_col: str,
    window: int = 14,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Flag points outside rolling mean +/- threshold * rolling std."""
    frame = data[[date_col, target_col]].copy()
    rolling = frame[target_col].rolling(window=window, min_periods=max(3, window // 2))
    frame["rolling_mean"] = rolling.mean()
    frame["rolling_std"] = rolling.std().fillna(0)
    frame["z_score"] = (frame[target_col] - frame["rolling_mean"]) / frame["rolling_std"].replace(0, pd.NA)
    anomalies = frame[frame["z_score"].abs() >= threshold].copy()
    return anomalies[[date_col, target_col, "rolling_mean", "z_score"]].tail(20)

