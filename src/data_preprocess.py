from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PreparedSeries:
    data: pd.DataFrame
    date_col: str
    target_col: str
    frequency: str


def infer_frequency(dates: pd.Series) -> str:
    inferred = pd.infer_freq(pd.to_datetime(dates).sort_values())
    return inferred or "D"


def prepare_time_series(
    frame: pd.DataFrame,
    date_col: str,
    target_col: str,
    frequency: str | None = None,
) -> PreparedSeries:
    """清洗单目标时间序列，并补齐连续时间索引。"""
    if date_col not in frame.columns:
        raise ValueError(f"日期列 `{date_col}` 不存在。")
    if target_col not in frame.columns:
        raise ValueError(f"目标值列 `{target_col}` 不存在。")

    data = frame[[date_col, target_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[date_col, target_col])
    if data.empty:
        raise ValueError("日期列和目标值列解析后没有可用数据，请检查 CSV 格式。")

    data = (
        data.groupby(date_col, as_index=False)[target_col]
        .mean()
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    freq = frequency or infer_frequency(data[date_col])
    full_dates = pd.date_range(data[date_col].min(), data[date_col].max(), freq=freq)
    data = data.set_index(date_col).reindex(full_dates)
    data.index.name = date_col
    data[target_col] = data[target_col].interpolate(method="time").ffill().bfill()
    data = data.reset_index().rename(columns={"index": date_col})

    return PreparedSeries(data=data, date_col=date_col, target_col=target_col, frequency=freq)


def split_train_test(data: pd.DataFrame, target_col: str, test_size: int = 14):
    if len(data) <= test_size + 10:
        test_size = max(3, len(data) // 5)
    train = data.iloc[:-test_size].copy()
    test = data.iloc[-test_size:].copy()
    if train.empty or test.empty:
        raise ValueError("数据行数不足，无法划分训练集和测试集。")
    return train, test
