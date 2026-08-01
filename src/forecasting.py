from __future__ import annotations

from dataclasses import dataclass
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import regression_metrics


@dataclass
class ForecastResult:
    name: str
    fitted: pd.DataFrame
    future: pd.DataFrame
    metrics: dict[str, float]


@dataclass
class ModelRun:
    results: dict[str, ForecastResult]
    errors: dict[str, str]


MODEL_DISPLAY_NAMES = {
    "Moving Average": "Moving Average",
    "Seasonal Naive": "Seasonal Naive",
    "Linear Trend": "Linear Trend",
    "ARIMA": "ARIMA",
    "Prophet": "Prophet",
    "Prophet-like Decomposition": "Prophet-like Decomposition",
    "LSTM": "LSTM",
}


def display_model_name(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name)


def future_dates(last_date, periods: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.to_datetime(last_date), periods=periods + 1, freq=freq)[1:]


def dayofweek_values(dates) -> np.ndarray:
    parsed = pd.to_datetime(dates)
    if isinstance(parsed, pd.Series):
        return parsed.dt.dayofweek.to_numpy()
    return parsed.dayofweek


def _result(name, date_col, target_col, test, test_pred, future_index, future_values) -> ForecastResult:
    fitted = pd.DataFrame(
        {
            date_col: test[date_col].to_numpy(),
            "actual": test[target_col].to_numpy(),
            "prediction": np.asarray(test_pred, dtype=float),
        }
    )
    future = pd.DataFrame({date_col: future_index, "prediction": np.asarray(future_values, dtype=float)})
    return ForecastResult(
        name=name,
        fitted=fitted,
        future=future,
        metrics=regression_metrics(fitted["actual"], fitted["prediction"]),
    )


def moving_average_forecast(train, test, date_col, target_col, horizon, freq, window=7) -> ForecastResult:
    history = train[target_col].astype(float).tolist()
    test_pred = []
    for _ in range(len(test)):
        pred = float(np.mean(history[-window:]))
        test_pred.append(pred)
        history.append(pred)

    full_history = train[target_col].astype(float).tolist() + test[target_col].astype(float).tolist()
    future_values = []
    generated = full_history[:]
    for _ in range(horizon):
        pred = float(np.mean(generated[-window:]))
        future_values.append(pred)
        generated.append(pred)

    return _result(
        "Moving Average",
        date_col,
        target_col,
        test,
        test_pred,
        future_dates(test[date_col].iloc[-1], horizon, freq),
        future_values,
    )


def seasonal_naive_forecast(train, test, date_col, target_col, horizon, freq, season_length=7) -> ForecastResult:
    values = train[target_col].astype(float).to_numpy()
    if len(values) < season_length:
        season_length = max(1, len(values))
    pattern = values[-season_length:]
    test_pred = [float(pattern[i % season_length]) for i in range(len(test))]

    combined = pd.concat([train[[target_col]], test[[target_col]]], ignore_index=True)
    pattern = combined[target_col].astype(float).to_numpy()[-season_length:]
    future_values = [float(pattern[i % season_length]) for i in range(horizon)]

    return _result(
        "Seasonal Naive",
        date_col,
        target_col,
        test,
        test_pred,
        future_dates(test[date_col].iloc[-1], horizon, freq),
        future_values,
    )


def linear_trend_forecast(train, test, date_col, target_col, horizon, freq) -> ForecastResult:
    x_train = np.arange(len(train))
    y_train = train[target_col].astype(float).to_numpy()
    slope, intercept = np.polyfit(x_train, y_train, 1)
    test_x = np.arange(len(train), len(train) + len(test))
    test_pred = intercept + slope * test_x

    all_x = np.arange(len(train) + len(test))
    all_y = pd.concat([train[target_col], test[target_col]], ignore_index=True).astype(float).to_numpy()
    slope, intercept = np.polyfit(all_x, all_y, 1)
    future_x = np.arange(len(all_y), len(all_y) + horizon)
    future_values = intercept + slope * future_x

    return _result(
        "Linear Trend",
        date_col,
        target_col,
        test,
        test_pred,
        future_dates(test[date_col].iloc[-1], horizon, freq),
        future_values,
    )


def prophet_like_decomposition_forecast(train, test, date_col, target_col, horizon, freq) -> ForecastResult:
    """Prophet 不可用时使用的趋势与星期周期分解后备模型。"""
    x_train = np.arange(len(train))
    y_train = train[target_col].astype(float).to_numpy()
    slope, intercept = np.polyfit(x_train, y_train, 1)
    train_trend = intercept + slope * x_train
    residuals = y_train - train_trend

    seasonal = (
        pd.DataFrame(
            {
                "dayofweek": dayofweek_values(train[date_col]),
                "residual": residuals,
            }
        )
        .groupby("dayofweek")["residual"]
        .mean()
        .to_dict()
    )

    def predict_for_dates(dates, start_x):
        xs = np.arange(start_x, start_x + len(dates))
        trend = intercept + slope * xs
        day_offsets = np.asarray([seasonal.get(int(day), 0.0) for day in dayofweek_values(dates)])
        return trend + day_offsets

    test_pred = predict_for_dates(test[date_col], len(train))

    combined = pd.concat([train, test], ignore_index=True)
    x_all = np.arange(len(combined))
    y_all = combined[target_col].astype(float).to_numpy()
    slope, intercept = np.polyfit(x_all, y_all, 1)
    residuals = y_all - (intercept + slope * x_all)
    seasonal = (
        pd.DataFrame(
            {
                "dayofweek": dayofweek_values(combined[date_col]),
                "residual": residuals,
            }
        )
        .groupby("dayofweek")["residual"]
        .mean()
        .to_dict()
    )
    future_index = future_dates(test[date_col].iloc[-1], horizon, freq)
    future_values = predict_for_dates(future_index, len(combined))
    return _result(
        "Prophet-like Decomposition",
        date_col,
        target_col,
        test,
        test_pred,
        future_index,
        future_values,
    )


def make_supervised(values: np.ndarray, lookback: int):
    x, y = [], []
    for idx in range(lookback, len(values)):
        x.append(values[idx - lookback : idx])
        y.append(values[idx])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def lstm_forecast(
    train,
    test,
    date_col,
    target_col,
    horizon,
    freq,
    lookback=14,
    epochs=80,
    hidden_size=24,
) -> ForecastResult:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("当前环境未安装 torch，无法运行 LSTM。请执行：pip install torch") from exc

    series = pd.concat([train[target_col], test[target_col]], ignore_index=True).astype(float).to_numpy()
    train_values = train[target_col].astype(float).to_numpy()
    if len(train_values) <= lookback + 5:
        lookback = max(3, min(lookback, len(train_values) // 3))
    if len(train_values) <= lookback:
        raise RuntimeError("训练数据行数不足，无法运行 LSTM。")

    min_value = float(train_values.min())
    max_value = float(train_values.max())
    scale = max(max_value - min_value, 1e-8)

    def normalize(values):
        return (np.asarray(values, dtype=np.float32) - min_value) / scale

    def denormalize(values):
        return np.asarray(values, dtype=float) * scale + min_value

    train_scaled = normalize(train_values)
    x_train, y_train = make_supervised(train_scaled, lookback)
    x_tensor = torch.tensor(x_train).unsqueeze(-1)
    y_tensor = torch.tensor(y_train).unsqueeze(-1)

    class LSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, inputs):
            output, _ = self.lstm(inputs)
            return self.head(output[:, -1, :])

    torch.manual_seed(42)
    model = LSTMRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    def recursive_predict(history_values, steps):
        history = normalize(history_values).astype(np.float32).tolist()
        predictions = []
        model.eval()
        with torch.no_grad():
            for _ in range(steps):
                window = torch.tensor(history[-lookback:], dtype=torch.float32).view(1, lookback, 1)
                pred_scaled = float(model(window).item())
                predictions.append(pred_scaled)
                history.append(pred_scaled)
        return denormalize(predictions)

    test_pred = recursive_predict(train_values, len(test))
    future_values = recursive_predict(series, horizon)
    return _result(
        "LSTM",
        date_col,
        target_col,
        test,
        test_pred,
        future_dates(test[date_col].iloc[-1], horizon, freq),
        future_values,
    )


def arima_forecast(train, test, date_col, target_col, horizon, freq, p=2, d=1, q=2) -> ForecastResult:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise RuntimeError("当前环境未安装 statsmodels，无法运行 ARIMA。请执行：pip install statsmodels") from exc

    order = (int(p), int(d), int(q))
    model = ARIMA(train[target_col].astype(float), order=order).fit()
    test_pred = model.forecast(steps=len(test)).to_numpy()
    combined = pd.concat([train[target_col], test[target_col]], ignore_index=True).astype(float)
    future_model = ARIMA(combined, order=order).fit()
    future_values = future_model.forecast(steps=horizon).to_numpy()
    return _result(
        "ARIMA",
        date_col,
        target_col,
        test,
        test_pred,
        future_dates(test[date_col].iloc[-1], horizon, freq),
        future_values,
    )


def prophet_forecast(train, test, date_col, target_col, horizon, freq) -> ForecastResult:
    if platform.system() == "Windows":
        return prophet_like_decomposition_forecast(train, test, date_col, target_col, horizon, freq)

    matplotlib_dir = Path.cwd() / ".matplotlib"
    matplotlib_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise RuntimeError("当前环境未安装 prophet，无法运行 Prophet。请执行：pip install prophet") from exc

    try:
        prophet_train = train[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
        model = Prophet()
        model.fit(prophet_train)
        test_future = pd.DataFrame({"ds": test[date_col]})
        test_pred = model.predict(test_future)["yhat"].to_numpy()

        all_data = pd.concat([train, test], ignore_index=True)
        final_train = all_data[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
        final_model = Prophet()
        final_model.fit(final_train)
        future_index = future_dates(test[date_col].iloc[-1], horizon, freq)
        future_values = final_model.predict(pd.DataFrame({"ds": future_index}))["yhat"].to_numpy()
        return _result("Prophet", date_col, target_col, test, test_pred, future_index, future_values)
    except Exception:
        return prophet_like_decomposition_forecast(train, test, date_col, target_col, horizon, freq)



def run_selected_models(
    train,
    test,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    selected: list[str],
    model_options: dict | None = None,
) -> dict[str, ForecastResult]:
    run = run_selected_models_with_errors(train, test, date_col, target_col, horizon, freq, selected, model_options)
    if not run.results:
        raise RuntimeError(f"没有任何模型成功完成运行。错误信息：{run.errors}")
    return run.results


def run_selected_models_with_errors(
    train,
    test,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    selected: list[str],
    model_options: dict | None = None,
) -> ModelRun:
    model_options = model_options or {}
    runners = {
        "Moving Average": moving_average_forecast,
        "Seasonal Naive": seasonal_naive_forecast,
        "Linear Trend": linear_trend_forecast,
        "ARIMA": arima_forecast,
        "Prophet-like Decomposition": prophet_like_decomposition_forecast,
        "Prophet": prophet_forecast,
        "LSTM": lstm_forecast,
    }
    results = {}
    errors = {}
    for name in selected:
        try:
            options = model_options.get(name, {})
            forecast = runners[name](train, test, date_col, target_col, horizon, freq, **options)
            results[forecast.name] = forecast
        except Exception as exc:
            errors[name] = str(exc)
    if not results:
        raise RuntimeError(f"没有任何模型成功完成运行。错误信息：{errors}")
    return ModelRun(results=results, errors=errors)


def choose_best_model(results: dict[str, ForecastResult], metric: str = "MAPE") -> ForecastResult:
    return sorted(results.values(), key=lambda result: result.metrics.get(metric, float("inf")))[0]
