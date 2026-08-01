from __future__ import annotations

import numpy as np
import pandas as pd

from .forecasting import ForecastResult


def fitted_diagnostics_frame(result: ForecastResult) -> pd.DataFrame:
    frame = result.fitted.copy()
    frame["residual"] = frame["actual"] - frame["prediction"]
    frame["absolute_error"] = frame["residual"].abs()
    actual = frame["actual"].replace(0, np.nan)
    frame["absolute_percentage_error"] = (frame["absolute_error"] / actual).abs() * 100
    return frame


def residual_summary_frame(result: ForecastResult) -> pd.DataFrame:
    diagnostics = fitted_diagnostics_frame(result)
    residuals = diagnostics["residual"]
    return pd.DataFrame(
        {
            "metric": ["mean", "std", "min", "median", "max"],
            "value": [
                residuals.mean(),
                residuals.std(ddof=0),
                residuals.min(),
                residuals.median(),
                residuals.max(),
            ],
        }
    )


def future_forecast_frame(result: ForecastResult) -> pd.DataFrame:
    return result.future.copy()
