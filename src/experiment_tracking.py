from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pandas as pd

from .forecasting import ForecastResult, display_model_name


def compact_model_options(model_options: dict) -> dict:
    return {name: options for name, options in model_options.items() if options}


def build_experiment_record(
    *,
    dataset_name: str,
    target_col: str,
    horizon: int,
    test_size: int,
    selected_models: list[str],
    best_result: ForecastResult,
    elapsed_seconds: float,
    row_count: int,
    model_options: dict,
    run_type: str = "single",
) -> dict:
    metrics = best_result.metrics
    return {
        "experiment_id": uuid4().hex[:8],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_type": run_type,
        "dataset": dataset_name,
        "target_col": target_col,
        "row_count": row_count,
        "horizon": int(horizon),
        "test_size": int(test_size),
        "selected_models": ", ".join(display_model_name(name) for name in selected_models),
        "best_model": display_model_name(best_result.name),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "model_options": str(compact_model_options(model_options)),
    }


def experiments_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    metric_cols = [col for col in ["MAE", "RMSE", "MAPE", "elapsed_seconds"] if col in frame.columns]
    for col in metric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values(["created_at", "experiment_id"], ascending=False).reset_index(drop=True)
