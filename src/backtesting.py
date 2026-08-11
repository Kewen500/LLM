from __future__ import annotations

import pandas as pd

from .forecasting import display_model_name, run_selected_models_with_errors


def rolling_backtest(
    *,
    data: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    selected_models: list[str],
    model_options: dict,
    n_splits: int,
    test_size: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    try:
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError as exc:
        raise RuntimeError(
            "当前环境未安装 Scikit-learn，无法运行 Rolling Backtest。请执行：pip install scikit-learn"
        ) from exc

    splitter = TimeSeriesSplit(n_splits=int(n_splits), test_size=int(test_size))
    records = []
    errors = {}
    for fold, (train_indices, test_indices) in enumerate(splitter.split(data), start=1):
        train = data.iloc[train_indices].reset_index(drop=True)
        test = data.iloc[test_indices].reset_index(drop=True)
        try:
            model_run = run_selected_models_with_errors(
                train=train,
                test=test,
                date_col=date_col,
                target_col=target_col,
                horizon=horizon,
                freq=freq,
                selected=selected_models,
                model_options=model_options,
            )
            for model_name, result in model_run.results.items():
                records.append(
                    {
                        "fold": fold,
                        "model": display_model_name(model_name),
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "MAE": result.metrics["MAE"],
                        "RMSE": result.metrics["RMSE"],
                        "MAPE": result.metrics["MAPE"],
                    }
                )
            errors.update(
                {
                    f"fold={fold}, {display_model_name(model_name)}": message
                    for model_name, message in model_run.errors.items()
                }
            )
        except Exception as exc:
            errors[f"fold={fold}"] = str(exc)
    return pd.DataFrame(records), errors


def summarize_backtest(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    return (
        records.groupby("model")
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAPE_mean=("MAPE", "mean"),
            MAPE_std=("MAPE", "std"),
        )
        .reset_index()
    )
