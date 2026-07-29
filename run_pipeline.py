from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.anomaly_detection import detect_anomalies
from src.data_preprocess import prepare_time_series, split_train_test
from src.forecasting import choose_best_model, display_model_name, run_selected_models_with_errors
from src.metrics import metrics_frame
from src.report_generator import ReportInput, generate_template_report


def main():
    raw = pd.read_csv(Path(__file__).parent / "data" / "sample_sales.csv")
    prepared = prepare_time_series(raw, "date", "sales")
    train, test = split_train_test(prepared.data, "sales", test_size=14)
    model_run = run_selected_models_with_errors(
        train=train,
        test=test,
        date_col="date",
        target_col="sales",
        horizon=30,
        freq=prepared.frequency,
        selected=["Moving Average", "Seasonal Naive", "Linear Trend", "ARIMA", "Prophet", "LSTM"],
    )
    results = model_run.results
    best = choose_best_model(results)
    anomalies = detect_anomalies(prepared.data, "date", "sales")
    report = generate_template_report(
        ReportInput(
            date_col="date",
            target_col="sales",
            prepared_data=prepared.data,
            anomalies=anomalies,
            best_result=best,
            all_results=results,
        )
    )
    metric_table = metrics_frame({display_model_name(name): result.metrics for name, result in results.items()})
    metric_table = metric_table.rename(columns={"model": "模型"})
    print("最佳模型：", display_model_name(best.name))
    print(metric_table)
    if model_run.errors:
        print("已跳过的模型：", {display_model_name(name): message for name, message in model_run.errors.items()})
    print(report[:900])


if __name__ == "__main__":
    main()
