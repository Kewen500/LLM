from __future__ import annotations

import pandas as pd

from src.data_preprocess import prepare_time_series, split_train_test
from src.exporters import dataframe_to_csv_bytes, markdown_to_docx_bytes, markdown_to_pdf_bytes
from src.forecasting import choose_best_model, run_selected_models_with_errors
from src.metrics import regression_metrics
from src.rag import retrieve_relevant_context, split_knowledge_text
from src.report_generator import ReportInput, generate_template_report
from src.report_quality import evaluate_report_quality, quality_score


def test_prepare_time_series_sorts_groups_and_interpolates():
    raw = pd.DataFrame(
        {
            "date": ["2025-01-03", "2025-01-01", "2025-01-01", "2025-01-04"],
            "sales": [30, 10, 20, 40],
        }
    )

    prepared = prepare_time_series(raw, "date", "sales")

    assert prepared.frequency == "D"
    assert len(prepared.data) == 4
    assert prepared.data["sales"].tolist() == [15.0, 22.5, 30.0, 40.0]


def test_baseline_models_complete_without_external_services():
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=40, freq="D"),
            "sales": [100 + index * 2 for index in range(40)],
        }
    )
    prepared = prepare_time_series(raw, "date", "sales")
    train, test = split_train_test(prepared.data, "sales", test_size=7)

    run = run_selected_models_with_errors(
        train=train,
        test=test,
        date_col="date",
        target_col="sales",
        horizon=7,
        freq=prepared.frequency,
        selected=["Moving Average", "Seasonal Naive", "Linear Trend"],
    )

    assert set(run.results) == {"Moving Average", "Seasonal Naive", "Linear Trend"}
    assert run.errors == {}


def test_metrics_and_exporters_return_downloadable_bytes():
    metrics = regression_metrics([100, 200, 300], [110, 190, 330])
    assert metrics["MAE"] == 16.6667

    csv_bytes = dataframe_to_csv_bytes(pd.DataFrame({"a": [1], "b": [2]}))
    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    assert b"a,b" in csv_bytes

    report = "# 中文报告\n\n## 摘要\n\n- 测试条目\n\n这是一段中文正文。"
    assert markdown_to_docx_bytes(report).startswith(b"PK")
    assert markdown_to_pdf_bytes(report).startswith(b"%PDF")


def test_rag_context_and_report_quality_checks():
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=40, freq="D"),
            "HUFL": [100 + index * 2 for index in range(40)],
        }
    )
    prepared = prepare_time_series(raw, "date", "HUFL")
    train, test = split_train_test(prepared.data, "HUFL", test_size=7)
    run = run_selected_models_with_errors(
        train=train,
        test=test,
        date_col="date",
        target_col="HUFL",
        horizon=7,
        freq=prepared.frequency,
        selected=["Moving Average", "Seasonal Naive", "Linear Trend"],
    )
    best = choose_best_model(run.results)
    chunks = split_knowledge_text("业务说明.md", "HUFL 表示高压有用负载。异常点需要结合设备检修记录判断。")
    context = retrieve_relevant_context(chunks, target_col="HUFL")
    report_input = ReportInput(
        date_col="date",
        target_col="HUFL",
        prepared_data=prepared.data,
        anomalies=pd.DataFrame(),
        best_result=best,
        all_results=run.results,
        knowledge_context=context,
    )

    report = generate_template_report(report_input)
    checks = evaluate_report_quality(report, report_input)

    assert "HUFL 表示高压有用负载" in context
    assert quality_score(checks) >= 80
