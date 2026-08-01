from __future__ import annotations

import time
import importlib.util
import platform
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.anomaly_detection import detect_anomalies
from src.data_preprocess import prepare_time_series, split_train_test
from src.diagnostics import fitted_diagnostics_frame, future_forecast_frame, residual_summary_frame
from src.experiment_tracking import build_experiment_record, experiments_frame
from src.exporters import dataframe_to_csv_bytes, markdown_to_docx_bytes, markdown_to_pdf_bytes
from src.forecasting import MODEL_DISPLAY_NAMES, choose_best_model, display_model_name, run_selected_models_with_errors
from src.history_store import save_analysis_run
from src.llm_client import generate_openai_compatible_report
from src.llm_presets import (
    CUSTOM_MODEL_LABEL,
    CUSTOM_PROVIDER_LABEL,
    LLM_PROVIDER_PRESETS,
    load_custom_llm_presets,
    save_custom_llm_preset,
)
from src.metrics import metrics_frame
from src.rag import read_knowledge_file, retrieve_relevant_context, split_knowledge_text
from src.report_quality import evaluate_report_quality, quality_checks_frame, quality_score
from src.report_generator import ReportInput, build_llm_prompt, generate_template_report


st.set_page_config(page_title="时间序列预测与 LLM 报告生成", layout="wide")

FORECAST_MODEL_OPTIONS = [
    "Moving Average",
    "Seasonal Naive",
    "Linear Trend",
    "ARIMA",
    "Prophet-like Decomposition",
    "Prophet",
    "LSTM",
]
MODEL_DEPENDENCIES = {
    "ARIMA": ("statsmodels", "statsmodels"),
    "Prophet": ("prophet", "prophet"),
    "LSTM": ("torch", "torch"),
}
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


def load_default_data() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "data" / "sample_sales.csv")


def load_uploaded_or_sample(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return load_default_data()
    return pd.read_csv(uploaded_file)


def profile_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "字段名": frame.columns,
            "数据类型": [str(frame[column].dtype) for column in frame.columns],
            "非空数量": [int(frame[column].notna().sum()) for column in frame.columns],
            "缺失数量": [int(frame[column].isna().sum()) for column in frame.columns],
            "缺失率(%)": [round(float(frame[column].isna().mean() * 100), 2) for column in frame.columns],
        }
    )


def build_chart_frame(prepared, best_result):
    actual = prepared.data[[prepared.date_col, prepared.target_col]].rename(
        columns={prepared.date_col: "日期", prepared.target_col: "实际值"}
    )
    future = best_result.future.rename(columns={prepared.date_col: "日期"})
    future["实际值"] = None
    future = future.rename(columns={"prediction": "预测值"})
    chart = actual.merge(future[["日期", "预测值"]], on="日期", how="outer")
    return chart.set_index("日期")


def build_forecast_export_frame(prepared, best_result):
    fitted = best_result.fitted.rename(columns={"actual": "实际值", "prediction": "预测值"}).copy()
    fitted["数据区间"] = "测试集"
    future = best_result.future.rename(columns={"prediction": "预测值"}).copy()
    future["实际值"] = pd.NA
    future["数据区间"] = "未来预测"
    export_frame = pd.concat(
        [
            fitted[[prepared.date_col, "数据区间", "实际值", "预测值"]],
            future[[prepared.date_col, "数据区间", "实际值", "预测值"]],
        ],
        ignore_index=True,
    )
    return export_frame.rename(columns={prepared.date_col: "日期"})


def localize_metrics_table(metric_table: pd.DataFrame) -> pd.DataFrame:
    localized = metric_table.copy()
    if "model" in localized.columns:
        localized["model"] = localized["model"].map(display_model_name)
        localized = localized.rename(columns={"model": "模型"})
    return localized


def localize_model_errors(errors: dict[str, str]) -> dict[str, str]:
    return {display_model_name(name): message for name, message in errors.items()}


def localize_anomaly_table(anomalies: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    if anomalies.empty:
        return anomalies
    return anomalies.rename(
        columns={
            date_col: "日期",
            target_col: "目标值",
            "rolling_mean": "滚动均值",
            "z_score": "标准化偏差",
        }
    )


def available_llm_presets():
    if "custom_llm_presets" not in st.session_state:
        st.session_state["custom_llm_presets"] = load_custom_llm_presets()
    custom_presets = st.session_state["custom_llm_presets"]
    return {**LLM_PROVIDER_PRESETS, **custom_presets}


def render_llm_api_settings(default_max_tokens: int):
    presets = available_llm_presets()
    provider_options = list(presets.keys()) + [CUSTOM_PROVIDER_LABEL]
    provider = st.selectbox("API 服务商", provider_options)

    if provider == CUSTOM_PROVIDER_LABEL:
        api_url = st.text_input("自定义 API URL", value="https://api.example.com/v1")
        model_name = st.text_input("自定义 Model Name", value="")
        preset_name = st.text_input("保存为预设名称（可选）", value="")
        if st.button("保存为本地预设"):
            if api_url.strip() and model_name.strip():
                name = preset_name.strip() or f"自定义 - {model_name.strip()}"
                st.session_state["custom_llm_presets"] = save_custom_llm_preset(
                    name=name,
                    api_url=api_url.strip(),
                    model_name=model_name.strip(),
                )
                st.success("已保存为本地预设，刷新或重启后仍可选用。")
            else:
                st.warning("请先填写 API URL 和 Model Name。")
    else:
        preset = presets[provider]
        st.caption(preset.get("note", ""))
        api_url = st.text_input("API URL", value=preset["api_url"])
        model_options = list(dict.fromkeys(preset["models"] + [CUSTOM_MODEL_LABEL]))
        model_choice = st.selectbox("Model Name", model_options)
        if model_choice == CUSTOM_MODEL_LABEL:
            model_name = st.text_input("自定义 Model Name", value="")
        else:
            model_name = model_choice

    api_key = st.text_input("API Key", type="password")
    max_tokens = st.slider("报告最大输出长度（token）", min_value=600, max_value=3000, value=default_max_tokens, step=200)
    show_prompt = st.checkbox("显示发送给 LLM 的 Prompt", value=False)
    return api_url, model_name, api_key, max_tokens, show_prompt


def render_knowledge_settings():
    with st.expander("RAG 知识库（可选）", expanded=False):
        st.caption("项目会自动读取 knowledge/ 目录中的 .md、.txt、.csv 文件；也可以临时上传或粘贴知识。")
        use_local_knowledge = st.checkbox("使用项目内置知识库", value=True)
        if KNOWLEDGE_DIR.exists():
            local_files = sorted(path.name for path in KNOWLEDGE_DIR.iterdir() if path.suffix.lower() in {".md", ".txt", ".csv"})
            if local_files:
                st.caption("已发现：" + "、".join(local_files))
        knowledge_files = st.file_uploader(
            "上传业务知识文件",
            type=["txt", "md", "csv"],
            accept_multiple_files=True,
        )
        manual_knowledge = st.text_area(
            "粘贴业务知识",
            value="",
            height=120,
            placeholder="例如：HUFL 表示高压负载特征，节假日前后可能出现波动；异常点需要结合检修记录判断。",
        )
    return use_local_knowledge, knowledge_files, manual_knowledge


def render_history_settings():
    with st.expander("历史记录保存（Supabase，可选）", expanded=False):
        enabled = st.checkbox("保存本次分析到 Supabase", value=False)
        supabase_url = st.text_input("Supabase Project URL", value="", placeholder="https://xxxx.supabase.co")
        supabase_key = st.text_input("Supabase anon key", value="", type="password")
        table_name = st.text_input("表名", value="analysis_runs")
        st.caption("先在 Supabase SQL Editor 执行项目里的 supabase-schema.sql。不要在公开仓库中保存 API Key。")
    return enabled, supabase_url, supabase_key, table_name


def load_local_knowledge_chunks():
    chunks = []
    if not KNOWLEDGE_DIR.exists():
        return chunks
    for path in sorted(KNOWLEDGE_DIR.iterdir()):
        if path.suffix.lower() not in {".md", ".txt", ".csv"}:
            continue
        text = read_knowledge_file(path.name, path.read_bytes())
        chunks.extend(split_knowledge_text(path.name, text))
    return chunks


def build_knowledge_context(use_local_knowledge: bool, knowledge_files, manual_knowledge: str, target_col: str) -> str:
    chunks = []
    if use_local_knowledge:
        chunks.extend(load_local_knowledge_chunks())
    for uploaded in knowledge_files or []:
        text = read_knowledge_file(uploaded.name, uploaded.getvalue())
        chunks.extend(split_knowledge_text(uploaded.name, text))
    if manual_knowledge.strip():
        chunks.extend(split_knowledge_text("手动输入", manual_knowledge))
    return retrieve_relevant_context(chunks, target_col=target_col, extra_terms=manual_knowledge)


def is_model_available(model_name: str) -> bool:
    if model_name == "Prophet":
        return platform.system() != "Windows" and importlib.util.find_spec("prophet") is not None
    dependency = MODEL_DEPENDENCIES.get(model_name)
    if dependency is None:
        return True
    package_name, _ = dependency
    return importlib.util.find_spec(package_name) is not None


def available_forecast_models() -> list[str]:
    return [model for model in FORECAST_MODEL_OPTIONS if is_model_available(model)]


def unavailable_forecast_models() -> list[str]:
    return [model for model in FORECAST_MODEL_OPTIONS if not is_model_available(model)]


def default_selected_models() -> list[str]:
    return available_forecast_models()


def render_model_dependency_notice(missing_models: list[str]) -> None:
    if not missing_models:
        return
    details = []
    for model in missing_models:
        _, pip_name = MODEL_DEPENDENCIES[model]
        details.append(f"{model} 需要 `{pip_name}`")
    st.info(
        "当前环境未安装部分高级模型依赖，已自动隐藏："
        + "；".join(details)
        + "。本地完整安装可执行：`pip install -r requirements-advanced.txt`。"
    )


def render_model_options(selected_models: list[str]) -> dict:
    options = {}
    with st.expander("Model 参数配置", expanded=False):
        st.caption("参数只会传给本次选中的模型；批量实验会复用同一组参数。")
        if "Moving Average" in selected_models:
            options["Moving Average"] = {
                "window": st.slider("Moving Average window", min_value=2, max_value=60, value=7, step=1)
            }
        if "Seasonal Naive" in selected_models:
            options["Seasonal Naive"] = {
                "season_length": st.slider("Seasonal Naive season_length", min_value=2, max_value=60, value=7, step=1)
            }
        if "ARIMA" in selected_models:
            c1, c2, c3 = st.columns(3)
            options["ARIMA"] = {
                "p": c1.number_input("ARIMA p", min_value=0, max_value=5, value=2, step=1),
                "d": c2.number_input("ARIMA d", min_value=0, max_value=2, value=1, step=1),
                "q": c3.number_input("ARIMA q", min_value=0, max_value=5, value=2, step=1),
            }
        if "LSTM" in selected_models:
            c1, c2, c3 = st.columns(3)
            options["LSTM"] = {
                "lookback": c1.slider("LSTM lookback", min_value=3, max_value=60, value=14, step=1),
                "epochs": c2.slider("LSTM epochs", min_value=10, max_value=200, value=80, step=10),
                "hidden_size": c3.slider("LSTM hidden_size", min_value=8, max_value=128, value=24, step=8),
            }
        if not options:
            st.info("当前选中的模型没有可配置参数。")
    return options


def parse_int_list(raw_text: str, fallback: list[int]) -> list[int]:
    values = []
    for item in re.split(r"[,，\s]+", raw_text.strip()):
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values or fallback


def append_experiment_record(record: dict) -> None:
    if "experiment_records" not in st.session_state:
        st.session_state["experiment_records"] = []
    st.session_state["experiment_records"].append(record)


def run_batch_experiments(
    *,
    raw: pd.DataFrame,
    date_col: str,
    target_col: str,
    dataset_name: str,
    horizons: list[int],
    test_sizes: list[int],
    selected_models: list[str],
    model_options: dict,
) -> tuple[pd.DataFrame, dict[str, str]]:
    records = []
    errors = {}
    prepared = prepare_time_series(raw, date_col, target_col)
    for batch_horizon in horizons:
        for batch_test_size in test_sizes:
            started_at = time.perf_counter()
            try:
                train, test = split_train_test(prepared.data, target_col, test_size=batch_test_size)
                model_run = run_selected_models_with_errors(
                    train=train,
                    test=test,
                    date_col=prepared.date_col,
                    target_col=prepared.target_col,
                    horizon=batch_horizon,
                    freq=prepared.frequency,
                    selected=selected_models,
                    model_options=model_options,
                )
                best = choose_best_model(model_run.results)
                records.append(
                    build_experiment_record(
                        dataset_name=dataset_name,
                        target_col=prepared.target_col,
                        horizon=batch_horizon,
                        test_size=batch_test_size,
                        selected_models=selected_models,
                        best_result=best,
                        elapsed_seconds=time.perf_counter() - started_at,
                        row_count=len(prepared.data),
                        model_options=model_options,
                        run_type="batch",
                    )
                )
                errors.update({f"h={batch_horizon}, test={batch_test_size}, {name}": msg for name, msg in model_run.errors.items()})
            except Exception as exc:
                errors[f"h={batch_horizon}, test={batch_test_size}"] = str(exc)
    return experiments_frame(records), errors


st.title("时间序列预测与 LLM 自动分析报告生成系统")

with st.sidebar:
    st.header("数据与参数")
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    raw = load_uploaded_or_sample(uploaded_file)
    st.caption(f"已解析：{raw.shape[0]:,} 行 × {raw.shape[1]:,} 列")
    date_col = st.selectbox("日期列", raw.columns, index=0)
    numeric_candidates = raw.select_dtypes(include="number").columns.tolist()
    default_target_index = raw.columns.get_loc(numeric_candidates[0]) if numeric_candidates else min(1, len(raw.columns) - 1)
    target_col = st.selectbox("目标值列", raw.columns, index=default_target_index)
    horizon = st.slider("预测周期", min_value=7, max_value=90, value=30, step=7)
    test_size = st.slider("测试集长度", min_value=7, max_value=60, value=14, step=7)
    available_models = available_forecast_models()
    render_model_dependency_notice(unavailable_forecast_models())
    selected_models = st.multiselect(
        "选择模型",
        available_models,
        default=default_selected_models(),
        format_func=display_model_name,
    )
    model_options = render_model_options(selected_models)
    with st.expander("批量实验", expanded=False):
        batch_horizons_text = st.text_input("预测周期组合", value=str(horizon))
        batch_test_sizes_text = st.text_input("测试集长度组合", value=str(test_size))
        run_batch_button = st.button("运行批量实验")
    report_mode = st.radio("报告生成方式", ["本地模板报告", "LLM API"], horizontal=True)
    api_url = "https://api.deepseek.com"
    model_name = "deepseek-v4-flash"
    api_key = ""
    max_tokens = 1600
    show_prompt = False
    if report_mode == "LLM API":
        api_url, model_name, api_key, max_tokens, show_prompt = render_llm_api_settings(max_tokens)
    use_local_knowledge, knowledge_files, manual_knowledge = render_knowledge_settings()
    save_history, supabase_url, supabase_key, history_table = render_history_settings()
    run_button = st.button("开始分析", type="primary")

st.subheader("原始数据预览")
info_cols = st.columns(4)
info_cols[0].metric("数据行数", f"{raw.shape[0]:,}")
info_cols[1].metric("数据列数", f"{raw.shape[1]:,}")
info_cols[2].metric("数值列", len(raw.select_dtypes(include="number").columns))
info_cols[3].metric("缺失值", f"{int(raw.isna().sum().sum()):,}")
with st.expander("查看字段解析结果", expanded=False):
    st.dataframe(profile_dataframe(raw), width="stretch")
st.dataframe(raw.head(20), width="stretch")

dataset_name = uploaded_file.name if uploaded_file else "sample_sales.csv"

if run_batch_button:
    if not selected_models:
        st.warning("请至少选择一个模型后再运行批量实验。")
    else:
        batch_horizons = parse_int_list(batch_horizons_text, [horizon])
        batch_test_sizes = parse_int_list(batch_test_sizes_text, [test_size])
        batch_frame, batch_errors = run_batch_experiments(
            raw=raw,
            date_col=date_col,
            target_col=target_col,
            dataset_name=dataset_name,
            horizons=batch_horizons,
            test_sizes=batch_test_sizes,
            selected_models=selected_models,
            model_options=model_options,
        )
        for record in batch_frame.to_dict(orient="records"):
            append_experiment_record(record)
        st.success(f"批量实验完成：{len(batch_frame)} 条记录。")
        st.dataframe(batch_frame, width="stretch")
        if batch_errors:
            with st.expander("批量实验错误详情", expanded=False):
                st.json(batch_errors)

if run_button or uploaded_file is None:
    try:
        run_started_at = time.perf_counter()
        prepared = prepare_time_series(raw, date_col, target_col)
        train, test = split_train_test(prepared.data, target_col, test_size=test_size)
        model_run = run_selected_models_with_errors(
            train=train,
            test=test,
            date_col=prepared.date_col,
            target_col=prepared.target_col,
            horizon=horizon,
            freq=prepared.frequency,
            selected=selected_models,
            model_options=model_options,
        )
        results = model_run.results
        best = choose_best_model(results)
        anomalies = detect_anomalies(prepared.data, prepared.date_col, prepared.target_col)
        knowledge_context = build_knowledge_context(use_local_knowledge, knowledge_files, manual_knowledge, prepared.target_col)
        report_input = ReportInput(
            date_col=prepared.date_col,
            target_col=prepared.target_col,
            prepared_data=prepared.data,
            anomalies=anomalies,
            best_result=best,
            all_results=results,
            knowledge_context=knowledge_context,
        )
        prompt = build_llm_prompt(report_input)
        template_report = generate_template_report(report_input)
        report = template_report
        report_source = "本地模板报告"
        llm_error = None
        if report_mode == "LLM API":
            try:
                report = generate_openai_compatible_report(
                    prompt=prompt,
                    api_key=api_key,
                    model=model_name,
                    api_url=api_url,
                    max_tokens=max_tokens,
                )
                report_source = f"LLM 生成报告：{model_name}"
            except Exception as exc:
                llm_error = str(exc)
        quality_checks = evaluate_report_quality(report, report_input)
        quality_table = quality_checks_frame(quality_checks)
        quality_pass_rate = quality_score(quality_checks)

        elapsed_seconds = time.perf_counter() - run_started_at
        if run_button:
            append_experiment_record(
                build_experiment_record(
                    dataset_name=dataset_name,
                    target_col=prepared.target_col,
                    horizon=horizon,
                    test_size=test_size,
                    selected_models=selected_models,
                    best_result=best,
                    elapsed_seconds=elapsed_seconds,
                    row_count=len(prepared.data),
                    model_options=model_options,
                    run_type="single",
                )
            )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("最佳模型", display_model_name(best.name))
        c2.metric("MAE", best.metrics["MAE"])
        c3.metric("RMSE", best.metrics["RMSE"])
        c4.metric("MAPE", f"{best.metrics['MAPE']}%")
        c5.metric("运行时间", f"{elapsed_seconds:.2f}s")

        st.subheader("预测曲线")
        st.line_chart(build_chart_frame(prepared, best), width="stretch")

        st.subheader("模型评估对比")
        metric_table = metrics_frame({name: result.metrics for name, result in results.items()})
        localized_metric_table = localize_metrics_table(metric_table)
        st.dataframe(localized_metric_table, width="stretch")
        if model_run.errors:
            st.warning("部分模型运行失败，系统已跳过。")
            st.json(localize_model_errors(model_run.errors))

        st.subheader("异常点")
        if anomalies.empty:
            st.info("未检测到明显异常点。")
        else:
            localized_anomalies = localize_anomaly_table(anomalies, prepared.date_col, prepared.target_col)
            st.dataframe(localized_anomalies, width="stretch")

        st.subheader("自动分析报告")
        st.caption(report_source)
        if knowledge_context:
            with st.expander("本次检索到的业务知识", expanded=False):
                st.text(knowledge_context)
        if llm_error:
            st.warning("LLM 报告生成失败，已自动回退为本地模板报告。")
            st.code(llm_error)
        if show_prompt:
            with st.expander("发送给 LLM 的 Prompt", expanded=False):
                st.text_area("Prompt", value=prompt, height=260)
                st.download_button(
                    "下载 Prompt",
                    data=prompt.encode("utf-8"),
                    file_name="llm_prompt.txt",
                    mime="text/plain",
                )
        st.markdown(report)

        st.subheader("报告事实一致性校验")
        score_cols = st.columns(2)
        score_cols[0].metric("校验通过率", f"{quality_pass_rate}%")
        score_cols[1].metric("检查项数量", len(quality_checks))
        st.dataframe(quality_table, width="stretch")

        diagnostic_tab, experiment_tab = st.tabs(["可视化诊断", "实验记录"])
        with diagnostic_tab:
            diagnostic_model = st.selectbox(
                "诊断模型",
                list(results.keys()),
                index=list(results.keys()).index(best.name) if best.name in results else 0,
                format_func=display_model_name,
            )
            diagnostic_result = results[diagnostic_model]
            diagnostic_frame = fitted_diagnostics_frame(diagnostic_result)
            chart_frame = diagnostic_frame.rename(
                columns={"actual": "Actual", "prediction": "Prediction", "residual": "Residual"}
            ).set_index(prepared.date_col)
            st.line_chart(chart_frame[["Actual", "Prediction"]], width="stretch")
            st.line_chart(chart_frame[["Residual"]], width="stretch")
            st.bar_chart(diagnostic_frame["absolute_error"], width="stretch")
            d1, d2 = st.columns(2)
            d1.dataframe(residual_summary_frame(diagnostic_result), width="stretch")
            d2.dataframe(future_forecast_frame(diagnostic_result), width="stretch")
            with st.expander("测试集诊断明细", expanded=False):
                st.dataframe(diagnostic_frame, width="stretch")

        with experiment_tab:
            records = st.session_state.get("experiment_records", [])
            record_frame = experiments_frame(records)
            if record_frame.empty:
                st.info("还没有实验记录。点击“开始分析”或“运行批量实验”后会自动记录。")
            else:
                st.dataframe(record_frame, width="stretch")
                best_records = record_frame.sort_values("MAPE").head(10)
                st.bar_chart(best_records.set_index("experiment_id")[["MAE", "RMSE", "MAPE"]], width="stretch")
                c_exp1, c_exp2 = st.columns(2)
                c_exp1.download_button(
                    "下载实验记录 CSV",
                    data=dataframe_to_csv_bytes(record_frame),
                    file_name="experiment_records.csv",
                    mime="text/csv",
                )
                if c_exp2.button("清空实验记录"):
                    st.session_state["experiment_records"] = []
                    st.rerun()

        history_saved = None
        history_error = None
        if save_history:
            try:
                history_saved = save_analysis_run(
                    supabase_url=supabase_url,
                    supabase_key=supabase_key,
                    table=history_table,
                    payload={
                        "dataset_name": uploaded_file.name if uploaded_file else "sample_sales.csv",
                        "target_col": prepared.target_col,
                        "horizon": horizon,
                        "test_size": test_size,
                        "best_model": display_model_name(best.name),
                        "best_metrics": best.metrics,
                        "model_metrics": localize_metrics_table(
                            metrics_frame({name: result.metrics for name, result in results.items()})
                        ).to_dict(orient="records"),
                        "anomalies": localize_anomaly_table(anomalies, prepared.date_col, prepared.target_col).to_dict(orient="records"),
                        "report_quality": quality_table.to_dict(orient="records"),
                        "knowledge_context": knowledge_context,
                        "report": report,
                    },
                )
            except Exception as exc:
                history_error = str(exc)
        if history_saved:
            st.success("本次分析已保存到 Supabase。")
            st.json(history_saved)
        if history_error:
            st.warning("Supabase 历史记录保存失败。")
            st.code(history_error)

        st.download_button(
            "下载分析报告",
            data=report.encode("utf-8"),
            file_name="时间序列分析报告.md",
            mime="text/markdown",
        )

        forecast_export = build_forecast_export_frame(prepared, best)
        localized_anomalies = localize_anomaly_table(anomalies, prepared.date_col, prepared.target_col)
        report_cols = st.columns(2)
        try:
            report_cols[0].download_button(
                "下载 DOCX 报告",
                data=markdown_to_docx_bytes(report),
                file_name="时间序列分析报告.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except RuntimeError as exc:
            report_cols[0].warning(str(exc))
        try:
            report_cols[1].download_button(
                "下载 PDF 报告",
                data=markdown_to_pdf_bytes(report),
                file_name="时间序列分析报告.pdf",
                mime="application/pdf",
            )
        except RuntimeError as exc:
            report_cols[1].warning(str(exc))

        data_cols = st.columns(3)
        data_cols[0].download_button(
            "下载预测结果 CSV",
            data=dataframe_to_csv_bytes(forecast_export),
            file_name="预测结果.csv",
            mime="text/csv",
        )
        data_cols[1].download_button(
            "下载模型评估 CSV",
            data=dataframe_to_csv_bytes(localized_metric_table),
            file_name="模型评估.csv",
            mime="text/csv",
        )
        data_cols[2].download_button(
            "下载异常点 CSV",
            data=dataframe_to_csv_bytes(localized_anomalies),
            file_name="异常点.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(str(exc))
