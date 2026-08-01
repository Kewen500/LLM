from __future__ import annotations

import time
import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st

from src.anomaly_detection import detect_anomalies
from src.data_preprocess import prepare_time_series, split_train_test
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


st.set_page_config(page_title="时间序列预测与大模型报告生成", layout="wide")

FORECAST_MODEL_OPTIONS = ["Moving Average", "Seasonal Naive", "Linear Trend", "ARIMA", "Prophet", "LSTM"]
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
        api_url = st.text_input("自定义接口地址", value="https://api.example.com/v1")
        model_name = st.text_input("自定义模型名称", value="")
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
                st.warning("请先填写 API URL 和模型名称。")
    else:
        preset = presets[provider]
        st.caption(preset.get("note", ""))
        api_url = st.text_input("接口地址", value=preset["api_url"])
        model_options = list(dict.fromkeys(preset["models"] + [CUSTOM_MODEL_LABEL]))
        model_choice = st.selectbox("模型名称", model_options)
        if model_choice == CUSTOM_MODEL_LABEL:
            model_name = st.text_input("自定义模型名称", value="")
        else:
            model_name = model_choice

    api_key = st.text_input("API 密钥", type="password")
    max_tokens = st.slider("报告最大输出长度（token）", min_value=600, max_value=3000, value=default_max_tokens, step=200)
    show_prompt = st.checkbox("显示发送给大模型的提示词", value=False)
    return api_url, model_name, api_key, max_tokens, show_prompt


def render_knowledge_settings():
    with st.expander("业务知识库（RAG，可选）", expanded=False):
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


def default_selected_models() -> list[str]:
    defaults = ["Moving Average", "Seasonal Naive", "Linear Trend"]
    if importlib.util.find_spec("statsmodels"):
        defaults.append("ARIMA")
    if importlib.util.find_spec("prophet"):
        defaults.append("Prophet")
    if importlib.util.find_spec("torch"):
        defaults.append("LSTM")
    return defaults


st.title("时间序列预测与大模型自动分析报告生成系统")

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
    selected_models = st.multiselect(
        "选择模型",
        FORECAST_MODEL_OPTIONS,
        default=default_selected_models(),
        format_func=display_model_name,
    )
    report_mode = st.radio("报告生成方式", ["本地模板报告", "大模型 API"], horizontal=True)
    api_url = "https://api.deepseek.com"
    model_name = "deepseek-v4-flash"
    api_key = ""
    max_tokens = 1600
    show_prompt = False
    if report_mode == "大模型 API":
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
        if report_mode == "大模型 API":
            try:
                report = generate_openai_compatible_report(
                    prompt=prompt,
                    api_key=api_key,
                    model=model_name,
                    api_url=api_url,
                    max_tokens=max_tokens,
                )
                report_source = f"大模型生成报告：{model_name}"
            except Exception as exc:
                llm_error = str(exc)
        quality_checks = evaluate_report_quality(report, report_input)
        quality_table = quality_checks_frame(quality_checks)
        quality_pass_rate = quality_score(quality_checks)

        elapsed_seconds = time.perf_counter() - run_started_at

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
            st.warning("大模型报告生成失败，已自动回退为本地模板报告。")
            st.code(llm_error)
        if show_prompt:
            with st.expander("发送给大模型的提示词", expanded=False):
                st.text_area("提示词", value=prompt, height=260)
                st.download_button(
                    "下载提示词",
                    data=prompt.encode("utf-8"),
                    file_name="大模型提示词.txt",
                    mime="text/plain",
                )
        st.markdown(report)

        st.subheader("报告事实一致性校验")
        score_cols = st.columns(2)
        score_cols[0].metric("校验通过率", f"{quality_pass_rate}%")
        score_cols[1].metric("检查项数量", len(quality_checks))
        st.dataframe(quality_table, width="stretch")

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
