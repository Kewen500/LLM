from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .forecasting import ForecastResult, display_model_name


@dataclass
class ReportInput:
    date_col: str
    target_col: str
    prepared_data: pd.DataFrame
    anomalies: pd.DataFrame
    best_result: ForecastResult
    all_results: dict[str, ForecastResult]


def recent_change_percent(data: pd.DataFrame, target_col: str) -> float:
    recent_mean = data[target_col].tail(14).mean()
    previous_mean = data[target_col].iloc[-28:-14].mean() if len(data) >= 28 else data[target_col].mean()
    return float((recent_mean - previous_mean) / previous_mean * 100) if previous_mean else 0.0


def trend_label(change: float) -> str:
    if change > 3:
        return "增长"
    if change < -3:
        return "下降"
    return "基本稳定"


def anomaly_summary(report_input: ReportInput) -> str:
    if report_input.anomalies.empty:
        return "未发现明显异常点。"
    rows = report_input.anomalies.tail(5)
    items = [
        f"{row[report_input.date_col].date()} 的 {report_input.target_col}={row[report_input.target_col]:.2f}，"
        f"标准化偏差={row['z_score']:.2f}"
        for _, row in rows.iterrows()
    ]
    return "；".join(items)


def build_model_summary(report_input: ReportInput) -> str:
    data = report_input.prepared_data
    first_date = data[report_input.date_col].min().date()
    last_date = data[report_input.date_col].max().date()
    change = recent_change_percent(data, report_input.target_col)
    metrics = report_input.best_result.metrics
    model_lines = [
        f"- {display_model_name(name)}：MAE={res.metrics['MAE']}，RMSE={res.metrics['RMSE']}，MAPE={res.metrics['MAPE']}%"
        for name, res in report_input.all_results.items()
    ]
    future = report_input.best_result.future["prediction"]
    return "\n".join(
        [
            f"分析字段：{report_input.target_col}",
            f"数据范围：{first_date} 至 {last_date}",
            f"样本量：{len(data)}",
            f"最近 14 个周期相对前一阶段变化：{change:.2f}%",
            f"近期趋势判断：{trend_label(change)}",
            f"异常点数量：{len(report_input.anomalies)}",
            f"异常点摘要：{anomaly_summary(report_input)}",
            f"最佳模型：{display_model_name(report_input.best_result.name)}",
            f"最佳模型指标：MAE={metrics['MAE']}，RMSE={metrics['RMSE']}，MAPE={metrics['MAPE']}%",
            f"未来预测均值：{future.mean():.2f}",
            f"未来预测最小值：{future.min():.2f}",
            f"未来预测最大值：{future.max():.2f}",
            "模型对比：",
            *model_lines,
        ]
    )


def generate_template_report(report_input: ReportInput) -> str:
    summary = build_model_summary(report_input)
    best = report_input.best_result
    future = best.future["prediction"]
    data = report_input.prepared_data
    change = recent_change_percent(data, report_input.target_col)
    direction = trend_label(change)

    return f"""# 时间序列自动分析报告

## 1. 数据概况
本次分析对象为 `{report_input.target_col}`，样本覆盖 {data[report_input.date_col].min().date()} 至 {data[report_input.date_col].max().date()}，共 {len(data)} 条连续时间记录。系统完成了日期解析、重复日期聚合、缺失值插补和训练/测试集划分。

## 2. 趋势分析
最近 14 个周期均值较前一阶段变化约 {change:.2f}%，整体表现为{direction}。如果该指标对应销售额、访问量或业务需求量，说明近期经营或使用强度出现了可观测变化，需要结合节假日、活动投放或外部环境进一步解释。

## 3. 模型预测
本次对比后表现最好的模型为 **{display_model_name(best.name)}**，测试集 MAE 为 {best.metrics['MAE']}，RMSE 为 {best.metrics['RMSE']}，MAPE 为 {best.metrics['MAPE']}%。未来预测均值约为 {future.mean():.2f}，预测区间大致位于 {future.min():.2f} 至 {future.max():.2f}。

## 4. 异常点分析
{anomaly_summary(report_input)} 这些点可能对应促销、系统故障、数据录入误差或突发业务事件，建议在真实业务中补充事件标签后再判断是否需要剔除或单独建模。

## 5. 建议
建议优先使用当前最佳模型作为基线模型，并持续记录预测误差。当 MAPE 连续升高时，应重新训练模型或加入节假日、价格、活动、天气等外生变量。对于异常点较多的时间段，建议先做数据质量核查，再进行业务归因。

---

结构化输入摘要：

{summary}
"""


def build_llm_prompt(report_input: ReportInput) -> str:
    summary = build_model_summary(report_input)
    return f"""你是一名数据分析师。请根据以下时间序列预测结果，生成一份中文业务分析报告。

请严格遵守：
1. 使用 Markdown 输出。
2. 必须包含：数据概况、趋势分析、模型对比、预测结论、异常点分析、风险提示、业务建议。
3. 必须引用输入摘要中的关键指标，例如最佳模型、MAE、RMSE、MAPE、预测均值和异常点数量。
4. 不要编造输入摘要中没有出现的具体数字、事件、业务背景或外部原因。
5. 如果某些模型被跳过或没有出现在模型对比中，不要假设它们的结果。
6. 语气专业、简洁，适合作为项目演示中的自动报告。

输入摘要：
{summary}
"""
