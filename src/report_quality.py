from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from .forecasting import display_model_name
from .report_generator import ReportInput


@dataclass
class QualityCheck:
    item: str
    passed: bool
    detail: str


def _contains_number(text: str, value: float, decimals: int = 2) -> bool:
    candidates = {
        f"{value:.{decimals}f}",
        f"{value:.4f}",
        str(round(float(value), decimals)),
        str(round(float(value), 4)),
    }
    return any(candidate in text for candidate in candidates)


def evaluate_report_quality(report: str, report_input: ReportInput) -> list[QualityCheck]:
    metrics = report_input.best_result.metrics
    future = report_input.best_result.future["prediction"]
    best_model = display_model_name(report_input.best_result.name)
    checks = [
        QualityCheck("包含数据概况", "数据概况" in report, "报告应说明数据范围、样本量和分析对象。"),
        QualityCheck("包含趋势分析", "趋势" in report, "报告应解释近期趋势变化。"),
        QualityCheck("包含模型预测", "模型" in report and "预测" in report, "报告应说明最佳模型和预测结果。"),
        QualityCheck("包含异常点分析", "异常" in report, "报告应说明异常点数量或异常点判断。"),
        QualityCheck("包含业务建议", "建议" in report, "报告应给出可执行建议。"),
        QualityCheck("引用最佳模型", best_model in report, f"报告应引用最佳模型：{best_model}。"),
        QualityCheck("引用 MAE", str(metrics["MAE"]) in report, f"报告应引用 MAE={metrics['MAE']}。"),
        QualityCheck("引用 RMSE", str(metrics["RMSE"]) in report, f"报告应引用 RMSE={metrics['RMSE']}。"),
        QualityCheck("引用 MAPE", str(metrics["MAPE"]) in report, f"报告应引用 MAPE={metrics['MAPE']}%。"),
        QualityCheck("引用预测均值", _contains_number(report, future.mean()), f"报告应引用未来预测均值 {future.mean():.2f}。"),
        QualityCheck("引用异常点数量", str(len(report_input.anomalies)) in report, f"报告应引用异常点数量 {len(report_input.anomalies)}。"),
    ]
    if report_input.knowledge_context:
        checks.append(
            QualityCheck(
                "使用业务知识库",
                "业务知识" in report or "知识库" in report or any(token in report for token in _knowledge_keywords(report_input.knowledge_context)),
                "启用 RAG 时，报告应体现业务知识库中的口径、规则或背景。",
            )
        )
    known_model_names = {display_model_name(name) for name in report_input.all_results}
    suspicious_models = {"TimeGPT", "DeepAR", "XGBoost", "LightGBM", "Transformer"}
    hallucinated = sorted(name for name in suspicious_models if name in report and name not in known_model_names)
    checks.append(
        QualityCheck(
            "未编造未运行模型",
            not hallucinated,
            "发现未运行却被提及的模型：" + "、".join(hallucinated) if hallucinated else "未发现明显模型编造。",
        )
    )
    return checks


def quality_checks_frame(checks: list[QualityCheck]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "检查项": [check.item for check in checks],
            "结果": ["通过" if check.passed else "需检查" for check in checks],
            "说明": [check.detail for check in checks],
        }
    )


def quality_score(checks: list[QualityCheck]) -> float:
    if not checks:
        return 0.0
    return round(sum(1 for check in checks if check.passed) / len(checks) * 100, 2)


def _knowledge_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_]{2,}", text)
    return tokens[:8]
