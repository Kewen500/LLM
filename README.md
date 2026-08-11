# 时间序列预测与 LLM 自动分析报告生成系统

这个项目用于展示“Time Series Forecasting + LLM Report Generation”的完整流程：上传 CSV 数据，完成数据清洗、模型预测、指标评估、异常检测，并自动生成中文业务分析报告和可下载交付物。

## 功能

- CSV 数据上传、字段选择、行列解析与数据预览
- 日期解析、重复日期聚合、缺失值插值
- 训练集 / 测试集划分
- 多模型预测对比：Moving Average、Seasonal Naive、Linear Trend
- 传统时间序列模型：ARIMA、Prophet
- 深度学习模型：基于 PyTorch 的 LSTM 滑动窗口预测
- TimesFM 本地高级模型：支持 Google TimesFM zero-shot forecasting，可选安装 `timesfm[torch]`
- Prophet-like Decomposition：当 Windows 环境中的 Stan 优化器不可用时，自动切换到趋势 + 星期周期分解模型
- Model 参数配置：支持 Moving Average window、Seasonal Naive season_length、ARIMA p/d/q、LSTM lookback/epochs/hidden_size
- 批量实验记录：支持多组 horizon 与 test_size 组合实验，自动记录最佳模型、MAE、RMSE、MAPE、耗时和参数
- Rolling Backtest：基于 Scikit-learn `TimeSeriesSplit` 进行多折时间序列验证
- 可视化诊断页：展示 Actual vs Prediction、Residual、绝对误差、残差统计和未来预测明细
- MAE、RMSE、MAPE 指标评估
- 滚动窗口异常点检测
- 本地模板报告或 OpenAI-compatible LLM API 报告生成，API 失败时自动回退本地报告
- 轻量 RAG 知识库：自动读取 `knowledge/` 目录，并支持页面上传或粘贴业务知识
- 报告事实一致性校验：检查报告是否引用真实模型、指标、预测均值和异常点数量
- 内置 LLM API 预设：DeepSeek、Kimi 国内、Kimi 国际、OpenAI、Alibaba Cloud Bailian / Qwen
- 支持自定义 OpenAI-compatible API URL 和 Model Name，并可保存为本地预设
- 分析运行耗时展示
- 报告导出：Markdown、DOCX、PDF
- 数据导出：预测结果 CSV、模型评估 CSV、异常点 CSV
- Pytest 自动化测试覆盖核心数据处理、基础模型和导出能力

## 项目结构

```text
time-series-llm-report/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── requirements-advanced.txt
├── data/
│   └── sample_sales.csv
├── knowledge/
│   └── ETT数据集业务说明.md
├── src/
│   ├── anomaly_detection.py
│   ├── data_preprocess.py
│   ├── exporters.py
│   ├── forecasting.py
│   ├── llm_client.py
│   ├── llm_presets.py
│   ├── metrics.py
│   └── report_generator.py
└── tests/
    └── test_core_pipeline.py
```

## 快速运行

先安装基础依赖：

```bash
pip install -r requirements.txt
```

如果要启用 ARIMA、Prophet、LSTM、DOCX/PDF 导出和测试能力，再安装：

```bash
pip install -r requirements-advanced.txt
```

说明：公开部署默认使用轻量依赖，页面会自动隐藏当前环境未安装的高级模型。`Prophet` 需要安装 `prophet`，`LSTM` 需要安装 `torch`，`TimesFM` 需要安装 `timesfm[torch]` 并在首次运行时下载模型权重；如果希望本地运行完整模型，请使用上面的 `requirements-advanced.txt`。

命令行验证核心流程：

```bash
python run_pipeline.py
```

运行测试：

```bash
pytest -q
```

启动网页应用：

```bash
streamlit run app.py
```

如果没有 API Key，可以选择“本地模板报告”。如果有 OpenAI-compatible API，在侧边栏选择“LLM API”，再从“API 服务商”中选择 DeepSeek、Kimi、OpenAI、Qwen 等预设。系统会自动带出 API URL 和常用 Model Name，也可以选择“自定义”手动填写 API URL 和 Model Name，并保存为本地预设。`API URL` 可以填写完整的 `/chat/completions` 地址，也可以只填写到服务商的 OpenAI-compatible base URL。系统只保存 URL 和 Model Name，不保存 API Key。

LLM 报告生成失败时，系统会自动回退为本地模板报告，并在页面显示错误原因和发送给 LLM 的 Prompt，便于调试。

## 本机私有配置

本地运行时，可以在页面中点击“保存 LLM API 设置到本机”和“保存 Supabase 设置到本机”。系统会把 `API Key`、`Supabase Project URL`、`Supabase anon key` 等内容保存到：

```text
data/local_private_settings.json
```

这个文件已经写入 `.gitignore`，不会上传到 GitHub。注意：该文件是本机明文保存，适合个人电脑免重复输入；公开部署给他人使用时，不建议保存访客的 API Key。

项目会自动识别常见公开部署环境，例如 Streamlit Community Cloud 和 Render。在公开部署环境中，页面不会读取或写入 `data/local_private_settings.json`，也不会显示保存按钮；访客输入的 API Key 只在当前会话临时使用。

## RAG 知识库放置位置

RAG 解释文档统一放在项目根目录的 `knowledge/` 文件夹中：

```text
knowledge/
└── ETT数据集业务说明.md
```

支持 `.md`、`.txt`、`.csv` 文件。文档中可以写字段含义、指标口径、异常原因、业务规则和报告写作口径。系统会自动读取 `knowledge/` 目录，也可以在页面的“RAG 知识库（可选）”区域临时上传文件或粘贴文本。

示例：

```md
# 字段说明
HUFL：高压有用负载特征。
OT：油温。

# 分析规则
如果负载和油温同时异常，需要优先关注设备温度和检修记录。
```

## 部署

这个项目是 Streamlit 应用，需要部署到支持常驻 Python Web 服务的平台。推荐两种方式：

默认部署使用 `requirements-deploy.txt`，用于保证免费环境启动稳定。这个轻量版本会保留 Moving Average、Seasonal Naive、Linear Trend、ARIMA、LLM API、RAG、导出和历史记录等核心功能；如果部署环境资源充足，可以把 `prophet`、`torch`、`scikit-learn`、`timesfm[torch]` 加入部署依赖来启用 Prophet、LSTM、Rolling Backtest 与 TimesFM。

### 方式一：Render

项目已提供 `render.yaml`，推送到 GitHub 后，在 Render 中选择 New Web Service，连接仓库即可。Render 会读取：

```bash
pip install -r requirements-deploy.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

部署成功后会得到一个公开 URL，任何人都可以打开。

### 方式二：Streamlit Community Cloud

把项目推送到 GitHub，在 Streamlit Community Cloud 中选择仓库和 `app.py`。如果希望使用轻量公开演示依赖，可以把 `requirements-deploy.txt` 的内容复制到 `requirements.txt`；如果希望本地保留完整高级模型，则继续使用 `requirements-advanced.txt`。

### 关于 Vercel 和 Supabase

Vercel 适合部署 Next.js/React 前端，但不适合直接运行 Streamlit 这种常驻 Python 服务。可以把 Streamlit 部署到 Render 或 Streamlit Community Cloud，再用 Vercel 做项目展示页、文档页或自定义域名跳转。

Supabase 当前不是必需项。后续如果要增加用户登录、保存历史分析记录、共享报告链接，可以再接入 Supabase。

## CSV 格式

至少包含一列日期和一列数值：

```csv
date,sales
2025-01-01,126
2025-01-02,132
```

## 阶段进度

1. 已完成：基础模型 Moving Average、Seasonal Naive、Linear Trend。
2. 已完成：加入 `statsmodels` 的 ARIMA 模型。
3. 已完成：加入 `prophet` 模型，并提供 Prophet-like 后备模型处理趋势与星期周期。
4. 已完成：使用 `torch` 实现 LSTM 滑动窗口预测。
5. 已完成：接入 OpenAI-compatible LLM API，把模板报告升级为 LLM 生成报告。
6. 已完成：增加 DeepSeek、Kimi、OpenAI、Qwen 等 LLM API 预设，并支持自定义 API URL 和 Model Name。
7. 已完成：增加 Markdown、DOCX、PDF 报告导出，以及预测结果、模型指标、异常点 CSV 导出。
8. 已完成：增加自动化测试，覆盖数据清洗、基础模型运行和导出能力。
9. 已完成：增加批量实验记录、Model 参数配置和可视化诊断页。
10. 下一阶段：增加用户登录、实验记录云端查询页和共享报告链接。

## 简历写法

基于 Time Series Forecasting 与 LLM 构建自动化数据分析系统，支持 CSV 数据上传、数据清洗、趋势预测、异常检测、模型评估和自然语言分析报告生成。使用 Moving Average、Seasonal Naive、Linear Trend、ARIMA、Prophet-like Decomposition、Prophet、PyTorch LSTM 与 TimesFM（本地可选高级依赖）等模型完成预测对比，采用 MAE、RMSE、MAPE 等指标评估模型表现，并支持 Model 参数配置、批量实验记录、基于 Scikit-learn TimeSeriesSplit 的 Rolling Backtest 和残差可视化诊断。项目接入 OpenAI-compatible LLM API，内置 DeepSeek、Kimi、OpenAI、Qwen 等服务商预设，根据结构化预测结果自动生成业务分析报告，支持 Markdown、DOCX、PDF 报告导出及预测结果、评估指标、异常点 CSV 下载，并通过 Pytest 覆盖核心数据处理、模型运行、实验记录和导出流程。
