from __future__ import annotations

import json
from pathlib import Path


CUSTOM_PROVIDER_LABEL = "自定义"
CUSTOM_MODEL_LABEL = "自定义 Model Name"
CUSTOM_PRESETS_PATH = Path(__file__).resolve().parents[1] / "data" / "custom_llm_presets.json"


LLM_PROVIDER_PRESETS = {
    "DeepSeek": {
        "api_url": "https://api.deepseek.com",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"],
        "note": "OpenAI-compatible API；不要填写 /anthropic。",
    },
    "Kimi 国内": {
        "api_url": "https://api.moonshot.cn/v1",
        "models": ["kimi-k3", "kimi-k2.7-code-highspeed", "kimi-k2.6"],
        "note": "适合 platform.kimi.com 创建的 Moonshot API Key。",
    },
    "Kimi 国际": {
        "api_url": "https://api.moonshot.ai/v1",
        "models": ["kimi-k3", "kimi-k2.7-code-highspeed", "kimi-k2.6"],
        "note": "适合 platform.kimi.ai 创建的 Moonshot API Key。",
    },
    "OpenAI": {
        "api_url": "https://api.openai.com/v1",
        "models": ["gpt-5.2", "gpt-5.2-mini", "gpt-5.1", "gpt-4o-mini"],
        "note": "OpenAI official API；不同账号可用 model id 可能不同。",
    },
    "Alibaba Cloud Bailian / Qwen": {
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-plus", "qwen-max", "qwen-turbo"],
        "note": "Bailian OpenAI-compatible mode；企业工作空间域名可在自定义里填写。",
    },
}


def load_custom_llm_presets() -> dict:
    if not CUSTOM_PRESETS_PATH.exists():
        return {}
    try:
        return json.loads(CUSTOM_PRESETS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_custom_llm_preset(name: str, api_url: str, model_name: str) -> dict:
    presets = load_custom_llm_presets()
    presets[name] = {
        "api_url": api_url,
        "models": [model_name],
        "note": "本地手动添加的 OpenAI-compatible API 预设。",
    }
    CUSTOM_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_PRESETS_PATH.write_text(
        json.dumps(presets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return presets
