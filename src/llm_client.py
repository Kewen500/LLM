from __future__ import annotations

import json
import urllib.error
import urllib.request


def normalize_chat_completions_url(api_url: str) -> str:
    """兼容完整 chat-completions 地址或服务商 base URL。"""
    cleaned = (api_url or "").strip().rstrip("/")
    if not cleaned:
        return "https://api.openai.com/v1/chat/completions"
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def generate_openai_compatible_report(
    prompt: str,
    api_key: str,
    model: str,
    api_url: str = "https://api.openai.com/v1/chat/completions",
    timeout: int = 60,
    temperature: float = 0.2,
    max_tokens: int = 1600,
) -> str:
    """调用 OpenAI 兼容的聊天补全接口。"""
    if not api_key:
        raise ValueError("生成大模型报告需要填写 API Key。")
    if not model:
        raise ValueError("生成大模型报告需要填写模型名称。")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一名严谨的数据分析师，只能基于用户提供的结构化预测结果生成中文分析报告。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        normalize_chat_completions_url(api_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"大模型接口请求失败：HTTP {exc.code}。{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"大模型接口请求失败：{exc.reason}") from exc

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"大模型接口返回格式不符合预期：{data}") from exc
