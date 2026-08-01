from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlparse


def validate_supabase_project_url(supabase_url: str) -> str:
    url = supabase_url.strip().rstrip("/")
    if not url:
        raise ValueError("请填写 Supabase Project URL。")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Supabase Project URL 应该形如：https://xxxx.supabase.co")
    if parsed.netloc == "supabase.com" or parsed.netloc.endswith(".supabase.com") or "/dashboard" in parsed.path:
        raise ValueError("请填写 Supabase Project URL，不要填写 Supabase Dashboard 页面地址。Project URL 通常形如：https://xxxx.supabase.co")
    return url


def save_analysis_run(
    supabase_url: str,
    supabase_key: str,
    payload: dict,
    table: str = "analysis_runs",
    timeout: int = 20,
) -> dict:
    url = validate_supabase_project_url(supabase_url)
    key = supabase_key.strip()
    if not key:
        raise ValueError("请填写 Supabase API Key。")
    endpoint = f"{url}/rest/v1/{table}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body) if body else []
            return data[0] if isinstance(data, list) and data else {"status": "saved"}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        if exc.code == 404 and "<html" in body.lower():
            raise RuntimeError(
                "Supabase 保存失败：HTTP 404。请检查 Project URL 和表名。"
                "Project URL 应形如 https://xxxx.supabase.co，表名默认是 analysis_runs；不要填写 Dashboard 页面地址。"
            ) from exc
        raise RuntimeError(f"Supabase 保存失败：HTTP {exc.code}。{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Supabase 保存失败：{exc.reason}") from exc
