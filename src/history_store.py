from __future__ import annotations

import json
import urllib.error
import urllib.request


def save_analysis_run(
    supabase_url: str,
    supabase_key: str,
    payload: dict,
    table: str = "analysis_runs",
    timeout: int = 20,
) -> dict:
    url = supabase_url.strip().rstrip("/")
    key = supabase_key.strip()
    if not url:
        raise ValueError("请填写 Supabase Project URL。")
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
        raise RuntimeError(f"Supabase 保存失败：HTTP {exc.code}。{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Supabase 保存失败：{exc.reason}") from exc
