from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any


LOCAL_PRIVATE_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "local_private_settings.json"
CLOUD_ENV_KEYS = {
    "RENDER",
    "STREAMLIT_CLOUD",
    "STREAMLIT_SHARING_MODE",
    "STREAMLIT_COMMUNITY_CLOUD",
    "VERCEL",
    "RAILWAY_ENVIRONMENT",
    "FLY_APP_NAME",
}
TRUE_VALUES = {"1", "true", "yes", "on"}


def is_cloud_deployment() -> bool:
    app_env = os.getenv("APP_ENV", "").strip().lower()
    if app_env in {"cloud", "prod", "production"}:
        return True
    if any(os.getenv(key) for key in CLOUD_ENV_KEYS):
        return True
    cwd = str(Path.cwd()).replace("\\", "/")
    return cwd.startswith("/mount/src/") or cwd.startswith("/opt/render/")


def can_persist_local_private_settings() -> bool:
    app_env = os.getenv("APP_ENV", "").strip().lower()
    explicit_enable = os.getenv("ENABLE_LOCAL_PRIVATE_SETTINGS", "").strip().lower() in TRUE_VALUES
    if explicit_enable or app_env == "local":
        return True
    if is_cloud_deployment():
        return False
    return platform.system() in {"Windows", "Darwin"} or str(Path.cwd()).startswith(str(Path.home()))


def load_local_private_settings() -> dict[str, Any]:
    if not can_persist_local_private_settings():
        return {}
    if not LOCAL_PRIVATE_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(LOCAL_PRIVATE_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_local_private_settings(settings: dict[str, Any]) -> None:
    if not can_persist_local_private_settings():
        raise RuntimeError("当前运行环境不允许把私密配置保存到服务器文件。")
    LOCAL_PRIVATE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_PRIVATE_SETTINGS_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_local_private_section(section: str, values: dict[str, Any]) -> dict[str, Any]:
    settings = load_local_private_settings()
    settings[section] = values
    save_local_private_settings(settings)
    return settings
