from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOCAL_PRIVATE_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "local_private_settings.json"


def load_local_private_settings() -> dict[str, Any]:
    if not LOCAL_PRIVATE_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(LOCAL_PRIVATE_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_local_private_settings(settings: dict[str, Any]) -> None:
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
