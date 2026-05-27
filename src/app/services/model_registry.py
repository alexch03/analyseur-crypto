"""Local registry for strategy parameter profiles ("models")."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REGISTRY_FILE = Path(".runtime_models.json")


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _default_registry() -> dict[str, Any]:
    return {"active_model_id": None, "items": [], "updated_at": _now()}


def load_registry() -> dict[str, Any]:
    if not REGISTRY_FILE.exists():
        data = _default_registry()
        save_registry(data)
        return data
    with REGISTRY_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "items" not in data:
        data["items"] = []
    data.setdefault("active_model_id", None)
    data.setdefault("updated_at", _now())
    return data


def save_registry(data: dict[str, Any]) -> dict[str, Any]:
    data["updated_at"] = _now()
    with REGISTRY_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
    return data


def list_models() -> list[dict[str, Any]]:
    return load_registry()["items"]


def add_model(name: str, params: dict[str, Any], stats: dict[str, Any] | None = None) -> dict[str, Any]:
    reg = load_registry()
    item = {
        "id": str(uuid.uuid4()),
        "name": name,
        "params": params,
        "stats": stats or {},
        "paper_enabled": False,
        "created_at": _now(),
        "updated_at": _now(),
    }
    reg["items"].append(item)
    save_registry(reg)
    return item


def update_model(model_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    reg = load_registry()
    for item in reg["items"]:
        if item["id"] == model_id:
            item.update(patch)
            item["updated_at"] = _now()
            save_registry(reg)
            return item
    return None


def set_active_model(model_id: str) -> dict[str, Any] | None:
    reg = load_registry()
    found = None
    for item in reg["items"]:
        if item["id"] == model_id:
            found = item
            break
    if found is None:
        return None
    reg["active_model_id"] = model_id
    save_registry(reg)
    return found


def get_active_model() -> dict[str, Any] | None:
    reg = load_registry()
    active_id = reg.get("active_model_id")
    if not active_id:
        return None
    for item in reg["items"]:
        if item["id"] == active_id:
            return item
    return None
