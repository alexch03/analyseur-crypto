"""Telegram bot config - reads .env (TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID)."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parents[3]  # D:\Analyseur crypto\
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ADMIN_CHAT_ID = os.environ.get("TELEGRAM_ADMIN_CHAT_ID", "")

ALLOWED_USER_IDS: set[int] = set()
if ADMIN_CHAT_ID:
    try:
        ALLOWED_USER_IDS.add(int(ADMIN_CHAT_ID))
    except ValueError:
        pass

# Toggle global pour activer/desactiver les notifs automatiques
NOTIFY_ON_TRIGGER = os.environ.get("TG_NOTIFY_TRIGGER", "true").lower() == "true"
NOTIFY_ON_CLOSE = os.environ.get("TG_NOTIFY_CLOSE", "true").lower() == "true"

# URL du dashboard local pour les liens
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:8000/patterns")
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000/api/v1")


def is_ready() -> bool:
    return bool(BOT_TOKEN and ALLOWED_USER_IDS)
