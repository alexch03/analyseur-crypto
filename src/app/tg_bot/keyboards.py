"""Menus inline pour le bot Telegram."""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 Perf globale", callback_data="perf"),
            InlineKeyboardButton("🔍 Trades open", callback_data="trades_open"),
        ],
        [
            InlineKeyboardButton("📈 Hypotheses actives", callback_data="hyps_active"),
            InlineKeyboardButton("🧮 Last 10 trades", callback_data="trades_recent"),
        ],
        [
            InlineKeyboardButton("▶️ Scan immediat", callback_data="scan_now"),
            InlineKeyboardButton("⏪ Backfill 7j", callback_data="backfill_7"),
        ],
        [
            InlineKeyboardButton("📋 Top patterns", callback_data="patterns_stats"),
            InlineKeyboardButton("🔔 Notifs", callback_data="notif_menu"),
        ],
        [
            InlineKeyboardButton("🌐 Open dashboard", callback_data="dashboard_link"),
            InlineKeyboardButton("🔄 Refresh", callback_data="menu"),
        ],
    ])


def back_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ Menu", callback_data="menu")]
    ])


def confirm_kb(action: str, target: str = "") -> InlineKeyboardMarkup:
    cb = f"confirm:{action}" + (f":{target}" if target else "")
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Confirmer", callback_data=cb),
        InlineKeyboardButton("❌ Annuler", callback_data="menu"),
    ]])


def notif_menu_kb(prefs: dict) -> InlineKeyboardMarkup:
    """Menu toggle notifications."""
    def _on_off(v: bool) -> str:
        return "ON" if v else "OFF"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(
            f"Trigger: {_on_off(prefs.get('trigger', True))}",
            callback_data="notif_toggle:trigger",
        )],
        [InlineKeyboardButton(
            f"Close (TARGET/STOPPED): {_on_off(prefs.get('close', True))}",
            callback_data="notif_toggle:close",
        )],
        [InlineKeyboardButton(
            f"Errors: {_on_off(prefs.get('error', True))}",
            callback_data="notif_toggle:error",
        )],
        [InlineKeyboardButton("⬅️ Menu", callback_data="menu")],
    ])
