"""Bot Telegram - interface mobile pour piloter l'Analyseur Crypto.

Commandes :
    /start, /menu       - Menu principal
    /perf               - Performance globale (cumul, winrate)
    /trades             - Trades cloturés récents
    /open               - Trades ouverts en ce moment
    /hyps               - Hypothèses actives (FORMING/ARMED/TRIGGERED)
    /scan               - Lance un scan immédiat
    /backfill <days>    - Lance un backfill historique
    /patterns           - Stats par pattern
    /notif on|off       - Toggle notifs
    /test               - Envoi message test
    /help

Usage :
    python -m app.tg_bot.bot
ou bien (recommande) :
    start_telegram.bat
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import Conflict, NetworkError, RetryAfter, TimedOut
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes,
)

from app.tg_bot import config as cfg
from app.tg_bot.keyboards import back_kb, confirm_kb, main_menu_kb, notif_menu_kb

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

_PID_FILE = ROOT / "data" / "tg_bot.pid"
_PREFS_FILE = ROOT / "data" / "tg_bot_prefs.json"


# ─────────────────────────────────────────────────────────────────────────────
# Lockfile + prefs
# ─────────────────────────────────────────────────────────────────────────────

def _acquire_pid_lock() -> bool:
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    my_pid = os.getpid()
    if _PID_FILE.exists():
        try:
            old_pid = int(_PID_FILE.read_text().strip())
            try:
                import psutil
                if psutil.pid_exists(old_pid) and old_pid != my_pid:
                    print(f"ERREUR: Bot deja en cours (PID {old_pid})")
                    return False
            except ImportError:
                pass
        except Exception:
            pass
    _PID_FILE.write_text(str(my_pid))
    return True


def _release_pid_lock() -> None:
    try:
        if _PID_FILE.exists():
            _PID_FILE.unlink()
    except Exception:
        pass


def _load_prefs() -> dict:
    if _PREFS_FILE.exists():
        try:
            return json.loads(_PREFS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"trigger": True, "close": True, "error": True}


def _save_prefs(prefs: dict) -> None:
    _PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PREFS_FILE.write_text(json.dumps(prefs, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Auth decorator
# ─────────────────────────────────────────────────────────────────────────────

def auth_required(handler):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id if update.effective_user else None
        # FAIL-CLOSED : refus si ALLOWED_USER_IDS est vide OU si user_id n'y est pas.
        # Empeche toute prise de controle quand TELEGRAM_ADMIN_CHAT_ID n'est pas configure.
        if not cfg.ALLOWED_USER_IDS or user_id not in cfg.ALLOWED_USER_IDS:
            if not cfg.ALLOWED_USER_IDS:
                msg = (
                    f"❌ Bot non configure (TELEGRAM_ADMIN_CHAT_ID manquant).\n"
                    f"Ton chat_id: `{user_id}`. Ajoute-le dans `.env`."
                )
            else:
                msg = (
                    f"❌ Acces refuse. Ton chat_id: `{user_id}`.\n"
                    f"Ajoute-le dans `.env` (TELEGRAM_ADMIN_CHAT_ID) si tu es l'admin."
                )
            if update.message:
                await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
            return
        return await handler(update, context)
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers (HTTP vers FastAPI local)
# ─────────────────────────────────────────────────────────────────────────────

async def _api_get(path: str, params: dict | None = None) -> dict | None:
    """GET sur l'API FastAPI locale."""
    url = f"{cfg.API_BASE}{path}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as cli:
            r = await cli.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.warning("API GET %s failed: %s", url, e)
        return None


async def _api_post(path: str, params: dict | None = None) -> dict | None:
    url = f"{cfg.API_BASE}{path}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as cli:
            r = await cli.post(url, params=params)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.warning("API POST %s failed: %s", url, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────────────────────

async def _reply(update: Update, text: str, kb=None, edit: bool = False) -> None:
    if edit and update.callback_query:
        try:
            await update.callback_query.edit_message_text(
                text, reply_markup=kb, parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
        except Exception:
            pass
    elif update.message:
        await update.message.reply_text(
            text, reply_markup=kb, parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )


async def _send_perf(update: Update, edit: bool = False) -> None:
    bd = await _api_get("/analytics/breakdown")
    if not bd:
        await _reply(update, "❌ API offline ou pas de données", back_kb(), edit)
        return
    overall = bd.get("overall", [{}])[0] if bd.get("overall") else {}
    text = "📊 *PERFORMANCE GLOBALE*\n\n"
    text += f"Trades: `{overall.get('count', 0)}`\n"
    text += f"Win rate: `{overall.get('win_rate_pct', 0):.1f}%`\n"
    text += f"Avg gain: `{overall.get('avg_pct', 0):+.3f}%`\n"
    text += f"Expectancy: `{overall.get('expectancy_pct', 0):+.3f}%`\n"
    text += f"Cumul simple: `{overall.get('cumul_simple_pct', 0):+.2f}%`\n"
    text += f"Cumul compound: `{overall.get('cumul_compound_pct', 0):+.2f}%`\n"
    text += f"Best: `{overall.get('best_pct', 0):+.2f}%`  Worst: `{overall.get('worst_pct', 0):+.2f}%`\n\n"

    text += "*Top 3 patterns:*\n"
    by_pat = bd.get("by_pattern", [])[:3]
    for p in by_pat:
        text += (f"  • `{p['label']}` N={p['count']} "
                 f"WR={p['win_rate_pct']:.0f}% cumul={p['cumul_compound_pct']:+.1f}%\n")
    await _reply(update, text, main_menu_kb(), edit)


async def _send_trades_open(update: Update, edit: bool = False) -> None:
    data = await _api_get("/hypotheses", params={"state": "TRIGGERED", "limit": 10})
    if data is None:
        await _reply(update, "❌ API offline", back_kb(), edit)
        return
    items = data if isinstance(data, list) else data.get("items", [])
    if not items:
        await _reply(update, "📂 Aucun trade ouvert", back_kb(), edit)
        return
    text = f"📂 *{len(items)} trade(s) ouvert(s)*\n\n"
    for h in items[:10]:
        sym = h.get("symbol", "?")
        side = h.get("side", "?")
        tf = h.get("timeframe", "?")
        entry = h.get("triggered_price") or h.get("entry_price") or 0
        target = h.get("target_price", 0)
        sl = h.get("invalidation_price", 0)
        pat = h.get("pattern_kind", "?")
        text += (f"*{sym}* `{tf}` {side} _{pat}_\n"
                 f"  Entry: `${entry:.4f}`  TP: `${target:.4f}`  SL: `${sl:.4f}`\n\n")
    await _reply(update, text, back_kb(), edit)


async def _send_hyps_active(update: Update, edit: bool = False) -> None:
    """Toutes les hypothèses non terminales (FORMING + ARMED + TRIGGERED)."""
    total = 0
    text = "📈 *HYPOTHESES ACTIVES*\n\n"
    for state in ("FORMING", "ARMED", "TRIGGERED"):
        data = await _api_get("/hypotheses", params={"state": state, "limit": 50})
        items = data if isinstance(data, list) else (data or {}).get("items", [])
        count = len(items)
        total += count
        emoji = {"FORMING": "⏳", "ARMED": "🎯", "TRIGGERED": "🔥"}[state]
        text += f"{emoji} `{state}` : *{count}*\n"
    if total == 0:
        text = "📈 *Aucune hypothese active*\n\nLance un scan ou attends une nouvelle bougie."
    await _reply(update, text, main_menu_kb(), edit)


async def _send_trades_recent(update: Update, edit: bool = False) -> None:
    data = await _api_get("/unit_paper/trades", params={"limit": 10})
    if data is None:
        await _reply(update, "❌ API offline", back_kb(), edit)
        return
    items = data if isinstance(data, list) else (data or {}).get("items", [])
    if not items:
        await _reply(update, "🧮 Aucun trade ferme", back_kb(), edit)
        return
    text = f"🧮 *{min(len(items), 10)} derniers trades*\n\n"
    for t in items[:10]:
        sym = t.get("symbol", "?")
        side = t.get("side", "?")
        pat = t.get("pattern_kind", "?")
        pct = t.get("pct_gain") or 0.0
        outcome = t.get("outcome", "?")
        emoji = "✅" if pct > 0 else ("⏸" if pct == 0 else "❌")
        text += f"{emoji} `{sym}` {side} _{pat[:14]}_  `{pct:+.2f}%`  ({outcome})\n"
    await _reply(update, text, back_kb(), edit)


async def _send_patterns_stats(update: Update, edit: bool = False) -> None:
    bd = await _api_get("/analytics/breakdown")
    if not bd:
        await _reply(update, "❌ API offline", back_kb(), edit)
        return
    by_pat = bd.get("by_pattern", [])
    if not by_pat:
        await _reply(update, "📋 Pas de stats par pattern (DB vide)", back_kb(), edit)
        return
    text = "📋 *STATS PAR PATTERN*\n\n"
    for p in by_pat[:10]:
        emoji = "✅" if p["cumul_compound_pct"] > 0 else "❌"
        text += (f"{emoji} `{p['label'][:20]}` N={p['count']} "
                 f"WR={p['win_rate_pct']:.0f}% cumul={p['cumul_compound_pct']:+.1f}%\n")
    await _reply(update, text, back_kb(), edit)


# ─────────────────────────────────────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────────────────────────────────────

@auth_required
async def cmd_start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = (
        f"🤖 *Analyseur Crypto*\n\n"
        f"Salut {user.first_name} !\n"
        f"Chat ID: `{user.id}`\n\n"
        f"Choisis une action :"
    )
    await _reply(update, text, main_menu_kb())


@auth_required
async def cmd_help(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "*Commandes disponibles*\n\n"
        "/menu — Menu principal\n"
        "/perf — Performance globale\n"
        "/open — Trades ouverts\n"
        "/trades — Derniers trades fermes\n"
        "/hyps — Hypotheses actives\n"
        "/patterns — Stats par pattern\n"
        "/scan — Lance scan immediat\n"
        "/backfill <jours> — Backfill (ex: /backfill 7)\n"
        "/notif — Toggle notifications\n"
        "/test — Test notification\n"
        "/dashboard — Lien dashboard"
    )
    await _reply(update, text)


@auth_required
async def cmd_perf(update: Update, _ctx): await _send_perf(update)


@auth_required
async def cmd_open(update: Update, _ctx): await _send_trades_open(update)


@auth_required
async def cmd_trades(update: Update, _ctx): await _send_trades_recent(update)


@auth_required
async def cmd_hyps(update: Update, _ctx): await _send_hyps_active(update)


@auth_required
async def cmd_patterns(update: Update, _ctx): await _send_patterns_stats(update)


@auth_required
async def cmd_scan(update: Update, _ctx):
    await _reply(update, "▶️ Scan immediat en cours...")
    res = await _api_post("/scanner/scan-now")
    if res:
        await _reply(update, f"✅ Job lance : `{res.get('job_id', '?')}`", back_kb())
    else:
        await _reply(update, "❌ Erreur API (scanner offline?)", back_kb())


@auth_required
async def cmd_backfill(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    days = 7
    if args:
        try:
            days = int(args[0])
        except ValueError:
            pass
    history_bars = days * 96  # 96 bars/jour en 15m
    await _reply(update, f"⏪ Backfill {days}j ({history_bars} bougies)...")
    res = await _api_post("/scanner/backfill", params={
        "history_bars": min(history_bars, 1000),  # cap API limite
    })
    if res:
        await _reply(update, f"✅ Job lance : `{res.get('job_id', '?')}`", back_kb())
    else:
        await _reply(update, "❌ Erreur API", back_kb())


@auth_required
async def cmd_notif(update: Update, _ctx):
    prefs = _load_prefs()
    text = "🔔 *Toggle notifications*\n\nCliquer pour activer/desactiver :"
    await _reply(update, text, notif_menu_kb(prefs))


@auth_required
async def cmd_test(update: Update, _ctx):
    from app.tg_bot.notifier import dispatch_test
    ok = dispatch_test()
    if ok:
        await _reply(update, "🧪 Test envoye OK")
    else:
        await _reply(update, "❌ Echec test (token/chat_id config ?)")


@auth_required
async def cmd_dashboard(update: Update, _ctx):
    text = f"🌐 *Dashboard*\n\n[Ouvrir]({cfg.DASHBOARD_URL})"
    await _reply(update, text)


@auth_required
async def cmd_exec_status(update: Update, _ctx):
    """Status execution : mode, balance, positions ouvertes, safety."""
    data = await _api_get("/execution/status")
    if not data:
        await _reply(update, "❌ API offline ou execution non configuree", back_kb())
        return
    bal = data.get("balance", {})
    mode = data.get("mode", "?")
    emoji = {"disabled": "⏸", "paper": "📝", "demo": "🟡", "live": "🔴"}.get(mode, "❔")
    ks = "🛑 TRIPPED" if data.get("killswitch") else "✓ OK"
    text = (
        f"⚙️ *EXEC STATUS* — {emoji} `{mode}`\n\n"
        f"Executor: `{data.get('executor', '?')}`\n"
        f"Killswitch: {ks}\n"
        f"Daily PnL: `{data.get('daily_pnl_usd', 0):+.2f}$` / "
        f"max -{data.get('max_daily_loss_usd', 0):.0f}$\n"
        f"Consec losses: `{data.get('consecutive_losses', 0)}`\n"
        f"Open positions: `{data.get('open_positions_count', 0)}` / "
        f"max {data.get('max_open_positions', 0)}\n"
        f"Max position size: `{data.get('max_position_usd', 0):.0f}$`\n\n"
        f"💰 Balance: `{bal.get('total', 0):.2f}$` "
        f"(free `{bal.get('free', 0):.2f}$`)"
    )
    if data.get("killswitch_reason"):
        text += f"\n\n*Killswitch reason:*\n`{data['killswitch_reason']}`"
    await _reply(update, text, main_menu_kb())


@auth_required
async def cmd_emergency_stop(update: Update, _ctx):
    """Ferme TOUTES les positions ET active killswitch (urgence)."""
    text = "🚨 *EMERGENCY STOP*\n\nFermeture immediate de TOUTES les positions ?"
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("🚨 OUI - Tout fermer", callback_data="confirm:emergency"),
        InlineKeyboardButton("❌ Annuler", callback_data="menu"),
    ]])
    await _reply(update, text, kb)


@auth_required
async def cmd_reset_killswitch(update: Update, _ctx):
    """Reset le killswitch apres review humaine."""
    data = await _api_post("/execution/reset_killswitch")
    if data and data.get("ok"):
        await _reply(update, f"✅ Killswitch RESET\n\n```\n{data.get('status', '')}\n```",
                       main_menu_kb())
    else:
        await _reply(update, "❌ Reset failed (API offline ?)", back_kb())


# ─────────────────────────────────────────────────────────────────────────────
# Callback handler
# ─────────────────────────────────────────────────────────────────────────────

@auth_required
async def callback_handler(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    if data == "menu":
        await q.edit_message_text(
            "🤖 *Menu principal*\nChoisis une action :",
            reply_markup=main_menu_kb(), parse_mode=ParseMode.MARKDOWN,
        )
    elif data == "perf":
        await _send_perf(update, edit=True)
    elif data == "trades_open":
        await _send_trades_open(update, edit=True)
    elif data == "trades_recent":
        await _send_trades_recent(update, edit=True)
    elif data == "hyps_active":
        await _send_hyps_active(update, edit=True)
    elif data == "patterns_stats":
        await _send_patterns_stats(update, edit=True)
    elif data == "scan_now":
        await q.edit_message_text("▶️ Scan en cours...", parse_mode=ParseMode.MARKDOWN)
        res = await _api_post("/scanner/scan-now")
        if res:
            await q.edit_message_text(
                f"✅ Scan lance : `{res.get('job_id', '?')}`",
                reply_markup=back_kb(), parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await q.edit_message_text(
                "❌ Erreur API (FastAPI offline?)",
                reply_markup=back_kb(), parse_mode=ParseMode.MARKDOWN,
            )
    elif data == "backfill_7":
        await q.edit_message_text("⏪ Backfill 7j en cours...", parse_mode=ParseMode.MARKDOWN)
        res = await _api_post("/scanner/backfill", params={"history_bars": 672})
        if res:
            await q.edit_message_text(
                f"✅ Backfill lance : `{res.get('job_id', '?')}`",
                reply_markup=back_kb(), parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await q.edit_message_text(
                "❌ Erreur API",
                reply_markup=back_kb(), parse_mode=ParseMode.MARKDOWN,
            )
    elif data == "dashboard_link":
        await q.edit_message_text(
            f"🌐 *Dashboard*\n\n[Ouvrir]({cfg.DASHBOARD_URL})",
            reply_markup=back_kb(), parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=False,
        )
    elif data == "notif_menu":
        prefs = _load_prefs()
        await q.edit_message_text(
            "🔔 *Notifications*\n\nCliquer pour toggle :",
            reply_markup=notif_menu_kb(prefs), parse_mode=ParseMode.MARKDOWN,
        )
    elif data == "confirm:emergency":
        await q.edit_message_text("🚨 Fermeture en cours...", parse_mode=ParseMode.MARKDOWN)
        res = await _api_post("/execution/emergency_stop")
        if res and res.get("ok"):
            n = len(res.get("closed", []))
            await q.edit_message_text(
                f"✅ *Emergency stop OK*\n\n{n} position(s) fermee(s)\nKillswitch TRIPPED.",
                reply_markup=back_kb(), parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await q.edit_message_text(
                f"❌ Emergency stop failed\n`{res}`",
                reply_markup=back_kb(), parse_mode=ParseMode.MARKDOWN,
            )
    elif data.startswith("notif_toggle:"):
        key = data.split(":", 1)[1]
        prefs = _load_prefs()
        prefs[key] = not prefs.get(key, True)
        _save_prefs(prefs)
        await q.edit_message_text(
            f"🔔 `{key}` -> *{'ON' if prefs[key] else 'OFF'}*",
            reply_markup=notif_menu_kb(prefs), parse_mode=ParseMode.MARKDOWN,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Error handler
# ─────────────────────────────────────────────────────────────────────────────

async def error_handler(update, context: ContextTypes.DEFAULT_TYPE):
    err = context.error
    if isinstance(err, (TimedOut, NetworkError)):
        logger.warning("Reseau (auto-retry): %s", err)
        return
    if isinstance(err, RetryAfter):
        logger.warning("Rate-limited, retry %ss", err.retry_after)
        return
    if isinstance(err, Conflict):
        print(f"\nERREUR: {err}\nUne autre instance tourne deja.")
        _release_pid_lock()
        os._exit(1)
    logger.error("Exception non geree: %s", err, exc_info=err)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    if not _acquire_pid_lock():
        return 1

    if not cfg.BOT_TOKEN:
        print("ERREUR: TELEGRAM_BOT_TOKEN manquant dans .env")
        print()
        print("Setup :")
        print("  1. Ouvre @BotFather sur Telegram")
        print("  2. /newbot, recupere le token")
        print("  3. .env :")
        print("       TELEGRAM_BOT_TOKEN=xxx")
        print("       TELEGRAM_ADMIN_CHAT_ID=xxx")
        print("  4. Ton chat_id : envoie /start au bot, regarde la sortie console")
        return 1

    print("=" * 60)
    print("  ANALYSEUR CRYPTO - Bot Telegram")
    print("=" * 60)
    # Ne loggue pas de fragment de token (meme partiel). Affiche juste sa presence.
    print(f"Token: {'configure (' + str(len(cfg.BOT_TOKEN)) + ' chars)' if cfg.BOT_TOKEN else 'MANQUANT'}")
    if cfg.ALLOWED_USER_IDS:
        print(f"Admin chat_id: {cfg.ALLOWED_USER_IDS}")
    else:
        print("Admin chat_id: NON CONFIGURE — toutes les commandes seront REFUSEES")
        print("  Ajoute TELEGRAM_ADMIN_CHAT_ID=<ton_chat_id> dans .env puis redemarre.")
    print(f"API base: {cfg.API_BASE}")
    print()
    print("Bot demarre - Ctrl+C pour stopper")
    print()

    app = (
        Application.builder()
        .token(cfg.BOT_TOKEN)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .pool_timeout(30.0)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler(["start", "menu"], cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("perf", cmd_perf))
    app.add_handler(CommandHandler("open", cmd_open))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("hyps", cmd_hyps))
    app.add_handler(CommandHandler("patterns", cmd_patterns))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("backfill", cmd_backfill))
    app.add_handler(CommandHandler("notif", cmd_notif))
    app.add_handler(CommandHandler("test", cmd_test))
    app.add_handler(CommandHandler("dashboard", cmd_dashboard))
    app.add_handler(CommandHandler("exec", cmd_exec_status))
    app.add_handler(CommandHandler("emergency_stop", cmd_emergency_stop))
    app.add_handler(CommandHandler("emergency", cmd_emergency_stop))
    app.add_handler(CommandHandler("reset_killswitch", cmd_reset_killswitch))

    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_error_handler(error_handler)

    try:
        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
            poll_interval=2.0,
            timeout=20,
        )
    finally:
        _release_pid_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
