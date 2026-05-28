"""Notifier centralise - envoie messages Telegram (silent fail si non configure).

Appele par HypothesisEngine ou Scanner pour notifier:
  - Hypothesis TRIGGERED (nouveau trade ouvert)
  - Hypothesis TARGET_HIT / STOPPED / INVALIDATED (trade ferme)
  - Erreurs systeme

Utilise un thread daemon + file pour ne PAS bloquer le scanner si Telegram
est lent ou hors-ligne.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any

logger = logging.getLogger(__name__)

# Queue pour les messages a envoyer (thread-safe)
_send_queue: Queue[tuple[str, str]] = Queue()
_worker_started = False
_worker_lock = threading.Lock()


def _send_sync(text: str, parse_mode: str = "Markdown") -> bool:
    """Envoi synchrone d'un message Telegram. Retourne True si OK."""
    from app.tg_bot import config as cfg

    if not cfg.is_ready():
        return False

    try:
        from telegram import Bot

        bot = Bot(token=cfg.BOT_TOKEN)

        async def _send():
            for chat_id in cfg.ALLOWED_USER_IDS:
                try:
                    await bot.send_message(
                        chat_id=chat_id, text=text, parse_mode=parse_mode,
                        disable_web_page_preview=True,
                    )
                except Exception as e:
                    logger.warning("Telegram send failed (chat %s): %s", chat_id, e)

        # Cree un nouvel event loop pour l'envoi (on est dans un thread daemon)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_send())
            loop.close()
        except Exception as e:
            logger.warning("Telegram event loop error: %s", e)
            return False
        return True
    except Exception as e:
        logger.warning("Telegram bot init failed: %s", e)
        return False


def _worker_loop() -> None:
    """Thread daemon : depile les messages et les envoie."""
    while True:
        try:
            text, parse_mode = _send_queue.get(timeout=5.0)
        except Empty:
            continue
        try:
            _send_sync(text, parse_mode)
        except Exception:
            logger.exception("Telegram worker crash")


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_worker_loop, daemon=True, name="tg-notifier")
        t.start()
        _worker_started = True


def send_async(text: str, parse_mode: str = "Markdown") -> None:
    """Empile un message a envoyer en arriere-plan. Ne bloque pas."""
    _ensure_worker()
    _send_queue.put((text, parse_mode))


# ─────────────────────────────────────────────────────────────────────────────
# Dispatchers de haut niveau
# ─────────────────────────────────────────────────────────────────────────────


def dispatch_hypothesis_triggered(
    *,
    symbol: str,
    timeframe: str,
    pattern: str,
    side: str,
    entry: float,
    target: float,
    invalidation: float,
    confluence_score: float,
) -> None:
    """Notif Telegram quand une hypothese passe a TRIGGERED."""
    from app.tg_bot import config as cfg
    if not cfg.NOTIFY_ON_TRIGGER or not cfg.is_ready():
        return
    rr = abs(target - entry) / max(1e-9, abs(invalidation - entry))
    emoji = "🟢" if side == "LONG" else "🔴"
    text = (
        f"{emoji} *TRIGGERED* — `{pattern}`\n\n"
        f"`{symbol}` {timeframe}  *{side}*\n"
        f"  Entry: `${entry:,.4f}`\n"
        f"  Target: `${target:,.4f}`\n"
        f"  SL: `${invalidation:,.4f}`\n"
        f"  R:R `{rr:.2f}`  Score `{confluence_score:.2f}`"
    )
    send_async(text)


def dispatch_hypothesis_closed(
    *,
    symbol: str,
    timeframe: str,
    pattern: str,
    side: str,
    outcome: str,
    entry: float,
    exit_price: float,
    pct_gain: float,
) -> None:
    """Notif Telegram quand une hypothese se ferme (TARGET_HIT, STOPPED, etc)."""
    from app.tg_bot import config as cfg
    if not cfg.NOTIFY_ON_CLOSE or not cfg.is_ready():
        return

    if outcome == "TARGET_HIT":
        emoji = "✅"
    elif outcome == "STOPPED":
        emoji = "❌"
    elif outcome == "INVALIDATED":
        emoji = "⏸"
    elif outcome == "EXPIRED":
        emoji = "⏱"
    else:
        emoji = "❔"

    text = (
        f"{emoji} *CLOSED {outcome}* — `{pattern}`\n\n"
        f"`{symbol}` {timeframe}  {side}\n"
        f"  Entry: `${entry:,.4f}` -> Exit: `${exit_price:,.4f}`\n"
        f"  PnL: `{pct_gain:+.2f}%`"
    )
    send_async(text)


def dispatch_error(message: str) -> None:
    from app.tg_bot import config as cfg
    if not cfg.is_ready():
        return
    text = f"🚨 *ERROR*\n\n`{message[:500]}`"
    send_async(text)


def dispatch_test() -> bool:
    """Envoi sync d'un message test. Retourne True/False."""
    text = (
        f"🧪 *TEST Analyseur Crypto*\n\n"
        f"Envoye a `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`\n"
        f"Bot OK ✓"
    )
    return _send_sync(text)
