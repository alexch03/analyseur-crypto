"""Telegram notifier: sends annotated chart images + signal text to a Telegram chat.

Supports both single-setup messages and a grouped summary of multiple setups
in a single message to avoid spamming the chat.
"""

from __future__ import annotations

import io
import logging

from telegram import Bot

from app.schemas.domain import Side, TradeSetupDTO
from app.telegram.interfaces import DeliveryReceipt

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot = Bot(token=bot_token)
        self._chat_id = chat_id

    async def send(
        self,
        setup: TradeSetupDTO,
        chart_png: bytes | None = None,
        signal_id: int = 0,
    ) -> DeliveryReceipt:
        text = self._format_message(setup)

        try:
            if chart_png:
                photo = io.BytesIO(chart_png)
                photo.name = "chart.png"
                await self._bot.send_photo(
                    chat_id=self._chat_id,
                    photo=photo,
                    caption=text,
                    parse_mode="HTML",
                )
            else:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode="HTML",
                )

            return DeliveryReceipt(
                signal_id=signal_id,
                chat_id=self._chat_id,
                success=True,
            )

        except Exception as exc:
            logger.exception("Failed to send Telegram message")
            return DeliveryReceipt(
                signal_id=signal_id,
                chat_id=self._chat_id,
                success=False,
                error=str(exc),
            )

    async def send_summary(
        self,
        setups: list[TradeSetupDTO],
        chart_png: bytes | None = None,
        trend: str = "",
    ) -> DeliveryReceipt:
        """Send a single grouped message summarizing all setups."""
        if not setups:
            return DeliveryReceipt(signal_id=0, chat_id=self._chat_id, success=True)

        text = self._format_summary(setups, trend=trend)
        try:
            if chart_png:
                photo = io.BytesIO(chart_png)
                photo.name = "chart.png"
                caption = text[:1024]
                await self._bot.send_photo(
                    chat_id=self._chat_id,
                    photo=photo,
                    caption=caption,
                    parse_mode="HTML",
                )
                if len(text) > 1024:
                    await self._bot.send_message(
                        chat_id=self._chat_id,
                        text=text,
                        parse_mode="HTML",
                    )
            else:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode="HTML",
                )
            return DeliveryReceipt(signal_id=0, chat_id=self._chat_id, success=True)
        except Exception as exc:
            logger.exception("Failed to send Telegram summary")
            return DeliveryReceipt(
                signal_id=0,
                chat_id=self._chat_id,
                success=False,
                error=str(exc),
            )

    async def send_plain_html(self, html: str) -> DeliveryReceipt:
        """Message texte seul (ex. événements paper live : entrée / clôture trade simulé)."""
        if not html.strip():
            return DeliveryReceipt(signal_id=0, chat_id=self._chat_id, success=True)
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=html,
                parse_mode="HTML",
            )
            return DeliveryReceipt(signal_id=0, chat_id=self._chat_id, success=True)
        except Exception as exc:
            logger.exception("Failed to send Telegram plain HTML message")
            return DeliveryReceipt(
                signal_id=0,
                chat_id=self._chat_id,
                success=False,
                error=str(exc),
            )

    @staticmethod
    def _format_message(setup: TradeSetupDTO) -> str:
        direction = "\U0001f7e2 LONG" if setup.side == Side.LONG else "\U0001f534 SHORT"
        tps = "\n".join(f"  TP{i+1}: <b>{tp:.2f}</b>" for i, tp in enumerate(setup.take_profits))

        return (
            f"<b>\U0001f4ca {setup.symbol} | {setup.timeframe}</b>\n"
            f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            f"{direction} \u2014 <b>{setup.setup_type}</b>\n\n"
            f"\u25b8 Entry: <b>{setup.entry:.2f}</b>\n"
            f"\u25b8 Stop Loss: <b>{setup.stop_loss:.2f}</b>\n"
            f"{tps}\n\n"
            f"\u25b8 R:R: <b>{setup.risk_reward:.1f}</b>\n"
            f"\u25b8 Confidence: <b>{setup.confidence:.0%}</b>\n\n"
            f"<i>{setup.rationale}</i>\n"
            f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
        )

    @staticmethod
    def _format_summary(setups: list[TradeSetupDTO], *, trend: str = "") -> str:
        sym = setups[0].symbol
        tf = setups[0].timeframe
        header = f"<b>\U0001f4ca {sym} | {tf}</b>"
        if trend:
            header += f"  \u2014  Trend: <b>{trend}</b>"
        header += f"\n{len(setups)} signal(s) detect\u00e9(s)\n"
        header += "\u2501" * 18 + "\n"

        lines: list[str] = [header]
        for i, s in enumerate(setups, 1):
            d = "\U0001f7e2" if s.side == Side.LONG else "\U0001f534"
            tp_str = " / ".join(f"{tp:.2f}" for tp in s.take_profits)
            lines.append(
                f"{i}. {d} <b>{s.setup_type}</b> ({s.confidence:.0%})\n"
                f"   Entry {s.entry:.2f}  SL {s.stop_loss:.2f}  TP {tp_str}\n"
                f"   R:R {s.risk_reward:.1f}  \u2014  <i>{s.rationale}</i>\n"
            )

        lines.append("\u2501" * 18)
        return "\n".join(lines)
