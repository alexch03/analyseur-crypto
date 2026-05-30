"""Safety guards CRITIQUES pour le trading live.

Verifie avant CHAQUE ouverture de position :
    1. Mode (disabled / paper / demo / live)
    2. Max position size en USDT
    3. Max nombre de positions ouvertes
    4. Max perte journaliere
    5. Killswitch : N pertes consecutives
    6. Symbole en blacklist
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import json

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
STATE_FILE = ROOT / "data" / "safety_state.json"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class SafetyConfig:
    mode: str = "disabled"  # disabled | paper | demo | live
    max_position_usd: float = 100.0      # taille max par trade
    max_open_positions: int = 5          # max positions ouvertes simultanees
    max_daily_loss_usd: float = 200.0    # killswitch pertes du jour
    max_consecutive_losses: int = 5      # killswitch losses consecutives
    blacklist_symbols: list[str] = field(default_factory=list)
    whitelist_symbols: list[str] = field(default_factory=list)  # si non vide, SEULS ces symbols sont autorises
    allowed_sides: tuple[str, ...] = ("LONG", "SHORT")  # ("LONG",) pour spot-only
    min_balance_usd: float = 10.0        # n'ouvre pas si balance < min


@dataclass
class SafetyState:
    """Etat persisté entre redemarrages."""
    day: str = ""                        # YYYY-MM-DD du jour courant
    daily_pnl_usd: float = 0.0
    consecutive_losses: int = 0
    killswitch_tripped: bool = False
    killswitch_reason: str = ""
    last_updated: str = ""


def _load_state() -> SafetyState:
    if not STATE_FILE.exists():
        return SafetyState(day=date.today().isoformat())
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return SafetyState(**data)
    except Exception as e:
        logger.warning("Cannot load safety state: %s", e)
        return SafetyState(day=date.today().isoformat())


def _save_state(state: SafetyState) -> None:
    state.last_updated = datetime.utcnow().isoformat()
    try:
        STATE_FILE.write_text(
            json.dumps(state.__dict__, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning("Cannot save safety state: %s", e)


def _ensure_today(state: SafetyState) -> SafetyState:
    """Reset journalier si on a change de jour."""
    today = date.today().isoformat()
    if state.day != today:
        state.day = today
        state.daily_pnl_usd = 0.0
        # consecutive_losses reste (carry across days)
        # killswitch_tripped reste aussi (l'humain doit le reset)
        _save_state(state)
    return state


class SafetyGuard:
    """Verifie chaque ordre avant execution."""

    def __init__(self, config: SafetyConfig) -> None:
        self._cfg = config
        self._state = _ensure_today(_load_state())

    @property
    def state(self) -> SafetyState:
        return self._state

    @property
    def config(self) -> SafetyConfig:
        return self._cfg

    def is_disabled(self) -> bool:
        return self._cfg.mode == "disabled"

    def is_live(self) -> bool:
        return self._cfg.mode == "live"

    def killswitch_tripped(self) -> bool:
        return self._state.killswitch_tripped

    def can_open(self, *, symbol: str, side: str, size_usd: float,
                  balance_usd: float, open_positions_count: int) -> tuple[bool, str]:
        """Verifie si une nouvelle position peut etre ouverte.

        Retourne (ok, reason). ok=False = refuser, reason explique pourquoi.
        """
        self._state = _ensure_today(self._state)
        self._evaluate_killswitch()  # trip si un seuil est deja franchi

        if self._cfg.mode == "disabled":
            return False, "execution disabled (set EXECUTION_MODE in .env)"

        if self._state.killswitch_tripped:
            return False, f"killswitch active: {self._state.killswitch_reason}"

        if side not in self._cfg.allowed_sides:
            return False, f"side {side} not in allowed_sides {self._cfg.allowed_sides}"

        if symbol in self._cfg.blacklist_symbols:
            return False, f"symbol {symbol} in blacklist"

        # Whitelist : si non vide, SEULS ces symbols passent (autres = paper only)
        if self._cfg.whitelist_symbols and symbol not in self._cfg.whitelist_symbols:
            return False, f"symbol {symbol} not in demo whitelist ({len(self._cfg.whitelist_symbols)} symbols)"

        if size_usd <= 0:
            return False, "size_usd must be > 0"

        if size_usd > self._cfg.max_position_usd:
            return False, f"size {size_usd}$ > max_position_usd {self._cfg.max_position_usd}$"

        if open_positions_count >= self._cfg.max_open_positions:
            return False, f"max_open_positions {self._cfg.max_open_positions} reached"

        if balance_usd < self._cfg.min_balance_usd:
            return False, f"balance {balance_usd}$ < min {self._cfg.min_balance_usd}$"

        return True, ""

    def record_close(self, *, pnl_usd: float) -> None:
        """Enregistre le PnL d'une cloture pour suivi killswitches.

        IMPORTANT : le killswitch est evalue ICI (a chaque cloture), pas seulement
        dans ``can_open``. Sinon il ne se declenche jamais en mode paper/disabled
        (ou ``can_open`` court-circuite avant les seuils) — exactement le bug
        observe : -598$ / 26 pertes consecutives sans declenchement.
        """
        self._state = _ensure_today(self._state)
        self._state.daily_pnl_usd += pnl_usd
        if pnl_usd < 0:
            self._state.consecutive_losses += 1
        elif pnl_usd > 0:
            self._state.consecutive_losses = 0
        self._evaluate_killswitch()
        _save_state(self._state)

    def _evaluate_killswitch(self) -> None:
        """Declenche le killswitch si un seuil de perte est franchi.

        Independant du mode : on veut savoir qu'on AURAIT ete coupe meme en paper.
        """
        if self._state.killswitch_tripped:
            return
        if self._state.daily_pnl_usd <= -self._cfg.max_daily_loss_usd:
            self._trip_killswitch(
                f"daily loss {self._state.daily_pnl_usd:.2f}$ <= -{self._cfg.max_daily_loss_usd}$"
            )
        elif self._state.consecutive_losses >= self._cfg.max_consecutive_losses:
            self._trip_killswitch(
                f"{self._state.consecutive_losses} consecutive losses "
                f">= {self._cfg.max_consecutive_losses}"
            )

    def _trip_killswitch(self, reason: str) -> None:
        if not self._state.killswitch_tripped:
            self._state.killswitch_tripped = True
            self._state.killswitch_reason = reason
            _save_state(self._state)
            logger.error("KILLSWITCH TRIPPED: %s", reason)

    def reset_killswitch(self) -> None:
        """Reset manuel apres review."""
        self._state.killswitch_tripped = False
        self._state.killswitch_reason = ""
        self._state.consecutive_losses = 0
        _save_state(self._state)
        logger.warning("Killswitch RESET")

    def status_text(self) -> str:
        s = self._state
        c = self._cfg
        ks = "TRIPPED" if s.killswitch_tripped else "OK"
        return (
            f"mode={c.mode}  killswitch={ks}\n"
            f"daily_pnl={s.daily_pnl_usd:+.2f}$ / -{c.max_daily_loss_usd}$\n"
            f"consec_losses={s.consecutive_losses}/{c.max_consecutive_losses}\n"
            f"max_pos_size={c.max_position_usd}$  "
            f"max_open={c.max_open_positions}"
        )
