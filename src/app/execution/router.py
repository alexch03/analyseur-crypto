"""Router d'execution : decide quel executor utiliser selon EXECUTION_MODE.

Mode :
    disabled : pas d'execution du tout (juste tracking interne paper)
    paper    : PaperExecutor (DB only, zero risque)
    demo     : BitgetDemoExecutor (testnet Bitget, vrais ordres sandbox)
    live     : BitgetLiveExecutor (VRAIS ORDRES VRAI ARGENT)
"""

from __future__ import annotations

import logging
import os

from app.execution.base import Executor
from app.execution.paper_executor import PaperExecutor
from app.execution.safety import SafetyConfig, SafetyGuard

logger = logging.getLogger(__name__)


_executor_singleton: Executor | None = None
_safety_singleton: SafetyGuard | None = None


def _build_safety_from_env() -> SafetyGuard:
    # DEMO_SYMBOLS sert de whitelist : si configure, seuls ces symbols
    # passent l'ordre exchange (paper continue pour TOUS via DB).
    whitelist = [
        s.strip() for s in os.environ.get("DEMO_SYMBOLS", "").split(",")
        if s.strip()
    ]
    cfg = SafetyConfig(
        mode=os.environ.get("EXECUTION_MODE", "disabled").lower(),
        max_position_usd=float(os.environ.get("MAX_POSITION_USD", "50")),
        max_open_positions=int(os.environ.get("MAX_OPEN_POSITIONS", "5")),
        max_daily_loss_usd=float(os.environ.get("MAX_DAILY_LOSS_USD", "100")),
        max_consecutive_losses=int(os.environ.get("MAX_CONSECUTIVE_LOSSES", "5")),
        blacklist_symbols=[
            s.strip() for s in os.environ.get("BLACKLIST_SYMBOLS", "").split(",")
            if s.strip()
        ],
        whitelist_symbols=whitelist,
        min_balance_usd=float(os.environ.get("MIN_BALANCE_USD", "10")),
    )
    return SafetyGuard(cfg)


def get_safety() -> SafetyGuard:
    global _safety_singleton
    if _safety_singleton is None:
        _safety_singleton = _build_safety_from_env()
    return _safety_singleton


def reset_safety() -> None:
    global _safety_singleton, _executor_singleton
    _safety_singleton = None
    _executor_singleton = None


async def get_executor() -> Executor:
    """Retourne le bon executor selon EXECUTION_MODE (lazy init).

    Si Bitget credentials manquants en mode demo/live, fallback paper.
    """
    global _executor_singleton
    if _executor_singleton is not None:
        return _executor_singleton

    safety = get_safety()
    mode = safety.config.mode

    if mode in ("disabled", "paper"):
        _executor_singleton = PaperExecutor()
        logger.info("Executor: PaperExecutor (mode=%s)", mode)
        return _executor_singleton

    # Bitget demo ou live : import lazy pour ne pas casser si ccxt bitget pas dispo
    try:
        from app.execution.bitget_client import BitgetClient
        from app.execution.bitget_executor import BitgetExecutor
    except ImportError as e:
        logger.error("Bitget executor unavailable (%s), fallback paper", e)
        _executor_singleton = PaperExecutor()
        return _executor_singleton

    try:
        client = BitgetClient(demo=(mode == "demo"))
        leverage = int(os.environ.get("BITGET_LEVERAGE", "1"))
        _executor_singleton = BitgetExecutor(
            client=client, safety=safety, default_leverage=leverage,
        )
        logger.warning(
            "Executor: BitgetExecutor mode=%s leverage=%dx max_pos=%s$ "
            "max_daily_loss=%s$ killswitch=%d",
            mode, leverage,
            safety.config.max_position_usd,
            safety.config.max_daily_loss_usd,
            safety.config.max_consecutive_losses,
        )
        return _executor_singleton
    except Exception as e:
        logger.error("Cannot init Bitget client (%s), fallback PaperExecutor", e)
        _executor_singleton = PaperExecutor()
        return _executor_singleton


async def shutdown_executor() -> None:
    global _executor_singleton
    if _executor_singleton is not None:
        try:
            await _executor_singleton.close()
        except Exception:
            pass
        _executor_singleton = None
