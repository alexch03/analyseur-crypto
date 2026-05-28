"""Interface Executor : abstrait l'execution des trades.

Trois implementations :
    - PaperExecutor : simulation interne (DB only, zero risque)
    - BitgetDemoExecutor : ordres sur Bitget Demo testnet (apprentissage)
    - BitgetLiveExecutor : VRAIS ORDRES VRAI ARGENT (sois prudent)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OrderRequest:
    """Requete d'ouverture d'une position."""
    hypothesis_id: str
    symbol: str           # ex: "BTC/USDT:USDT" pour futures
    side: str             # "LONG" ou "SHORT"
    entry_price: float    # prix de declenchement (limite, peut diff de prix execution)
    target_price: float
    invalidation_price: float
    size_usd: float       # taille en USDT (notional)
    leverage: int = 1


@dataclass
class CloseRequest:
    """Requete de cloture (TARGET_HIT, STOPPED, INVALIDATED, EXPIRED, MANUAL)."""
    hypothesis_id: str
    symbol: str
    side: str             # cote actuelle de la position
    reason: str           # raison de cloture


@dataclass
class OrderResult:
    """Resultat d'execution (succes ou echec)."""
    ok: bool
    order_id: str | None = None
    filled_price: float | None = None
    filled_qty: float | None = None
    exchange: str = "internal"   # binance, bitget, internal
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: str | None = None


class Executor(ABC):
    """Interface abstraite d'execution."""

    name: str = "base"

    @abstractmethod
    async def open_position(self, req: OrderRequest) -> OrderResult:
        """Ouvre une nouvelle position selon req. Retourne OrderResult."""
        ...

    @abstractmethod
    async def close_position(self, req: CloseRequest) -> OrderResult:
        """Ferme une position existante."""
        ...

    @abstractmethod
    async def fetch_open_positions(self) -> list[dict]:
        """Liste les positions actuellement ouvertes (raw dict)."""
        ...

    @abstractmethod
    async def fetch_balance(self) -> dict:
        """Solde courant (USDT free/used/total)."""
        ...

    async def close(self) -> None:
        """Cleanup ressources (sessions HTTP, etc.)."""
        return
