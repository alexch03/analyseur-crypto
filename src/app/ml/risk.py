"""Plan de trade ajusté à la volatilité (ATR).

Corrige la cause racine des 85 % de stop-out : le coefficient #1 du modèle est
``stop_dist_pct`` (+0.87) — les stops sont trop serrés et se font sortir par le
bruit. On élargit donc le stop à **au moins k×ATR**, et on fixe le target à un
**RR constant** depuis ce stop (au lieu d'un target trop lointain via
``target_multiplier`` 1.3, dont le coef ``tgt_dist_pct`` était négatif).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TradePlan:
    entry: float
    stop: float
    target: float


def atr_trade_plan(
    *,
    side: str,
    entry: float,
    raw_invalidation: float,
    atr: float,
    k_stop: float = 2.0,
    rr_target: float = 2.0,
) -> TradePlan:
    """Stop = le plus large entre l'invalidation du pattern et entry ∓ k×ATR ;
    target = entry ± rr_target × distance_stop.

    ``side`` : 'LONG' ou 'SHORT'. Si ATR indisponible (<=0), on garde le stop du
    pattern et on aligne juste le target sur le RR demandé.
    """
    s = side.upper()
    if atr is None or atr <= 0 or entry <= 0:
        stop = raw_invalidation
        dist = abs(entry - stop)
        target = entry + rr_target * dist if s == "LONG" else entry - rr_target * dist
        return TradePlan(entry, stop, target)

    if s == "LONG":
        stop = min(raw_invalidation, entry - k_stop * atr)  # le plus bas = le plus large
        dist = entry - stop
        target = entry + rr_target * dist
    else:
        stop = max(raw_invalidation, entry + k_stop * atr)  # le plus haut = le plus large
        dist = stop - entry
        target = entry - rr_target * dist
    return TradePlan(entry, stop, target)
