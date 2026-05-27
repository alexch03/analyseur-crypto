"""Analytics : decompose les trades cloturés par segment pour identifier
les sous-ensembles profitables et orienter le tuning des filtres.

Segments :
    - par pattern_kind (DOUBLE_TOP, FLAG_BULL, ...)
    - par tag de confluence (volume_expansion, trend_aligned, ...)
    - par bucket de confluence_score (0-0.3, 0.3-0.5, 0.5-0.7, 0.7-1.0)
    - par timeframe
    - par symbol (top N)
    - par side (LONG / SHORT)

Pour chaque segment : count, win_rate, avg_pct, expectancy, cumul_simple, cumul_compound.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Symbol, Timeframe, UnitTrade


@dataclass(frozen=True, slots=True)
class SegmentStats:
    label: str
    count: int
    win_count: int
    loss_count: int
    win_rate_pct: float
    avg_pct: float
    cumul_simple_pct: float
    cumul_compound_pct: float
    best_pct: float
    worst_pct: float
    expectancy_pct: float


def _compute(label: str, gains: list[float]) -> SegmentStats:
    if not gains:
        return SegmentStats(label, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wins = [g for g in gains if g > 0]
    losses = [g for g in gains if g < 0]
    n = len(gains)
    win_rate = len(wins) / n
    avg = sum(gains) / n
    compound = 1.0
    for g in gains:
        compound *= 1.0 + g / 100.0
    compound_pct = (compound - 1.0) * 100.0
    avg_w = sum(wins) / len(wins) if wins else 0.0
    avg_l = sum(losses) / len(losses) if losses else 0.0
    exp = win_rate * avg_w + (1.0 - win_rate) * avg_l
    return SegmentStats(
        label=label,
        count=n,
        win_count=len(wins),
        loss_count=len(losses),
        win_rate_pct=round(win_rate * 100.0, 2),
        avg_pct=round(avg, 3),
        cumul_simple_pct=round(sum(gains), 2),
        cumul_compound_pct=round(compound_pct, 2),
        best_pct=round(max(gains), 2),
        worst_pct=round(min(gains), 2),
        expectancy_pct=round(exp, 3),
    )


def _bucket_score(score: float) -> str:
    if score < 0.3:
        return "0.0-0.3"
    if score < 0.5:
        return "0.3-0.5"
    if score < 0.7:
        return "0.5-0.7"
    return "0.7-1.0"


@dataclass
class _ClosedTrade:
    symbol: str
    timeframe: str
    pattern_kind: str
    side: str
    pct_gain: float
    confluence_score: float
    confluence_tags: list[str]


async def load_closed_trades(session: AsyncSession) -> list[_ClosedTrade]:
    q = (
        select(UnitTrade, Symbol, Timeframe)
        .join(Symbol, Symbol.id == UnitTrade.symbol_id)
        .join(Timeframe, Timeframe.id == UnitTrade.timeframe_id)
        .where(UnitTrade.exit_price.is_not(None))
        .where(UnitTrade.pct_gain.is_not(None))
    )
    rows = (await session.execute(q)).all()
    return [
        _ClosedTrade(
            symbol=f"{s.base}/{s.quote}",
            timeframe=t.code,
            pattern_kind=ut.pattern_kind,
            side=ut.side,
            pct_gain=float(ut.pct_gain or 0.0),
            confluence_score=float(ut.confluence_score or 0.0),
            confluence_tags=list(ut.confluence_tags or []),
        )
        for ut, s, t in rows
    ]


def _group_by(
    trades: Iterable[_ClosedTrade], key_fn
) -> list[SegmentStats]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        for key in key_fn(t):  # key_fn peut retourner plusieurs labels (ex: tags)
            buckets[key].append(t.pct_gain)
    out = [_compute(k, gains) for k, gains in buckets.items()]
    return sorted(out, key=lambda s: -s.expectancy_pct)


async def compute_breakdowns(session: AsyncSession) -> dict[str, list[dict]]:
    trades = await load_closed_trades(session)
    overall = _compute("ALL", [t.pct_gain for t in trades])

    return {
        "overall": [_segment_to_dict(overall)],
        "by_pattern": [
            _segment_to_dict(s)
            for s in _group_by(trades, lambda t: [t.pattern_kind])
        ],
        "by_tag": [
            _segment_to_dict(s)
            for s in _group_by(trades, lambda t: t.confluence_tags or ["<no_tag>"])
        ],
        "by_score_bucket": [
            _segment_to_dict(s)
            for s in _group_by(trades, lambda t: [_bucket_score(t.confluence_score)])
        ],
        "by_timeframe": [
            _segment_to_dict(s)
            for s in _group_by(trades, lambda t: [t.timeframe])
        ],
        "by_side": [
            _segment_to_dict(s)
            for s in _group_by(trades, lambda t: [t.side])
        ],
        "by_symbol_top20": [
            _segment_to_dict(s)
            for s in _group_by(trades, lambda t: [t.symbol])[:20]
        ],
        "by_pattern_x_tag": [
            _segment_to_dict(s)
            for s in _group_by(
                trades,
                lambda t: [f"{t.pattern_kind} + {tag}" for tag in (t.confluence_tags or ["<no_tag>"])],
            )[:30]
        ],
    }


def _segment_to_dict(s: SegmentStats) -> dict:
    return {
        "label": s.label,
        "count": s.count,
        "win_count": s.win_count,
        "loss_count": s.loss_count,
        "win_rate_pct": s.win_rate_pct,
        "avg_pct": s.avg_pct,
        "cumul_simple_pct": s.cumul_simple_pct,
        "cumul_compound_pct": s.cumul_compound_pct,
        "best_pct": s.best_pct,
        "worst_pct": s.worst_pct,
        "expectancy_pct": s.expectancy_pct,
    }


# ----------------------------------------------------------------------
# Optimization : grid search "virtuel" sur les filtres
# ----------------------------------------------------------------------

def _filter_trades(
    trades: list[_ClosedTrade],
    *,
    min_score: float,
    reject_tags: tuple[str, ...],
    required_tags: tuple[str, ...],
    pattern_kinds: tuple[str, ...] | None,
) -> list[float]:
    out: list[float] = []
    for t in trades:
        if t.confluence_score < min_score:
            continue
        if reject_tags and any(tag in t.confluence_tags for tag in reject_tags):
            continue
        if required_tags and not all(tag in t.confluence_tags for tag in required_tags):
            continue
        if pattern_kinds and t.pattern_kind not in pattern_kinds:
            continue
        out.append(t.pct_gain)
    return out


async def optimize_filters(session: AsyncSession, top_n: int = 20) -> dict:
    """Grid search sur les filtres applicables a posteriori aux trades cloturés.

    On ne peut pas simuler le breakeven (besoin de l'evolution intra-trade).
    Mais on peut simuler tous les autres filtres : min_score, reject_tags, etc.
    Retourne le top N configs par cumul compound.
    """
    trades = await load_closed_trades(session)
    if not trades:
        return {"trades_available": 0, "results": []}

    score_levels = [0.0, 0.35, 0.45, 0.55, 0.65]
    reject_combos = [
        (),
        ("trend_counter",),
        ("volume_weak",),
        ("trend_counter", "volume_weak"),
    ]
    require_combos = [
        (),
        ("volume_expansion",),
        ("trend_aligned",),
        ("volume_expansion", "trend_aligned"),
    ]
    pattern_combos = [None]   # all patterns - on pourrait drill down par pattern

    candidates: list[dict] = []
    for ms in score_levels:
        for rj in reject_combos:
            for rq in require_combos:
                gains = _filter_trades(
                    trades,
                    min_score=ms,
                    reject_tags=rj,
                    required_tags=rq,
                    pattern_kinds=pattern_combos[0],
                )
                if len(gains) < 20:   # minimum stat pour etre fiable
                    continue
                s = _compute(
                    label=f"score>={ms} reject={rj or '-'} require={rq or '-'}",
                    gains=gains,
                )
                candidates.append({
                    **_segment_to_dict(s),
                    "config": {
                        "min_confluence_score": ms,
                        "reject_tags": list(rj),
                        "required_tags": list(rq),
                    },
                })

    # Tri : cumul compound > expectancy (les deux importent, on optimise le cumul d'abord)
    candidates.sort(key=lambda x: (-x["cumul_compound_pct"], -x["expectancy_pct"]))

    baseline = _compute("baseline", [t.pct_gain for t in trades])

    return {
        "trades_available": len(trades),
        "baseline": _segment_to_dict(baseline),
        "top_configs": candidates[:top_n],
    }
