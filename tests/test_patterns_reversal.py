"""Tests pour les patterns de retournement : Double Top/Bottom + H&S/iH&S."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.market_structure.swings import detect_swings
from app.patterns.reversal import ReversalDetector
from app.schemas.patterns import BreakoutDirection, PatternKind


def _interp(pivots: list[tuple[int, float]], n: int) -> list[float]:
    pivots = sorted(pivots, key=lambda x: x[0])
    closes = [0.0] * n
    for i, p in pivots:
        closes[i] = p
    for k in range(len(pivots) - 1):
        i0, p0 = pivots[k]
        i1, p1 = pivots[k + 1]
        for j in range(i0 + 1, i1):
            t = (j - i0) / (i1 - i0)
            closes[j] = p0 + (p1 - p0) * t
    fi, fp = pivots[0]
    li, lp = pivots[-1]
    for j in range(0, fi):
        closes[j] = fp
    for j in range(li + 1, n):
        closes[j] = lp
    return closes


def _df(closes: list[float], wick: float = 0.3) -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return pd.DataFrame([
        {
            "timestamp": start + timedelta(hours=i),
            "open": c, "high": c + wick, "low": c - wick, "close": c, "volume": 100.0,
        }
        for i, c in enumerate(closes)
    ])


def _detector() -> ReversalDetector:
    """Détecteur configuré pour tester la GÉOMÉTRIE des retournements en isolation.

    Les filtres de confluence dépendant d'un long historique ou d'indicateurs sont
    désactivés car les fixtures synthétiques (~22-28 bougies, interpolation linéaire)
    ne peuvent structurellement pas les satisfaire :
      - ``require_pre_trend`` : exige 20 bougies de tendance AVANT le 1er swing, qui
        se trouve ici à l'index 4.
      - ``require_rsi_divergence`` : une divergence RSI de 2 pts est infaisable sur des
        rampes linéaires symétriques.
      - ``min_swing_prominence_atr`` : les sommets/creux jumeaux sont distants de ~0.5pt
        alors que 0.5×ATR ≈ 1.2pt, donc le 2e swing serait filtré.
    On vérifie ici la détection de la FORME (neckline, sens de cassure, invalidation,
    target) ; ces filtres ont leur propre couverture.
    """
    return ReversalDetector(
        require_pre_trend=False,
        require_rsi_divergence=False,
        min_swing_prominence_atr=0.0,
    )


def test_double_top_detected():
    # H pivot 110 idx 4 ; L pivot 100 idx 9 ; H pivot 110.5 idx 14
    # close encore au-dessus de la neckline (100)
    pivots = [(0, 100.0), (4, 110.0), (9, 100.0), (14, 110.5), (18, 104.0)]
    closes = _interp(pivots, 22)
    df = _df(closes)
    swings = detect_swings(df, left=2, right=2)
    p = _detector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    dt = [x for x in p if x.kind == PatternKind.DOUBLE_TOP]
    assert len(dt) == 1, f"Got patterns: {[(x.kind, x.payload) for x in p]}"
    assert dt[0].breakout_direction == BreakoutDirection.DOWN
    assert 99.5 <= dt[0].breakout_level <= 100.5
    assert dt[0].invalidation_level >= 110.0
    assert dt[0].target is not None and dt[0].target < dt[0].breakout_level


def test_double_bottom_detected():
    # L 90 idx 4 ; H 100 idx 9 ; L 90.5 idx 14 ; close encore sous neckline
    pivots = [(0, 100.0), (4, 90.0), (9, 100.0), (14, 90.5), (18, 96.0)]
    closes = _interp(pivots, 22)
    df = _df(closes)
    swings = detect_swings(df, left=2, right=2)
    p = _detector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    db = [x for x in p if x.kind == PatternKind.DOUBLE_BOTTOM]
    assert len(db) == 1
    assert db[0].breakout_direction == BreakoutDirection.UP
    assert 99.5 <= db[0].breakout_level <= 100.5
    assert db[0].invalidation_level <= 90.5


def test_head_shoulders_detected():
    # LS 110 idx 4 ; neck-L 100 idx 8 ; HEAD 120 idx 12 ; neck-R 100 idx 16 ; RS 112 idx 20
    # Neckline horizontale (100/100) : une neckline ascendante (100→101) se projette
    # au-dessus du dernier close et serait lue comme "déjà cassée à la baisse" → rejet.
    pivots = [
        (0, 100.0),
        (4, 110.0),
        (8, 100.0),
        (12, 120.0),
        (16, 100.0),
        (20, 112.0),
        (24, 103.0),
    ]
    closes = _interp(pivots, 28)
    df = _df(closes)
    swings = detect_swings(df, left=2, right=2)
    p = _detector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    hs = [x for x in p if x.kind == PatternKind.HEAD_SHOULDERS]
    assert len(hs) == 1
    assert hs[0].breakout_direction == BreakoutDirection.DOWN
    # Neckline ~100-101 ; target = neck − height
    assert 99.0 <= hs[0].breakout_level <= 105.0
    assert hs[0].invalidation_level >= 120.0


def test_inverse_head_shoulders_detected():
    # LS 90 idx 4 ; neck-H 100 idx 8 ; HEAD 80 idx 12 ; neck-H 99 idx 16 ; RS 91 idx 20
    pivots = [
        (0, 100.0),
        (4, 90.0),
        (8, 100.0),
        (12, 80.0),
        (16, 99.0),
        (20, 91.0),
        (24, 97.0),
    ]
    closes = _interp(pivots, 28)
    df = _df(closes)
    swings = detect_swings(df, left=2, right=2)
    p = _detector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    ihs = [x for x in p if x.kind == PatternKind.INVERSE_HEAD_SHOULDERS]
    assert len(ihs) == 1
    assert ihs[0].breakout_direction == BreakoutDirection.UP
    assert ihs[0].invalidation_level <= 80.5


def test_no_double_top_when_highs_too_different():
    # Top1 = 110, Top2 = 130 → 18% d'écart, dépasse tolerance
    pivots = [(0, 100.0), (4, 110.0), (9, 100.0), (14, 130.0), (18, 105.0)]
    closes = _interp(pivots, 22)
    df = _df(closes)
    swings = detect_swings(df, left=2, right=2)
    p = _detector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    assert all(x.kind != PatternKind.DOUBLE_TOP for x in p)
