"""Tests pour le détecteur de triangles."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.market_structure.swings import detect_swings
from app.patterns.triangles import TriangleDetector
from app.schemas.patterns import BreakoutDirection, PatternKind


def _make_ohlcv(prices: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        [
            {
                "timestamp": start + timedelta(hours=i),
                "open": o, "high": h, "low": l, "close": c, "volume": 100.0,
            }
            for i, (o, h, l, c) in enumerate(prices)
        ]
    )


def test_ascending_triangle_detected() -> None:
    """Highs aplatis ~110, lows ascendants 95 → 103. Doit être TRIANGLE_ASC."""
    prices = [
        (100, 105, 99, 104),
        (104, 109, 103, 108),
        (108, 110.0, 105, 107),    # H pivot 110 (idx 2)
        (107, 108, 100, 103),
        (103, 104, 95.0, 98),      # L pivot 95 (idx 4)
        (98, 103, 98, 102),
        (102, 109, 102, 108),
        (108, 110.1, 106, 107),    # H pivot 110.1 (idx 7)
        (107, 108, 102, 104),
        (104, 105, 98.0, 99),      # L pivot 98 (idx 9)
        (99, 104, 99, 103),
        (103, 108, 103, 107),
        (107, 110.0, 105, 106),    # H pivot 110 (idx 12)
        (106, 107, 103, 105),
        (105, 106, 101.0, 102),    # L pivot 101 (idx 14)
        (102, 105, 102, 104),
        (104, 109, 104, 108),
        (108, 110.2, 106, 107),    # H pivot 110.2 (idx 17)
        (107, 108, 104, 105),
        (105, 106, 103.0, 104),    # L pivot 103 (idx 19)
        (104, 108, 104, 107),
        (107, 109, 105, 108),
    ]
    df = _make_ohlcv(prices)
    swings = detect_swings(df, left=2, right=2)

    patterns = TriangleDetector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    assert len(patterns) == 1, f"Attendu 1 triangle, trouvé {len(patterns)}"
    p = patterns[0]
    assert p.kind == PatternKind.TRIANGLE_ASC
    assert p.breakout_direction == BreakoutDirection.UP
    assert 109.5 <= p.breakout_level <= 110.5
    assert p.invalidation_level < p.breakout_level
    assert p.target is not None and p.target > p.breakout_level
    assert p.confidence > 0.5


def test_descending_triangle_detected() -> None:
    """Lows aplatis ~90, highs descendants 110 → 102. Doit être TRIANGLE_DESC."""
    prices = [
        (100, 105, 99, 104),
        (104, 109, 103, 108),
        (108, 110.0, 105, 107),    # H pivot 110 (idx 2)
        (107, 108, 92, 95),
        (95, 96, 90.0, 91),        # L pivot 90 (idx 4)
        (91, 96, 91, 95),
        (95, 107, 95, 106),
        (106, 108.0, 102, 103),    # H pivot 108 (idx 7)
        (103, 104, 92, 94),
        (94, 95, 90.1, 91),        # L pivot 90.1 (idx 9)
        (91, 96, 91, 95),
        (95, 105, 95, 104),
        (104, 106.0, 100, 101),    # H pivot 106 (idx 12)
        (101, 102, 92, 93),
        (93, 94, 90.0, 91),        # L pivot 90 (idx 14)
        (91, 96, 91, 95),
        (95, 103, 95, 102),
        (102, 104.0, 99, 100),     # H pivot 104 (idx 17)
        (100, 101, 92, 93),
        (93, 94, 90.2, 91),        # L pivot 90.2 (idx 19)
        (91, 95, 91, 94),
        (94, 96, 91, 93),
    ]
    df = _make_ohlcv(prices)
    swings = detect_swings(df, left=2, right=2)
    patterns = TriangleDetector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    assert len(patterns) == 1, f"Attendu 1 triangle, trouvé {len(patterns)}"
    p = patterns[0]
    assert p.kind == PatternKind.TRIANGLE_DESC
    assert p.breakout_direction == BreakoutDirection.DOWN
    assert 89.5 <= p.breakout_level <= 90.7
    assert p.invalidation_level > p.breakout_level
    assert p.target is not None and p.target < p.breakout_level


def test_symmetrical_triangle_detected() -> None:
    """Highs descendants 115 → 108, lows ascendants 95 → 102 : convergence symétrique."""
    prices = [
        (100, 105, 99, 104),
        (104, 110, 103, 109),
        (109, 115.0, 108, 110),   # H pivot 115 (idx 2)
        (110, 111, 99, 100),
        (100, 101, 95.0, 96),     # L pivot 95 (idx 4)
        (96, 105, 96, 104),
        (104, 112, 104, 111),
        (111, 113.0, 109, 110),   # H pivot 113 (idx 7)
        (110, 111, 99, 100),
        (100, 101, 97.0, 98),     # L pivot 97 (idx 9)
        (98, 105, 98, 104),
        (104, 110, 104, 109),
        (109, 111.0, 107, 108),   # H pivot 111 (idx 12)
        (108, 109, 100, 101),
        (101, 102, 99.0, 100),    # L pivot 99 (idx 14)
        (100, 104, 100, 103),
        (103, 108, 103, 107),
        (107, 109.0, 105, 106),   # H pivot 109 (idx 17)
        (106, 107, 101, 102),
        (102, 103, 101.0, 102),   # L pivot 101 (idx 19)
        (102, 106, 102, 105),
        (105, 107, 103, 106),
    ]
    df = _make_ohlcv(prices)
    swings = detect_swings(df, left=2, right=2)
    patterns = TriangleDetector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    assert len(patterns) == 1, f"Attendu 1 triangle, trouvé {len(patterns)}"
    p = patterns[0]
    assert p.kind == PatternKind.TRIANGLE_SYM
    assert p.breakout_direction == BreakoutDirection.UNDETERMINED
    assert p.target is None
    assert p.invalidation_level < p.breakout_level
    assert p.height > 0


def test_random_walk_no_triangle() -> None:
    """Une série quasi-aléatoire sans structure ne doit pas produire de triangle."""
    import random
    random.seed(42)
    prices: list[tuple[float, float, float, float]] = []
    last = 100.0
    for _ in range(30):
        change = random.uniform(-3, 3)
        o = last
        c = last + change
        h = max(o, c) + random.uniform(0.5, 2.5)
        l = min(o, c) - random.uniform(0.5, 2.5)
        prices.append((o, h, l, c))
        last = c
    df = _make_ohlcv(prices)
    swings = detect_swings(df, left=2, right=2)
    patterns = TriangleDetector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    assert patterns == [], "Une série aléatoire ne devrait pas matcher un triangle"


def test_broken_triangle_not_returned() -> None:
    """Si le prix actuel est largement au-dessus de la résistance, le pattern n'est plus actif."""
    prices = [
        (100, 105, 99, 104),
        (104, 109, 103, 108),
        (108, 110.0, 105, 107),
        (107, 108, 100, 103),
        (103, 104, 95.0, 98),
        (98, 103, 98, 102),
        (102, 109, 102, 108),
        (108, 110.1, 106, 107),
        (107, 108, 102, 104),
        (104, 105, 98.0, 99),
        (99, 104, 99, 103),
        (103, 108, 103, 107),
        (107, 110.0, 105, 106),
        (106, 107, 103, 105),
        (105, 106, 101.0, 102),
        (102, 105, 102, 104),
        (104, 109, 104, 108),
        (108, 110.2, 106, 107),
        (107, 108, 104, 105),
        (105, 106, 103.0, 104),
        (104, 115, 104, 114),    # cassure haussière violente
        (114, 120, 113, 119),    # close 119 → bien au-dessus du niveau 110 avec tol 0.2%
    ]
    df = _make_ohlcv(prices)
    swings = detect_swings(df, left=2, right=2)
    patterns = TriangleDetector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    assert patterns == [], "Un triangle déjà cassé ne devrait plus être retourné"
