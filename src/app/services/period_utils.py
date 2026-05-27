"""Convert calendar period presets into approximate OHLCV bar counts per timeframe."""

from __future__ import annotations

import math
import re


_TF_RE = re.compile(r"^(\d+)([mhdw])$", re.IGNORECASE)


def timeframe_bar_seconds(timeframe: str) -> float:
    """Return seconds per candle for ccxt-style timeframe codes (5m, 15m, 1h, 4h, 1d, 1w)."""
    tf = timeframe.strip().lower()
    m = _TF_RE.match(tf)
    if not m:
        return 3600.0
    n, unit = int(m.group(1)), m.group(2).lower()
    mult = {"m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    return float(n * mult)


def preset_to_days(preset: str) -> float | None:
    """Return number of calendar days for a preset like '30d', or None if custom."""
    p = preset.strip().lower()
    if p == "custom":
        return None
    m = re.match(r"^(\d+)d$", p)
    if m:
        return float(m.group(1))
    # legacy / short forms
    if p.endswith("d") and p[:-1].isdigit():
        return float(p[:-1])
    return None


def calendar_bar_count_uncapped(preset: str, timeframe: str, *, min_bars: int = 300) -> int | None:
    """Nombre de bougies pour un preset calendaire (7d, 90d, …) sans plafond max_bars.

    Retourne None si le preset est custom ou sans jours calendaires.
    """
    days = preset_to_days(preset)
    if days is None:
        return None
    bar_sec = timeframe_bar_seconds(timeframe)
    if bar_sec <= 0:
        return min_bars
    n = int(math.ceil(days * 86400.0 / bar_sec))
    return max(min_bars, n)


def bars_for_calendar_days(
    days: int,
    timeframe: str,
    *,
    min_bars: int = 100,
    max_bars: int = 20000,
) -> int:
    """Nombre de bougies pour couvrir ``days`` jours calendaires à partir du timeframe."""
    d = max(1, int(days))
    bar_sec = timeframe_bar_seconds(timeframe)
    if bar_sec <= 0:
        return min_bars
    n = int(math.ceil(d * 86400.0 / bar_sec))
    return max(min_bars, min(max_bars, n))


def bars_for_period(preset: str, timeframe: str, *, min_bars: int = 300, max_bars: int = 20000) -> int:
    """Approximate bars needed to cover `preset` calendar days at given timeframe."""
    days = preset_to_days(preset)
    if days is None:
        return min_bars
    bar_sec = timeframe_bar_seconds(timeframe)
    if bar_sec <= 0:
        return min_bars
    total_sec = days * 86400.0
    n = int(math.ceil(total_sec / bar_sec))
    return max(min_bars, min(max_bars, n))
