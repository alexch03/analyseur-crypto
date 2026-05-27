"""BOS (Break of Structure) and CHOCH (Change of Character) detection.

Single-pass algorithm: trend is tracked incrementally as each swing is
processed.  BOS events are emitted *at the moment* a close breaches the
reference swing, using the trend that was active *when that swing existed*
(not the final trend of the series — that was the v1 bug).

Trend tracking:
    BULLISH: swings form HH / HL.
      - BOS bullish: close > last swing HIGH → continuation.
      - CHOCH bearish: new swing LOW < prev swing LOW → reversal.
    BEARISH: swings form LL / LH.
      - BOS bearish: close < last swing LOW → continuation.
      - CHOCH bullish: new swing HIGH > prev swing HIGH → reversal.

Close-based confirmation:
    BOS requires a *close* (not wick) beyond the reference swing.
    CHOCH is emitted at the violating swing point itself.
"""

from __future__ import annotations

import pandas as pd

from app.schemas.domain import StructureEvent, StructureEventType, SwingKind, SwingPoint, Trend


def _find_bos_bar(
    closes: pd.Series,
    ref_price: float,
    search_start: int,
    search_end: int,
    direction: Trend,
) -> int | None:
    """Return first bar index where close breaks *ref_price* in *direction*."""
    n = len(closes)
    end = min(search_end, n)
    for bar_idx in range(search_start, end):
        c = closes.iloc[bar_idx]
        if direction == Trend.BULLISH and c > ref_price:
            return bar_idx
        if direction == Trend.BEARISH and c < ref_price:
            return bar_idx
    return None


def detect_bos_choch(
    swings: list[SwingPoint],
    closes: pd.Series,
) -> list[StructureEvent]:
    """Produce BOS / CHOCH events in a single chronological pass.

    Both CHOCH and BOS are emitted inline while iterating swings so BOS
    uses the trend that was active *at the time of the swing*, not the
    trend at the end of the series.
    """
    if len(swings) < 2:
        return []

    events: list[StructureEvent] = []
    trend = Trend.UNDEFINED

    last_high: SwingPoint | None = None
    last_low: SwingPoint | None = None
    prev_high: SwingPoint | None = None
    prev_low: SwingPoint | None = None

    bos_emitted_for: set[int] = set()
    bos_max_search = 50

    for sw in swings:
        # -- BOS: check *before* updating swing refs so the reference is
        #    the *previous* swing (whose break we are looking for).
        if trend == Trend.BULLISH and last_high is not None and sw.kind == SwingKind.HIGH:
            bar = _find_bos_bar(
                closes, last_high.price,
                last_high.index + 1,
                min(sw.index + 1, last_high.index + bos_max_search),
                Trend.BULLISH,
            )
            if bar is not None and bar not in bos_emitted_for:
                ts = closes.index[bar] if hasattr(closes.index, "__getitem__") else last_high.timestamp
                events.append(StructureEvent(
                    index=bar, timestamp=ts,
                    event_type=StructureEventType.BOS,
                    direction=Trend.BULLISH,
                    swing_ref=last_high,
                ))
                bos_emitted_for.add(bar)

        if trend == Trend.BEARISH and last_low is not None and sw.kind == SwingKind.LOW:
            bar = _find_bos_bar(
                closes, last_low.price,
                last_low.index + 1,
                min(sw.index + 1, last_low.index + bos_max_search),
                Trend.BEARISH,
            )
            if bar is not None and bar not in bos_emitted_for:
                ts = closes.index[bar] if hasattr(closes.index, "__getitem__") else last_low.timestamp
                events.append(StructureEvent(
                    index=bar, timestamp=ts,
                    event_type=StructureEventType.BOS,
                    direction=Trend.BEARISH,
                    swing_ref=last_low,
                ))
                bos_emitted_for.add(bar)

        # -- Update swing references --
        if sw.kind == SwingKind.HIGH:
            prev_high = last_high
            last_high = sw
        else:
            prev_low = last_low
            last_low = sw

        # -- Establish initial trend --
        if trend == Trend.UNDEFINED:
            if last_high is not None and last_low is not None:
                trend = Trend.BEARISH if last_high.index < last_low.index else Trend.BULLISH
            continue

        # -- CHOCH detection --
        if trend == Trend.BULLISH and sw.kind == SwingKind.LOW:
            if prev_low is not None and sw.price < prev_low.price:
                events.append(StructureEvent(
                    index=sw.index, timestamp=sw.timestamp,
                    event_type=StructureEventType.CHOCH,
                    direction=Trend.BEARISH,
                    swing_ref=sw,
                ))
                trend = Trend.BEARISH
                continue

        if trend == Trend.BEARISH and sw.kind == SwingKind.HIGH:
            if prev_high is not None and sw.price > prev_high.price:
                events.append(StructureEvent(
                    index=sw.index, timestamp=sw.timestamp,
                    event_type=StructureEventType.CHOCH,
                    direction=Trend.BULLISH,
                    swing_ref=sw,
                ))
                trend = Trend.BULLISH
                continue

    # -- Final pending BOS: last swing may not have had a successor yet --
    if trend == Trend.BULLISH and last_high is not None:
        bar = _find_bos_bar(
            closes, last_high.price,
            last_high.index + 1,
            last_high.index + bos_max_search,
            Trend.BULLISH,
        )
        if bar is not None and bar not in bos_emitted_for:
            ts = closes.index[bar] if hasattr(closes.index, "__getitem__") else last_high.timestamp
            events.append(StructureEvent(
                index=bar, timestamp=ts,
                event_type=StructureEventType.BOS,
                direction=Trend.BULLISH,
                swing_ref=last_high,
            ))

    if trend == Trend.BEARISH and last_low is not None:
        bar = _find_bos_bar(
            closes, last_low.price,
            last_low.index + 1,
            last_low.index + bos_max_search,
            Trend.BEARISH,
        )
        if bar is not None and bar not in bos_emitted_for:
            ts = closes.index[bar] if hasattr(closes.index, "__getitem__") else last_low.timestamp
            events.append(StructureEvent(
                index=bar, timestamp=ts,
                event_type=StructureEventType.BOS,
                direction=Trend.BEARISH,
                swing_ref=last_low,
            ))

    events.sort(key=lambda e: e.index)
    return events
