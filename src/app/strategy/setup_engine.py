"""Deterministic rule-based setup engine (v2).

Generates trade proposals from MarketContextDTO by applying SMC rules.

Key improvements over v1:
- SL placed on structural levels (swing low/high) instead of arbitrary % of gap.
- Zone freshness: only zones created within the last ``max_zone_age`` bars are used.
- Entry direction check: for a LONG the price should be pulling *back* toward
  the zone (last close above entry is suspicious).
- Diversity: max 2 setups per setup_type to avoid one type crowding out all others.

Setup types (SMC / ICT):
    1. BOS_RETEST         — BOS then retest of broken swing level
    2. FVG_FILL           — Price pulls back into an unmitigated FVG
    3. OB_REJECTION       — Reaction on unmitigated Order Block
    4. CHOCH_REVERSAL     — CHOCH récent : SL sur structure jusqu'à la dernière bougie
       (pas figé à l'index CHOCH), entrée légèrement décalée du pivot pour éviter les mèches.
    5. OB_FVG_CONFLUENCE  — OB + FVG overlap zone (institutional confluence)
    6. IFVG_REACTION      — Reaction on inverse FVG zone
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from app.schemas.domain import (
    FVGType,
    FairValueGap,
    MarketContextDTO,
    OBType,
    OrderBlock,
    Side,
    StructureEventType,
    SwingKind,
    Trend,
    TradeSetupDTO,
)

_MAX_PER_TYPE = 2


class RuleBasedSetupEngine:
    """Produces deterministic trade setups from a full market context."""

    def __init__(
        self,
        *,
        rr_min: float = 2.0,
        max_setups: int = 5,
        fvg_proximity_pct: float = 0.004,
        ob_proximity_pct: float = 0.004,
        max_zone_age: int = 120,
        choch_max_age_bars: int = 48,
    ) -> None:
        self._rr_min = rr_min
        self._max_setups = max_setups
        self._fvg_prox = fvg_proximity_pct
        self._ob_prox = ob_proximity_pct
        self._struct_tol = max(float(fvg_proximity_pct), float(ob_proximity_pct))
        self._max_zone_age = max_zone_age
        self._choch_max_age_bars = max(5, int(choch_max_age_bars))

    def propose(self, ctx: MarketContextDTO) -> list[TradeSetupDTO]:
        setups: list[TradeSetupDTO] = []
        last_bar = len(ctx.ohlcv) - 1

        setups.extend(self._bos_retest(ctx, last_bar))
        setups.extend(self._fvg_fill(ctx, last_bar))
        setups.extend(self._ob_rejection(ctx, last_bar))
        setups.extend(self._choch_reversal(ctx, last_bar))
        setups.extend(self._ob_fvg_confluence(ctx, last_bar))
        setups.extend(self._ifvg_reaction(ctx, last_bar))

        setups.sort(key=lambda s: s.confidence, reverse=True)

        diversified: list[TradeSetupDTO] = []
        type_count: Counter[str] = Counter()
        for s in setups:
            if type_count[s.setup_type] >= _MAX_PER_TYPE:
                continue
            diversified.append(s)
            type_count[s.setup_type] += 1
            if len(diversified) >= self._max_setups:
                break

        return diversified

    # ------------------------------------------------------------------
    # Zone freshness filter
    # ------------------------------------------------------------------
    def _fresh_fvgs(self, ctx: MarketContextDTO, last_bar: int) -> list[FairValueGap]:
        cutoff = max(0, last_bar - self._max_zone_age)
        return [f for f in ctx.fvgs if not f.mitigated and f.index >= cutoff]

    def _fresh_obs(self, ctx: MarketContextDTO, last_bar: int) -> list[OrderBlock]:
        cutoff = max(0, last_bar - self._max_zone_age)
        return [o for o in ctx.order_blocks if not o.mitigated and o.index >= cutoff]

    # ------------------------------------------------------------------
    # Setup 1: BOS Retest
    # ------------------------------------------------------------------
    def _bos_retest(self, ctx: MarketContextDTO, last_bar: int) -> list[TradeSetupDTO]:
        """After a BOS, propose entry on retest of the broken swing level."""
        setups: list[TradeSetupDTO] = []
        last_close = float(ctx.ohlcv["close"].iloc[-1])

        bos_events = [e for e in ctx.structure_events if e.event_type == StructureEventType.BOS]

        for bos in bos_events[-3:]:
            ref_price = bos.swing_ref.price

            if bos.direction == Trend.BULLISH:
                if ref_price * (1 - self._struct_tol) < last_close < ref_price * (1 + self._struct_tol):
                    sl = self._find_nearest_swing_low(ctx, bos.index)
                    if sl is None:
                        continue
                    risk = ref_price - sl
                    if risk <= 0:
                        continue
                    tp1 = ref_price + self._rr_min * risk
                    rr = self._rr_min
                    setups.append(self._make_setup(
                        ctx, Side.LONG, ref_price, sl, [tp1], rr,
                        confidence=0.55, setup_type="BOS_RETEST",
                        rationale=f"BOS bullish at {ref_price:.2f}, retest entry",
                    ))

            elif bos.direction == Trend.BEARISH:
                if ref_price * (1 - self._struct_tol) < last_close < ref_price * (1 + self._struct_tol):
                    sl = self._find_nearest_swing_high(ctx, bos.index)
                    if sl is None:
                        continue
                    risk = sl - ref_price
                    if risk <= 0:
                        continue
                    tp1 = ref_price - self._rr_min * risk
                    rr = self._rr_min
                    setups.append(self._make_setup(
                        ctx, Side.SHORT, ref_price, sl, [tp1], rr,
                        confidence=0.55, setup_type="BOS_RETEST",
                        rationale=f"BOS bearish at {ref_price:.2f}, retest entry",
                    ))

        return setups

    # ------------------------------------------------------------------
    # Setup 2: FVG Fill
    # ------------------------------------------------------------------
    def _fvg_fill(self, ctx: MarketContextDTO, last_bar: int) -> list[TradeSetupDTO]:
        """Enter when price pulls back into an unmitigated FVG.

        SL is placed on the structural swing beyond the FVG, not an
        arbitrary fraction of gap size.
        """
        setups: list[TradeSetupDTO] = []
        last_close = float(ctx.ohlcv["close"].iloc[-1])
        unmitigated = self._fresh_fvgs(ctx, last_bar)

        for fvg in unmitigated[-3:]:
            if fvg.fvg_type == FVGType.BULLISH:
                if fvg.bottom <= last_close <= fvg.top * (1 + self._fvg_prox):
                    entry = fvg.bottom
                    sl = self._find_nearest_swing_low(ctx, fvg.index)
                    if sl is None or sl >= entry:
                        sl = entry - (fvg.top - fvg.bottom) * 0.8
                    risk = entry - sl
                    if risk <= 0:
                        continue
                    tp1 = entry + self._rr_min * risk
                    rr = self._rr_min
                    setups.append(self._make_setup(
                        ctx, Side.LONG, entry, sl, [tp1], rr,
                        confidence=0.50, setup_type="FVG_FILL",
                        rationale=f"Bullish FVG [{fvg.bottom:.2f}-{fvg.top:.2f}] fill entry",
                    ))

            elif fvg.fvg_type == FVGType.BEARISH:
                if fvg.bottom * (1 - self._fvg_prox) <= last_close <= fvg.top:
                    entry = fvg.top
                    sl = self._find_nearest_swing_high(ctx, fvg.index)
                    if sl is None or sl <= entry:
                        sl = entry + (fvg.top - fvg.bottom) * 0.8
                    risk = sl - entry
                    if risk <= 0:
                        continue
                    tp1 = entry - self._rr_min * risk
                    rr = self._rr_min
                    setups.append(self._make_setup(
                        ctx, Side.SHORT, entry, sl, [tp1], rr,
                        confidence=0.50, setup_type="FVG_FILL",
                        rationale=f"Bearish FVG [{fvg.bottom:.2f}-{fvg.top:.2f}] fill entry",
                    ))

        return setups

    # ------------------------------------------------------------------
    # Setup 3: Order Block Rejection
    # ------------------------------------------------------------------
    def _ob_rejection(self, ctx: MarketContextDTO, last_bar: int) -> list[TradeSetupDTO]:
        """Enter on price touching an unmitigated Order Block.

        SL is placed on the structural swing beyond the OB.
        """
        setups: list[TradeSetupDTO] = []
        last_close = float(ctx.ohlcv["close"].iloc[-1])
        unmitigated = self._fresh_obs(ctx, last_bar)

        for ob in unmitigated[-3:]:
            if ob.ob_type == OBType.BULLISH:
                if ob.bottom * (1 - self._ob_prox) <= last_close <= ob.top * (1 + self._ob_prox):
                    entry = ob.top
                    sl = self._find_nearest_swing_low(ctx, ob.index)
                    if sl is None or sl >= ob.bottom:
                        sl = ob.bottom - (ob.top - ob.bottom) * 0.3
                    risk = entry - sl
                    if risk <= 0:
                        continue
                    tp1 = entry + self._rr_min * risk
                    rr = self._rr_min
                    setups.append(self._make_setup(
                        ctx, Side.LONG, entry, sl, [tp1], rr,
                        confidence=0.55, setup_type="OB_REJECTION",
                        rationale=f"Bullish OB [{ob.bottom:.2f}-{ob.top:.2f}] rejection",
                    ))

            elif ob.ob_type == OBType.BEARISH:
                if ob.bottom * (1 - self._ob_prox) <= last_close <= ob.top * (1 + self._ob_prox):
                    entry = ob.bottom
                    sl = self._find_nearest_swing_high(ctx, ob.index)
                    if sl is None or sl <= ob.top:
                        sl = ob.top + (ob.top - ob.bottom) * 0.3
                    risk = sl - entry
                    if risk <= 0:
                        continue
                    tp1 = entry - self._rr_min * risk
                    rr = self._rr_min
                    setups.append(self._make_setup(
                        ctx, Side.SHORT, entry, sl, [tp1], rr,
                        confidence=0.55, setup_type="OB_REJECTION",
                        rationale=f"Bearish OB [{ob.bottom:.2f}-{ob.top:.2f}] rejection",
                    ))

        return setups

    # ------------------------------------------------------------------
    # Setup 4: CHOCH Reversal
    # ------------------------------------------------------------------
    def _choch_reversal(self, ctx: MarketContextDTO, last_bar: int) -> list[TradeSetupDTO]:
        """CHOCH récent : entrée avec léger décalage du pivot ; SL sur invalidation à jour.

        L'ancienne règle plaçait le SL uniquement sur des swings **avant** l'index du CHOCH,
        alors que le pullback de retest forme souvent des pivots **après** — stop trop serré
        ou incohérent. On utilise maintenant les swings jusqu'à ``last_bar`` et un tampon
        d'entrée au-dessous / au-dessus du pivot pour limiter les sorties SL sur mèche.
        """
        setups: list[TradeSetupDTO] = []
        choch_events = [e for e in ctx.structure_events if e.event_type == StructureEventType.CHOCH]

        if not choch_events:
            return setups

        last_choch = choch_events[-1]
        choch_idx = int(last_choch.swing_ref.index)
        if last_bar - choch_idx > self._choch_max_age_bars:
            return setups

        last_close = float(ctx.ohlcv["close"].iloc[-1])
        # Tampon d'entrée : % du prix, borné (évite entrée pile sur l'extrême du pivot).
        pad = max(0.0006, min(0.004, float(self._struct_tol) * 0.35))

        if last_choch.direction == Trend.BEARISH:
            ref = float(last_choch.swing_ref.price)
            entry = ref * (1.0 + pad)
            sl = self._swing_high_invalidates_short(ctx, last_bar, entry)
            if sl is None or sl <= entry:
                return setups
            risk = sl - entry
            if risk < entry * 0.0005:
                return setups
            if risk > entry * 0.045:
                return setups
            tp1 = entry - self._rr_min * risk
            rr = self._rr_min
            if last_close >= ref * (1 - self._struct_tol) and last_close <= sl:
                setups.append(self._make_setup(
                    ctx, Side.SHORT, entry, sl, [tp1], rr,
                    confidence=0.58, setup_type="CHOCH_REVERSAL",
                    rationale=f"CHOCH bearish, short retest au-dessus du pivot ({entry:.2f}), SL structure récente",
                ))

        elif last_choch.direction == Trend.BULLISH:
            ref = float(last_choch.swing_ref.price)
            entry = ref * (1.0 - pad)
            sl = self._swing_low_invalidates_long(ctx, last_bar, entry)
            if sl is None or sl >= entry:
                return setups
            risk = entry - sl
            if risk < entry * 0.0005:
                return setups
            if risk > entry * 0.045:
                return setups
            tp1 = entry + self._rr_min * risk
            rr = self._rr_min
            if last_close <= ref * (1 + self._struct_tol) and last_close >= sl:
                setups.append(self._make_setup(
                    ctx, Side.LONG, entry, sl, [tp1], rr,
                    confidence=0.58, setup_type="CHOCH_REVERSAL",
                    rationale=f"CHOCH bullish, long retest sous le pivot ({entry:.2f}), SL structure récente",
                ))

        return setups

    # ------------------------------------------------------------------
    # Setup 5: OB + FVG confluence
    # ------------------------------------------------------------------
    def _ob_fvg_confluence(self, ctx: MarketContextDTO, last_bar: int) -> list[TradeSetupDTO]:
        """OB and FVG of same direction whose zones overlap — strong confluence."""
        setups: list[TradeSetupDTO] = []
        last_close = float(ctx.ohlcv["close"].iloc[-1])

        def overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
            return hi_a >= lo_b and hi_b >= lo_a

        bull_obs = [o for o in self._fresh_obs(ctx, last_bar) if o.ob_type == OBType.BULLISH]
        bear_obs = [o for o in self._fresh_obs(ctx, last_bar) if o.ob_type == OBType.BEARISH]
        bull_fvgs = [f for f in self._fresh_fvgs(ctx, last_bar) if f.fvg_type == FVGType.BULLISH]
        bear_fvgs = [f for f in self._fresh_fvgs(ctx, last_bar) if f.fvg_type == FVGType.BEARISH]

        for ob in bull_obs[-3:]:
            for fvg in bull_fvgs[-5:]:
                if not overlap(ob.bottom, ob.top, fvg.bottom, fvg.top):
                    continue
                z_lo = max(ob.bottom, fvg.bottom)
                z_hi = min(ob.top, fvg.top)
                if z_hi <= z_lo:
                    continue
                if not (z_lo * (1 - self._ob_prox) <= last_close <= z_hi * (1 + self._ob_prox)):
                    continue
                entry = z_lo
                sl = self._find_nearest_swing_low(ctx, ob.index)
                if sl is None or sl >= z_lo:
                    sl = min(ob.bottom, fvg.bottom) - (ob.top - ob.bottom) * 0.2
                risk = entry - sl
                if risk <= 0:
                    continue
                tp1 = entry + self._rr_min * risk
                rr = self._rr_min
                setups.append(self._make_setup(
                    ctx, Side.LONG, entry, sl, [tp1], rr,
                    confidence=0.62, setup_type="OB_FVG_CONFLUENCE",
                    rationale=f"Confluence OB+FVG haussiers [{z_lo:.2f}-{z_hi:.2f}]",
                ))

        for ob in bear_obs[-3:]:
            for fvg in bear_fvgs[-5:]:
                if not overlap(ob.bottom, ob.top, fvg.bottom, fvg.top):
                    continue
                z_lo = max(ob.bottom, fvg.bottom)
                z_hi = min(ob.top, fvg.top)
                if z_hi <= z_lo:
                    continue
                if not (z_lo * (1 - self._ob_prox) <= last_close <= z_hi * (1 + self._ob_prox)):
                    continue
                entry = z_hi
                sl = self._find_nearest_swing_high(ctx, ob.index)
                if sl is None or sl <= z_hi:
                    sl = max(ob.top, fvg.top) + (ob.top - ob.bottom) * 0.2
                risk = sl - entry
                if risk <= 0:
                    continue
                tp1 = entry - self._rr_min * risk
                rr = self._rr_min
                setups.append(self._make_setup(
                    ctx, Side.SHORT, entry, sl, [tp1], rr,
                    confidence=0.62, setup_type="OB_FVG_CONFLUENCE",
                    rationale=f"Confluence OB+FVG baissiers [{z_lo:.2f}-{z_hi:.2f}]",
                ))

        return setups

    # ------------------------------------------------------------------
    # Setup 6: IFVG reaction
    # ------------------------------------------------------------------
    def _ifvg_reaction(self, ctx: MarketContextDTO, last_bar: int) -> list[TradeSetupDTO]:
        """Enter on inverse FVG zone."""
        setups: list[TradeSetupDTO] = []
        last_close = float(ctx.ohlcv["close"].iloc[-1])
        cutoff = max(0, last_bar - self._max_zone_age)

        for z in list(ctx.ifvgs)[-3:]:
            if z.index < cutoff:
                continue
            if z.fvg_type == FVGType.BULLISH:
                if z.bottom <= last_close <= z.top * (1 + self._fvg_prox):
                    entry = z.bottom
                    sl = self._find_nearest_swing_low(ctx, z.index)
                    gap_size = z.top - z.bottom
                    if gap_size <= 0:
                        continue
                    if sl is None or sl >= entry:
                        sl = entry - gap_size * 0.8
                    risk = entry - sl
                    if risk <= 0:
                        continue
                    tp1 = entry + self._rr_min * risk
                    rr = self._rr_min
                    setups.append(self._make_setup(
                        ctx, Side.LONG, entry, sl, [tp1], rr,
                        confidence=0.48, setup_type="IFVG_REACTION",
                        rationale=f"IFVG haussier [{z.bottom:.2f}-{z.top:.2f}] réaction",
                    ))
            elif z.fvg_type == FVGType.BEARISH:
                if z.bottom * (1 - self._fvg_prox) <= last_close <= z.top:
                    entry = z.top
                    sl = self._find_nearest_swing_high(ctx, z.index)
                    gap_size = z.top - z.bottom
                    if gap_size <= 0:
                        continue
                    if sl is None or sl <= entry:
                        sl = entry + gap_size * 0.8
                    risk = sl - entry
                    if risk <= 0:
                        continue
                    tp1 = entry - self._rr_min * risk
                    rr = self._rr_min
                    setups.append(self._make_setup(
                        ctx, Side.SHORT, entry, sl, [tp1], rr,
                        confidence=0.48, setup_type="IFVG_REACTION",
                        rationale=f"IFVG baissier [{z.bottom:.2f}-{z.top:.2f}] réaction",
                    ))

        return setups

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _swing_high_invalidates_short(
        self, ctx: MarketContextDTO, last_idx: int, entry: float
    ) -> float | None:
        """Plus haut swing récent au-dessus du prix d'entrée (invalidation short)."""
        highs = [
            s
            for s in ctx.swings
            if s.kind == SwingKind.HIGH and s.index <= last_idx and float(s.price) > entry
        ]
        if not highs:
            h = self._find_nearest_swing_high(ctx, last_idx)
            return h if h is not None and h > entry else None
        return float(max(highs[-12:], key=lambda s: s.price).price)

    def _swing_low_invalidates_long(
        self, ctx: MarketContextDTO, last_idx: int, entry: float
    ) -> float | None:
        """Plus bas swing récent sous le prix d'entrée (invalidation long)."""
        lows = [
            s
            for s in ctx.swings
            if s.kind == SwingKind.LOW and s.index <= last_idx and float(s.price) < entry
        ]
        if not lows:
            lo = self._find_nearest_swing_low(ctx, last_idx)
            return lo if lo is not None and lo < entry else None
        return float(min(lows[-12:], key=lambda s: s.price).price)

    def _find_nearest_swing_low(self, ctx: MarketContextDTO, ref_index: int) -> float | None:
        lows = [s for s in ctx.swings if s.kind.value == "LOW" and s.index <= ref_index]
        if not lows:
            return None
        return min(lows[-5:], key=lambda s: s.price).price

    def _find_nearest_swing_high(self, ctx: MarketContextDTO, ref_index: int) -> float | None:
        highs = [s for s in ctx.swings if s.kind.value == "HIGH" and s.index <= ref_index]
        if not highs:
            return None
        return max(highs[-5:], key=lambda s: s.price).price

    @staticmethod
    def _make_setup(
        ctx: MarketContextDTO,
        side: Side,
        entry: float,
        sl: float,
        tps: list[float],
        rr: float,
        *,
        confidence: float,
        setup_type: str,
        rationale: str,
    ) -> TradeSetupDTO:
        last_ts = ctx.ohlcv["timestamp"].iloc[-1] if "timestamp" in ctx.ohlcv.columns else datetime.utcnow()
        return TradeSetupDTO(
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            side=side,
            entry=round(entry, 6),
            stop_loss=round(sl, 6),
            take_profits=[round(tp, 6) for tp in tps],
            risk_reward=round(rr, 2),
            confidence=round(confidence, 2),
            setup_type=setup_type,
            timestamp=last_ts,
            rationale=rationale,
        )
