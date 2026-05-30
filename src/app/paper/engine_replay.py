"""Replay backtest engine for deterministic setup validation.

Walk-forward methodology:
- At each bar i (after warmup/training period), analyze candles [0..i].
- Generate setups without looking at future candles.
- Keep top-N setups by confidence for that bar.
- Simulate executions on future bars with explicit execution rules.
- Intrabar : pas de break-even/trailing sur la bougie d'entrée ; si SL et TP coexistent dans une
  même bougie, sortie déterministe (bougie haussière → TP d'abord pour un long, etc.).
- TIMEOUT intelligent (``replay_timeout_smart_extend``) : à l'échéance ``max_holding_bars``, on ne
  clôt pas si le PnL latent au close est **> 0** et que **Bollinger** (prix vs bande milieu / bornes)
  et **tendance** (SMA rapide vs SMA lente) restent favorables au sens du trade ; on prolonge alors
  la fenêtre de ``replay_timeout_grace_bars`` (défaut = même valeur que ``max_holding_bars``), jusqu'à
  ``replay_timeout_max_extensions`` fois. Sinon clôture TIMEOUT au close (PnL latent ≤ 0 ou conditions
  non réunies).

Backtest mode uses unit quantity (default 1 crypto unit), includes:
- opening/closing fees: fee_quote = price * unit_size * fee_rate (fee_rate is a CCXT-style fraction, e.g. 0.0006 = 0.06% of quote notional).
- funding: per 8h interval, funding_quote = entry_notional * funding_rate_8h * n_intervals (rate can be negative).
- detailed per-trade stats (duration, time in drawdown/negative, MAE/MFE)

PARITÉ AVEC LE MOTEUR LIVE (HypothesisEngine)
---------------------------------------------
Ce moteur et ``app.services.hypothesis_engine.HypothesisEngine`` implementent la meme
logique de trade de DEUX facons differentes. Les divergences (entree LIMITE-au-niveau
ici vs entree CLOTURE-de-cassure + confirmation body + filtre regime cote live) sont
mesurees et verrouillees par ``tests/test_engine_parity.py``. La porte d'entree opt-in
``entry_*`` (voir ``__init__``) permet de rapprocher la SELECTION des trades de celle du
live ; la divergence de PRIX d'execution residuelle (fill au niveau vs au close) reste,
par choix de modele, documentee dans ce test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.domain import Side, TradeSetupDTO
from app.services.analysis_pipeline import build_context_and_setups, precompute_all_structures, setups_at_bar


@dataclass(slots=True, frozen=True)
class BacktestTrade:
    setup_type: str
    side: Side
    opened_index: int
    closed_index: int
    opened_at: str
    closed_at: str
    bars_held: int
    entry: float
    stop_loss: float
    take_profit: float
    close_price: float
    quantity: float
    fees_open_quote: float
    fees_close_quote: float
    funding_quote: float
    gross_pnl_quote: float
    net_pnl_quote: float
    pnl_pct_on_notional: float
    pnl_r: float
    outcome: str
    max_adverse_excursion_quote: float
    max_favorable_excursion_quote: float
    max_drawdown_quote: float
    time_in_negative_bars: int
    time_in_negative_pct: float


@dataclass(slots=True, frozen=True)
class BacktestReport:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    net_r: float
    max_drawdown_r: float
    gross_pnl_quote: float
    net_pnl_quote: float
    total_fees_quote: float
    total_funding_quote: float
    # Somme des PnL nets des trades gagnants / magnitude des trades perdants (quote).
    realized_gains_quote: float
    realized_losses_quote: float
    avg_trade_duration_bars: float
    avg_time_in_negative_pct: float
    max_drawdown_quote: float
    trades: list[BacktestTrade]


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = float("-inf")
    dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return abs(dd)


def _timeframe_to_hours(timeframe: str) -> float:
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return max(1.0 / 60.0, float(tf[:-1]) / 60.0)
    if tf.endswith("h"):
        return max(1.0, float(tf[:-1]))
    if tf.endswith("d"):
        return max(24.0, float(tf[:-1]) * 24.0)
    if tf.endswith("w"):
        return max(168.0, float(tf[:-1]) * 168.0)
    return 1.0


def _funding_events_count(bars: int, bar_hours: float, funding_interval_hours: float) -> int:
    if bars <= 0 or bar_hours <= 0 or funding_interval_hours <= 0:
        return 0
    total_hours = bars * bar_hours
    return int(total_hours // funding_interval_hours)


def _be_entry_epsilon(entry: float) -> float:
    return max(1e-9, abs(float(entry)) * 1e-10)


def _resolve_intrabar_long(
    lo: float,
    hi: float,
    o: float,
    c: float,
    *,
    effective_sl: float,
    tp: float,
    entry: float,
    trail_armed: bool,
) -> tuple[str, float]:
    """SL et TP touches dans la MEME bougie : on ne connait pas l'ordre intra-bougie
    a partir des seuls OHLC.

    Hypothese CONSERVATRICE (worst-case) : le cote ADVERSE (SL / SL_BE / TRAIL) est
    resolu EN PREMIER. Garantit que le backtest ne surestime jamais le live.

    NB: l'ancienne regle "bougie haussiere -> TP d'abord" etait un biais OPTIMISTE
    (et meme a l'envers : une bougie haussiere fait souvent son creux avant son
    sommet, donc le SL d'un LONG serait touche avant le TP). Voir test
    test_intrabar_resolution_is_conservative.

    Exception gap : si l'open est deja au-dela du TP (gap favorable franc), le TP
    est reellement le premier prix disponible -> on l'accorde.
    """
    eps = _be_entry_epsilon(entry)
    # Gap d'ouverture franchement au-dela du TP : le 1er prix tradable est >= tp.
    if o >= tp and o > float(effective_sl):
        return "TP", float(tp)
    if trail_armed:
        return "TRAIL", float(effective_sl)
    if effective_sl >= float(entry) - eps:
        return "SL_BE", float(effective_sl)
    return "SL", float(effective_sl)


def _resolve_intrabar_short(
    lo: float,
    hi: float,
    o: float,
    c: float,
    *,
    effective_sl: float,
    tp: float,
    entry: float,
    trail_armed: bool,
) -> tuple[str, float]:
    """SL et TP touches dans la MEME bougie (SHORT) : hypothese CONSERVATRICE,
    le cote adverse (SL / SL_BE / TRAIL) est resolu en premier.

    Exception gap : si l'open est deja sous le TP short, on accorde le TP.
    """
    eps = _be_entry_epsilon(entry)
    if o <= tp and o < float(effective_sl):
        return "TP", float(tp)
    if trail_armed:
        return "TRAIL", float(effective_sl)
    if effective_sl <= float(entry) + eps:
        return "SL_BE", float(effective_sl)
    return "SL", float(effective_sl)


def _atr_at_bar(highs: Any, lows: Any, closes: Any, idx: int, *, period: int = 14) -> float:
    """ATR (moyenne des true range) jusqu'à la bougie ``idx`` incluse."""
    p = max(2, period)
    start = max(1, idx - p + 1)
    trs: list[float] = []
    for k in range(start, idx + 1):
        h, l, c_prev = float(highs[k]), float(lows[k]), float(closes[k - 1])
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    return float(np.mean(trs)) if trs else max(1e-12, float(highs[idx]) - float(lows[idx]))


def _sma_window(closes: Any, end_idx: int, length: int) -> float | None:
    """SMA des closes sur ``length`` bougies se terminant à ``end_idx`` (inclus)."""
    if length <= 0 or end_idx < length - 1:
        return None
    s = 0.0
    for k in range(end_idx - length + 1, end_idx + 1):
        s += float(closes[k])
    return s / float(length)


def _bollinger_at(
    closes: Any,
    idx: int,
    *,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[float, float, float] | None:
    """Bande de Bollinger (milieu, haut, bas) à la clôture ``idx``."""
    p = max(2, period)
    if idx < p - 1:
        return None
    window = np.array([float(closes[k]) for k in range(idx - p + 1, idx + 1)], dtype=float)
    mid = float(np.mean(window))
    sd = float(np.std(window, ddof=0))
    if sd <= 0:
        sd = 1e-12 * max(abs(mid), 1.0)
    u = mid + num_std * sd
    lo = mid - num_std * sd
    return mid, u, lo


def _unrealized_gross_quote(setup: TradeSetupDTO, mark_close: float, quantity: float) -> float:
    """PnL latent brut (sans frais de clôture) au prix mark."""
    if setup.side == Side.LONG:
        return (float(mark_close) - float(setup.entry)) * quantity
    return (float(setup.entry) - float(mark_close)) * quantity


def _timeout_extend_conditions_met(
    setup: TradeSetupDTO,
    idx: int,
    closes: Any,
    *,
    quantity: float,
    bb_period: int,
    sma_fast: int,
    sma_slow: int,
) -> bool:
    """True si on prolonge au lieu de clôturer au TIMEOUT (PnL latent > 0 + BB + tendance)."""
    c = float(closes[idx])
    if _unrealized_gross_quote(setup, c, quantity) <= 0:
        return False

    bb = _bollinger_at(closes, idx, period=bb_period, num_std=2.0)
    if bb is None:
        return False
    mid, upper, lower = bb

    f = _sma_window(closes, idx, sma_fast)
    s = _sma_window(closes, idx, sma_slow)
    trend_ok: bool
    bb_ok: bool
    if setup.side == Side.LONG:
        bb_ok = c >= mid and c <= upper * 1.002
        trend_ok = (f is not None and s is not None and f > s) or (
            (f is None or s is None) and c >= mid
        )
    else:
        bb_ok = c <= mid and c >= lower * 0.998
        trend_ok = (f is not None and s is not None and f < s) or (
            (f is None or s is None) and c <= mid
        )

    return bool(bb_ok and trend_ok)


def _optional_positive_float(v: Any) -> float | None:
    if v is None or v is False:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x <= 0:
        return None
    return x


def replay_engine_from_bt_cfg(bt: dict[str, Any]) -> ReplayBacktestEngine:
    """Instancie le moteur replay à partir du même dict que l'optim / walk-forward OOS."""
    period = int(bt.get("replay_trail_atr_period", 14))
    period = max(2, min(period, 500))
    mspb = int(bt.get("max_setups_per_bar", 1))
    mspb = max(1, min(10, mspb))
    grace = bt.get("replay_timeout_grace_bars")
    grace_i = int(grace) if grace is not None else None

    return ReplayBacktestEngine(
        warmup_bars=int(bt.get("warmup_bars", 120)),
        max_holding_bars=int(bt.get("max_holding_bars", 120)),
        max_setups_per_bar=mspb,
        unit_size=float(bt.get("unit_size", 1.0)),
        entry_fee_rate=float(bt.get("entry_fee_rate", 0.0004)),
        exit_fee_rate=float(bt.get("exit_fee_rate", 0.0004)),
        funding_rate_8h=float(bt.get("funding_rate_8h", bt.get("funding_rate_per_bar", 0.0))),
        break_even_r=_optional_positive_float(bt.get("replay_break_even_r")),
        trail_after_r=_optional_positive_float(bt.get("replay_trail_after_r")),
        trail_atr_mult=_optional_positive_float(bt.get("replay_trail_atr_mult")),
        trail_atr_period=period,
        timeout_smart_extend=bool(bt.get("replay_timeout_smart_extend", True)),
        timeout_grace_bars=grace_i,
        timeout_max_extensions=max(0, int(bt.get("replay_timeout_max_extensions", 3))),
        timeout_bb_period=max(2, int(bt.get("replay_timeout_bb_period", 20))),
        timeout_sma_fast=max(2, int(bt.get("replay_timeout_sma_fast", 10))),
        timeout_sma_slow=max(2, int(bt.get("replay_timeout_sma_slow", 20))),
        # Porte d'entree fidele au live (opt-in ; defaut off => comportement historique).
        # Le filtre regime exige un objet MarketRegime, non transmissible via ce dict scalaire :
        # il reste a cabler par construction directe (cf. tests/test_engine_parity.py).
        entry_require_close_breakout=bool(bt.get("replay_entry_require_close_breakout", False)),
        entry_min_body_ratio=float(bt.get("replay_entry_min_body_ratio", 0.0)),
        entry_breakout_buffer_pct=float(bt.get("replay_entry_breakout_buffer_pct", 0.0)),
        entry_abort_on_invalidation=bool(bt.get("replay_entry_abort_on_invalidation", False)),
    )


class ReplayBacktestEngine:
    def __init__(
        self,
        *,
        warmup_bars: int = 120,
        max_holding_bars: int = 120,
        max_setups_per_bar: int = 1,
        unit_size: float = 1.0,
        entry_fee_rate: float = 0.0004,
        exit_fee_rate: float = 0.0004,
        funding_rate_8h: float = 0.0,
        funding_interval_hours: float = 8.0,
        break_even_r: float | None = None,
        trail_after_r: float | None = None,
        trail_atr_mult: float | None = None,
        trail_atr_period: int = 14,
        # --- Porte d'entree "fidele au live" (opt-in ; tout off => comportement historique) ---
        # Replique les conditions d'ARMEMENT/TRIGGER du HypothesisEngine live, absentes du
        # replay par defaut (cf. docstring du module et tests/test_engine_parity.py) :
        #   - entry_require_close_breakout : exige une CLOTURE au-dela du niveau (pas une
        #     simple meche qui traverse le prix d'entree),
        #   - entry_min_body_ratio : body ratio minimum de la bougie d'entree (anti-wick),
        #     equivalent de min_breakout_body_ratio cote live,
        #   - entry_regime / entry_regime_min_score : filtre regime au moment de l'entree
        #     (pattern_regime_score(setup_type, regime) >= seuil), comme _still_passes_regime_filter,
        #   - entry_abort_on_invalidation : abandonne le trade (NO_TRADE) si la cloture franchit
        #     l'invalidation avant l'entree, comme la transition live FORMING/ARMED -> INVALIDATED.
        entry_require_close_breakout: bool = False,
        entry_min_body_ratio: float = 0.0,
        entry_breakout_buffer_pct: float = 0.0,
        entry_abort_on_invalidation: bool = False,
        entry_regime: object | None = None,
        entry_regime_min_score: float = 0.70,
        timeout_smart_extend: bool = True,
        timeout_grace_bars: int | None = None,
        timeout_max_extensions: int = 3,
        timeout_bb_period: int = 20,
        timeout_sma_fast: int = 10,
        timeout_sma_slow: int = 20,
    ) -> None:
        self.warmup_bars = warmup_bars
        self.max_holding_bars = max_holding_bars
        self.max_setups_per_bar = max_setups_per_bar
        self.unit_size = unit_size
        self.entry_fee_rate = entry_fee_rate
        self.exit_fee_rate = exit_fee_rate
        self.funding_rate_8h = funding_rate_8h
        self.funding_interval_hours = funding_interval_hours
        self.break_even_r = break_even_r
        self.trail_after_r = trail_after_r
        self.trail_atr_mult = trail_atr_mult
        self.trail_atr_period = max(2, trail_atr_period)
        self.entry_require_close_breakout = bool(entry_require_close_breakout)
        self.entry_min_body_ratio = float(entry_min_body_ratio)
        self.entry_breakout_buffer_pct = float(entry_breakout_buffer_pct)
        self.entry_abort_on_invalidation = bool(entry_abort_on_invalidation)
        self.entry_regime = entry_regime
        self.entry_regime_min_score = float(entry_regime_min_score)
        self.timeout_smart_extend = timeout_smart_extend
        self.timeout_grace_bars = timeout_grace_bars
        self.timeout_max_extensions = max(0, timeout_max_extensions)
        self.timeout_bb_period = max(2, timeout_bb_period)
        self.timeout_sma_fast = max(2, timeout_sma_fast)
        self.timeout_sma_slow = max(2, timeout_sma_slow)

    def run_walkforward(
        self,
        ohlcv_df: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        swing_left: int = 3,
        swing_right: int = 3,
        engine_params: dict[str, Any] | None = None,
    ) -> BacktestReport:
        """Run walk-forward backtest — O(n) thanks to pre-computed structures.

        All market-structure (swings, BOS/CHOCH, FVG, OB …) is computed once
        over the whole DataFrame.  At each bar *i* we filter to structures
        confirmed at or before *i* (swing confirmation = index + swing_right).
        This is mathematically equivalent to the old expanding-window approach
        but avoids re-running detection ~n times.
        """
        trades: list[BacktestTrade] = []
        n = len(ohlcv_df)
        if n <= self.warmup_bars + 2:
            return ReplayBacktestEngine._empty_report()

        highs = ohlcv_df["high"].to_numpy(dtype=float)
        lows = ohlcv_df["low"].to_numpy(dtype=float)
        closes = ohlcv_df["close"].to_numpy(dtype=float)
        opens = (
            ohlcv_df["open"].to_numpy(dtype=float) if "open" in ohlcv_df.columns else None
        )
        bar_hours = _timeframe_to_hours(timeframe)
        timestamps = ohlcv_df["timestamp"] if "timestamp" in ohlcv_df.columns else ohlcv_df.index

        pre = precompute_all_structures(
            ohlcv_df,
            symbol=symbol,
            timeframe=timeframe,
            swing_left=swing_left,
            swing_right=swing_right,
        )

        last_exit_idx = -1
        for i in range(self.warmup_bars, n - 2):
            if i <= last_exit_idx:
                continue

            setups = setups_at_bar(
                pre, i,
                engine_params=engine_params,
                swing_right=swing_right,
            )
            if not setups:
                continue

            setups = sorted(setups, key=lambda s: s.confidence, reverse=True)[: self.max_setups_per_bar]
            for setup in setups:
                trade = self._simulate_one_trade(
                    setup=setup,
                    signal_index=i,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    opens=opens,
                    timestamps=timestamps,
                    bar_hours=bar_hours,
                )
                if trade is not None:
                    trades.append(trade)
                    last_exit_idx = max(last_exit_idx, trade.closed_index)

        return self._build_report(trades)

    def _simulate_one_trade(
        self,
        *,
        setup: TradeSetupDTO,
        signal_index: int,
        highs: list[float] | pd.Series | Any,
        lows: list[float] | pd.Series | Any,
        closes: list[float] | pd.Series | Any,
        opens: list[float] | pd.Series | Any | None = None,
        timestamps: list[Any] | pd.Series | Any,
        bar_hours: float,
    ) -> BacktestTrade | None:
        """Unified LONG/SHORT trade simulation using a directional multiplier."""
        n = len(closes)
        end = min(signal_index + self.max_holding_bars, n - 1)
        is_long = setup.side == Side.LONG
        # Porte d'entree fidele au live : active des qu'au moins un critere est configure.
        # Off (defaut) => simple "le prix touche entry" comme historiquement.
        gate_on = (
            self.entry_require_close_breakout
            or self.entry_min_body_ratio > 0.0
            or self.entry_regime is not None
        )

        entry_idx: int | None = None
        for j in range(signal_index + 1, end + 1):
            c_j = float(closes[j])
            # Abandon avant entree si la CLOTURE franchit l'invalidation (mirroir du passage
            # live FORMING/ARMED -> INVALIDATED : l'ordre est annule, AUCUN trade n'a lieu).
            if self.entry_abort_on_invalidation and (
                (is_long and c_j <= float(setup.stop_loss))
                or (not is_long and c_j >= float(setup.stop_loss))
            ):
                return None
            if not (lows[j] <= setup.entry <= highs[j]):
                continue
            if gate_on:
                o_j = float(opens[j]) if opens is not None else c_j
                if not self._entry_confirmed(
                    o_j, float(highs[j]), float(lows[j]), c_j, setup, is_long
                ):
                    continue
            entry_idx = j
            break
        if entry_idx is None:
            return None

        if not setup.take_profits:
            return None
        tp = float(setup.take_profits[0])
        risk = abs(setup.entry - setup.stop_loss)
        if risk <= 0:
            return None

        atr_entry = _atr_at_bar(highs, lows, closes, entry_idx, period=self.trail_atr_period)
        initial_sl = float(setup.stop_loss)
        effective_sl = initial_sl
        trail_armed = False

        d = 1.0 if is_long else -1.0
        running_extreme = float(setup.entry)
        resolve_fn = _resolve_intrabar_long if is_long else _resolve_intrabar_short

        grace_bars = self.timeout_grace_bars if self.timeout_grace_bars is not None else self.max_holding_bars
        grace_bars = max(1, grace_bars)
        extensions_used = 0
        deadline = min(signal_index + self.max_holding_bars, n - 1)
        segment_start = entry_idx

        while True:
            for j in range(segment_start, deadline + 1):
                hi = float(highs[j])
                lo = float(lows[j])
                o = float(opens[j]) if opens is not None else float(closes[j])
                c = float(closes[j])

                if is_long:
                    running_extreme = max(running_extreme, hi)
                else:
                    running_extreme = min(running_extreme, lo)
                mfe = d * (running_extreme - float(setup.entry))

                if j > entry_idx:
                    if self.break_even_r is not None and mfe >= self.break_even_r * risk:
                        if is_long:
                            effective_sl = max(effective_sl, float(setup.entry))
                        else:
                            effective_sl = min(effective_sl, float(setup.entry))

                    if (
                        self.trail_after_r is not None
                        and self.trail_atr_mult is not None
                        and mfe >= self.trail_after_r * risk
                        and atr_entry > 0
                    ):
                        trail_armed = True
                        trail_line = running_extreme - d * self.trail_atr_mult * atr_entry
                        if is_long:
                            effective_sl = max(effective_sl, trail_line)
                        else:
                            effective_sl = min(effective_sl, trail_line)
                else:
                    effective_sl = initial_sl
                    trail_armed = False

                sl_hit = lo <= effective_sl if is_long else hi >= effective_sl
                tp_hit = hi >= tp if is_long else lo <= tp

                if sl_hit and tp_hit:
                    oc, xp = resolve_fn(
                        lo, hi, o, c,
                        effective_sl=effective_sl, tp=tp,
                        entry=float(setup.entry), trail_armed=trail_armed,
                    )
                    return self._finalize_trade(
                        setup=setup, entry_idx=entry_idx, exit_idx=j,
                        exit_price=xp, outcome=oc,
                        highs=highs, lows=lows, closes=closes,
                        timestamps=timestamps, bar_hours=bar_hours,
                    )
                if sl_hit:
                    eps = _be_entry_epsilon(setup.entry)
                    if trail_armed:
                        oc = "TRAIL"
                    elif (is_long and effective_sl >= float(setup.entry) - eps) or (
                        not is_long and effective_sl <= float(setup.entry) + eps
                    ):
                        oc = "SL_BE"
                    else:
                        oc = "SL"
                    return self._finalize_trade(
                        setup=setup, entry_idx=entry_idx, exit_idx=j,
                        exit_price=float(effective_sl), outcome=oc,
                        highs=highs, lows=lows, closes=closes,
                        timestamps=timestamps, bar_hours=bar_hours,
                    )
                if tp_hit:
                    return self._finalize_trade(
                        setup=setup, entry_idx=entry_idx, exit_idx=j,
                        exit_price=tp, outcome="TP",
                        highs=highs, lows=lows, closes=closes,
                        timestamps=timestamps, bar_hours=bar_hours,
                    )

            if not self.timeout_smart_extend:
                return self._finalize_trade(
                    setup=setup,
                    entry_idx=entry_idx,
                    exit_idx=deadline,
                    exit_price=float(closes[deadline]),
                    outcome="TIMEOUT",
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    timestamps=timestamps,
                    bar_hours=bar_hours,
                )

            mark = float(closes[deadline])
            gross = _unrealized_gross_quote(setup, mark, self.unit_size)
            if gross <= 0:
                return self._finalize_trade(
                    setup=setup,
                    entry_idx=entry_idx,
                    exit_idx=deadline,
                    exit_price=mark,
                    outcome="TIMEOUT",
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    timestamps=timestamps,
                    bar_hours=bar_hours,
                )

            extend_ok = _timeout_extend_conditions_met(
                setup,
                deadline,
                closes,
                quantity=self.unit_size,
                bb_period=self.timeout_bb_period,
                sma_fast=self.timeout_sma_fast,
                sma_slow=self.timeout_sma_slow,
            )
            if not extend_ok or extensions_used >= self.timeout_max_extensions or deadline >= n - 1:
                return self._finalize_trade(
                    setup=setup,
                    entry_idx=entry_idx,
                    exit_idx=deadline,
                    exit_price=mark,
                    outcome="TIMEOUT",
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    timestamps=timestamps,
                    bar_hours=bar_hours,
                )

            extensions_used += 1
            segment_start = deadline + 1
            new_deadline = min(deadline + grace_bars, n - 1)
            if new_deadline <= deadline or segment_start > new_deadline:
                return self._finalize_trade(
                    setup=setup,
                    entry_idx=entry_idx,
                    exit_idx=deadline,
                    exit_price=mark,
                    outcome="TIMEOUT",
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    timestamps=timestamps,
                    bar_hours=bar_hours,
                )
            deadline = new_deadline

    def _entry_confirmed(
        self,
        o: float,
        h: float,
        l: float,
        c: float,
        setup: TradeSetupDTO,
        is_long: bool,
    ) -> bool:
        """Porte d'entree fidele au live (cf. constructeur). Renvoie True si la bougie
        confirme l'entree au sens du HypothesisEngine : cloture au-dela du niveau,
        body solide, et regime favorable. Inerte tant qu'aucun critere n'est configure.

        NB : le prix de remplissage reste ``setup.entry`` (modele d'ordre LIMITE pose au
        niveau de cassure). Le live, lui, remplit a la CLOTURE de la bougie de cassure
        (ordre marche). Cette difference de prix d'execution est la divergence
        structurelle residuelle documentee dans tests/test_engine_parity.py ; la porte
        n'aligne que la SELECTION des trades (quels trades sont pris), pas leur fill.
        """
        if self.entry_require_close_breakout or self.entry_min_body_ratio > 0.0:
            # 1) Cloture au-dela du niveau de cassure (+ buffer optionnel), comme _breakout_confirmed.
            if is_long:
                if c < float(setup.entry) * (1.0 + self.entry_breakout_buffer_pct):
                    return False
            else:
                if c > float(setup.entry) * (1.0 - self.entry_breakout_buffer_pct):
                    return False
            # 2) Body ratio (anti-wick).
            if self.entry_min_body_ratio > 0.0:
                rng = h - l
                body_ratio = abs(c - o) / rng if rng > 0 else 0.0
                if body_ratio < self.entry_min_body_ratio:
                    return False
        # 3) Filtre regime au moment de l'entree (setups non types pattern => score 1.0 => jamais bloque).
        if self.entry_regime is not None:
            from app.services.market_regime import pattern_regime_score
            score = pattern_regime_score(setup.setup_type, self.entry_regime)
            if score < self.entry_regime_min_score:
                return False
        return True

    def seek_entry_after_signal(
        self,
        setup: TradeSetupDTO,
        signal_index: int,
        highs: Any,
        lows: Any,
        n: int,
    ) -> int | None:
        """Même règle que ``_simulate_one_trade`` : entrée sur la 1re bougie après le signal où le prix touche ``setup.entry``."""
        end = min(signal_index + self.max_holding_bars, n - 1)
        for j in range(signal_index + 1, end + 1):
            if float(lows[j]) <= setup.entry <= float(highs[j]):
                return j
        return None

    def paper_live_init_exit_state(
        self,
        setup: TradeSetupDTO,
        entry_idx: int,
        highs: Any,
        lows: Any,
        closes: Any,
        *,
        signal_index: int,
        n_bars: int,
    ) -> dict[str, Any] | None:
        """État mutable pour reprendre la boucle de sortie (break-even / trailing) entre deux ticks paper live."""
        if not setup.take_profits:
            return None
        tp = float(setup.take_profits[0])
        risk = abs(float(setup.entry) - float(setup.stop_loss))
        if risk <= 0:
            return None
        atr_entry = _atr_at_bar(highs, lows, closes, entry_idx, period=self.trail_atr_period)
        sl0 = float(setup.stop_loss)
        deadline = min(int(signal_index) + self.max_holding_bars, max(0, n_bars - 1))
        base = {
            "entry_idx": entry_idx,
            "j": entry_idx,
            "effective_sl": sl0,
            "initial_sl": sl0,
            "trail_armed": False,
            "atr_entry": float(atr_entry),
            "risk": float(risk),
            "tp": tp,
            "signal_index": int(signal_index),
            "deadline_idx": deadline,
            "timeout_extensions_used": 0,
        }
        if setup.side == Side.LONG:
            return {
                **base,
                "side": "LONG",
                "running_high": float(setup.entry),
            }
        return {
            **base,
            "side": "SHORT",
            "running_low": float(setup.entry),
        }

    def _paper_try_extend_deadline(
        self,
        setup: TradeSetupDTO,
        state: dict[str, Any],
        closes: Any,
        *,
        n_bars: int,
    ) -> bool:
        """Prolonge ``deadline_idx`` si PnL latent > 0 + Bollinger + tendance. Mutate ``state``."""
        if not self.timeout_smart_extend:
            return False
        deadline = int(state["deadline_idx"])
        if deadline < 0 or deadline >= n_bars:
            return False
        gross = _unrealized_gross_quote(setup, float(closes[deadline]), self.unit_size)
        if gross <= 0:
            return False
        if not _timeout_extend_conditions_met(
            setup,
            deadline,
            closes,
            quantity=self.unit_size,
            bb_period=self.timeout_bb_period,
            sma_fast=self.timeout_sma_fast,
            sma_slow=self.timeout_sma_slow,
        ):
            return False
        ext = int(state.get("timeout_extensions_used", 0))
        if ext >= self.timeout_max_extensions:
            return False
        if deadline >= n_bars - 1:
            return False
        grace = self.timeout_grace_bars if self.timeout_grace_bars is not None else self.max_holding_bars
        grace = max(1, int(grace))
        new_d = min(deadline + grace, n_bars - 1)
        if new_d <= deadline:
            return False
        state["deadline_idx"] = new_d
        state["timeout_extensions_used"] = ext + 1
        return True

    def paper_live_advance_exit_state(
        self,
        setup: TradeSetupDTO,
        state: dict[str, Any],
        highs: Any,
        lows: Any,
        closes: Any,
        timestamps: Any,
        bar_hours: float,
        max_bar_idx_inclusive: int,
        opens: Any | None = None,
    ) -> tuple[BacktestTrade | None, dict[str, Any] | None]:
        """Avance la simulation de sortie bar par bar jusqu'à ``max_bar_idx_inclusive`` (dernière bougie connue)."""
        start_j = int(state["j"])
        if start_j > max_bar_idx_inclusive:
            return None, state
        n_bars = int(len(closes))
        if "deadline_idx" not in state:
            sig = int(state.get("signal_index", int(state.get("entry_idx", 0))))
            state["deadline_idx"] = min(sig + self.max_holding_bars, max(0, n_bars - 1))
            state.setdefault("timeout_extensions_used", 0)
        deadline_idx = int(state["deadline_idx"])
        loop_end = min(max_bar_idx_inclusive, deadline_idx)
        if start_j > loop_end:
            return None, state
        tp = float(state["tp"])
        risk = float(state["risk"])
        atr_entry = float(state["atr_entry"])
        side = str(state["side"])
        entry_idx = int(state["entry_idx"])
        initial_sl = float(state.get("initial_sl", state["effective_sl"]))

        if side == "LONG":
            for j in range(start_j, loop_end + 1):
                hi = float(highs[j])
                lo = float(lows[j])
                o = float(opens[j]) if opens is not None else float(closes[j])
                c = float(closes[j])
                state["running_high"] = max(float(state["running_high"]), hi)
                mfe = float(state["running_high"]) - float(setup.entry)

                if j > entry_idx:
                    if self.break_even_r is not None and mfe >= self.break_even_r * risk:
                        state["effective_sl"] = max(float(state["effective_sl"]), float(setup.entry))

                    if (
                        self.trail_after_r is not None
                        and self.trail_atr_mult is not None
                        and mfe >= self.trail_after_r * risk
                        and atr_entry > 0
                    ):
                        state["trail_armed"] = True
                        trail_line = float(state["running_high"]) - self.trail_atr_mult * atr_entry
                        state["effective_sl"] = max(float(state["effective_sl"]), trail_line)
                else:
                    state["effective_sl"] = initial_sl
                    state["trail_armed"] = False

                effective_sl = float(state["effective_sl"])
                sl_hit = lo <= effective_sl
                tp_hit = hi >= tp
                if sl_hit and tp_hit:
                    oc, xp = _resolve_intrabar_long(
                        lo,
                        hi,
                        o,
                        c,
                        effective_sl=effective_sl,
                        tp=tp,
                        entry=float(setup.entry),
                        trail_armed=bool(state["trail_armed"]),
                    )
                    tr = self._finalize_trade(
                        setup=setup,
                        entry_idx=entry_idx,
                        exit_idx=j,
                        exit_price=xp,
                        outcome=oc,
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        timestamps=timestamps,
                        bar_hours=bar_hours,
                    )
                    return tr, None
                if sl_hit:
                    oc = "SL"
                    if state["trail_armed"]:
                        oc = "TRAIL"
                    elif effective_sl >= float(setup.entry) - 1e-12 * max(1.0, abs(float(setup.entry))):
                        oc = "SL_BE"
                    tr = self._finalize_trade(
                        setup=setup,
                        entry_idx=entry_idx,
                        exit_idx=j,
                        exit_price=float(effective_sl),
                        outcome=oc,
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        timestamps=timestamps,
                        bar_hours=bar_hours,
                    )
                    return tr, None
                if tp_hit:
                    tr = self._finalize_trade(
                        setup=setup,
                        entry_idx=entry_idx,
                        exit_idx=j,
                        exit_price=tp,
                        outcome="TP",
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        timestamps=timestamps,
                        bar_hours=bar_hours,
                    )
                    return tr, None
                state["j"] = j + 1
        else:
            for j in range(start_j, loop_end + 1):
                hi = float(highs[j])
                lo = float(lows[j])
                o = float(opens[j]) if opens is not None else float(closes[j])
                c = float(closes[j])
                state["running_low"] = min(float(state["running_low"]), lo)
                mfe = float(setup.entry) - float(state["running_low"])

                if j > entry_idx:
                    if self.break_even_r is not None and mfe >= self.break_even_r * risk:
                        state["effective_sl"] = min(float(state["effective_sl"]), float(setup.entry))

                    if (
                        self.trail_after_r is not None
                        and self.trail_atr_mult is not None
                        and mfe >= self.trail_after_r * risk
                        and atr_entry > 0
                    ):
                        state["trail_armed"] = True
                        trail_line = float(state["running_low"]) + self.trail_atr_mult * atr_entry
                        state["effective_sl"] = min(float(state["effective_sl"]), trail_line)
                else:
                    state["effective_sl"] = initial_sl
                    state["trail_armed"] = False

                effective_sl = float(state["effective_sl"])
                sl_hit = hi >= effective_sl
                tp_hit = lo <= tp
                if sl_hit and tp_hit:
                    oc, xp = _resolve_intrabar_short(
                        lo,
                        hi,
                        o,
                        c,
                        effective_sl=effective_sl,
                        tp=tp,
                        entry=float(setup.entry),
                        trail_armed=bool(state["trail_armed"]),
                    )
                    tr = self._finalize_trade(
                        setup=setup,
                        entry_idx=entry_idx,
                        exit_idx=j,
                        exit_price=xp,
                        outcome=oc,
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        timestamps=timestamps,
                        bar_hours=bar_hours,
                    )
                    return tr, None
                if sl_hit:
                    oc = "SL"
                    if state["trail_armed"]:
                        oc = "TRAIL"
                    elif effective_sl <= float(setup.entry) + 1e-12 * max(1.0, abs(float(setup.entry))):
                        oc = "SL_BE"
                    tr = self._finalize_trade(
                        setup=setup,
                        entry_idx=entry_idx,
                        exit_idx=j,
                        exit_price=float(effective_sl),
                        outcome=oc,
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        timestamps=timestamps,
                        bar_hours=bar_hours,
                    )
                    return tr, None
                if tp_hit:
                    tr = self._finalize_trade(
                        setup=setup,
                        entry_idx=entry_idx,
                        exit_idx=j,
                        exit_price=tp,
                        outcome="TP",
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        timestamps=timestamps,
                        bar_hours=bar_hours,
                    )
                    return tr, None
                state["j"] = j + 1

        dln = int(state["deadline_idx"])
        if int(state["j"]) > dln:
            if self._paper_try_extend_deadline(setup, state, closes, n_bars=n_bars):
                return None, state
            tr = self.paper_live_timeout_at_bar(
                setup, entry_idx, dln, highs, lows, closes, timestamps, bar_hours
            )
            return tr, None
        return None, state

    def paper_live_timeout_at_bar(
        self,
        setup: TradeSetupDTO,
        entry_idx: int,
        exit_idx: int,
        highs: Any,
        lows: Any,
        closes: Any,
        timestamps: Any,
        bar_hours: float,
    ) -> BacktestTrade:
        """Clôture TIMEOUT au prix de clôture de ``exit_idx`` (bougie d'indice max connue)."""
        return self._finalize_trade(
            setup=setup,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            exit_price=float(closes[exit_idx]),
            outcome="TIMEOUT",
            highs=highs,
            lows=lows,
            closes=closes,
            timestamps=timestamps,
            bar_hours=bar_hours,
        )

    def _finalize_trade(
        self,
        *,
        setup: TradeSetupDTO,
        entry_idx: int,
        exit_idx: int,
        exit_price: float,
        outcome: str,
        highs: list[float] | pd.Series | Any,
        lows: list[float] | pd.Series | Any,
        closes: list[float] | pd.Series | Any,
        timestamps: list[Any] | pd.Series | Any,
        bar_hours: float,
    ) -> BacktestTrade:
        quantity = float(self.unit_size)
        bars_held = max(1, exit_idx - entry_idx + 1)
        entry_notional = setup.entry * quantity
        exit_notional = exit_price * quantity

        fees_open = entry_notional * self.entry_fee_rate
        fees_close = exit_notional * self.exit_fee_rate
        funding_events = _funding_events_count(
            bars=bars_held,
            bar_hours=bar_hours,
            funding_interval_hours=self.funding_interval_hours,
        )
        funding_quote = entry_notional * self.funding_rate_8h * funding_events

        if setup.side == Side.LONG:
            gross = (exit_price - setup.entry) * quantity
        else:
            gross = (setup.entry - exit_price) * quantity
        net = gross - fees_open - fees_close - funding_quote

        risk_quote = abs(setup.entry - setup.stop_loss) * quantity
        pnl_r = net / risk_quote if risk_quote > 0 else 0.0
        pnl_pct_notional = net / entry_notional if entry_notional > 0 else 0.0

        # In-trade path stats.
        min_move = 0.0
        max_move = 0.0
        unrealized_net_path: list[float] = []
        neg_bars = 0

        for k in range(entry_idx, exit_idx + 1):
            hi = float(highs[k])
            lo = float(lows[k])
            cl = float(closes[k])
            if setup.side == Side.LONG:
                min_move = min(min_move, (lo - setup.entry) * quantity)
                max_move = max(max_move, (hi - setup.entry) * quantity)
                unreal_gross = (cl - setup.entry) * quantity
            else:
                min_move = min(min_move, (setup.entry - hi) * quantity)
                max_move = max(max_move, (setup.entry - lo) * quantity)
                unreal_gross = (setup.entry - cl) * quantity

            elapsed = max(1, k - entry_idx + 1)
            elapsed_funding_events = _funding_events_count(
                bars=elapsed,
                bar_hours=bar_hours,
                funding_interval_hours=self.funding_interval_hours,
            )
            unreal_funding = entry_notional * self.funding_rate_8h * elapsed_funding_events
            unreal_net = unreal_gross - fees_open - unreal_funding
            unrealized_net_path.append(unreal_net)
            if unreal_net < 0:
                neg_bars += 1

        peak = float("-inf")
        dd = 0.0
        for v in unrealized_net_path:
            peak = max(peak, v)
            dd = min(dd, v - peak)
        max_trade_dd = abs(dd)

        open_ts = timestamps.iloc[entry_idx] if hasattr(timestamps, "iloc") else timestamps[entry_idx]
        close_ts = timestamps.iloc[exit_idx] if hasattr(timestamps, "iloc") else timestamps[exit_idx]

        return BacktestTrade(
            setup_type=setup.setup_type,
            side=setup.side,
            opened_index=entry_idx,
            closed_index=exit_idx,
            opened_at=str(open_ts),
            closed_at=str(close_ts),
            bars_held=bars_held,
            entry=setup.entry,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profits[0] if setup.take_profits else exit_price,
            close_price=float(exit_price),
            quantity=quantity,
            fees_open_quote=float(fees_open),
            fees_close_quote=float(fees_close),
            funding_quote=float(funding_quote),
            gross_pnl_quote=float(gross),
            net_pnl_quote=float(net),
            pnl_pct_on_notional=float(pnl_pct_notional),
            pnl_r=float(pnl_r),
            outcome=outcome,
            max_adverse_excursion_quote=abs(float(min_move)),
            max_favorable_excursion_quote=float(max_move),
            max_drawdown_quote=float(max_trade_dd),
            time_in_negative_bars=neg_bars,
            time_in_negative_pct=(neg_bars / bars_held) if bars_held > 0 else 0.0,
        )

    @staticmethod
    def _empty_report() -> BacktestReport:
        return BacktestReport(
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy_r=0.0,
            net_r=0.0,
            max_drawdown_r=0.0,
            gross_pnl_quote=0.0,
            net_pnl_quote=0.0,
            total_fees_quote=0.0,
            total_funding_quote=0.0,
            realized_gains_quote=0.0,
            realized_losses_quote=0.0,
            avg_trade_duration_bars=0.0,
            avg_time_in_negative_pct=0.0,
            max_drawdown_quote=0.0,
            trades=[],
        )

    @staticmethod
    def _build_report(trades: list[BacktestTrade]) -> BacktestReport:
        total = len(trades)
        if total == 0:
            return ReplayBacktestEngine._empty_report()

        wins = sum(1 for t in trades if t.net_pnl_quote > 0)
        losses = total - wins
        win_rate = wins / total

        gross_win = sum(t.net_pnl_quote for t in trades if t.net_pnl_quote > 0)
        gross_loss = abs(sum(t.net_pnl_quote for t in trades if t.net_pnl_quote < 0))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

        net_r = sum(t.pnl_r for t in trades)
        expectancy = net_r / total

        equity_r: list[float] = []
        equity_quote: list[float] = []
        cur_r = 0.0
        cur_quote = 0.0
        for t in trades:
            cur_r += t.pnl_r
            cur_quote += t.net_pnl_quote
            equity_r.append(cur_r)
            equity_quote.append(cur_quote)
        max_dd_r = _max_drawdown(equity_r)
        max_dd_quote = _max_drawdown(equity_quote)

        gross_pnl_quote = sum(t.gross_pnl_quote for t in trades)
        net_pnl_quote = sum(t.net_pnl_quote for t in trades)
        total_fees = sum(t.fees_open_quote + t.fees_close_quote for t in trades)
        total_funding = sum(t.funding_quote for t in trades)
        avg_duration = sum(t.bars_held for t in trades) / total
        avg_time_negative = sum(t.time_in_negative_pct for t in trades) / total

        return BacktestReport(
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy_r=expectancy,
            net_r=net_r,
            max_drawdown_r=max_dd_r,
            gross_pnl_quote=gross_pnl_quote,
            net_pnl_quote=net_pnl_quote,
            total_fees_quote=total_fees,
            total_funding_quote=total_funding,
            realized_gains_quote=gross_win,
            realized_losses_quote=gross_loss,
            avg_trade_duration_bars=avg_duration,
            avg_time_in_negative_pct=avg_time_negative,
            max_drawdown_quote=max_dd_quote,
            trades=trades,
        )
