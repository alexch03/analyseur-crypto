"""Tests d'integrite : traitement des donnees, exactitude mathematique,
et parite backtest/live (absence de look-ahead bias).

Ces tests verrouillent les proprietes critiques d'un systeme de trading :
  1. DONNEES   : validation OHLCV, detection de trous, strip bougie non close
  2. MATHS     : ATR (Wilder), RSI bornes, signe P&L long/short, compound
  3. BACKTEST  : entree a signal+1 (pas de look-ahead), resolution intrabar
                 conservatrice (SL d'abord), signe P&L short via le moteur

Si un de ces tests casse, c'est qu'une regression a introduit soit un biais
optimiste dans le backtest, soit une erreur de signe/formule.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.ingestion.data_quality import (
    detect_gaps,
    strip_unclosed_candle,
    validate_ohlcv_df,
)
from app.patterns._indicators import compute_atr, compute_rsi
from app.paper.engine_replay import ReplayBacktestEngine
from app.paper.unit_tracker import compute_pct_gain
from app.schemas.domain import Side, TradeSetupDTO


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _df(rows: list[dict], start: datetime | None = None, bar_min: int = 15) -> pd.DataFrame:
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    out = []
    for i, r in enumerate(rows):
        out.append({
            "timestamp": start + timedelta(minutes=bar_min * i),
            "open": r["o"], "high": r["h"], "low": r["l"],
            "close": r["c"], "volume": r.get("v", 100.0),
        })
    return pd.DataFrame(out)


def _ohlc(o, h, l, c, v=100.0) -> dict:
    return {"o": o, "h": h, "l": l, "c": c, "v": v}


# ═════════════════════════════════════════════════════════════════════════════
# 1. INTEGRITE DES DONNEES
# ═════════════════════════════════════════════════════════════════════════════

def test_validate_ohlcv_clean_passes():
    df = _df([_ohlc(100, 101, 99, 100.5), _ohlc(100.5, 102, 100, 101)])
    assert validate_ohlcv_df(df) == []


def test_validate_ohlcv_catches_high_lt_low():
    df = _df([_ohlc(100, 98, 99, 99)])  # high=98 < low=99
    issues = validate_ohlcv_df(df)
    assert any(i.kind == "high_lt_low" for i in issues)


def test_validate_ohlcv_catches_out_of_range_and_neg_volume():
    df = _df([
        _ohlc(105, 101, 99, 100),       # open=105 > high=101
        _ohlc(100, 101, 99, 103),       # close=103 > high=101
        _ohlc(100, 101, 99, 100, v=-5), # volume negatif
    ])
    kinds = {i.kind for i in validate_ohlcv_df(df)}
    assert "open_out_of_range" in kinds
    assert "close_out_of_range" in kinds
    assert "neg_volume" in kinds


def test_validate_ohlcv_catches_nan():
    df = _df([_ohlc(100, 101, 99, 100)])
    df.loc[0, "close"] = np.nan
    assert any(i.kind == "nan" for i in validate_ohlcv_df(df))


def test_detect_gaps_finds_hole():
    # 15m bars, mais on saute de la bougie 1 a une bougie 3 bars plus loin
    base = datetime(2026, 1, 1, tzinfo=UTC)
    df = pd.DataFrame([
        {"timestamp": base, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"timestamp": base + timedelta(minutes=15), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        # trou : on saute 14:30 et 14:45, prochaine bougie a +60min
        {"timestamp": base + timedelta(minutes=75), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
    ])
    gaps = detect_gaps(df, bar_seconds=900)  # 15m = 900s
    assert len(gaps) == 1
    assert gaps[0].missing_bars == 3  # 75min - 15min = 60min = 4 bars d'ecart -> 3 manquantes


def test_detect_gaps_continuous_series_is_clean():
    df = _df([_ohlc(100, 101, 99, 100) for _ in range(10)], bar_min=15)
    assert detect_gaps(df, bar_seconds=900) == []


def test_strip_unclosed_candle_removes_forming():
    # Derniere bougie ouverte a 12:00, bar 1h. Si "now" = 12:30, elle est en formation.
    base = datetime(2026, 1, 1, 11, 0, tzinfo=UTC)
    df = _df([_ohlc(100, 101, 99, 100), _ohlc(100, 102, 100, 101)], start=base, bar_min=60)
    now = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)  # 12:00 bar pas encore close (close a 13:00)
    stripped = strip_unclosed_candle(df, bar_seconds=3600, now=now)
    assert len(stripped) == 1


def test_strip_unclosed_candle_keeps_closed():
    base = datetime(2026, 1, 1, 11, 0, tzinfo=UTC)
    df = _df([_ohlc(100, 101, 99, 100), _ohlc(100, 102, 100, 101)], start=base, bar_min=60)
    now = datetime(2026, 1, 1, 13, 30, tzinfo=UTC)  # 12:00 bar close a 13:00 < now -> complete
    stripped = strip_unclosed_candle(df, bar_seconds=3600, now=now)
    assert len(stripped) == 2


# ═════════════════════════════════════════════════════════════════════════════
# 2. EXACTITUDE MATHEMATIQUE
# ═════════════════════════════════════════════════════════════════════════════

def test_atr_constant_true_range_converges():
    # Toutes les bougies ont un range constant de 2.0, closes egaux -> TR=2 partout.
    closes = [100.0] * 30
    df = pd.DataFrame({
        "high": [101.0] * 30,
        "low": [99.0] * 30,
        "close": closes,
    })
    atr = compute_atr(df, period=14)
    # ATR de Wilder sur un TR constant = ce TR exactement.
    assert atr[-1] == pytest.approx(2.0, abs=1e-6)


def test_atr_non_negative():
    rng = np.random.RandomState(0)
    base = 100 + np.cumsum(rng.normal(0, 1, 100))
    df = pd.DataFrame({
        "high": base + 1.0, "low": base - 1.0, "close": base,
    })
    atr = compute_atr(df, period=14)
    valid = atr[~np.isnan(atr)]
    assert (valid >= 0).all()


def test_rsi_bounded_and_no_nan_explosion():
    # Serie mixte (gains et pertes) -> RSI strictement dans [0,100], pas de crash.
    rng = np.random.RandomState(1)
    closes = 100 + np.cumsum(rng.normal(0, 1, 100))
    rsi = compute_rsi(closes, period=14)
    assert len(rsi) == len(closes)
    assert np.all(rsi >= 0.0) and np.all(rsi <= 100.0)
    assert not np.any(np.isnan(rsi))


def test_rsi_up_biased_above_50():
    # Tendance haussiere avec quelques respirations baissieres -> RSI final > 50.
    closes = []
    p = 100.0
    for i in range(60):
        p += 1.0 if i % 5 != 0 else -0.3  # majoritairement haussier
        closes.append(p)
    rsi = compute_rsi(np.array(closes), period=14)
    assert rsi[-1] > 50.0


def test_pct_gain_long_sign():
    # LONG gagnant : exit > entry -> positif
    assert compute_pct_gain(Side.LONG, 100.0, 110.0) == pytest.approx(10.0)
    # LONG perdant : exit < entry -> negatif
    assert compute_pct_gain(Side.LONG, 100.0, 95.0) == pytest.approx(-5.0)


def test_pct_gain_short_sign():
    # SHORT gagnant : exit < entry -> positif (prix baisse = profit)
    assert compute_pct_gain(Side.SHORT, 100.0, 90.0) == pytest.approx(11.1111, abs=1e-3)
    # SHORT perdant : exit > entry -> negatif
    assert compute_pct_gain(Side.SHORT, 100.0, 110.0) == pytest.approx(-9.0909, abs=1e-3)


def test_compound_is_multiplicative():
    # +10% puis -10% = -1% (compound), PAS 0% (additif)
    gains = [10.0, -10.0]
    factor = 1.0
    for g in gains:
        factor *= 1.0 + g / 100.0
    compound_pct = (factor - 1.0) * 100.0
    assert compound_pct == pytest.approx(-1.0)


# ═════════════════════════════════════════════════════════════════════════════
# 3. PARITE BACKTEST / LIVE — ABSENCE DE LOOK-AHEAD
# ═════════════════════════════════════════════════════════════════════════════

def _setup_long(entry=100.0, sl=98.0, tp=104.0) -> TradeSetupDTO:
    return TradeSetupDTO(
        symbol="X/USDT", timeframe="15m", side=Side.LONG,
        entry=entry, stop_loss=sl, take_profits=[tp],
        risk_reward=(tp - entry) / (entry - sl), confidence=0.7,
        setup_type="TEST", timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _setup_short(entry=100.0, sl=102.0, tp=96.0) -> TradeSetupDTO:
    return TradeSetupDTO(
        symbol="X/USDT", timeframe="15m", side=Side.SHORT,
        entry=entry, stop_loss=sl, take_profits=[tp],
        risk_reward=(entry - tp) / (sl - entry), confidence=0.7,
        setup_type="TEST", timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _arrays(rows: list[dict]):
    highs = [r["h"] for r in rows]
    lows = [r["l"] for r in rows]
    closes = [r["c"] for r in rows]
    opens = [r["o"] for r in rows]
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * i) for i in range(len(rows))]
    return highs, lows, closes, opens, ts


def test_entry_only_after_signal_bar():
    """L'entree ne doit JAMAIS avoir lieu sur la bougie du signal (look-ahead)."""
    engine = ReplayBacktestEngine(warmup_bars=0, max_holding_bars=20)
    setup = _setup_long(entry=100.0, sl=98.0, tp=104.0)
    # Bar 2 (= signal_index) contient deja le prix d'entree, mais on ne doit PAS entrer dessus.
    rows = [
        _ohlc(100, 100.5, 99.5, 100),   # 0
        _ohlc(100, 100.5, 99.5, 100),   # 1
        _ohlc(100, 100.5, 99.5, 100),   # 2  <- signal bar, contient entry=100
        _ohlc(100, 100.5, 99.5, 100),   # 3  <- entree autorisee ici au plus tot
        _ohlc(100, 104.5, 99.5, 104),   # 4  TP
    ]
    highs, lows, closes, opens, ts = _arrays(rows)
    trade = engine._simulate_one_trade(
        setup=setup, signal_index=2,
        highs=highs, lows=lows, closes=closes, opens=opens,
        timestamps=ts, bar_hours=0.25,
    )
    assert trade is not None
    assert trade.opened_index >= 3, "entree avant signal+1 = look-ahead bias"


def test_intrabar_resolution_is_conservative():
    """Si SL et TP sont dans la MEME bougie haussiere, le SL doit gagner (worst-case).

    C'est le test anti-biais : l'ancienne regle 'bougie haussiere -> TP' surestimait
    le winrate. La regle conservatrice resout le cote adverse en premier.
    """
    engine = ReplayBacktestEngine(warmup_bars=0, max_holding_bars=20)
    setup = _setup_long(entry=100.0, sl=98.0, tp=104.0)
    rows = [
        _ohlc(100, 100.5, 99.5, 100),    # 0 signal
        _ohlc(100, 100.5, 99.5, 100),    # 1 entree (touche 100)
        # 2 : bougie HAUSSIERE (close>open) qui touche SL (97<=98) ET TP (105>=104)
        _ohlc(99, 105, 97, 104),
    ]
    highs, lows, closes, opens, ts = _arrays(rows)
    trade = engine._simulate_one_trade(
        setup=setup, signal_index=0,
        highs=highs, lows=lows, closes=closes, opens=opens,
        timestamps=ts, bar_hours=0.25,
    )
    assert trade is not None
    assert trade.outcome in ("SL", "SL_BE"), f"attendu SL conservateur, obtenu {trade.outcome}"
    assert trade.close_price == pytest.approx(98.0)


def test_intrabar_gap_through_tp_allows_tp():
    """Exception gap : si l'open est deja au-dela du TP, le TP est le 1er prix dispo."""
    engine = ReplayBacktestEngine(warmup_bars=0, max_holding_bars=20)
    setup = _setup_long(entry=100.0, sl=98.0, tp=104.0)
    rows = [
        _ohlc(100, 100.5, 99.5, 100),    # 0 signal
        _ohlc(100, 100.5, 99.5, 100),    # 1 entree
        # 2 : open=104.5 deja au-dessus du TP (gap), low repasse a 97
        _ohlc(104.5, 106, 97, 100),
    ]
    highs, lows, closes, opens, ts = _arrays(rows)
    trade = engine._simulate_one_trade(
        setup=setup, signal_index=0,
        highs=highs, lows=lows, closes=closes, opens=opens,
        timestamps=ts, bar_hours=0.25,
    )
    assert trade is not None
    assert trade.outcome == "TP"
    assert trade.close_price == pytest.approx(104.0)


def test_short_trade_tp_gives_positive_pnl():
    """Un SHORT qui atteint son TP (prix baisse) doit donner un PnL net positif."""
    engine = ReplayBacktestEngine(warmup_bars=0, max_holding_bars=20,
                                  entry_fee_rate=0.0, exit_fee_rate=0.0)
    setup = _setup_short(entry=100.0, sl=102.0, tp=96.0)
    rows = [
        _ohlc(100, 100.5, 99.5, 100),    # 0 signal
        _ohlc(100, 100.5, 99.5, 100),    # 1 entree (touche 100)
        _ohlc(99, 99.5, 95.5, 96),       # 2 baisse, touche TP=96
    ]
    highs, lows, closes, opens, ts = _arrays(rows)
    trade = engine._simulate_one_trade(
        setup=setup, signal_index=0,
        highs=highs, lows=lows, closes=closes, opens=opens,
        timestamps=ts, bar_hours=0.25,
    )
    assert trade is not None
    assert trade.outcome == "TP"
    assert trade.net_pnl_quote > 0, "SHORT gagnant doit avoir PnL > 0 (verifie le signe)"
    assert trade.close_price == pytest.approx(96.0)


def test_short_trade_sl_gives_negative_pnl():
    """Un SHORT stoppe (prix monte au-dessus du SL) doit donner un PnL net negatif."""
    engine = ReplayBacktestEngine(warmup_bars=0, max_holding_bars=20,
                                  entry_fee_rate=0.0, exit_fee_rate=0.0)
    setup = _setup_short(entry=100.0, sl=102.0, tp=96.0)
    rows = [
        _ohlc(100, 100.5, 99.5, 100),    # 0 signal
        _ohlc(100, 100.5, 99.5, 100),    # 1 entree
        _ohlc(101, 102.5, 100.5, 102),   # 2 monte, touche SL=102
    ]
    highs, lows, closes, opens, ts = _arrays(rows)
    trade = engine._simulate_one_trade(
        setup=setup, signal_index=0,
        highs=highs, lows=lows, closes=closes, opens=opens,
        timestamps=ts, bar_hours=0.25,
    )
    assert trade is not None
    assert trade.outcome in ("SL", "SL_BE")
    assert trade.net_pnl_quote < 0
