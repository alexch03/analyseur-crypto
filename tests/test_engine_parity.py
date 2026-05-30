"""Test de PARITE entre le moteur de BACKTEST (replay) et le moteur LIVE.

Deux moteurs implementent la meme logique de trade de facons distinctes :

  * REPLAY  — ``app.paper.engine_replay.ReplayBacktestEngine._simulate_one_trade``
              Entree par ordre LIMITE : des qu'une bougie (apres le signal) voit son
              range toucher ``entry`` (``low <= entry <= high``), on entre AU PRIX
              ``entry``. Puis boucle SL/TP/trailing intrabar.

  * LIVE    — ``app.services.hypothesis_engine.HypothesisEngine.step`` (machine a etats
              FORMING -> ARMED -> TRIGGERED -> TARGET_HIT/STOPPED), pilote par
              ``continuous_scanner``. Entree par CASSURE CONFIRMEE EN CLOTURE : on ne
              declenche que lorsqu'une bougie CLOTURE au-dela du niveau, avec un body
              ratio suffisant et un filtre regime favorable. Le fill se fait AU CLOSE.

Ce test genere des series de bougies synthetiques a pattern connu, les passe dans LES
DEUX moteurs et compare (entree, sortie, outcome, PnL). Il QUANTIFIE et VERROUILLE les
divergences pour qu'aucune regression ne les elargisse silencieusement.

----------------------------------------------------------------------------------------
TAXONOMIE DES DIVERGENCES (mesurees ci-dessous)
----------------------------------------------------------------------------------------
1. MODELE D'ENTREE (prix de fill)
   - Replay : fill au NIVEAU ``entry`` (ordre limite).
   - Live   : fill a la CLOTURE de la bougie de cassure (ordre marche).
   => Sur un trade que LES DEUX prennent, le PnL diverge mecaniquement de l'ecart
      (close_cassure - entry). Scenarios A / E / F.

2. SELECTION DES TRADES (quels trades sont pris)
   a) Meche sans cloture : le replay entre sur une simple meche qui traverse ``entry`` ;
      le live exige une CLOTURE au-dela => le live n'entre pas. Scenario B (le replay
      encaisse une perte fantome que le live evite via INVALIDATED avant trigger).
   b) Body ratio : le live rejette les cassures a faible body (``min_breakout_body_ratio``,
      0.3 en prod) ; le replay n'a pas ce filtre => il prend un trade (gagnant fantome ici)
      que le live ignore. Scenario C.
   c) Filtre regime : le live re-verifie l'affinite pattern x regime AU MOMENT DU TRIGGER
      (``_still_passes_regime_filter``) et INVALIDE si contre-tendance ; le replay n'a pas
      ce filtre. Scenario D.

3. SORTIE / TIMEOUT (non illustre par des divergences ici, mais documente)
   - Replay : break-even et trailing en multiples de R, + TIMEOUT intelligent
     (``max_holding_bars`` + extensions Bollinger/tendance).
   - Live   : break-even et trailing en % / ATR, et AUCUN timeout (la position reste
     ouverte jusqu'a target ou stop).
   La detection d'atteinte SL/TP est en revanche identique (intrabar high/low vs niveaux
   absolus) et conservatrice des deux cotes (stop verifie en premier).

----------------------------------------------------------------------------------------
REDUCTION DE LA DIVERGENCE (objectif 3)
----------------------------------------------------------------------------------------
La porte d'entree opt-in du replay (``entry_min_body_ratio`` + ``entry_abort_on_invalidation``
+ ``entry_regime``) aligne la SELECTION des trades sur celle du live : avec elle, le replay
prend/rejette exactement les memes trades que le live (scenarios B/C/D -> NO_TRADE comme le
live). La divergence (1) de PRIX d'execution (fill au niveau vs au close) reste, par choix de
modele d'ordre, et est mesuree ici pour rester visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

from app.paper.engine_replay import ReplayBacktestEngine
from app.paper.unit_tracker import compute_pct_gain
from app.schemas.domain import Side, TradeSetupDTO
from app.schemas.hypothesis import HypothesisState
from app.schemas.patterns import BreakoutDirection, ChartPatternDTO, PatternKind
from app.services.hypothesis_engine import HypothesisEngine
from app.services.market_regime import MarketRegime

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_CONSOL = 20          # bougies de consolidation (idx 0..19) => spawn/signal a l'idx 19
_SPAWN = _CONSOL - 1  # 19
_BODY_PROD = 0.30     # min_breakout_body_ratio de production cote live


# ─────────────────────────────────────────────────────────────────────────────
# Construction de bougies / DTO
# ─────────────────────────────────────────────────────────────────────────────

def _df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = []
    for i, (o, h, l, c) in enumerate(bars):
        rows.append({
            "timestamp": _T0 + timedelta(minutes=15 * i),
            "open": float(o), "high": float(h), "low": float(l),
            "close": float(c), "volume": 1000.0,
        })
    return pd.DataFrame(rows)


def _consolidation(n: int, price: float, *, half_range: float = 1.0) -> list[tuple]:
    return [(price, price + half_range, price - half_range, price) for _ in range(n)]


def _pattern(kind: PatternKind, side: Side, entry: float, stop: float,
             target: float, *, end_idx: int) -> ChartPatternDTO:
    direction = BreakoutDirection.UP if side == Side.LONG else BreakoutDirection.DOWN
    return ChartPatternDTO(
        kind=kind, symbol="X/USDT", timeframe="15m",
        start_index=0, end_index=end_idx,
        start_timestamp=_T0, end_timestamp=_T0 + timedelta(minutes=15 * end_idx),
        breakout_level=entry, invalidation_level=stop,
        breakout_direction=direction, height=abs(target - entry),
        target=target, confidence=0.9,
    )


def _setup(side: Side, entry: float, stop: float, target: float,
           setup_type: str) -> TradeSetupDTO:
    rr = abs(target - entry) / abs(entry - stop)
    return TradeSetupDTO(
        symbol="X/USDT", timeframe="15m", side=side,
        entry=entry, stop_loss=stop, take_profits=[target],
        risk_reward=rr, confidence=0.9, setup_type=setup_type, timestamp=_T0,
    )


def _bear_regime(strength: float = 0.9) -> MarketRegime:
    """BEAR fort : pattern_regime_score(TRIANGLE_ASC) = 1 + (0.5-1)*0.9 = 0.55 < 0.70."""
    return MarketRegime(
        trend="BEAR", volatility="NORMAL", strength=strength,
        btc_change_24h_pct=-5.0, btc_above_sma50=False, btc_above_sma200=False,
        breadth_pct=25.0, atr_pct=1.0, detected_at=_T0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resultat normalise + runners
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Outcome:
    engine: str
    took_trade: bool
    bucket: str                  # TARGET | STOP | TIMEOUT | NO_TRADE
    entry_idx: int | None
    entry_price: float | None
    exit_idx: int | None
    exit_price: float | None
    pnl_pct: float | None        # brut (hors frais), signe selon le sens
    detail: str = ""


def run_replay(engine: ReplayBacktestEngine, df: pd.DataFrame,
               setup: TradeSetupDTO, signal_index: int) -> Outcome:
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    opens = df["open"].tolist()
    ts = df["timestamp"].tolist()
    trade = engine._simulate_one_trade(
        setup=setup, signal_index=signal_index,
        highs=highs, lows=lows, closes=closes, opens=opens,
        timestamps=ts, bar_hours=0.25,
    )
    if trade is None:
        return Outcome("replay", False, "NO_TRADE", None, None, None, None, None,
                       "entree jamais remplie / abandonnee")
    if trade.outcome == "TP":
        bucket = "TARGET"
    elif trade.outcome in ("SL", "SL_BE", "TRAIL"):
        bucket = "STOP"
    else:
        bucket = "TIMEOUT"
    entry_price = float(trade.entry)
    pnl = compute_pct_gain(setup.side, entry_price, float(trade.close_price))
    return Outcome("replay", True, bucket, trade.opened_index, entry_price,
                   trade.closed_index, float(trade.close_price), pnl, trade.outcome)


def run_live(engine: HypothesisEngine, df: pd.DataFrame, pattern: ChartPatternDTO,
             spawn_idx: int, *, regime_at=None) -> Outcome:
    """Pilote le moteur live bougie par bougie (slice df[:i+1]) comme le scanner."""
    existing: list = []
    current = None
    trigger_idx: int | None = None
    exit_idx: int | None = None
    n = len(df)
    for i in range(spawn_idx, n):
        if regime_at is not None:
            engine.set_market_regime(regime_at(i))
        sub = df.iloc[: i + 1].reset_index(drop=True)
        new_patterns = [pattern] if i == spawn_idx else []
        res = engine.step(sub, new_patterns, existing)
        pool = res.created + res.updated
        if current is None and res.created:
            current = res.created[0]
        match = next((h for h in pool if current is not None and h.id == current.id), None)
        if match is not None:
            current = match
        existing = [h for h in pool if not h.is_terminal]
        if current is not None:
            if current.triggered_at is not None and trigger_idx is None:
                trigger_idx = i
            if current.is_terminal:
                exit_idx = i
                break

    if current is None or current.triggered_at is None:
        state = current.state.value if current is not None else "NONE"
        return Outcome("live", False, "NO_TRADE", None, None, None, None, None,
                       f"state={state}")

    entry_price = float(current.triggered_price)
    if current.state == HypothesisState.TARGET_HIT:
        bucket, exit_price = "TARGET", float(current.outcome_price)
    elif current.state == HypothesisState.STOPPED:
        bucket, exit_price = "STOP", float(current.outcome_price)
    else:
        # Toujours TRIGGERED en fin de serie : le live n'a pas de timeout (position ouverte).
        bucket, exit_price = "TIMEOUT", float(df["close"].iloc[-1])
        exit_idx = n - 1
    pnl = compute_pct_gain(current.side, entry_price, exit_price)
    return Outcome("live", True, bucket, trigger_idx, entry_price, exit_idx, exit_price,
                   pnl, f"state={current.state.value}")


# ─────────────────────────────────────────────────────────────────────────────
# Fabriques de moteurs
# ─────────────────────────────────────────────────────────────────────────────

def _replay_default() -> ReplayBacktestEngine:
    return ReplayBacktestEngine(warmup_bars=0, max_holding_bars=40, timeout_smart_extend=False)


def _replay_faithful(*, regime: MarketRegime | None = None) -> ReplayBacktestEngine:
    """Replay avec la porte d'entree opt-in => SELECTION alignee sur le live."""
    return ReplayBacktestEngine(
        warmup_bars=0, max_holding_bars=40, timeout_smart_extend=False,
        entry_min_body_ratio=_BODY_PROD,
        entry_abort_on_invalidation=True,
        entry_regime=regime,
        entry_regime_min_score=0.70,
    )


def _live_engine(*, body: float = 0.0, regime_adaptive: bool = False) -> HypothesisEngine:
    return HypothesisEngine(
        arm_proximity_pct=0.005,
        min_confluence_score=0.0,
        reject_trend_counter=False,
        breakeven_trigger_pct=0.0,
        trailing_stop_atr_mult=0.0,
        min_breakout_body_ratio=body,
        regime_adaptive=regime_adaptive,
        regime_min_score=0.70,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios (LONG : entry/breakout=110, stop/inval=100, target=120)
#           (SHORT: entry/breakout=90,  stop/inval=100, target=80)
# ─────────────────────────────────────────────────────────────────────────────

_LONG = (110.0, 100.0, 120.0)
_SHORT = (90.0, 100.0, 80.0)


def _long_df(tail: list[tuple]) -> pd.DataFrame:
    # idx 0..19 consolidation @105 ; idx 20 approche (close 109.6 -> ARMED) ; idx 21+ = tail
    return _df(_consolidation(_CONSOL, 105.0) + [(105.0, 109.8, 104.8, 109.6)] + tail)


def _short_df(tail: list[tuple]) -> pd.DataFrame:
    return _df(_consolidation(_CONSOL, 95.0) + [(95.0, 95.2, 90.2, 90.4)] + tail)


# Tails par scenario (les commentaires donnent l'idx absolu)
_TAIL_A = [(109.6, 113.0, 109.4, 112.5), (112.5, 117.0, 112.0, 116.0), (116.0, 121.0, 115.5, 120.5)]
_TAIL_B = [(109.6, 112.0, 108.0, 108.5), (108.5, 109.0, 101.0, 102.0), (101.0, 101.5, 99.0, 99.0)]
_TAIL_C = [(109.8, 121.0, 109.5, 110.4), (110.4, 111.0, 105.0, 106.0), (106.0, 106.5, 104.5, 105.0)]
_TAIL_D = _TAIL_A
_TAIL_E = [(109.6, 113.0, 109.4, 112.5), (112.5, 112.5, 99.5, 101.0)]
_TAIL_F = [(90.4, 90.6, 87.0, 87.5), (87.5, 88.0, 83.0, 84.0), (84.0, 84.0, 79.0, 80.5)]


# ─────────────────────────────────────────────────────────────────────────────
# Scenario A — cassure propre vers target (LONG) : MEME outcome, PnL divergent (fill)
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_a_clean_win_entry_model_divergence():
    df = _long_df(_TAIL_A)
    entry, stop, target = _LONG
    rep = run_replay(_replay_default(), df, _setup(Side.LONG, entry, stop, target, "TRIANGLE_ASC"), _SPAWN)
    liv = run_live(_live_engine(), df, _pattern(PatternKind.TRIANGLE_ASC, Side.LONG, entry, stop, target, end_idx=_SPAWN), _SPAWN)

    # Les deux gagnent (TARGET) et entrent sur la MEME bougie de cassure (idx 21)...
    assert rep.bucket == "TARGET" and liv.bucket == "TARGET"
    assert rep.entry_idx == liv.entry_idx == 21
    # ... mais le replay remplit au NIVEAU (110) et le live au CLOSE (112.5).
    assert rep.entry_price == 110.0
    assert liv.entry_price == 112.5
    # => divergence de PnL purement due au modele d'entree (~2.4 points de %).
    assert rep.pnl_pct > liv.pnl_pct
    assert abs(rep.pnl_pct - liv.pnl_pct) > 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Scenario B — meche traverse entry sans cloture : replay = perte fantome ; live = NO_TRADE
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_b_wick_touch_replay_phantom_loss():
    df = _long_df(_TAIL_B)
    entry, stop, target = _LONG
    rep = run_replay(_replay_default(), df, _setup(Side.LONG, entry, stop, target, "TRIANGLE_ASC"), _SPAWN)
    liv = run_live(_live_engine(), df, _pattern(PatternKind.TRIANGLE_ASC, Side.LONG, entry, stop, target, end_idx=_SPAWN), _SPAWN)

    # Le replay entre sur la meche (low<=110<=high) puis se fait stopper : perte FANTOME.
    assert rep.took_trade and rep.bucket == "STOP"
    assert rep.pnl_pct < 0
    # Le live n'a jamais cloture au-dessus de 110 -> reste ARMED -> INVALIDATED en croisant
    # l'invalidation avant trigger -> AUCUN trade.
    assert liv.took_trade is False and liv.bucket == "NO_TRADE"

    # La porte opt-in du replay supprime la perte fantome (parite de selection avec le live).
    rep_f = run_replay(_replay_faithful(), df, _setup(Side.LONG, entry, stop, target, "TRIANGLE_ASC"), _SPAWN)
    assert rep_f.bucket == "NO_TRADE"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario C — cassure a faible body qui pique le target : replay = gain fantome ; live skip
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_c_weak_body_replay_phantom_win():
    df = _long_df(_TAIL_C)
    entry, stop, target = _LONG
    rep = run_replay(_replay_default(), df, _setup(Side.LONG, entry, stop, target, "TRIANGLE_ASC"), _SPAWN)
    liv = run_live(_live_engine(body=_BODY_PROD), df, _pattern(PatternKind.TRIANGLE_ASC, Side.LONG, entry, stop, target, end_idx=_SPAWN), _SPAWN)

    # Le replay entre au niveau et la meme bougie pique le target (high>=120) : gain FANTOME.
    assert rep.took_trade and rep.bucket == "TARGET" and rep.pnl_pct > 0
    # Le live rejette la cassure (body 0.05 < 0.30) -> n'entre jamais.
    assert liv.took_trade is False and liv.bucket == "NO_TRADE"

    # Porte opt-in (body 0.30) -> le replay rejette aussi la cassure.
    rep_f = run_replay(_replay_faithful(), df, _setup(Side.LONG, entry, stop, target, "TRIANGLE_ASC"), _SPAWN)
    assert rep_f.bucket == "NO_TRADE"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario D — filtre regime au trigger : live INVALIDE en BEAR ; replay prend le trade
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_d_regime_block_at_trigger():
    df = _long_df(_TAIL_D)
    entry, stop, target = _LONG
    rep = run_replay(_replay_default(), df, _setup(Side.LONG, entry, stop, target, "TRIANGLE_ASC"), _SPAWN)

    # Regime None jusqu'a la cassure (idx 21), puis BEAR (flip avant le trigger).
    regime_at = lambda i: (None if i < 21 else _bear_regime())  # noqa: E731
    liv = run_live(_live_engine(regime_adaptive=True), df,
                   _pattern(PatternKind.TRIANGLE_ASC, Side.LONG, entry, stop, target, end_idx=_SPAWN),
                   _SPAWN, regime_at=regime_at)

    assert rep.took_trade and rep.bucket == "TARGET"          # replay : aucun filtre regime
    assert liv.took_trade is False and liv.bucket == "NO_TRADE"  # live : regime-block at trigger

    # Porte opt-in avec regime BEAR -> le replay rejette aussi (parite de selection).
    rep_f = run_replay(_replay_faithful(regime=_bear_regime()), df,
                       _setup(Side.LONG, entry, stop, target, "TRIANGLE_ASC"), _SPAWN)
    assert rep_f.bucket == "NO_TRADE"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario E — stop propre (LONG) : MEME outcome (STOP@100), PnL divergent (fill)
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_e_clean_stop_same_exit_diff_pnl():
    df = _long_df(_TAIL_E)
    entry, stop, target = _LONG
    rep = run_replay(_replay_default(), df, _setup(Side.LONG, entry, stop, target, "TRIANGLE_ASC"), _SPAWN)
    liv = run_live(_live_engine(), df, _pattern(PatternKind.TRIANGLE_ASC, Side.LONG, entry, stop, target, end_idx=_SPAWN), _SPAWN)

    assert rep.bucket == "STOP" and liv.bucket == "STOP"
    # Sortie au MEME prix (l'invalidation = 100).
    assert rep.exit_price == liv.exit_price == 100.0
    # Mais l'entree differe (110 vs 112.5) => la perte du live est PLUS grande.
    assert rep.entry_price == 110.0 and liv.entry_price == 112.5
    assert liv.pnl_pct < rep.pnl_pct < 0


# ─────────────────────────────────────────────────────────────────────────────
# Scenario F — cassure propre vers target (SHORT) : MEME outcome, PnL divergent (fill)
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_f_short_clean_win_entry_model_divergence():
    df = _short_df(_TAIL_F)
    entry, stop, target = _SHORT
    rep = run_replay(_replay_default(), df, _setup(Side.SHORT, entry, stop, target, "TRIANGLE_DESC"), _SPAWN)
    liv = run_live(_live_engine(), df, _pattern(PatternKind.TRIANGLE_DESC, Side.SHORT, entry, stop, target, end_idx=_SPAWN), _SPAWN)

    assert rep.bucket == "TARGET" and liv.bucket == "TARGET"
    assert rep.entry_idx == liv.entry_idx == 21
    assert rep.entry_price == 90.0 and liv.entry_price == 87.5
    # SHORT gagnant : le replay (fill a 90) gagne plus que le live (fill a 87.5).
    assert rep.pnl_pct > liv.pnl_pct > 0
    assert abs(rep.pnl_pct - liv.pnl_pct) > 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Synthese quantifiee de la divergence (table + assertions d'agregat)
# ─────────────────────────────────────────────────────────────────────────────

def _all_scenarios() -> list[tuple[str, Outcome, Outcome, Outcome]]:
    """Renvoie [(nom, replay_defaut, live, replay_fidele), ...] pour tous les scenarios."""
    out: list[tuple[str, Outcome, Outcome, Outcome]] = []

    def long_case(name, tail, *, body=0.0, regime_adaptive=False, regime_at=None, faithful_regime=None):
        df = _long_df(tail)
        e, s, t = _LONG
        rep = run_replay(_replay_default(), df, _setup(Side.LONG, e, s, t, "TRIANGLE_ASC"), _SPAWN)
        liv = run_live(_live_engine(body=body, regime_adaptive=regime_adaptive), df,
                       _pattern(PatternKind.TRIANGLE_ASC, Side.LONG, e, s, t, end_idx=_SPAWN),
                       _SPAWN, regime_at=regime_at)
        repf = run_replay(_replay_faithful(regime=faithful_regime), df,
                          _setup(Side.LONG, e, s, t, "TRIANGLE_ASC"), _SPAWN)
        out.append((name, rep, liv, repf))

    long_case("A clean-win", _TAIL_A)
    long_case("B wick-touch", _TAIL_B)
    long_case("C weak-body", _TAIL_C, body=_BODY_PROD)
    long_case("D regime-block", _TAIL_D, regime_adaptive=True,
              regime_at=lambda i: (None if i < 21 else _bear_regime()),
              faithful_regime=_bear_regime())
    long_case("E clean-stop", _TAIL_E)

    df = _short_df(_TAIL_F)
    e, s, t = _SHORT
    out.append((
        "F short-win",
        run_replay(_replay_default(), df, _setup(Side.SHORT, e, s, t, "TRIANGLE_DESC"), _SPAWN),
        run_live(_live_engine(), df, _pattern(PatternKind.TRIANGLE_DESC, Side.SHORT, e, s, t, end_idx=_SPAWN), _SPAWN),
        run_replay(_replay_faithful(), df, _setup(Side.SHORT, e, s, t, "TRIANGLE_DESC"), _SPAWN),
    ))
    return out


def format_parity_table() -> str:
    """Table lisible (visible avec ``pytest -s``)."""
    rows = _all_scenarios()
    lines = [
        "",
        "DIVERGENCE REPLAY (defaut) vs LIVE - entree/sortie/outcome/PnL",
        "-" * 92,
        f"{'scenario':<16}{'replay':<22}{'live':<22}{'PnL rep':>9}{'PnL live':>9}{'parite':>10}",
        "-" * 92,
    ]

    def cell(o: Outcome) -> str:
        if not o.took_trade:
            return f"{o.bucket}"
        return f"{o.bucket}@{o.exit_price:g} e={o.entry_price:g}"

    for name, rep, liv, _ in rows:
        rp = f"{rep.pnl_pct:+.2f}" if rep.pnl_pct is not None else "  -  "
        lp = f"{liv.pnl_pct:+.2f}" if liv.pnl_pct is not None else "  -  "
        same = "OUI" if rep.bucket == liv.bucket else "NON"
        lines.append(f"{name:<16}{cell(rep):<22}{cell(liv):<22}{rp:>9}{lp:>9}{same:>10}")

    lines.append("-" * 92)
    lines.append("Note: B/C/D = le replay defaut prend un trade que le live ignore (meche/body/regime).")
    lines.append("      A/E/F = meme outcome mais PnL divergent (fill niveau vs close).")
    lines.append("")
    lines.append("PARITE DE SELECTION - replay FIDELE (porte opt-in) vs LIVE")
    lines.append("-" * 60)
    for name, _, liv, repf in rows:
        same = "OUI" if repf.bucket == liv.bucket else "NON"
        lines.append(f"{name:<16}replay_fidele={repf.bucket:<10} live={liv.bucket:<10} parite={same}")
    return "\n".join(lines)


def test_parity_summary_quantifies_divergence(capsys):
    rows = _all_scenarios()
    print(format_parity_table())

    # 1) DIVERGENCE DE SELECTION (replay defaut) : exactement B, C, D divergent de bucket.
    selection_div = [name for name, rep, liv, _ in rows if rep.bucket != liv.bucket]
    assert selection_div == ["B wick-touch", "C weak-body", "D regime-block"]
    for name in selection_div:
        rep, liv = next((r, l) for n, r, l, _ in rows if n == name)
        assert rep.took_trade is True and liv.took_trade is False  # replay prend, live ignore

    # 2) DIVERGENCE DE PRIX D'ENTREE : sur A/E/F (les deux prennent) l'entree differe toujours.
    both_take = [(name, rep, liv) for name, rep, liv, _ in rows if rep.took_trade and liv.took_trade]
    assert {n for n, _, _ in both_take} == {"A clean-win", "E clean-stop", "F short-win"}
    for name, rep, liv in both_take:
        assert rep.entry_price != liv.entry_price                       # modele d'entree
        assert abs(rep.pnl_pct - liv.pnl_pct) > 1.0                     # impact PnL non trivial

    # 3) PARITE DE SELECTION avec la porte opt-in : le replay fidele matche le live partout.
    for name, _, liv, repf in rows:
        assert repf.bucket == liv.bucket, f"{name}: replay fidele {repf.bucket} != live {liv.bucket}"

    # Sanity : la table a bien ete imprimee.
    assert "DIVERGENCE REPLAY" in capsys.readouterr().out
