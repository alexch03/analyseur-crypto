"""Moteur de cycle de vie d'hypothèse.

Stateless : prend en entrée l'OHLCV courant + les patterns fraîchement détectés
+ la liste des hypothèses persistées, retourne le nouvel état et les transitions.

Règles de transition :

    FORMING ─► ARMED         : distance(close, breakout_level) ≤ arm_proximity_pct
    FORMING ─► INVALIDATED   : close franchit invalidation_level dans le mauvais sens
    ARMED   ─► TRIGGERED     : close franchit breakout_level dans le bon sens
                              (+ confirmation volume si configurée)
    ARMED   ─► INVALIDATED   : close franchit invalidation_level avant trigger
    TRIGGERED ─► TARGET_HIT  : high(LONG) ≥ target ou low(SHORT) ≤ target
    TRIGGERED ─► STOPPED     : low(LONG) ≤ invalidation ou high(SHORT) ≥ invalidation
    FORMING|ARMED ─► EXPIRED : bars écoulés depuis détection > expiry_bars

L'invalidation **avant trigger** est essentielle : si le pattern se casse avant que
le trade ne soit pris, on annule l'ordre (= état INVALIDATED, sans STOPPED ni perte).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import datetime

import pandas as pd

from app.schemas.domain import Side
from app.schemas.hypothesis import (
    HypothesisDTO,
    HypothesisState,
    StateTransition,
    is_terminal,
)
from app.schemas.patterns import BreakoutDirection, ChartPatternDTO


@dataclass(frozen=True, slots=True)
class EngineStepResult:
    created: list[HypothesisDTO]
    updated: list[HypothesisDTO]
    transitions: list[tuple[str, StateTransition]]   # (hypothesis_id, transition)


_DEFAULT_ARM_PROXIMITY_PCT = 0.005
_DEFAULT_EXPIRY_BARS = 40
_DEFAULT_DEDUPE_PRICE_TOL_PCT = 0.003
# Defaults agressifs issus de l'analyse MFE/MAE :
# - min_conf 0.45 : grid search montre que >=0.55 est optimal (29t, +1.9%)
#   on prend 0.45 pour avoir un peu plus de signaux
# - BE 0.3 : sauve 45% des stops avec MFE+0.5% moyen
# - breakout_buffer 0.001 : close doit depasser breakout de 0.1% pour filtrer les wicks
_DEFAULT_MIN_CONFLUENCE = 0.40          # filtre les setups faibles (issue grid search ~0.55 optimal)
_DEFAULT_MIN_RR = 0.0
_DEFAULT_REJECT_TREND_COUNTER = True    # CRITIQUE : DOUBLE_BOTTOM en downtrend perd massivement
_DEFAULT_REQUIRE_VOLUME = False
_DEFAULT_BREAKEVEN_TRIGGER_PCT = 0.0    # BE desactive : sur ce dataset il convertit wins en BE
_DEFAULT_BREAKOUT_BUFFER_PCT = 0.0
_DEFAULT_TRAILING_ATR_MULT = 0.0        # 0 = pas de trailing ; ex 2.0 = SL trail a bar_low - 2*ATR (LONG)
_DEFAULT_TRAILING_ACTIVATION_PCT = 0.5  # active le trailing apres 50% du chemin vers target
# Adaptation au regime de marche : si activee, l'engine filtre/scale les patterns
# selon affinite avec regime (BULL/BEAR/RANGE). Voir market_regime.PATTERN_REGIME_AFFINITY.
_DEFAULT_REGIME_ADAPTIVE = True         # active par defaut
_DEFAULT_REGIME_MIN_SCORE = 0.70        # rejette pattern si score regime < 0.70 (etait 0.65)
# NB: avec strength=0.9 (BEAR fort), il faut affinity >= 0.67 pour passer (etait 0.61).
# Tous les patterns bullish (BEAR affinity=0.3) scorent 0.37 → bloques.
# Fail-closed quand le regime est inconnu : evite le scenario observe sur 24h ou
# 336 CUP_AND_HANDLE LONG sont passes en marche BEAR (filtre desactive par defaut
# si market_regime is None apres echec de fetch BTC).
# Default False pour ne pas casser les tests legacy ; le scanner production
# l'active explicitement (continuous_scanner).
_DEFAULT_REJECT_IF_REGIME_UNKNOWN = False
_HIGHLY_DIRECTIONAL_PATTERNS = frozenset({
    "CUP_AND_HANDLE", "INVERSE_CUP_AND_HANDLE",
    "PENNANT_BULL", "PENNANT_BEAR",
    "FLAG_BULL", "FLAG_BEAR",
    "TRIANGLE_ASC", "TRIANGLE_DESC",
    "CHANNEL_UP", "CHANNEL_DOWN",
    "EXPANDING_TRIANGLE_BULLISH", "EXPANDING_TRIANGLE_BEARISH",
})
# Body ratio minimum pour confirmer un breakout (filtre les wicks)
# 0.3 = au moins 30% de la bougie est du body (60% wicks max).
# 0.0 = desactive (legacy)
_DEFAULT_MIN_BREAKOUT_BODY_RATIO = 0.3


class HypothesisEngine:
    def __init__(
        self,
        *,
        arm_proximity_pct: float = _DEFAULT_ARM_PROXIMITY_PCT,
        expiry_bars: int = _DEFAULT_EXPIRY_BARS,
        dedupe_price_tol_pct: float = _DEFAULT_DEDUPE_PRICE_TOL_PCT,
        confluence_scorer: "ConfluenceScorer | None" = None,
        min_confluence_score: float = _DEFAULT_MIN_CONFLUENCE,
        min_rr_ratio: float = _DEFAULT_MIN_RR,
        reject_trend_counter: bool = _DEFAULT_REJECT_TREND_COUNTER,
        require_volume_expansion: bool = _DEFAULT_REQUIRE_VOLUME,
        breakeven_trigger_pct: float = _DEFAULT_BREAKEVEN_TRIGGER_PCT,
        breakout_buffer_pct: float = _DEFAULT_BREAKOUT_BUFFER_PCT,
        require_volume_weak_reject: bool = False,
        excluded_patterns: tuple[str, ...] = (),
        trailing_stop_atr_mult: float = _DEFAULT_TRAILING_ATR_MULT,
        trailing_activation_pct: float = _DEFAULT_TRAILING_ACTIVATION_PCT,
        min_breakout_body_ratio: float = _DEFAULT_MIN_BREAKOUT_BODY_RATIO,
        regime_adaptive: bool = _DEFAULT_REGIME_ADAPTIVE,
        regime_min_score: float = _DEFAULT_REGIME_MIN_SCORE,
        reject_if_regime_unknown: bool = _DEFAULT_REJECT_IF_REGIME_UNKNOWN,
        adaptive_enabled: bool = False,
        adaptive_extend_atr_mult: float = 1.5,
        adaptive_tighten_atr_mult: float = 0.7,
        # Stop/target initiaux dimensionnes a la volatilite, au SPAWN (distinct du
        # trailing et de la couche adaptive qui gerent un trade DEJA ouvert).
        # 0.0 = OFF = comportement legacy (stop = invalidation du pattern).
        spawn_atr_stop_mult: float = 0.0,   # ex 2.0 : stop elargi a >= 2x ATR (anti stops serres)
        spawn_atr_target_rr: float = 0.0,   # ex 2.0 : target = entry +/- 2x distance_stop
    ) -> None:
        self._arm_prox = arm_proximity_pct
        self._expiry_bars = expiry_bars
        self._dedupe_tol = dedupe_price_tol_pct
        self._confluence = confluence_scorer
        self._min_conf = float(min_confluence_score)
        self._min_rr = float(min_rr_ratio)
        self._reject_counter = bool(reject_trend_counter)
        self._require_volume = bool(require_volume_expansion)
        self._reject_volume_weak = bool(require_volume_weak_reject)
        self._be_trigger = float(breakeven_trigger_pct)
        self._breakout_buf = float(breakout_buffer_pct)
        self._excluded = tuple(excluded_patterns)
        self._trail_atr_mult = float(trailing_stop_atr_mult)
        self._trail_activate = float(trailing_activation_pct)
        self._regime_adaptive = bool(regime_adaptive)
        self._regime_min_score = float(regime_min_score)
        self._reject_if_regime_unknown = bool(reject_if_regime_unknown)
        self._market_regime = None  # MarketRegime | None, mis a jour par set_market_regime()
        self._min_breakout_body = float(min_breakout_body_ratio)
        # Couche adaptative (gestion dynamique du trade par features continues).
        self._adaptive_enabled = bool(adaptive_enabled)
        self._adaptive_extend_mult = float(adaptive_extend_atr_mult)
        self._adaptive_tighten_mult = float(adaptive_tighten_atr_mult)
        self._adaptive_arrays = None  # rempli par step() pour la duree de l'appel
        self._spawn_atr_stop_mult = float(spawn_atr_stop_mult)
        self._spawn_atr_target_rr = float(spawn_atr_target_rr)

    def set_market_regime(self, regime) -> None:
        """Hook appele par le scanner avec le regime detecte courant."""
        self._market_regime = regime

    def step(
        self,
        ohlcv: pd.DataFrame,
        new_patterns: list[ChartPatternDTO],
        existing: list[HypothesisDTO],
    ) -> EngineStepResult:
        if len(ohlcv) == 0:
            return EngineStepResult([], list(existing), [])

        last = ohlcv.iloc[-1]
        now_ts = _to_dt(last["timestamp"])
        bar_open = float(last["open"])
        bar_close = float(last["close"])
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        last_idx = len(ohlcv) - 1

        # Calcul ATR pour le trailing stop dynamique
        atr_val = _compute_atr(ohlcv, period=14)

        # Body ratio de la bougie : |close - open| / (high - low)
        # Utile pour confirmer un breakout (body fort = momentum reel, pas wick)
        bar_range = bar_high - bar_low
        bar_body_ratio = abs(bar_close - bar_open) / bar_range if bar_range > 0 else 0.0

        # Couche adaptative : prepare les arrays (passe uniquement) pour cette passe.
        if self._adaptive_enabled:
            self._adaptive_arrays = (
                ohlcv["high"].to_numpy(dtype=float),
                ohlcv["low"].to_numpy(dtype=float),
                ohlcv["close"].to_numpy(dtype=float),
                ohlcv["volume"].to_numpy(dtype=float) if "volume" in ohlcv.columns
                else None,
                ohlcv["open"].to_numpy(dtype=float),
            )
        else:
            self._adaptive_arrays = None

        active = [h for h in existing if not h.is_terminal]
        terminal = [h for h in existing if h.is_terminal]
        transitions: list[tuple[str, StateTransition]] = []

        # 1. Advance state machine for each active hypothesis
        progressed: list[HypothesisDTO] = []
        for h in active:
            new_h, h_trans = self._advance(
                h,
                bar_close=bar_close,
                bar_high=bar_high,
                bar_low=bar_low,
                now_ts=now_ts,
                last_idx=last_idx,
                atr_val=atr_val,
                bar_body_ratio=bar_body_ratio,
            )
            progressed.append(new_h)
            transitions.extend((new_h.id, t) for t in h_trans)

        # 2. Spawn new hypotheses from patterns not already tracked, then évalue
        # immédiatement contre la bougie courante (proximité, breakout déjà fait, etc.)
        spawned: list[HypothesisDTO] = []
        for p in new_patterns:
            if p.breakout_direction == BreakoutDirection.UNDETERMINED:
                # Pattern symétrique : on attend la cassure pour déterminer le sens.
                # Sera traité par une variante directionnelle au moment du breakout réel.
                continue
            if self._matches_existing(p, progressed + terminal):
                continue
            h_created = self._spawn_from_pattern(p, now_ts=now_ts, ohlcv=ohlcv)
            # Filtres qualite : on rejette les setups trop faibles avant persistance.
            if not self._passes_quality_filters(h_created):
                continue
            transitions.append((
                h_created.id,
                StateTransition(
                    from_state=HypothesisState.FORMING,
                    to_state=HypothesisState.FORMING,
                    timestamp=now_ts,
                    price=bar_close,
                    reason="hypothesis created from pattern",
                ),
            ))
            h_evaluated, h_trans = self._advance(
                h_created,
                bar_close=bar_close,
                bar_high=bar_high,
                bar_low=bar_low,
                now_ts=now_ts,
                last_idx=last_idx,
                atr_val=atr_val,
                bar_body_ratio=bar_body_ratio,
            )
            spawned.append(h_evaluated)
            transitions.extend((h_evaluated.id, t) for t in h_trans)

        return EngineStepResult(
            created=spawned,
            updated=progressed,
            transitions=transitions,
        )

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def _advance(
        self,
        h: HypothesisDTO,
        *,
        bar_close: float,
        bar_high: float,
        bar_low: float,
        now_ts: datetime,
        last_idx: int,
        atr_val: float = 0.0,
        bar_body_ratio: float = 1.0,
    ) -> tuple[HypothesisDTO, list[StateTransition]]:
        transitions: list[StateTransition] = []
        current = h
        was_triggered_at_entry = (current.state == HypothesisState.TRIGGERED)

        if current.state == HypothesisState.FORMING:
            new_state, reason = self._eval_forming(current, bar_close, bar_body_ratio)
            if new_state is not None:
                t = self._make_transition(current.state, new_state, now_ts, bar_close, reason)
                transitions.append(t)
                current = self._apply_transition(current, new_state, now_ts, bar_close, t)

        if current.state == HypothesisState.ARMED:
            # Couche adaptative : annuler l'ordre en attente si l'information
            # independante est franchement adverse AVANT le declenchement.
            if self._adaptive_enabled:
                feats = self._adaptive_features(current, bar_close)
                if feats is not None:
                    from app.strategy.adaptive import decide_pending
                    if decide_pending(feats):
                        t = self._make_transition(
                            current.state, HypothesisState.INVALIDATED, now_ts, bar_close,
                            "adaptive: ordre annule (conviction adverse avant trigger)",
                        )
                        transitions.append(t)
                        current = self._apply_transition(
                            current, HypothesisState.INVALIDATED, now_ts, bar_close, t
                        )
            if current.state == HypothesisState.ARMED:
                new_state, reason = self._eval_armed(current, bar_close, bar_body_ratio)
                if new_state is not None:
                    t = self._make_transition(current.state, new_state, now_ts, bar_close, reason)
                    transitions.append(t)
                    current = self._apply_transition(current, new_state, now_ts, bar_close, t)

        if current.state == HypothesisState.TRIGGERED:
            # IMPORTANT: ne pas appliquer le BE sur la meme bougie que le trigger
            # (sinon on bouge le SL au-dessus du low de cette bougie, stop immediat).
            # Le BE s'applique uniquement aux bougies suivantes.
            if was_triggered_at_entry:
                # 1) BE classique : SL passe a entry quand X% du target atteint
                be_updated = self._maybe_breakeven_trail(
                    current, bar_high=bar_high, bar_low=bar_low, bar_close=bar_close,
                )
                if be_updated is not None:
                    t = self._make_transition(
                        current.state, current.state, now_ts, bar_close,
                        "stop loss moved to entry (breakeven)",
                    )
                    transitions.append(t)
                    current = replace(be_updated, transitions=current.transitions + (t,), updated_at=now_ts)
                # 2) Trailing stop ATR : monte le SL avec le prix pour faire courir les winners
                trail_updated = self._maybe_atr_trailing(
                    current, bar_high=bar_high, bar_low=bar_low, bar_close=bar_close,
                    atr_val=atr_val,
                )
                if trail_updated is not None:
                    t = self._make_transition(
                        current.state, current.state, now_ts, bar_close,
                        f"trailing stop ATR @ {trail_updated.invalidation_price:.4f}",
                    )
                    transitions.append(t)
                    current = replace(trail_updated, transitions=current.transitions + (t,), updated_at=now_ts)

                # 3) Couche adaptative : laisser courir (EXTEND), verrouiller (TIGHTEN)
                #    ou couper tot (EXIT_NOW) selon la conviction des features.
                if self._adaptive_enabled:
                    adapted = self._apply_adaptive_open(
                        current, bar_close=bar_close, atr_val=atr_val
                    )
                    if adapted is not current and (
                        adapted.target_price != current.target_price
                        or adapted.invalidation_price != current.invalidation_price
                    ):
                        t = self._make_transition(
                            current.state, current.state, now_ts, bar_close,
                            "adaptive: gestion dynamique "
                            f"(tgt={adapted.target_price:.4f} sl={adapted.invalidation_price:.4f})",
                        )
                        transitions.append(t)
                        current = replace(
                            adapted, transitions=current.transitions + (t,), updated_at=now_ts
                        )

            new_state, exit_price, reason = self._eval_triggered(
                current, bar_high=bar_high, bar_low=bar_low
            )
            if new_state is not None:
                t = self._make_transition(current.state, new_state, now_ts, exit_price, reason)
                transitions.append(t)
                current = self._apply_transition(current, new_state, now_ts, exit_price, t)

        # Expiry seulement si encore non terminal et pas TRIGGERED
        if current.state in (HypothesisState.FORMING, HypothesisState.ARMED):
            bars_since = last_idx - current.pattern.end_index
            if bars_since >= current.expiry_bars:
                t = self._make_transition(
                    current.state,
                    HypothesisState.EXPIRED,
                    now_ts,
                    bar_close,
                    f"expired after {bars_since} bars without trigger",
                )
                transitions.append(t)
                current = self._apply_transition(
                    current, HypothesisState.EXPIRED, now_ts, bar_close, t
                )

        return current, transitions

    def _eval_forming(
        self, h: HypothesisDTO, close: float, body_ratio: float = 1.0
    ) -> tuple[HypothesisState | None, str]:
        if _invalidation_hit(h, close):
            return HypothesisState.INVALIDATED, "invalidation level hit before arming"
        if _breakout_confirmed(h, close, self._breakout_buf, body_ratio, self._min_breakout_body):
            # FIX: re-verifier le filtre regime au moment du trigger.
            # Une hypothese creee dans un regime favorable peut devenir counter-regime
            # entre temps. On INVALIDATE plutot que de TRIGGER un trade contre-tendance.
            ok, reason = self._still_passes_regime_filter(h)
            if not ok:
                return HypothesisState.INVALIDATED, f"regime-block at trigger: {reason}"
            return HypothesisState.TRIGGERED, (
                f"breakout fired from forming (body_ratio={body_ratio:.2f})"
            )
        if _within_arm_zone(h, close, self._arm_prox):
            return HypothesisState.ARMED, "close within arm proximity of breakout"
        return None, ""

    def _eval_armed(
        self, h: HypothesisDTO, close: float, body_ratio: float = 1.0
    ) -> tuple[HypothesisState | None, str]:
        if _invalidation_hit(h, close):
            return HypothesisState.INVALIDATED, "invalidation level hit before trigger (order cancelled)"
        if _breakout_confirmed(h, close, self._breakout_buf, body_ratio, self._min_breakout_body):
            # FIX: re-verifier le filtre regime au moment du trigger.
            # Bloque les hypotheses ARMED bullish dans un BEAR confirme (et vice-versa).
            ok, reason = self._still_passes_regime_filter(h)
            if not ok:
                return HypothesisState.INVALIDATED, f"regime-block at trigger: {reason}"
            return HypothesisState.TRIGGERED, (
                f"breakout confirmed (body_ratio={body_ratio:.2f} >= {self._min_breakout_body:.2f})"
            )
        return None, ""

    def _still_passes_regime_filter(self, h: HypothesisDTO) -> tuple[bool, str]:
        """Verifie que le pattern passe encore le filtre regime au moment du trigger.

        Returns:
            (True, "")  -> trade peut prendre place
            (False, reason)  -> trade bloque, hypothese a invalider
        """
        if not self._regime_adaptive or self._market_regime is None:
            return True, ""
        from app.services.market_regime import pattern_regime_score
        score = pattern_regime_score(h.pattern.kind.value, self._market_regime)
        if score < self._regime_min_score:
            return False, (
                f"{h.pattern.kind.value} score={score:.2f} "
                f"< min={self._regime_min_score:.2f} "
                f"in regime {self._market_regime.trend} (strength={self._market_regime.strength:.2f})"
            )
        return True, ""

    def _eval_triggered(
        self, h: HypothesisDTO, *, bar_high: float, bar_low: float
    ) -> tuple[HypothesisState | None, float, str]:
        if h.side == Side.LONG:
            if bar_low <= h.invalidation_price:
                return HypothesisState.STOPPED, h.invalidation_price, "stop hit (low)"
            if bar_high >= h.target_price:
                return HypothesisState.TARGET_HIT, h.target_price, "target hit (high)"
        else:
            if bar_high >= h.invalidation_price:
                return HypothesisState.STOPPED, h.invalidation_price, "stop hit (high)"
            if bar_low <= h.target_price:
                return HypothesisState.TARGET_HIT, h.target_price, "target hit (low)"
        return None, 0.0, ""

    def _maybe_atr_trailing(
        self, h: HypothesisDTO, *, bar_high: float, bar_low: float,
        bar_close: float, atr_val: float,
    ) -> HypothesisDTO | None:
        """Trailing stop dynamique base sur ATR. Maximise les winners en suivant la
        tendance : le SL monte avec le prix (jamais en arriere).

        Activation : trade au-dela de trail_activate_pct (defaut 50%) du target.
        Distance : bar_low - mult * ATR (LONG) / bar_high + mult * ATR (SHORT).

        Combinable avec BE classique : si BE deja active (be_locked), on continue
        a monter le SL via trailing.
        """
        if self._trail_atr_mult <= 0.0 or atr_val <= 0.0:
            return None
        if h.triggered_price is None:
            return None
        target_dist = abs(h.target_price - h.triggered_price)
        if target_dist <= 0:
            return None

        # Active seulement apres trail_activate_pct du chemin vers target
        if h.side == Side.LONG:
            progress = (bar_close - h.triggered_price) / target_dist
            if progress < self._trail_activate:
                return None
            # Trail SL : suivre bar_low avec un buffer ATR
            candidate_sl = bar_low - self._trail_atr_mult * atr_val
            # SL ne descend jamais ; doit etre > entry pour locker du profit
            if candidate_sl > h.invalidation_price:
                tags = h.confluence_tags
                if "trail_active" not in tags:
                    tags = tags + ("trail_active",)
                return replace(h, invalidation_price=candidate_sl, confluence_tags=tags)
        else:  # SHORT
            progress = (h.triggered_price - bar_close) / target_dist
            if progress < self._trail_activate:
                return None
            candidate_sl = bar_high + self._trail_atr_mult * atr_val
            if candidate_sl < h.invalidation_price:
                tags = h.confluence_tags
                if "trail_active" not in tags:
                    tags = tags + ("trail_active",)
                return replace(h, invalidation_price=candidate_sl, confluence_tags=tags)
        return None

    def _maybe_breakeven_trail(
        self, h: HypothesisDTO, *, bar_high: float, bar_low: float,
        bar_close: float | None = None,
    ) -> HypothesisDTO | None:
        """Si activé, remonte le SL au breakeven (= entry) une fois ``be_trigger_pct``
        du chemin vers le target atteint. Une fois fait, le tag "be_locked" indique
        qu'on ne peut plus subir de perte sèche.

        IMPORTANT : on calcule le progress sur le CLOSE et non sur le high/low pour
        eviter qu'un wick declenche le BE prematurement (causant BE-puis-stop
        sur la meme bougie). Le close = mouvement confirme.
        """
        if self._be_trigger <= 0.0 or self._be_trigger >= 1.0:
            return None
        if h.triggered_price is None:
            return None
        # Si deja au breakeven, rien a faire.
        if "be_locked" in h.confluence_tags:
            return None
        target_dist = abs(h.target_price - h.triggered_price)
        if target_dist <= 0:
            return None
        # Reference price : close si dispo (mouvement confirme), sinon high/low.
        if h.side == Side.LONG:
            ref = bar_close if bar_close is not None else bar_high
            progress = (ref - h.triggered_price) / target_dist
            if progress >= self._be_trigger and h.invalidation_price < h.triggered_price:
                return replace(
                    h,
                    invalidation_price=h.triggered_price,
                    confluence_tags=h.confluence_tags + ("be_locked",),
                )
        else:
            ref = bar_close if bar_close is not None else bar_low
            progress = (h.triggered_price - ref) / target_dist
            if progress >= self._be_trigger and h.invalidation_price > h.triggered_price:
                return replace(
                    h,
                    invalidation_price=h.triggered_price,
                    confluence_tags=h.confluence_tags + ("be_locked",),
                )
        return None

    def _passes_quality_filters(self, h: HypothesisDTO) -> bool:
        # Exclusion par type de pattern (issue de l'analyse stat)
        if self._excluded and h.pattern.kind.value in self._excluded:
            return False
        if self._min_conf > 0.0 and h.confluence_score < self._min_conf:
            return False
        if self._reject_counter and "trend_counter" in h.confluence_tags:
            return False
        if self._reject_volume_weak and "volume_weak" in h.confluence_tags:
            return False
        if self._require_volume and "volume_expansion" not in h.confluence_tags:
            return False
        if self._min_rr > 0.0:
            risk = abs(h.entry_price - h.invalidation_price)
            reward = abs(h.target_price - h.entry_price)
            if risk <= 0 or (reward / risk) < self._min_rr:
                return False
        # Filtrage adaptatif au regime de marche : rejette les patterns
        # mal alignes avec le regime courant.
        if self._regime_adaptive:
            if self._market_regime is None:
                # Fail-closed : si on ne connait pas le regime, on rejette les
                # patterns hautement directionnels pour eviter le scenario du
                # 28/05 (336 CUP_AND_HANDLE LONG en marche BEAR).
                if (
                    self._reject_if_regime_unknown
                    and h.pattern.kind.value in _HIGHLY_DIRECTIONAL_PATTERNS
                ):
                    return False
            else:
                from app.services.market_regime import pattern_regime_score
                score = pattern_regime_score(h.pattern.kind.value, self._market_regime)
                if score < self._regime_min_score:
                    return False
        return True

    # ------------------------------------------------------------------
    # Couche adaptative (gestion dynamique par features continues)
    # ------------------------------------------------------------------
    def _adaptive_features(self, h: HypothesisDTO, bar_close: float):
        """Calcule les features adaptatives pour l'hypothese ``h`` (ou None)."""
        if not self._adaptive_enabled or self._adaptive_arrays is None:
            return None
        highs, lows, closes, volumes, opens = self._adaptive_arrays
        if volumes is None or len(closes) < 6:
            return None
        from app.strategy.adaptive import compute_trade_features

        entry = h.triggered_price if h.triggered_price is not None else h.entry_price
        direction = 1 if h.side == Side.LONG else -1
        # running_extreme approxime par le close courant -> mfe_progress = progres actuel.
        return compute_trade_features(
            highs, lows, closes, volumes, opens,
            direction=direction, entry=float(entry), target=float(h.target_price),
            running_extreme=float(bar_close),
        )

    def _apply_adaptive_open(
        self, h: HypothesisDTO, *, bar_close: float, atr_val: float
    ) -> HypothesisDTO:
        """Applique la decision adaptative a une position TRIGGERED (renvoie un DTO modifie).

        - EXIT_NOW      : exit immediat au close (cible/SL ramene au close selon le P&L)
        - EXTEND_TARGET : repousse la cible de k*ATR (laisser courir)
        - TIGHTEN_STOP  : rapproche le SL du close (verrouille le profit, jamais relacher)
        """
        from app.strategy.adaptive import ManageAction, decide_open

        feats = self._adaptive_features(h, bar_close)
        if feats is None:
            return h
        action = decide_open(feats)
        is_long = h.side == Side.LONG
        d = 1.0 if is_long else -1.0

        if action == ManageAction.EXIT_NOW:
            tag = "adaptive_exit"
            entry = h.triggered_price if h.triggered_price is not None else h.entry_price
            favorable = d * (bar_close - entry) > 0
            tags = h.confluence_tags + (tag,) if tag not in h.confluence_tags else h.confluence_tags
            # Favorable -> on ramene la cible au close (sortie TARGET_HIT) ; sinon le SL (STOPPED).
            if favorable:
                return replace(h, target_price=bar_close, confluence_tags=tags)
            return replace(h, invalidation_price=bar_close, confluence_tags=tags)

        if action == ManageAction.EXTEND_TARGET and atr_val > 0:
            new_target = h.target_price + d * self._adaptive_extend_mult * atr_val
            # N'etend jamais vers l'interieur (uniquement plus loin).
            if (is_long and new_target > h.target_price) or (not is_long and new_target < h.target_price):
                tags = h.confluence_tags + ("adaptive_extend",) if "adaptive_extend" not in h.confluence_tags else h.confluence_tags
                return replace(h, target_price=new_target, confluence_tags=tags)

        if action == ManageAction.TIGHTEN_STOP and atr_val > 0:
            candidate = bar_close - d * self._adaptive_tighten_mult * atr_val
            # Resserre uniquement (SL ne relache jamais).
            if is_long and candidate > h.invalidation_price and candidate < bar_close:
                tags = h.confluence_tags + ("adaptive_tighten",) if "adaptive_tighten" not in h.confluence_tags else h.confluence_tags
                return replace(h, invalidation_price=candidate, confluence_tags=tags)
            if not is_long and candidate < h.invalidation_price and candidate > bar_close:
                tags = h.confluence_tags + ("adaptive_tighten",) if "adaptive_tighten" not in h.confluence_tags else h.confluence_tags
                return replace(h, invalidation_price=candidate, confluence_tags=tags)

        return h

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _spawn_from_pattern(
        self, p: ChartPatternDTO, *, now_ts: datetime, ohlcv: pd.DataFrame
    ) -> HypothesisDTO:
        side = Side.LONG if p.breakout_direction == BreakoutDirection.UP else Side.SHORT
        target = p.target if p.target is not None else (
            p.breakout_level + p.height if side == Side.LONG else p.breakout_level - p.height
        )
        tags: tuple[str, ...] = ()
        score = 0.0
        if self._confluence is not None:
            score, tags = self._confluence.score(p, ohlcv, market_regime=self._market_regime)
        # Stop/target initiaux ajustes a la volatilite (opt-in) : elargit le stop a
        # >= k×ATR (corrige les stops trop serres, coef #1 du modele) et fixe le
        # target a un RR constant. OFF par defaut (spawn_atr_stop_mult=0).
        entry_price = p.breakout_level
        invalidation_price = p.invalidation_level
        if self._spawn_atr_stop_mult > 0.0:
            from app.ml.risk import atr_trade_plan
            plan = atr_trade_plan(
                side=side.value,
                entry=entry_price,
                raw_invalidation=invalidation_price,
                atr=_compute_atr(ohlcv, period=14),
                k_stop=self._spawn_atr_stop_mult,
                rr_target=(self._spawn_atr_target_rr or 2.0),
            )
            invalidation_price = plan.stop
            if self._spawn_atr_target_rr > 0.0:
                target = plan.target
        return HypothesisDTO(
            id=str(uuid.uuid4()),
            pattern=p,
            symbol=p.symbol,
            timeframe=p.timeframe,
            side=side,
            entry_price=entry_price,
            target_price=target,
            invalidation_price=invalidation_price,
            state=HypothesisState.FORMING,
            created_at=now_ts,
            updated_at=now_ts,
            arm_proximity_pct=self._arm_prox,
            expiry_bars=self._expiry_bars,
            confluence_score=score,
            confluence_tags=tags,
        )

    def _matches_existing(self, p: ChartPatternDTO, existing: list[HypothesisDTO]) -> bool:
        for h in existing:
            if h.symbol != p.symbol or h.timeframe != p.timeframe:
                continue
            if h.pattern.kind != p.kind:
                continue
            if is_terminal(h.state):
                continue
            if _close_enough(h.entry_price, p.breakout_level, self._dedupe_tol):
                return True
        return False

    @staticmethod
    def _make_transition(
        from_s: HypothesisState,
        to_s: HypothesisState,
        ts: datetime,
        price: float,
        reason: str,
    ) -> StateTransition:
        return StateTransition(
            from_state=from_s, to_state=to_s, timestamp=ts, price=price, reason=reason
        )

    @staticmethod
    def _apply_transition(
        h: HypothesisDTO,
        new_state: HypothesisState,
        ts: datetime,
        price: float,
        t: StateTransition,
    ) -> HypothesisDTO:
        updates: dict = {
            "state": new_state,
            "updated_at": ts,
            "transitions": h.transitions + (t,),
        }
        if new_state == HypothesisState.TRIGGERED:
            updates["triggered_at"] = ts
            updates["triggered_price"] = price
        if new_state in (
            HypothesisState.TARGET_HIT,
            HypothesisState.STOPPED,
            HypothesisState.INVALIDATED,
            HypothesisState.EXPIRED,
        ):
            updates["closed_at"] = ts
            updates["outcome_price"] = price
        return replace(h, **updates)


# ----------------------------------------------------------------------
# Confluence scorer (optional injection — basic implementation here)
# ----------------------------------------------------------------------

class ConfluenceScorer:
    """Score 0-1 et tags de confluence pour une hypothèse.

    Quatre piliers (v3 — apres analyse live 24h qui a montre que SMA50 local
    sur 15m capte du bruit, et que trend_aligned correlait NEGATIVEMENT avec
    le winrate sur ces 24h en BEAR) :

    1. **Geometry confidence** — qualité du fit (deja calcule par le detecteur)
    2. **Volume expansion** — ratio volume bougie / moy 20 bars
    3. **HTF trend (SMA50 local)** — fallback quand le regime n'est pas fourni
    4. **Market regime affinity** — pattern_regime_score si MarketRegime fourni.
       Quand regime fourni, sa pondération domine le trend local (qui devient bruit).
    """

    def __init__(
        self,
        *,
        volume_strong_ratio: float = 1.5,
        volume_weak_ratio: float = 0.7,
        trend_sma_period: int = 50,
        trend_slope_min: float = 0.0008,   # 0.08% par bar pour qualifier un trend
        weight_geometry: float = 0.30,
        weight_volume: float = 0.30,
        weight_trend_local: float = 0.10,  # SMA50 local : faible (bruit sur LTF)
        weight_regime: float = 0.30,       # nouveau : affinite pattern x regime macro
    ) -> None:
        self._vol_strong = volume_strong_ratio
        self._vol_weak = volume_weak_ratio
        self._sma_n = trend_sma_period
        self._slope_min = trend_slope_min
        self._w_geo = weight_geometry
        self._w_vol = weight_volume
        self._w_trend = weight_trend_local
        self._w_regime = weight_regime

    def score(
        self,
        pattern: ChartPatternDTO,
        ohlcv: pd.DataFrame,
        market_regime=None,
    ) -> tuple[float, tuple[str, ...]]:
        tags: list[str] = []
        geo = float(pattern.confidence)

        vol_score, vol_tag = self._volume_component(ohlcv)
        if vol_tag:
            tags.append(vol_tag)

        trend_score, trend_tag = self._trend_component(ohlcv, pattern.breakout_direction)
        if trend_tag:
            tags.append(trend_tag)

        # Composante regime : si fournie, c'est le signal le plus discriminant
        # (utilise BTC 1h plutot que la SMA50 locale, qui est trop bruitee sur 15m).
        if market_regime is not None:
            from app.services.market_regime import pattern_regime_score
            raw = pattern_regime_score(pattern.kind.value, market_regime)
            # Borne entre 0 et 1 : raw va de 0.3 (anti) a 1.3 (favorable),
            # on normalise via (raw - 0.3) / 1.0 = (raw - 0.3) clip [0,1]
            regime_score = max(0.0, min(1.0, raw - 0.3))
            if raw >= 1.1:
                tags.append("regime_favorable")
            elif raw <= 0.5:
                tags.append("regime_adverse")
        else:
            regime_score = 0.5  # neutre : ni boost ni penalize

        if pattern.breakout_direction != BreakoutDirection.UNDETERMINED:
            tags.append("directional_bias")

        composite = (
            self._w_geo * geo
            + self._w_vol * vol_score
            + self._w_trend * trend_score
            + self._w_regime * regime_score
        )
        return round(min(1.0, max(0.0, composite)), 3), tuple(tags)

    def _volume_component(self, ohlcv: pd.DataFrame) -> tuple[float, str]:
        if "volume" not in ohlcv.columns or len(ohlcv) < 21:
            return 0.5, ""
        recent = ohlcv["volume"].iloc[-21:-1].mean()
        last = ohlcv["volume"].iloc[-1]
        if recent <= 0:
            return 0.5, ""
        ratio = float(last / recent)
        if ratio >= self._vol_strong:
            return 1.0, "volume_expansion"
        if ratio <= self._vol_weak:
            return 0.2, "volume_weak"
        # Interpolation linéaire dans la zone "normale"
        score = 0.3 + 0.5 * (ratio - self._vol_weak) / (self._vol_strong - self._vol_weak)
        return float(max(0.3, min(0.8, score))), ""

    def _trend_component(
        self, ohlcv: pd.DataFrame, direction: BreakoutDirection
    ) -> tuple[float, str]:
        if "close" not in ohlcv.columns or len(ohlcv) < self._sma_n + 2:
            return 0.5, ""
        sma = ohlcv["close"].rolling(self._sma_n).mean()
        if len(sma.dropna()) < 2:
            return 0.5, ""
        s_now = float(sma.iloc[-1])
        s_prev = float(sma.iloc[-2])
        if s_prev <= 0:
            return 0.5, ""
        # Pente relative a la SMA precedente (coherent avec market_regime.sma50_slope).
        # Avant : division par le prix spot (ref) = non standard et incoherent.
        slope_pct = (s_now - s_prev) / s_prev
        if abs(slope_pct) < self._slope_min:
            return 0.5, "trend_flat"
        is_up_trend = slope_pct > 0
        if direction == BreakoutDirection.UP and is_up_trend:
            return 1.0, "trend_aligned"
        if direction == BreakoutDirection.DOWN and not is_up_trend:
            return 1.0, "trend_aligned"
        if direction == BreakoutDirection.UNDETERMINED:
            return 0.6, ""
        return 0.2, "trend_counter"


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def _invalidation_hit(h: HypothesisDTO, close: float) -> bool:
    if h.side == Side.LONG:
        return close <= h.invalidation_price
    return close >= h.invalidation_price


def _breakout_hit(h: HypothesisDTO, close: float) -> bool:
    if h.side == Side.LONG:
        return close > h.entry_price
    return close < h.entry_price


def _breakout_confirmed(
    h: HypothesisDTO, close: float, buffer_pct: float,
    body_ratio: float = 1.0, min_body_ratio: float = 0.0,
) -> bool:
    """Breakout confirme si :
    1) la close est au-dela du breakout level (+ buffer_pct optionnel)
    2) le body de la bougie est >= min_body_ratio (filtre wicks)

    Pour LONG : close >= entry × (1+buffer)
    Pour SHORT : close <= entry × (1-buffer)

    body_ratio = |close - open| / (high - low). Une bougie de pure tendance
    a body_ratio ~ 1, un doji ~ 0.
    """
    # 1. Position de la close
    if h.side == Side.LONG:
        threshold = h.entry_price * (1.0 + buffer_pct)
        if close < threshold:
            return False
    else:
        threshold = h.entry_price * (1.0 - buffer_pct)
        if close > threshold:
            return False
    # 2. Body solide (anti-wick)
    if min_body_ratio > 0 and body_ratio < min_body_ratio:
        return False
    return True


def _within_arm_zone(h: HypothesisDTO, close: float, prox_pct: float) -> bool:
    if h.entry_price <= 0:
        return False
    return abs(close - h.entry_price) / h.entry_price <= prox_pct


def _close_enough(a: float, b: float, tol_pct: float) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / a <= tol_pct


def _to_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.Timestamp(value).to_pydatetime()


def _compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
    """ATR simple sur les ``period`` dernieres bougies (True Range moyenne).

    Renvoie 0.0 si pas assez de donnees ou si volatilite nulle.
    """
    if len(ohlcv) < period + 1:
        return 0.0
    try:
        high = ohlcv["high"].iloc[-(period + 1):].to_numpy()
        low = ohlcv["low"].iloc[-(period + 1):].to_numpy()
        close = ohlcv["close"].iloc[-(period + 1):].to_numpy()
        prev_close = close[:-1]
        h_l = high[1:] - low[1:]
        h_pc = abs(high[1:] - prev_close)
        l_pc = abs(low[1:] - prev_close)
        tr = pd.Series([max(a, b, c) for a, b, c in zip(h_l, h_pc, l_pc)])
        return float(tr.mean())
    except Exception:
        return 0.0
