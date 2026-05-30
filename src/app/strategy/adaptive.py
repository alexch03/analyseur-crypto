"""Couche ADAPTATIVE : features continues independantes + gestion dynamique du trade.

Philosophie (en rupture avec les filtres binaires a l'entree) :

    Les filtres statiques (reject si score < X, exclude pattern Y) sont du curve-fitting :
    ils marchent sur le passe et cassent au changement de regime. A la place, on calcule
    des FEATURES CONTINUES et INDEPENDANTES (chacune une source d'information orthogonale)
    qui pilotent la GESTION du trade, pas un go/no-go binaire :

      - sortir plus tard d'un trade gagnant (laisser courir)  -> EXTEND_TARGET
      - maximiser les profits quand le momentum faiblit         -> TIGHTEN_STOP
      - annuler un ordre en attente qui se degrade               -> decide_pending()
      - couper tot un trade qui tourne mal (avant le SL plein)   -> EXIT_NOW

Chaque feature est dans [-1, 1] (signee dans le sens du trade : + = favorable) ou [0, 1].
Le score de conviction est une combinaison ponderee. LES POIDS SONT DES PRIORS INITIAUX,
destines a etre REMPLACES par des poids APPRIS depuis FeatureSnapshot (table existante)
une fois assez de paires (features, outcome) collectees -> voir app.ml.

Toutes les features sont calculees sur des donnees PASSEES uniquement (slice jusqu'a la
bougie courante incluse) : aucun look-ahead.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from app.patterns._indicators import compute_rsi


class ManageAction(str, Enum):
    EXTEND_TARGET = "EXTEND_TARGET"   # laisser courir : repousser la cible
    HOLD = "HOLD"                     # ne rien changer
    TIGHTEN_STOP = "TIGHTEN_STOP"     # verrouiller le profit : rapprocher le stop
    EXIT_NOW = "EXIT_NOW"             # couper : l'information independante est franchement adverse


@dataclass(frozen=True, slots=True)
class TradeFeatures:
    """Features continues independantes, signees dans le sens du trade (+ = favorable)."""
    momentum: float          # [-1,1] RSI vs 50 + pente RSI + ROC
    volume_conviction: float # [-1,1] desequilibre volume haussier/baissier
    htf_alignment: float     # [-1,1] pente SMA longue (proxy HTF) vs direction
    structure_health: float  # [-1,1] higher-lows (long) / lower-highs (short)
    volatility_state: float  # [0,1] percentile ATR (0=calme, 1=explosif) — module l'agressivite du trail
    mfe_progress: float      # [0,1] avancee vers la cible (running extreme / distance cible)

    def as_dict(self) -> dict:
        return asdict(self)


# ── Poids initiaux (PRIORS). A remplacer par des poids appris (app.ml / FeatureSnapshot). ──
_W_MOMENTUM = 0.35
_W_VOLUME = 0.20
_W_HTF = 0.30
_W_STRUCTURE = 0.15

# Seuils de decision (conviction dans [-1,1]).
# Principe : posture par defaut = HOLD (zone morte large -> le trade respire,
# les BE/trail existants gerent le risque). On n'agit que sur signaux FORTS.
# L'A/B du 29/05 a montre qu'un TIGHTEN a -0.05 se declenchait 15x et etranglait
# les gagnants (WR 56%->47%) : la zone morte doit etre large.
CANCEL_PENDING_THRESHOLD = -0.45   # ARMED : annuler l'ordre si conviction franchement adverse
EXIT_OPEN_THRESHOLD = -0.60        # TRIGGERED : couper seulement si catastrophique
EXTEND_THRESHOLD = 0.50            # pres de la cible + forte conviction : laisser courir
# TIGHTEN desactive par defaut (seuil < EXIT -> jamais atteint) : le tuning du 29/05 a
# montre que la conviction n'a PAS de pouvoir predictif fiable (htf/structure correlent
# NEGATIVEMENT en regime BEAR/RANGE), donc plafonner les gagnants via tighten degrade le
# compound (A/B : +10.5% -> +7.7%). A reactiver avec des poids conditionnels au regime.
TIGHTEN_THRESHOLD = -0.99


def _safe(x: float, default: float = 0.0) -> float:
    return default if (x is None or not math.isfinite(x)) else float(x)


def _tanh(x: float) -> float:
    return math.tanh(x)


def compute_trade_features(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    opens: np.ndarray,
    *,
    direction: int,            # +1 LONG, -1 SHORT
    entry: float,
    target: float,
    running_extreme: float | None = None,
    lookback: int = 20,
    htf_sma_period: int = 50,
) -> TradeFeatures:
    """Calcule les features sur les donnees passees (arrays se terminant a la bougie courante)."""
    n = len(closes)
    d = 1.0 if direction >= 0 else -1.0
    if n < 6:
        return TradeFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    lb = min(lookback, n - 1)

    # 1) MOMENTUM : RSI vs 50 + pente RSI + ROC, signe par la direction.
    rsi = compute_rsi(closes, period=14)
    rsi_now = _safe(rsi[-1], 50.0)
    rsi_prev = _safe(rsi[-4] if n >= 4 else rsi[-1], 50.0)
    rsi_dev = (rsi_now - 50.0) / 50.0           # [-1,1]
    rsi_slope = (rsi_now - rsi_prev) / 50.0      # variation normalisee
    roc = (closes[-1] / closes[-lb] - 1.0) if closes[-lb] > 0 else 0.0
    momentum = _tanh(d * (1.2 * rsi_dev + 1.0 * rsi_slope + 8.0 * roc))

    # 2) VOLUME CONVICTION : volume sur bougies favorables vs adverses sur la fenetre.
    vol_fav = 0.0
    vol_adv = 0.0
    for k in range(n - lb, n):
        bar_dir = 1.0 if closes[k] >= opens[k] else -1.0
        if bar_dir * d > 0:
            vol_fav += volumes[k]
        else:
            vol_adv += volumes[k]
    vol_total = vol_fav + vol_adv
    volume_conviction = ((vol_fav - vol_adv) / vol_total) if vol_total > 0 else 0.0

    # 3) HTF ALIGNMENT : pente d'une SMA longue (proxy tendance superieure), signee.
    if n >= htf_sma_period + 3:
        sma_now = float(np.mean(closes[-htf_sma_period:]))
        sma_prev = float(np.mean(closes[-htf_sma_period - 3:-3]))
        slope = (sma_now / sma_prev - 1.0) if sma_prev > 0 else 0.0
        htf_alignment = _tanh(d * slope * 50.0)
    else:
        htf_alignment = 0.0

    # 4) STRUCTURE HEALTH : pour LONG, higher-lows ; pour SHORT, lower-highs.
    half = max(2, lb // 2)
    if direction >= 0:
        recent_low = float(np.min(lows[-half:]))
        older_low = float(np.min(lows[-2 * half:-half])) if n >= 2 * half else recent_low
        structure_health = _tanh((recent_low / older_low - 1.0) * 50.0) if older_low > 0 else 0.0
    else:
        recent_high = float(np.max(highs[-half:]))
        older_high = float(np.max(highs[-2 * half:-half])) if n >= 2 * half else recent_high
        structure_health = _tanh((older_high / recent_high - 1.0) * 50.0) if recent_high > 0 else 0.0

    # 5) VOLATILITY STATE [0,1] : ATR courant vs son percentile sur la fenetre.
    trs = []
    for k in range(max(1, n - 4 * lb), n):
        tr = max(highs[k] - lows[k], abs(highs[k] - closes[k - 1]), abs(lows[k] - closes[k - 1]))
        trs.append(tr)
    if len(trs) >= 5:
        atr_now = float(np.mean(trs[-14:])) if len(trs) >= 14 else float(np.mean(trs))
        lo, hi = float(np.min(trs)), float(np.max(trs))
        volatility_state = (atr_now - lo) / (hi - lo) if hi > lo else 0.5
    else:
        volatility_state = 0.5

    # 6) MFE PROGRESS [0,1] : avancee du prix extreme vers la cible.
    if running_extreme is not None and abs(target - entry) > 1e-12:
        prog = d * (running_extreme - entry) / (d * (target - entry))
        mfe_progress = float(min(1.0, max(0.0, prog)))
    else:
        mfe_progress = 0.0

    return TradeFeatures(
        momentum=round(momentum, 4),
        volume_conviction=round(volume_conviction, 4),
        htf_alignment=round(htf_alignment, 4),
        structure_health=round(structure_health, 4),
        volatility_state=round(float(min(1.0, max(0.0, volatility_state))), 4),
        mfe_progress=round(mfe_progress, 4),
    )


def compute_evaluation_features(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    opens: np.ndarray,
    *,
    direction: int,
    entry: float,
    lookback: int = 20,
) -> dict[str, float]:
    """Large jeu de VARIABLES CANDIDATES pour evaluer la probabilite de gain.

    Objectif : sortir du 50/50 en trouvant des variables EMPIRIQUEMENT predictives.
    Ces features ne pilotent PAS encore les decisions : elles sont mesurees par
    tune_adaptive_weights.py (correlation avec l'issue) pour identifier lesquelles
    deplacent reellement la proba de gain. Les gagnantes seront promues dans le
    modele de conviction (avec poids conditionnels au regime).

    Toutes calculees sur donnees PASSEES uniquement (slice jusqu'a l'entree).
    direction = +1 LONG, -1 SHORT. Conventions de signe documentees par feature.
    """
    n = len(closes)
    d = 1.0 if direction >= 0 else -1.0
    out: dict[str, float] = {}
    if n < 25:
        return out
    lb = min(lookback, n - 1)
    close = float(closes[-1])

    rsi = compute_rsi(closes, period=14)
    rsi_now = _safe(rsi[-1], 50.0)
    # 1. rsi_level : momentum brut, signe par la direction (+ = RSI dans le sens du trade)
    out["rsi_level"] = d * (rsi_now - 50.0) / 50.0
    # 2. rsi_extreme_revert : surachat/survente CONTRE le trade -> potentiel mean-reversion
    #    (+ = survente pour un long / surachat pour un short = favorable au reversal)
    out["rsi_extreme_revert"] = d * (50.0 - rsi_now) / 50.0 if abs(rsi_now - 50.0) > 15 else 0.0

    # 3. bb_zscore : position du prix dans les bandes de Bollinger (z-score), signe -d
    #    (+ = prix etire a l'oppose du trade -> reversion favorable)
    w = closes[-lb:]
    mid = float(np.mean(w))
    sd = float(np.std(w))
    bb_z = (close - mid) / sd if sd > 1e-12 else 0.0
    out["bb_zscore_revert"] = -d * bb_z

    # 4-5. range_position : ou est le prix dans le range recent [0,1], signe pour reversion
    for win in (20, 50):
        if n > win:
            hh = float(np.max(highs[-win:]))
            ll = float(np.min(lows[-win:]))
            pos = (close - ll) / (hh - ll) if hh > ll else 0.5
            # + = achete bas / vend haut (reversion) : long favorise si pos faible
            out[f"range_pos_{win}_revert"] = d * (0.5 - pos) * 2.0

    # 6. vwap_distance : ecart au VWAP roulant, signe -d (reversion vers le VWAP)
    tp = (highs[-lb:] + lows[-lb:] + closes[-lb:]) / 3.0
    vol_w = volumes[-lb:]
    vsum = float(np.sum(vol_w))
    vwap = float(np.sum(tp * vol_w) / vsum) if vsum > 0 else close
    out["vwap_dist_revert"] = -d * ((close - vwap) / vwap if vwap > 0 else 0.0) * 50.0

    # 7. volume_spike : volume de la derniere bougie vs moyenne (z-score, non signe)
    vmean = float(np.mean(volumes[-lb:]))
    vstd = float(np.std(volumes[-lb:]))
    out["volume_spike"] = (float(volumes[-1]) - vmean) / vstd if vstd > 1e-12 else 0.0

    # 8. trend_strength : |SMA_fast - SMA_slow| / ATR (force de tendance locale, non signe)
    if n >= 52:
        smaf = float(np.mean(closes[-10:]))
        smas = float(np.mean(closes[-50:]))
        trs = [max(highs[k] - lows[k], abs(highs[k] - closes[k - 1]), abs(lows[k] - closes[k - 1]))
               for k in range(n - 14, n)]
        atr = float(np.mean(trs)) if trs else 0.0
        out["trend_strength"] = abs(smaf - smas) / atr if atr > 1e-12 else 0.0

    # 9. price_acceleration : variation de ROC (2e derivee), signee par la direction
    if n > 2 * lb:
        roc_recent = closes[-1] / closes[-lb] - 1.0 if closes[-lb] > 0 else 0.0
        roc_older = closes[-lb] / closes[-2 * lb] - 1.0 if closes[-2 * lb] > 0 else 0.0
        out["price_accel"] = d * (roc_recent - roc_older) * 50.0

    # 10. consecutive_bars : sequence de bougies dans le sens du trade (exhaustion), signe -d
    cnt = 0
    for k in range(n - 1, max(0, n - 12), -1):
        bar_dir = 1.0 if closes[k] >= opens[k] else -1.0
        if bar_dir * d > 0:
            cnt += 1
        else:
            break
    # + = beaucoup de bougies DANS le sens (exhaustion -> defavorable a la continuation)
    out["consecutive_exhaustion"] = -float(cnt) / 6.0

    # 11. entry_body_ratio : conviction de la bougie d'entree
    rng = float(highs[-1] - lows[-1])
    out["entry_body_ratio"] = abs(close - float(opens[-1])) / rng if rng > 1e-12 else 0.0

    # 12. sma200_position : prix vs SMA200, signe par la direction (regime macro)
    if n >= 200:
        sma200 = float(np.mean(closes[-200:]))
        out["sma200_pos"] = d * (close / sma200 - 1.0) * 20.0 if sma200 > 0 else 0.0

    return {k: round(float(v), 4) for k, v in out.items()}


def conviction_score(f: TradeFeatures) -> float:
    """Combinaison ponderee des features signees -> conviction globale dans [-1,1].

    + = l'information independante soutient le trade ; - = elle s'y oppose.
    (Poids = priors initiaux, a apprendre depuis les donnees.)
    """
    raw = (
        _W_MOMENTUM * f.momentum
        + _W_VOLUME * f.volume_conviction
        + _W_HTF * f.htf_alignment
        + _W_STRUCTURE * f.structure_health
    )
    return float(min(1.0, max(-1.0, raw)))


def decide_pending(f: TradeFeatures) -> bool:
    """ARMED (ordre en attente, pas encore declenche) : True = ANNULER l'ordre.

    Si l'information independante est franchement adverse AVANT l'entree, on evite
    un trade qui aurait probablement ete perdant.
    """
    return conviction_score(f) <= CANCEL_PENDING_THRESHOLD


def decide_open(f: TradeFeatures, *, near_target_frac: float = 0.8) -> ManageAction:
    """TRIGGERED (position ouverte) : decide la gestion dynamique.

    - EXIT_NOW      : conviction franchement adverse -> couper avant le SL plein
    - EXTEND_TARGET : proche de la cible + forte conviction -> laisser courir
    - TIGHTEN_STOP  : momentum qui faiblit -> verrouiller le profit
    - HOLD          : sinon
    """
    c = conviction_score(f)
    if c <= EXIT_OPEN_THRESHOLD:
        return ManageAction.EXIT_NOW
    if f.mfe_progress >= near_target_frac and c >= EXTEND_THRESHOLD:
        return ManageAction.EXTEND_TARGET
    if c <= TIGHTEN_THRESHOLD:
        return ManageAction.TIGHTEN_STOP
    return ManageAction.HOLD
