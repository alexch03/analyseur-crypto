"""Politique de décision : P(gagnant) + R:R  ->  prendre / rejeter + sizing.

Remplace les seuils statiques (min_confluence, filtres régime binaires) par une
règle d'espérance mathématique fondée sur *plusieurs variables* (via la proba du
modèle) ET le profil risque/rendement propre au trade :

    EV (en R) = p · RR − (1 − p)          (gagner RR·R avec proba p, risquer 1R)
    => prendre ssi EV > 0  ⟺  p > 1 / (1 + RR)

Le sizing utilise un Kelly fractionnaire (prudent) : f* = (p·(b+1) − 1) / b, b=RR,
borné par ``max_size`` et atténué par ``kelly_fraction``. Sizing ∝ edge plutôt
qu'inclusion/exclusion binaire.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Decision:
    take: bool
    size: float       # fraction d'unité de risque (0..max_size) ; 0 si rejet
    prob: float       # P(gagnant) prédite
    ev_r: float       # espérance en multiples de R
    reason: str


@dataclass(frozen=True, slots=True)
class TradePolicy:
    """Politique configurable. Tous les seuils sont explicites et testables."""
    min_prob: float = 0.0        # plancher absolu sur P(gagnant)
    ev_margin_r: float = 0.0     # EV minimale requise (en R) au-dessus de 0
    min_rr: float = 0.3          # rejette les R:R dégénérés
    kelly_fraction: float = 0.25 # Kelly fractionnaire (1.0 = Kelly plein, agressif)
    max_size: float = 1.0

    def decide(self, prob: float, rr: float | None) -> Decision:
        if rr is None or rr <= 0 or not (0.0 <= prob <= 1.0):
            return Decision(False, 0.0, float(prob), 0.0, "entrées invalides")
        ev_r = prob * rr - (1.0 - prob)
        if prob < self.min_prob:
            return Decision(False, 0.0, prob, ev_r, f"prob {prob:.2f} < min {self.min_prob:.2f}")
        if rr < self.min_rr:
            return Decision(False, 0.0, prob, ev_r, f"RR {rr:.2f} < min {self.min_rr:.2f}")
        if ev_r <= self.ev_margin_r:
            return Decision(False, 0.0, prob, ev_r, f"EV {ev_r:+.3f}R <= marge {self.ev_margin_r:+.3f}")
        kelly = (prob * (rr + 1.0) - 1.0) / rr        # >0 garanti car EV>0
        size = max(0.0, min(self.max_size, self.kelly_fraction * kelly))
        return Decision(True, size, prob, ev_r, f"EV {ev_r:+.3f}R, size {size:.2f}")
