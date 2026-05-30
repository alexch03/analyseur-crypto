"""Gate d'inférence live : modèle + politique -> décision prendre / rejeter.

Destiné à être branché dans ``HypothesisEngine`` au moment du trigger, en
remplacement (ou complément) des filtres statiques. Sûr par défaut : si aucun
modèle actif n'est disponible, ``MlTradeGate.try_load`` renvoie ``None`` et
l'appelant garde son comportement legacy (aucun blocage surprise).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.ml.dataset import live_feature_row
from app.ml.model import load_model
from app.ml.policy import Decision, TradePolicy


@dataclass
class MlTradeGate:
    model: object
    policy: TradePolicy

    @classmethod
    def load(cls, model_path: str | Path, policy: TradePolicy | None = None) -> "MlTradeGate":
        return cls(model=load_model(model_path), policy=policy or TradePolicy())

    @classmethod
    def try_load(cls, model_path: str | Path, policy: TradePolicy | None = None) -> "MlTradeGate | None":
        """Charge le modèle si présent, sinon None (fallback legacy côté appelant)."""
        try:
            if Path(model_path).exists():
                return cls.load(model_path, policy)
        except Exception:
            pass
        return None

    def predict(self, **kw) -> tuple[float, float | None]:
        """Retourne (P(gagnant), RR) pour un setup décrit par ses composantes brutes."""
        X = live_feature_row(**kw)
        prob = float(self.model.predict_proba(X)[0, 1])
        rr = X["rr"].iloc[0]
        if pd.isna(rr):
            risk = abs(kw["entry"] - kw["invalidation"])
            reward = abs(kw["target"] - kw["entry"])
            rr = (reward / risk) if risk > 0 else None
        else:
            rr = float(rr)
        return prob, rr

    def evaluate(self, **kw) -> Decision:
        prob, rr = self.predict(**kw)
        return self.policy.decide(prob, rr)
