"""Protocol interfaces pour les détecteurs de patterns."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from app.schemas.domain import SwingPoint
from app.schemas.patterns import ChartPatternDTO


class PatternDetector(Protocol):
    """Détecte une famille de patterns sur une fenêtre OHLCV.

    L'implémentation reçoit l'OHLCV complet et la liste de swings déjà détectés
    (déjà filtrés par confirmation). Elle retourne les patterns détectés
    actuellement valides — i.e. non encore cassés ni mitigés —, avec leur
    fenêtre d'origine, target projeté et niveau d'invalidation.
    """

    def detect(
        self,
        ohlcv: pd.DataFrame,
        swings: list[SwingPoint],
        *,
        symbol: str,
        timeframe: str,
    ) -> list[ChartPatternDTO]: ...
