"""Pipeline d'entraînement + promotion sous garde-fous = boucle d'amélioration.

Étapes : reconstruire le dataset depuis la DB -> évaluer OUT-OF-SAMPLE ->
entraîner le modèle final sur toutes les données -> persister -> ne PROMOUVOIR
actif QUE si le critère « modèle suffisant » est atteint.

C'est le coeur de « améliorer les prédictions jusqu'à un modèle suffisant » :
à relancer périodiquement quand des trades s'accumulent (cron / après chaque
journée). Le garde-fou empêche de déployer un modèle non rentable — réponse
directe au risque « le système fait n'importe quoi ».
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.ml.dataset import DATASET_PATH, DEFAULT_DB, build_dataset, load_dataset
from app.ml.evaluate import evaluate_model
from app.ml.model import save_model, train_model

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models"

# Garde-fous : refuser d'entraîner sur trop peu de données (sur-apprentissage garanti).
MIN_SAMPLES = 200
MIN_POSITIVES = 20


def run_training(
    db_path: str | Path = DEFAULT_DB,
    model_type: str = "logreg",
    n_splits: int = 5,
    promote: bool = True,
    dataset_csv: str | Path | None = None,
) -> dict:
    if dataset_csv:
        df = load_dataset(dataset_csv)
        print(f"dataset chargé depuis {dataset_csv} ({len(df)} trades)")
    else:
        df = build_dataset(db_path)
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)

    n, pos = len(df), int(df["label"].sum())
    if n < MIN_SAMPLES or pos < MIN_POSITIVES:
        print(f"[garde-fou] données insuffisantes (n={n}, gagnants={pos}) "
              f"-> entraînement annulé (min {MIN_SAMPLES}/{MIN_POSITIVES}).")
        return {"trained": False, "n": n, "positives": pos}

    res = evaluate_model(df, model_type, n_splits=n_splits)
    pipe = train_model(df, model_type)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_type}.joblib"
    save_model(pipe, model_path)

    metrics = {
        "model_type": model_type,
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "n_samples": n,
        "n_positives": pos,
        "auc": res.get("auc"),
        "brier": res.get("brier"),
        "oos_ev_pf": res["ev"]["pf"],
        "oos_ev_pnl": res["ev"]["pnl"],
        "oos_ev_n": res["ev"]["n"],
        "sufficient": res["sufficient"],
        "model_path": str(model_path),
    }
    (MODEL_DIR / f"{model_type}.metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    active_path = MODEL_DIR / "active.json"
    if promote and res["sufficient"]:
        active_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\n[promotion] modèle SUFFISANT -> promu actif: {model_path}")
    else:
        print(f"\n[promotion] modèle NON suffisant -> NON promu (actif inchangé).")
        print("            Garde-fou : on ne déploie jamais un modèle non rentable.")
    return {"trained": True, **metrics}
