"""CLI : entraîne + évalue OOS + persiste le modèle ML de sélection de trades.

Usage:
    .venv/Scripts/python.exe scripts/ml_train.py [logreg|gbm] [db_path]

À relancer périodiquement (ex. quotidien) au fur et à mesure que des trades
s'accumulent — c'est la boucle d'amélioration. Le modèle n'est promu actif que
s'il passe le critère de rentabilité out-of-sample.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from app.ml.train_pipeline import run_training  # noqa: E402

if __name__ == "__main__":
    # Args positionnels : [model_type] [source]. Si source finit en .csv -> dataset_csv,
    # sinon c'est un chemin de DB. Ex: ml_train.py logreg data/ml/replay_dataset.csv
    args = sys.argv[1:]
    model_type = args[0] if args else "logreg"
    kwargs: dict = {"model_type": model_type}
    if len(args) > 1:
        src = args[1]
        kwargs["dataset_csv" if src.endswith(".csv") else "db_path"] = src
    run_training(**kwargs)
