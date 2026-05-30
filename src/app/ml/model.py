"""Modèle de prédiction P(gagnant) à partir de plusieurs variables indépendantes.

Deux modèles interchangeables :

  - ``logreg`` : régression logistique régularisée (L2). **Interprétable** : chaque
    coefficient = contribution signée d'une variable -> répond à « quelles variables
    indépendantes comptent ». Robuste sur petit échantillon.
  - ``gbm`` : HistGradientBoosting (non linéaire, interactions). Comparateur de
    puissance ; sur-apprend plus facilement sur peu de données.

Le préprocesseur (ColumnTransformer) impute, standardise les numériques et
one-hot encode les catégorielles : chaque modalité devient une variable
indépendante du modèle. C'est volontairement la MÊME spec (features.py) en
offline et en live, pour éviter tout décalage train/serve.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features import (
    ALL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL,
    NUMERIC_FEATURES,
)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extrait X (features, typées) et y (label) en coerçant proprement les dtypes."""
    X = df[ALL_FEATURES].copy()
    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    for c in BINARY_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int)
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str)
    y = df[LABEL].astype(int)
    return X, y


def build_pipeline(model_type: str = "logreg", *, C: float = 0.5,
                   min_category_freq: int = 10, random_state: int = 42) -> Pipeline:
    """Construit le pipeline préprocesseur + estimateur.

    ``min_category_freq`` regroupe les modalités rares (ex. patterns vus <10 fois)
    dans un bucket « infrequent » : anti-sur-apprentissage sur les petits effectifs.
    """
    numeric = Pipeline([
        # keep_empty_features : garde les colonnes 100% NaN (ex. indicateurs absents
        # du dataset DB) -> dimensionnalité stable entre train et serve.
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="NA")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=min_category_freq)),
    ])
    binary = SimpleImputer(strategy="constant", fill_value=0)

    pre = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("bin", binary, BINARY_FEATURES),
    ], remainder="drop", verbose_feature_names_out=True)

    if model_type == "logreg":
        clf = LogisticRegression(
            class_weight="balanced", C=C, max_iter=5000, random_state=random_state,
        )
    elif model_type == "gbm":
        clf = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
            min_samples_leaf=20, class_weight="balanced", random_state=random_state,
        )
    else:
        raise ValueError(f"model_type inconnu: {model_type}")

    return Pipeline([("pre", pre), ("clf", clf)])


def feature_names(pipe: Pipeline) -> list[str]:
    return list(pipe.named_steps["pre"].get_feature_names_out())


def logreg_coefficients(pipe: Pipeline) -> pd.DataFrame:
    """Coefficients signés de la régression logistique, triés par |coef|.

    coef > 0 : la variable augmente P(gagnant). coef < 0 : la diminue.
    (Numériques standardisées -> coefficients comparables entre eux.)
    """
    names = feature_names(pipe)
    coef = np.asarray(pipe.named_steps["clf"].coef_).ravel()
    out = pd.DataFrame({"feature": names, "coef": coef})
    out["abs"] = out["coef"].abs()
    return out.sort_values("abs", ascending=False).reset_index(drop=True)


def train_model(df: pd.DataFrame, model_type: str = "logreg", **kw) -> Pipeline:
    X, y = split_xy(df)
    pipe = build_pipeline(model_type, **kw)
    pipe.fit(X, y)
    return pipe


def save_model(pipe: Pipeline, path) -> None:
    joblib.dump(pipe, path)


def load_model(path) -> Pipeline:
    return joblib.load(path)
