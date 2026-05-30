"""Génère un dataset multi-régimes par replay sur les CSV OHLCV disponibles.

Resample le 5m en 1h (signal) + 4h (contexte HTF), rejoue les détecteurs avec
stops ATR, et produit data/ml/replay_dataset.csv — prêt pour ml_train.

Usage: .venv/Scripts/python.exe scripts/ml_replay.py [step]
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from app.ml.dataset import engineer_features  # noqa: E402
from app.ml.features import ALL_FEATURES, LABEL, META_COLUMNS  # noqa: E402
from app.ml.replay import replay_ohlcv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ml" / "replay_dataset.csv"

SOURCES = [
    ("BTC/USDT", ROOT / "data" / "dashboard_datasets" / "BTC_USDT__5m__80d.csv"),
    ("ETH/USDT", ROOT / "data" / "dashboard_datasets" / "ETH_USDT__5m__80d.csv"),
]


def _load(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return (df.set_index("timestamp").resample(rule).agg(agg).dropna().reset_index())


# (timeframe, higher-tf pour contexte, pas de balayage). Plus de TF = plus de trades.
TF_CONFIG = [("5m", "15m", 6), ("15m", "1h", 4), ("1h", "4h", 2), ("4h", None, 1)]
_RULES = {"5m": None, "15m": "15min", "1h": "1h", "4h": "4h"}


def main() -> None:
    all_rows = []
    for sym, path in SOURCES:
        if not path.exists():
            print(f"skip {sym}: {path} absent")
            continue
        base = _load(path)
        frames = {tf: (base if rule is None else _resample(base, rule))
                  for tf, rule in _RULES.items()}
        print(f"{sym}: " + ", ".join(f"{tf}={len(f)}" for tf, f in frames.items()))
        for tf, htf_name, step in TF_CONFIG:
            htf = frames.get(htf_name) if htf_name else None
            rows = replay_ohlcv(frames[tf], symbol=sym, timeframe=tf, htf=htf, step=step)
            print(f"  {sym} {tf}: {len(rows)} trades")
            all_rows.extend(rows)

    if not all_rows:
        print("Aucun trade généré.")
        return

    raw = pd.DataFrame(all_rows)
    df = engineer_features(raw)
    df[LABEL] = (raw["outcome"] == "TARGET_HIT").astype(int)
    keep = [c for c in META_COLUMNS if c in df.columns] + ALL_FEATURES + [LABEL]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(OUT, index=False)

    n, pos = len(df), int(df[LABEL].sum())
    print(f"\n{n} trades -> {OUT}")
    print(f"gagnants = {pos} ({100*pos/n:.1f}%)")
    print("\nWinrate par régime :")
    print(df.groupby('regime_trend')[LABEL].agg(['count', 'mean']).to_string())
    print("\nWinrate par pattern (top/bottom) :")
    g = df.groupby('pattern_kind')[LABEL].agg(['count', 'mean']).sort_values('mean')
    print(g.to_string())


if __name__ == "__main__":
    main()
