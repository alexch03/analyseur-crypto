from __future__ import annotations

from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        enable_decoding=False,
    )

    # Par defaut : SQLite local (zero install). Fichier cree au demarrage.
    # Pour repasser sur Postgres : DATABASE_URL=postgresql+asyncpg://user:pwd@host:5432/db
    database_url: str = "sqlite+aiosqlite:///./analyseur.db"
    exchange_id: str = "binance"
    # Dossier des CSV OHLCV (backtest source « fichier »), relatif au répertoire de lancement sauf chemin absolu.
    ohlcv_data_dir: str = "data/ohlcv"
    # Univers SMC historique : reste petit par défaut (analyses lourdes par paire).
    symbols: list[str] = ["BTC/USDT", "ETH/USDT"]
    timeframes: list[str] = ["5m", "15m", "1h", "4h", "1d"]
    # Univers du scanner continu de patterns chartistes (~50 paires × 15m/1h/4h).
    # Override possible via ANALYSEUR_SCAN_SYMBOLS / ANALYSEUR_SCAN_TIMEFRAMES.
    scan_symbols: list[str] = []
    scan_timeframes: list[str] = []
    scan_interval_seconds: int = 60
    # Univers : "50" (defaut), "100" (top 100 majors+alts) ou "200" (ultra-large).
    # Si scan_symbols est defini, ce parametre est ignore.
    scan_universe: str = "50"
    # Throttle entre paires (ms). Baisser pour scanner plus vite (attention rate limit).
    scan_pair_delay_ms: int = 100
    # Filtres de qualite (engine d'hypothese). Defaults = pas de filtre.
    # Active-les via .env pour rendre le bot selectif :
    #   MIN_CONFLUENCE_SCORE=0.55   # ne garde que les setups >= 0.55
    #   MIN_RR_RATIO=1.5            # exige R:R >= 1.5
    #   REJECT_TREND_COUNTER=true   # drop les trades contre la tendance HTF
    #   REQUIRE_VOLUME_EXPANSION=true  # exige tag volume_expansion
    #   BREAKEVEN_TRIGGER_PCT=0.5   # SL = entry quand prix a fait 50% vers target
    min_confluence_score: float = 0.0
    min_rr_ratio: float = 0.0
    reject_trend_counter: bool = False
    require_volume_expansion: bool = False
    breakeven_trigger_pct: float = 0.0
    api_key: str = "changeme"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    def effective_scan_symbols(self) -> list[str]:
        if self.scan_symbols:
            return list(self.scan_symbols)
        from app.universe import get_universe
        return list(get_universe(self.scan_universe))

    def effective_scan_timeframes(self) -> list[str]:
        from app.universe import DEFAULT_SCAN_TIMEFRAMES
        return list(self.scan_timeframes) if self.scan_timeframes else list(DEFAULT_SCAN_TIMEFRAMES)

    @field_validator("symbols", "timeframes", "scan_symbols", "scan_timeframes", mode="before")
    @classmethod
    def parse_csv_or_list(cls, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            # Accept both JSON list and CSV list from .env.
            v = value.strip()
            if v.startswith("[") and v.endswith("]"):
                # Let pydantic JSON parsing handle it by returning raw string.
                return [s.strip().strip('"').strip("'") for s in v.strip("[]").split(",") if s.strip()]
            return [s.strip() for s in value.split(",") if s.strip()]
        return []


settings = Settings()
