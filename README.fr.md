[🇬🇧 English](README.md) | [🇫🇷 Français](README.fr.md)

# Analyseur Crypto

Un **analyseur de structure de marché crypto** multi-timeframes pour Bitget,
écrit en Python, avec un moteur de paper trading, un générateur de setups
déterministe, un dashboard FastAPI, et un bot Telegram pour les alertes en
direct et le contrôle à distance. Outillage smart-money-concepts (détection
BOS / CHOCH / FVG / order blocks) + patterns chartistes classiques, le tout
basé sur des règles et couvert par des tests unitaires.

Le projet se situe entre le « bot de trading » et le « laboratoire de
recherche » : les setups sont produits par des règles transparentes (pas de
modèle opaque), chaque backtest est reproductible, et la méthodologie est
documentée dans [docs/RESEARCH_NOTES.fr.md](docs/RESEARCH_NOTES.fr.md) afin
qu'un lecteur puisse distinguer ce qui est étayé par des preuves et ce qui
n'est encore qu'une hypothèse.

> Outil de recherche et d'éducation. Pas un conseil financier. Lire
> [DISCLAIMER.fr.md](DISCLAIMER.fr.md).

---

## Ce qu'il fait

- **Détection de structure de marché.** Swings (pivots fractaux), clustering
  support / résistance, BOS, CHOCH, FVG, order blocks, divergences RSI,
  canaux, biseaux, drapeaux, rectangles, triangles, et patterns de
  retournement — tous déterministes, tous testés unitairement sur des
  datasets synthétiques.
- **Génération de setups.** Scoring de confluence basé sur des règles avec
  des poids explicites. Pas de boîte noire.
- **Moteur de paper trading.** Exécution paper en direct plus un backtest de
  replay historique avec tests de parité broker, killswitch, limites de
  drawdown.
- **API REST.** 17 routeurs sous FastAPI, docs OpenAPI à `/docs`.
- **Dashboards.** Deux vues HTML : un dashboard marché et un dashboard
  patterns. Templates Jinja simples, sans framework JS.
- **Bot Telegram.** Commandes : `/start`, `/help`, `/perf`, `/open`,
  `/trades`, `/scan`, `/dashboard`, `/exec_status`, `/emergency_stop`,
  `/reset_killswitch`. Claviers inline pour les callbacks.
- **Stockage.** SQLAlchemy 2 async, SQLite pour le dev, PostgreSQL pour la
  prod. Migrations Alembic.
- **206 tests unitaires.** Couvrant la structure de marché, le moteur de
  backtest, les détecteurs de patterns, le cycle de vie des hypothèses, le
  killswitch de sécurité, et le stub du pipeline ML.

## Ce qu'il ne fait PAS

- Il ne trade pas automatiquement avec votre argent par défaut. Le mode live
  est opt-in et requiert des credentials API Bitget et une configuration
  explicite.
- Il ne prédit pas les prix. Les patterns détectés sont descriptifs — les
  notes de recherche sont explicites sur ceux qui ont une valeur prédictive
  mesurable et ceux qui ont été réfutés sur holdout.
- Aucun modèle ML n'est actuellement utilisé dans le scoring en production.
  Il existe un writer de feature snapshots et un ranker placeholder
  (`app.ml`), mais le pipeline live est basé sur des règles.

---

## Démarrage rapide

```bash
git clone https://github.com/alexch03/analyseur-crypto
cd analyseur-crypto

python -m venv .venv
.venv/Scripts/activate           # Windows
# source .venv/bin/activate      # Linux/macOS
pip install -e ".[dev]"

cp .env.example .env             # puis éditer .env
alembic upgrade head             # construit le schéma SQLite

uvicorn src.app.main:app --reload
# -> http://localhost:8000
# -> http://localhost:8000/docs
```

Le bot Telegram tourne séparément :

```bash
python -m src.app.tg_bot.bot
```

(Windows : `start_telegram.bat` fait la même chose.)

## Tests

```bash
pytest
```

Les 206 tests passent en environ 10 secondes. Ils ne touchent pas le réseau —
les appels exchange et Telegram sont mockés.

## Configuration

Tout est dans `.env`. Principaux paramètres extraits de `.env.example` :

| Variable | Description |
|---|---|
| `DATABASE_URL` | SQLite (défaut) ou `postgresql+asyncpg://...` pour la prod |
| `EXCHANGE_ID` | `binance` ou `bitget` via ccxt |
| `SYMBOLS` | watchlist séparée par virgules, ex. `BTC/USDT,ETH/USDT,SOL/USDT` |
| `TIMEFRAMES` | `5m,15m,1h,4h,1d` |
| `SCAN_INTERVAL_SECONDS` | fréquence de poll du scanner live |
| `TELEGRAM_BOT_TOKEN` | depuis @BotFather (optionnel) |
| `TELEGRAM_ADMIN_CHAT_ID` | votre chat id |
| `EXECUTION_MODE` | `paper` (défaut), `dry`, ou `live` |
| `API_KEY` | header pour les endpoints admin FastAPI |

## Architecture

```
src/app/
  api/                17 routeurs FastAPI (backtest, candles, chart, control,
                       dashboard, health, hypotheses, admin, analytics,
                       execution, ingestion, regime, settings, scan,
                       scanner_ops, signals, unit_paper)
  market_structure/   swings, BOS, CHOCH, FVG, order blocks, RSI,
                       support/résistance, divergences (7 modules)
  paper/              moteur paper live + backtest de replay historique
  strategy/           générateur de setups basé sur des règles + scoring de confluence
  execution/          client Bitget (ccxt async)
  tg_bot/             bot Telegram, notifier, handlers de commandes
  db/                 modèles SQLAlchemy, migrations via Alembic
  web/                templates Jinja (dashboard + patterns)
  ml/                 writer de feature snapshots + ranker placeholder
```

Les modules communiquent via des interfaces `Protocol` typées et des DTOs
dataclass figées, donc chaque couche est mockable dans les tests.

## Roadmap

Voir [docs/RESEARCH_NOTES.fr.md](docs/RESEARCH_NOTES.fr.md) pour le plan
expérimental pré-enregistré : quelles hypothèses sont ouvertes, lesquelles
ont été réfutées sur holdout, et comment le holdout est verrouillé.

## Licence

MIT — voir [LICENSE](LICENSE).

## Avertissement

Recherche et usage éducatif uniquement. Trader des dérivés est risqué et ne
convient pas à tout le monde. Lire [DISCLAIMER.fr.md](DISCLAIMER.fr.md) avant
de s'approcher du mode live.
