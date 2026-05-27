# Analyseur Crypto

Multi-timeframe crypto analysis platform — market structure detection, deterministic setup generation, paper trading, chart API and Telegram alerts.

## Prerequisites

- **Python 3.12+**
- **PostgreSQL 16+** running locally or via Docker
- A `.env` file (copy `.env.example` and fill in your credentials)

## Installation

```bash
# Create a virtual environment and install dependencies
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -e ".[dev]"
```

## Database setup

```bash
# Run Alembic migrations
alembic upgrade head
```

## Running the server

```bash
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`. OpenAPI docs at `/docs`.

## Running tests

```bash
pytest -v
```

## Architecture

The platform is split into seven independent services communicating through typed Protocol interfaces and frozen dataclass DTOs.

| # | Service | Package | Role |
|---|---------|---------|------|
| 1 | Data Ingestion | `app.ingestion` | Fetch OHLCV candles via ccxt, normalize, store in PostgreSQL |
| 2 | Market Structure Analysis | `app.market_structure` | Swing detection, support/resistance clustering, BOS/CHOCH |
| 3 | Strategy Engine | `app.strategy` | Deterministic rule-based setup generation with confidence scoring |
| 4 | Paper Trading | `app.paper` | Live paper execution and historical replay backtest |
| 5 | Chart Rendering | `app.chart` | JSON DTO assembling OHLCV + overlays for frontend consumption |
| 6 | Telegram | `app.telegram` | Format and deliver trade signals to Telegram |
| 7 | ML Ranking (stub) | `app.ml` | Feature snapshot writer + no-op ranker placeholder |

## Algorithmic conventions (v1)

All detection algorithms follow strict, deterministic definitions documented in docstrings and validated by synthetic test datasets. See `src/app/market_structure/` for implementations:

- **Swings**: fractal pivot detection with configurable `left`/`right` window
- **Support/Resistance**: price clustering of swings within an ATR-based epsilon
- **BOS**: break of structure confirmed by close beyond last major swing
- **CHOCH**: change of character when trend sequence is first violated

## Roadmap

- FVG, Order Blocks, OTE Fibonacci, divergences
- Paper trading replay backtest engine
- Telegram signal delivery
- ML feature extraction and setup ranking
