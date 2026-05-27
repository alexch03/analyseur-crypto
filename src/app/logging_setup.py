"""Configuration de logging centralisee.

- Console : niveau INFO+
- Fichier : logs/scanner.log (rotation 5 MB x 3) niveau DEBUG+
- Tracebacks complets sur les erreurs

A appeler tot au demarrage (worker.py, main.py FastAPI lifespan).
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(log_dir: str | Path = "logs", level_console: int = logging.INFO) -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "scanner.log"

    root = logging.getLogger()
    # Evite la duplication si appele plusieurs fois
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(logging.DEBUG)

    fmt_short = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt_long = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(process)d] %(name)s.%(funcName)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level_console)
    console.setFormatter(fmt_short)
    root.addHandler(console)

    file_h = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt_long)
    root.addHandler(file_h)

    # Modules trop bavards : on calme.
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    root.info("Logging initialise: console=%s, file=%s",
              logging.getLevelName(level_console), log_file)
