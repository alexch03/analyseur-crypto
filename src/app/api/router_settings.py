"""API settings : lit/modifie .env a chaud (runtime) sans redemarrer.

Endpoints :
    GET  /settings              : current config (univers, mode, demo_symbols, etc.)
    GET  /settings/universes    : liste des univers disponibles + nb symbols
    POST /settings/universe     : change SCAN_UNIVERSE
    POST /settings/demo_symbols : set whitelist DEMO_SYMBOLS
    POST /settings/execution    : change EXECUTION_MODE
    POST /settings/refresh_bitget : relance fetch_bitget_symbols.py

Modifie .env sur disque + recharge la config (executor singleton reset).
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.universe import get_universe

router = APIRouter(prefix="/settings", tags=["settings"])

ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = ROOT / ".env"


def _read_env() -> dict[str, str]:
    """Parse .env en dict (gere commentaires inline VAR=val  # comment)."""
    if not ENV_FILE.exists():
        return {}
    out: dict[str, str] = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        # Strip inline comment : "value  # comment" -> "value"
        if "#" in v and not (v.startswith('"') or v.startswith("'")):
            v = v.split("#", 1)[0].strip()
        v = v.strip('"').strip("'").strip()
        out[k.strip()] = v
    return out


def _set_env_var(key: str, value: str) -> None:
    """Met a jour .env (in-place, preserve commentaires/ordre)."""
    if not ENV_FILE.exists():
        ENV_FILE.write_text(f"{key}={value}\n", encoding="utf-8")
        os.environ[key] = value
        return
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    new_lines: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        # Conserve commentaires/lignes vides
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if "=" in stripped and stripped.split("=", 1)[0].strip() == key:
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    os.environ[key] = value


@router.get("")
async def get_settings() -> dict:
    env = _read_env()
    universe_name = env.get("SCAN_UNIVERSE", "50")
    return {
        "scan": {
            "universe": universe_name,
            "universe_size": len(get_universe(universe_name)),
            "timeframes": env.get("SCAN_TIMEFRAMES", "").split(",")
            if env.get("SCAN_TIMEFRAMES") else ["15m", "1h", "4h"],
            "interval_seconds": int(env.get("SCAN_INTERVAL_SECONDS", "60")),
            "pair_delay_ms": int(env.get("SCAN_PAIR_DELAY_MS", "100")),
        },
        "execution": {
            "mode": env.get("EXECUTION_MODE", "disabled"),
            "max_position_usd": float(env.get("MAX_POSITION_USD", "50")),
            "max_open_positions": int(env.get("MAX_OPEN_POSITIONS", "5")),
            "max_daily_loss_usd": float(env.get("MAX_DAILY_LOSS_USD", "100")),
            "max_consecutive_losses": int(env.get("MAX_CONSECUTIVE_LOSSES", "5")),
            "leverage": int(env.get("BITGET_LEVERAGE", "1")),
            "demo_symbols": [
                s.strip() for s in env.get("DEMO_SYMBOLS", "").split(",")
                if s.strip()
            ],
        },
        "exchange_id": env.get("EXCHANGE_ID", "binance"),
        "bitget_demo_configured": bool(
            env.get("BITGET_API_KEY_demo") or env.get("BITGET_DEMO_API_KEY")
        ),
        "bitget_live_configured": bool(
            env.get("BITGET_API_KEY") or env.get("BITGET_LIVE_API_KEY")
        ),
        "telegram_configured": bool(env.get("TELEGRAM_BOT_TOKEN")),
    }


@router.get("/universes")
async def list_universes() -> dict:
    """Liste les univers disponibles avec leur taille."""
    static_universes = ["50", "100", "200"]
    dynamic_universes: list[str] = []
    # Verifie si bitget files existent
    data_dir = ROOT / "data"
    if (data_dir / "bitget_symbols_spot.json").exists():
        dynamic_universes.extend([
            "bitget_spot_top50", "bitget_spot_top100", "bitget_spot_top200",
            "bitget_spot_tier12", "bitget_spot_all",
        ])
    if (data_dir / "bitget_symbols_futures.json").exists():
        dynamic_universes.extend([
            "bitget_futures_top50", "bitget_futures_top100", "bitget_futures_top200",
            "bitget_futures_tier12", "bitget_futures_all",
        ])
    all_universes = static_universes + dynamic_universes
    items = []
    for name in all_universes:
        u = get_universe(name)
        items.append({"name": name, "count": len(u), "preview": u[:3]})
    return {"items": items}


@router.post("/universe")
async def set_universe(name: str) -> dict:
    """Change SCAN_UNIVERSE dans .env (necessite redemarrage scanner)."""
    if not get_universe(name):
        raise HTTPException(400, f"Univers '{name}' inconnu ou liste vide")
    _set_env_var("SCAN_UNIVERSE", name)
    return {
        "ok": True,
        "universe": name,
        "count": len(get_universe(name)),
        "note": "Redemarre le scanner pour appliquer (stop.bat + start.bat)",
    }


@router.post("/execution")
async def set_execution_mode(mode: str) -> dict:
    """Change EXECUTION_MODE : disabled | paper | demo | live."""
    if mode not in ("disabled", "paper", "demo", "live"):
        raise HTTPException(400, "mode must be: disabled, paper, demo, live")
    _set_env_var("EXECUTION_MODE", mode)
    # Reset executor singleton pour qu'il reprenne le nouveau mode
    try:
        from app.execution.router import reset_safety
        reset_safety()
    except Exception:
        pass
    return {"ok": True, "mode": mode}


@router.post("/demo_symbols")
async def set_demo_symbols(symbols: str) -> dict:
    """CSV de symbols autorises en demo (vide = tous)."""
    _set_env_var("DEMO_SYMBOLS", symbols.strip())
    parts = [s.strip() for s in symbols.split(",") if s.strip()]
    return {"ok": True, "symbols": parts, "count": len(parts)}


@router.post("/scan_interval")
async def set_scan_interval(seconds: int) -> dict:
    if seconds < 30 or seconds > 3600:
        raise HTTPException(400, "seconds doit etre dans [30, 3600]")
    _set_env_var("SCAN_INTERVAL_SECONDS", str(seconds))
    return {"ok": True, "interval_seconds": seconds}


@router.post("/refresh_bitget")
async def refresh_bitget_symbols(market: str = "futures") -> dict:
    """Relance fetch_bitget_symbols.py pour mettre a jour la liste."""
    import subprocess
    script = ROOT / "scripts" / "fetch_bitget_symbols.py"
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        venv_py = "python"
    args = [str(venv_py), str(script)]
    if market == "futures":
        args.append("--futures")
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=60)
        return {
            "ok": r.returncode == 0,
            "stdout_tail": r.stdout.splitlines()[-10:] if r.stdout else [],
            "stderr_tail": r.stderr.splitlines()[-5:] if r.stderr else [],
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "fetch timeout")
    except Exception as e:
        raise HTTPException(500, f"fetch failed: {e}")
