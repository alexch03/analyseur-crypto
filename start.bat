@echo off
REM ============================================================
REM Analyseur Crypto - Lanceur (Windows, SQLite local)
REM Double-clique. Lance le scanner + l'API en arriere plan.
REM Preserve ton .env existant (y compris cles Bitget, Telegram).
REM ============================================================

cd /d "%~dp0"
title Analyseur - Lanceur

echo.
echo  ============================================================
echo   ANALYSEUR CRYPTO - Demarrage
echo  ============================================================
echo.

REM --- 0. Verifie qu'aucune instance ne tourne deja ---
tasklist /FI "WINDOWTITLE eq Analyseur - Scanner*" 2>nul | find /I "cmd.exe" >nul
if not errorlevel 1 (
    echo [INFO] Scanner deja en cours.
    echo Si tu veux relancer proprement : double-clic stop.bat d'abord.
    pause
    exit /b 0
)

REM --- 1. Python ---
where python >nul 2>&1
if errorlevel 1 goto :err_python

REM --- 2. .env : on PRESERVE l'existant (cles Bitget, Telegram, etc.) ---
REM Cree un .env minimal SEULEMENT si absent.
if exist ".env" goto :after_env_check
echo [SETUP] Aucun .env detecte. Creation d'un .env minimal...
> .env echo DATABASE_URL=sqlite+aiosqlite:///./analyseur.db
>> .env echo EXCHANGE_ID=binance
>> .env echo API_KEY=changeme
>> .env echo SCAN_UNIVERSE=50
>> .env echo SCAN_INTERVAL_SECONDS=60
>> .env echo EXECUTION_MODE=disabled
echo [INFO] Edite .env pour ajouter tes cles Bitget/Telegram puis relance.
:after_env_check
REM Verifie l'ancien format Postgres et l'arrete
findstr /I /C:"postgresql" ".env" >nul 2>&1
if not errorlevel 1 (
    echo [WARN] Ancien .env Postgres detecte. Backup vers .env.postgres.bak
    copy /Y ".env" ".env.postgres.bak" >nul
    echo [WARN] Edite .env pour le passer en SQLite : DATABASE_URL=sqlite+aiosqlite:///./analyseur.db
    pause
    exit /b 1
)

REM --- 3. venv ---
if exist ".venv\Scripts\python.exe" goto :after_venv
echo [INSTALL] Creation de l'environnement virtuel...
python -m venv .venv
if errorlevel 1 goto :err_venv
:after_venv

REM --- 4. Activer venv ---
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :err_activate

REM --- 5. Dependances ---
if exist ".venv\Lib\site-packages\fastapi" goto :after_deps
echo [INSTALL] Installation des dependances (2-3 min premiere fois)...
python -m pip install --upgrade pip
pip install -e ".[dev]"
if errorlevel 1 goto :err_deps
:after_deps

REM --- 5b. Verifier aiosqlite ---
python -c "import aiosqlite" >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Ajout d'aiosqlite...
    pip install aiosqlite
)

REM --- 6. DB : init/repair sans rien casser ---
echo [DB] Verification du schema...
python scripts\init_db.py
if errorlevel 1 goto :err_db

REM --- 7. Scanner (24/7 boucle continue) ---
echo [LAUNCH] Demarrage scanner continu (lit SCAN_UNIVERSE depuis .env)...
start "Analyseur - Scanner" cmd /k "call .venv\Scripts\activate.bat && python -m app.worker --scan-daemon"

REM --- 8. API + Dashboard ---
echo [LAUNCH] Demarrage API + Dashboard...
start "Analyseur - API" cmd /k "call .venv\Scripts\activate.bat && uvicorn app.main:app --host 127.0.0.1 --port 8000"

REM --- 9. Attente puis navigateur ---
echo [WAIT] Attente du demarrage API (6s)...
timeout /t 6 /nobreak >nul
echo [OPEN] Ouverture du dashboard...
start "" "http://127.0.0.1:8000/patterns"

REM --- 10. Telegram bot (optionnel - si configure) ---
findstr /I /C:"TELEGRAM_BOT_TOKEN=" ".env" | findstr /V /C:"TELEGRAM_BOT_TOKEN=$" | findstr /V /C:"TELEGRAM_BOT_TOKEN= " >nul 2>&1
if errorlevel 1 goto :skip_telegram
echo [LAUNCH] Telegram bot detecte - demarrage...
start "Analyseur - Telegram" cmd /k "call .venv\Scripts\activate.bat && python -m app.tg_bot.bot"
:skip_telegram

echo.
echo  ============================================================
echo   TOUT EST LANCE - 24/7 jusqu'a stop.bat
echo  ============================================================
echo   Scanner   : fenetre "Analyseur - Scanner" (scan continu)
echo   API       : fenetre "Analyseur - API"
echo   Telegram  : fenetre "Analyseur - Telegram" (si configure)
echo   Dashboard : http://127.0.0.1:8000/patterns
echo.
echo   Le scanner tourne en BOUCLE INFINIE (cycle ~3 min).
echo   Pas besoin de cliquer "Scan immediat" - il scanne tout seul.
echo.
echo   Pour ARRETER tout : double-clic stop.bat
echo  ============================================================
echo.
timeout /t 5 /nobreak >nul
exit /b 0


REM ============================================================
REM Erreurs
REM ============================================================

:err_python
echo.
echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
echo Installe Python 3.11+ depuis https://www.python.org/downloads/
echo (Coche "Add Python to PATH" pendant l'installation.)
echo.
pause
exit /b 1

:err_venv
echo.
echo [ERREUR] Echec creation venv. Verifie les droits d'ecriture.
echo.
pause
exit /b 1

:err_activate
echo.
echo [ERREUR] Echec activation venv. Supprime .venv\ et relance.
echo.
pause
exit /b 1

:err_deps
echo.
echo [ERREUR] Echec installation des dependances.
echo Verifie ta connexion internet et relance.
echo.
pause
exit /b 1

:err_db
echo.
echo [ERREUR] Echec init DB.
echo Si le probleme persiste, supprime analyseur.db et relance.
echo.
pause
exit /b 1
