@echo off
REM ============================================================
REM Analyseur Crypto - Lanceur tout-en-un (Windows, SQLite local)
REM Double-clique. Premier lancement = installation. Ensuite : direct.
REM Aucun Postgres requis. Une base "analyseur.db" est creee au demarrage.
REM ============================================================

cd /d "%~dp0"
title Analyseur - Lanceur

echo.
echo  ============================================================
echo   ANALYSEUR CRYPTO - Demarrage (mode SQLite local)
echo  ============================================================
echo.

REM --- 1. Python ---
where python >nul 2>&1
if errorlevel 1 goto :err_python

REM --- 2. .env minimal (cree si absent ou si vieille config Postgres) ---
if not exist ".env" goto :write_env
REM Detecte une vieille config pointant sur Postgres et la remplace.
findstr /I /C:"postgresql" ".env" >nul 2>&1
if errorlevel 1 goto :after_env
echo [SETUP] Ancien .env Postgres detecte - sauvegarde dans .env.postgres.bak
copy /Y ".env" ".env.postgres.bak" >nul
:write_env
echo [SETUP] Creation du .env par defaut (SQLite local)...
> .env echo DATABASE_URL=sqlite+aiosqlite:///./analyseur.db
>> .env echo EXCHANGE_ID=binance
>> .env echo API_KEY=changeme
>> .env echo SCAN_INTERVAL_SECONDS=60
:after_env

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
echo [INSTALL] Installation des dependances (2-3 min la premiere fois)...
python -m pip install --upgrade pip
pip install -e ".[dev]"
if errorlevel 1 goto :err_deps
:after_deps

REM --- 5b. S'assurer que aiosqlite est present (cas mise a jour) ---
python -c "import aiosqlite" >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Ajout d'aiosqlite...
    pip install aiosqlite
)

REM --- 6. Creer / reparer les tables SQLite ---
echo [DB] Verification et creation des tables (SQLite)...
echo       (repare automatiquement le schema si necessaire)
python scripts\init_db.py
if errorlevel 1 goto :err_db

REM --- 7. Scanner dans une nouvelle fenetre ---
echo [LAUNCH] Demarrage du scanner continu (50 cryptos x 15m/1h/4h)...
start "Analyseur - Scanner" cmd /k "call .venv\Scripts\activate.bat && python -m app.worker --scan-daemon -v"

REM --- 8. API dans une nouvelle fenetre ---
echo [LAUNCH] Demarrage de l'API FastAPI...
start "Analyseur - API" cmd /k "call .venv\Scripts\activate.bat && uvicorn app.main:app --host 127.0.0.1 --port 8000"

REM --- 9. Attente puis navigateur ---
echo [WAIT] Attente du demarrage API (5s)...
timeout /t 5 /nobreak >nul
echo [OPEN] Ouverture du dashboard...
start "" "http://127.0.0.1:8000/patterns"

echo.
echo  ============================================================
echo   TOUT EST LANCE
echo  ============================================================
echo   Scanner   : fenetre "Analyseur - Scanner"
echo   API       : fenetre "Analyseur - API"
echo   Dashboard : http://127.0.0.1:8000/patterns
echo   Base      : %CD%\analyseur.db
echo.
echo   Scripts utiles (ouvre une invite .venv activee) :
echo     Backtest 14j + optimisation :
echo       python scripts\run_loop.py --days 14
echo     Backtest rapide (4 symboles, 7j) :
echo       python scripts\run_loop.py --days 7 --quick
echo     Analyse MFE/MAE trades :
echo       python scripts\analyze_trades.py
echo     Tests de sante :
echo       python scripts\test_suite.py --no-network
echo.
echo   Bot Telegram (mobile) :
echo     1. Configure TELEGRAM_BOT_TOKEN + TELEGRAM_ADMIN_CHAT_ID dans .env
echo     2. Double-clic start_telegram.bat
echo     Le bot expose : /perf /open /trades /scan /backfill /patterns
echo.
echo   Pour arreter : ferme les deux fenetres.
echo  ============================================================
echo.
pause
exit /b 0


REM ============================================================
REM Erreurs
REM ============================================================

:err_python
echo.
echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
echo Installe Python 3.12+ depuis https://www.python.org/downloads/
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
echo [ERREUR] Echec activation venv. Supprime le dossier .venv et relance.
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
echo [ERREUR] Echec creation/reparation des tables SQLite.
echo Le script init_db.py tente de reparer automatiquement le schema.
echo Si l'erreur persiste, supprime analyseur.db et relance :
echo   del analyseur.db
echo.
pause
exit /b 1
