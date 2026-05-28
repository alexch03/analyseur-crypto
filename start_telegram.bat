@echo off
REM ============================================================
REM Bot Telegram - Analyseur Crypto
REM Lance le bot Telegram qui te permet de piloter le scanner
REM depuis ton telephone (commandes /scan, /perf, /trades, etc).
REM
REM Prerequis dans .env :
REM   TELEGRAM_BOT_TOKEN=xxx       (depuis @BotFather)
REM   TELEGRAM_ADMIN_CHAT_ID=xxx   (ton chat_id perso)
REM ============================================================

cd /d "%~dp0"
title Analyseur - Telegram Bot

echo.
echo  ============================================================
echo   ANALYSEUR CRYPTO - Bot Telegram
echo  ============================================================
echo.

if not exist ".venv\Scripts\python.exe" goto :err_venv
if not exist ".env" goto :err_env

REM Verifie token Telegram present
findstr /I /C:"TELEGRAM_BOT_TOKEN=" ".env" >nul 2>&1
if errorlevel 1 goto :err_token

REM Installe python-telegram-bot et httpx si absents
.venv\Scripts\python.exe -c "import telegram, httpx" >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Installation python-telegram-bot + httpx...
    .venv\Scripts\python.exe -m pip install python-telegram-bot httpx psutil
)

REM Lance le bot
echo [LAUNCH] Demarrage du bot Telegram...
echo Le bot est actif sur Telegram. Envoie /start dans le chat.
echo.
.venv\Scripts\python.exe -m app.tg_bot.bot

pause
exit /b 0


:err_venv
echo.
echo [ERREUR] Venv absent. Lance d'abord start.bat pour installer.
echo.
pause
exit /b 1

:err_env
echo.
echo [ERREUR] Fichier .env absent. Lance d'abord start.bat.
echo.
pause
exit /b 1

:err_token
echo.
echo [ERREUR] TELEGRAM_BOT_TOKEN absent dans .env
echo.
echo Setup :
echo   1. Ouvre @BotFather sur Telegram
echo   2. Envoie /newbot, suis les instructions
echo   3. Recupere le token
echo   4. Ajoute dans .env :
echo        TELEGRAM_BOT_TOKEN=xxxx:yyyyyy
echo        TELEGRAM_ADMIN_CHAT_ID=123456789
echo   5. Ton chat_id : envoie /start au bot une fois lance,
echo      il s'affichera dans la console.
echo.
pause
exit /b 1
