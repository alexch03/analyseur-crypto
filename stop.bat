@echo off
REM ============================================================
REM Analyseur Crypto - Arret propre
REM Ferme les 3 fenetres (Scanner, API, Telegram).
REM Ne touche PAS aux autres pythons (autres projets) - filtre
REM par titre de fenetre.
REM ============================================================

cd /d "%~dp0"
title Analyseur - Arret

echo.
echo  ============================================================
echo   ANALYSEUR CRYPTO - Arret
echo  ============================================================
echo.

set ANY_KILLED=0

REM --- Scanner ---
tasklist /FI "WINDOWTITLE eq Analyseur - Scanner*" 2>nul | find /I "cmd.exe" >nul
if not errorlevel 1 (
    echo [STOP] Fermeture du scanner...
    taskkill /FI "WINDOWTITLE eq Analyseur - Scanner*" /T /F >nul 2>&1
    set ANY_KILLED=1
) else (
    echo [INFO] Scanner deja arrete.
)

REM --- API ---
tasklist /FI "WINDOWTITLE eq Analyseur - API*" 2>nul | find /I "cmd.exe" >nul
if not errorlevel 1 (
    echo [STOP] Fermeture de l'API...
    taskkill /FI "WINDOWTITLE eq Analyseur - API*" /T /F >nul 2>&1
    set ANY_KILLED=1
) else (
    echo [INFO] API deja arretee.
)

REM --- Telegram bot ---
tasklist /FI "WINDOWTITLE eq Analyseur - Telegram*" 2>nul | find /I "cmd.exe" >nul
if not errorlevel 1 (
    echo [STOP] Fermeture du bot Telegram...
    taskkill /FI "WINDOWTITLE eq Analyseur - Telegram*" /T /F >nul 2>&1
    set ANY_KILLED=1
) else (
    echo [INFO] Telegram bot deja arrete (ou jamais lance).
)

echo.
if "%ANY_KILLED%"=="1" (
    echo  ============================================================
    echo   Tout est arrete proprement.
    echo   La DB analyseur.db conserve tes 808+ trades.
    echo   Au prochain start.bat, le bot reprend ou il s'etait arrete.
    echo  ============================================================
) else (
    echo  ============================================================
    echo   Rien a arreter (le bot n'etait pas en cours).
    echo  ============================================================
)
echo.
timeout /t 3 /nobreak >nul
exit /b 0
