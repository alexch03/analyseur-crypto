@echo off
REM ============================================================
REM Analyseur Crypto - Arret propre
REM Double-clique pour fermer le scanner + l'API.
REM ============================================================

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

REM --- Filet de securite : tue tout uvicorn/python orphelin lance depuis ce dossier ---
REM (commente par defaut pour ne pas tuer d'autres projets Python en cours)
REM taskkill /F /IM uvicorn.exe >nul 2>&1

echo.
if "%ANY_KILLED%"=="1" (
    echo  ============================================================
    echo   Tout est arrete. Les hypotheses sont sauvegardees dans
    echo   analyseur.db et reprendront au prochain start.bat.
    echo  ============================================================
) else (
    echo  ============================================================
    echo   Rien a arreter (le bot n'etait pas en cours).
    echo  ============================================================
)
echo.
timeout /t 3 /nobreak >nul
exit /b 0
