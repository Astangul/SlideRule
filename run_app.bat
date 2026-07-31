@echo off
REM ---------------------------------------------------------------------
REM  SlideRule - lancement de l'application (Windows)
REM
REM  Double-cliquer ce fichier suffit : uv installe automatiquement
REM  Python 3.11 (si besoin) et les dependances de requirements.txt au
REM  premier lancement, puis demarre l'application Streamlit.
REM  Les lancements suivants reutilisent l'environnement mis en cache
REM  par uv (aucune reinstallation tant que requirements.txt ne change
REM  pas).
REM
REM  Prerequis : uv installe (une seule fois, dans PowerShell) :
REM      powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
REM ---------------------------------------------------------------------
chcp 65001 >nul
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERREUR] uv n'est pas installe ou pas dans le PATH.
    echo.
    echo Installer uv une fois pour toutes, dans PowerShell :
    echo    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo.
    echo Puis fermer et rouvrir cette fenetre.
    echo.
    pause
    exit /b 1
)

echo Preparation de l'environnement Python et lancement de SlideRule...
echo (le premier lancement peut prendre quelques minutes : telechargement
echo  de Python 3.11 et installation des dependances)
echo Fermer cette fenetre pour arreter l'application.
echo.

uv run --python 3.11 --with-requirements requirements.txt -- streamlit run "00_👋_SlideRule_app.py"
if errorlevel 1 (
    echo.
    echo [ERREUR] Le lancement de l'application a echoue.
    pause
    exit /b 1
)

pause
endlocal
