#!/usr/bin/env bash
# ---------------------------------------------------------------------
# SlideRule - lancement de l'application (Linux/macOS)
#
# Executer ce script suffit : uv installe automatiquement Python 3.11
# (si besoin) et les dependances de requirements.txt au premier
# lancement, puis demarre l'application Streamlit. Les lancements
# suivants reutilisent l'environnement mis en cache par uv (aucune
# reinstallation tant que requirements.txt ne change pas).
#
# Prerequis : uv installe (une seule fois) :
#     curl -LsSf https://astral.sh/uv/install.sh | sh
# ---------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo ""
    echo "[ERREUR] uv n'est pas installe ou pas dans le PATH."
    echo ""
    echo "Installer uv une fois pour toutes :"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Puis fermer et rouvrir ce terminal (ou 'source ~/.bashrc')."
    echo ""
    exit 1
fi

echo "Preparation de l'environnement Python et lancement de SlideRule..."
echo "(le premier lancement peut prendre quelques minutes : telechargement"
echo " de Python 3.11 et installation des dependances)"
echo "Fermer ce terminal (Ctrl+C) pour arreter l'application."
echo ""

uv run --python 3.11 --with-requirements requirements.txt -- streamlit run "00_👋_SlideRule_app.py"
