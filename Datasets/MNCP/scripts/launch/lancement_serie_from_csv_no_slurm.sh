#!/usr/bin/env bash
#
# Version sans SLURM du script lancement_serie_from_csv.sh
# Usage: ./lancement_serie_from_csv_no_slurm.sh [generated_files.csv] [MAX_PARALLEL]
#

set -eEuo pipefail

# --- Paramètres ---
CSV_FILE="${1:-generated_files.csv}"
MAX_PAR="${2:-5}"          # Nombre de calculs simultanés (par défaut 5)
CPT_ACTUAL="${3:-4}"       # CPUs par calcul (pour info, non utilisé directement)

[ -f "$CSV_FILE" ] || { echo "✖ CSV introuvable: $CSV_FILE"; exit 1; }

# --- Payload par défaut : script qui exécute 1 calcul ---
DEFAULT_PAYLOAD="${DEFAULT_PAYLOAD:-/share_snc/snc/Johann/Etudes/ACC_CRIT/SlideRule/MCNP/scripts/run_advantg_mcnp.sh}"

echo "========================================="
echo "Lancement série sans SLURM"
echo "========================================="
echo "Machine   : $(hostname)"
echo "Parallèle : max ${MAX_PAR} calculs simultanés"
echo "CPUs/calc : ${CPT_ACTUAL} (info)"
echo "CSV       : $CSV_FILE"
echo "========================================="

# --- Lire CSV -> liste d'inputs absolus ---
CSV_FILE_ABS="$(realpath "$CSV_FILE")"
BASE_DIR="${BASE_DIR:-$PWD}"

mapfile -t INPUTS < <(
  CSV_FILE="$CSV_FILE_ABS" BASE_DIR="$BASE_DIR" python3 - <<'PY'
import csv, os
csv_file = os.environ["CSV_FILE"]
base_dir = os.environ.get("BASE_DIR", os.getcwd())
rows = []
with open(csv_file, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        p  = (row.get("path") or "").replace("\\", "/").strip()
        fn = (row.get("file") or "").strip()
        if not p or not fn:
            continue
        if not os.path.isabs(p):
            p = os.path.join(base_dir, p)
        rows.append(os.path.realpath(os.path.join(p, fn)))
print("\n".join(rows))
PY
)

N="${#INPUTS[@]}"
[ "$N" -gt 0 ] || { echo "✖ Aucune entrée valide lue depuis $CSV_FILE (colonnes attendues: path,file)"; exit 1; }
echo "Trouvé N=$N entrées"
echo "Première entrée : ${INPUTS[0]:-rien}"
echo "========================================="
echo ""

# --- Lancement en parallèle avec throttling manuel ---
pids=()
fail=0
running=0
completed=0

for INP in "${INPUTS[@]}"; do
  [ -n "${INP:-}" ] || continue
  CASE_DIR="$(dirname "$INP")"
  BASE="$(basename "$INP")"
  BASE_NOEXT="${BASE%.*}"

  # payload local prioritaire si présent à côté de l'input
  LOCAL_PAYLOAD="$CASE_DIR/run_advantg_mcnp.sh"
  PAYLOAD="$DEFAULT_PAYLOAD"
  [ -x "$LOCAL_PAYLOAD" ] && PAYLOAD="$LOCAL_PAYLOAD"
  [ -x "$PAYLOAD" ] || { echo "✖ Payload non exécutable: $PAYLOAD (case: $CASE_DIR)"; exit 1; }

  LOG_DIR="$CASE_DIR/logs"
  APP_LOG="$LOG_DIR/${BASE_NOEXT}.log"
  mkdir -p "$LOG_DIR"

  echo "[$(date +%H:%M:%S)] Démarrage: $BASE_NOEXT ($((completed+running+1))/$N)"

  (
    cd "$CASE_DIR"
    {
      echo "===== $(date -Is) START host=$(hostname) inp=\"$INP\" payload=\"$PAYLOAD\" ====="
      "$PAYLOAD" "$INP"
      rc=$?
      echo "===== $(date -Is) END rc=$rc ====="
      exit $rc
    } |& tee -a "$APP_LOG"
  ) & pids+=($!)
  running=$((running+1))

  # Ne pas dépasser MAX_PAR processus simultanés
  while [ "$running" -ge "$MAX_PAR" ]; do
    if wait -n; then
      completed=$((completed+1))
      echo "[$(date +%H:%M:%S)] Complété: $completed/$N (en cours: $((running-1)))"
    else
      fail=1
      completed=$((completed+1))
      echo "[$(date +%H:%M:%S)] ÉCHEC détecté: $completed/$N (en cours: $((running-1)))"
    fi
    running=$((running-1))
  done
done

# Attendre le reste
echo ""
echo "Attente des derniers processus..."
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    completed=$((completed+1))
    echo "[$(date +%H:%M:%S)] Complété: $completed/$N"
  else
    fail=1
    completed=$((completed+1))
    echo "[$(date +%H:%M:%S)] ÉCHEC détecté: $completed/$N"
  fi
done

echo ""
echo "========================================="
if [ "$fail" -eq 0 ]; then
  echo "✓ Tous les calculs ont réussi ($N/$N)"
else
  echo "✖ Certains calculs ont échoué"
fi
echo "========================================="
exit $fail
