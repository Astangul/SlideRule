#!/usr/bin/env bash
#SBATCH -J adv_batch
#SBATCH -p par_IB             # ← adapte à ta partition
#SBATCH -t 23-00:00:00        # ← adapte la durée
#SBATCH --ntasks=5            # ← K = concurrence (nb de calculs simultanés)
#SBATCH --cpus-per-task=4     # ← CPUs par calcul (ADVANTG/MCNP)
#SBATCH --mem-per-cpu=4G      # ← RAM par CPU

set -eEuo pipefail
[ -n "${SLURM_JOB_ID:-}" ] || { echo "✖ Submit with: sbatch lancement_serie_from_csv.sh [generated_files.csv]"; exit 1; }

# --- CSV en entrée (colonnes attendues: path,file) ---
CSV_FILE="${1:-generated_files.csv}"
[ -f "$CSV_FILE" ] || { echo "✖ CSV introuvable: $CSV_FILE"; exit 1; }

# --- Payload par défaut : script qui exécute 1 calcul ---
# Conseil: mets un CHEMIN ABSOLU ici. Tu peux aussi surcharger via l'env: DEFAULT_PAYLOAD=/abs/chemin sbatch ...
DEFAULT_PAYLOAD="${DEFAULT_PAYLOAD:-/SCRATCH/users/herth-joh/SLIDERULE/MCNP/launch_scripts/run_advantg_mcnp.sh}"

# --- Paramètres SRUN (1 step = 1 calcul) ---
CPT_ACTUAL="${SLURM_CPUS_PER_TASK:-4}"
SRUN_BASE="srun -n1 -c${CPT_ACTUAL} --exclusive --cpu-bind=cores --kill-on-bad-exit=1"

echo "Alloc : ntasks=${SLURM_NTASKS:-?}  cpt=${SLURM_CPUS_PER_TASK:-?}"
echo "CSV   : $CSV_FILE"

# --- Lire CSV -> liste d'inputs absolus ---
CSV_FILE_ABS="$(realpath "$CSV_FILE")"
BASE_DIR="${BASE_DIR:-$PWD}"                 # ancre pour 'path' si relatif (surcharge possible: BASE_DIR=/abs/dir sbatch ...)

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
echo "Trouvé N=$N entrées (ex: ${INPUTS[0]:-rien})"

# --- Lancement en parallèle avec throttling à K = --ntasks ---
pids=()
fail=0
MAX_PAR="${SLURM_NTASKS:-1}"
running=0

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

  (
    cd "$CASE_DIR"
    {
      echo "===== $(date -Is) START host=$(hostname) cpt=${CPT_ACTUAL} inp=\"$INP\" payload=\"$PAYLOAD\" ====="
      $SRUN_BASE "$PAYLOAD" "$INP"
      rc=$?
      echo "===== $(date -Is) END rc=$rc ====="
      exit $rc
    } |& tee -a "$APP_LOG"
  ) & pids+=($!)
  running=$((running+1))

  # Ne pas dépasser MAX_PAR steps simultanés
  if [ "$running" -ge "$MAX_PAR" ]; then
    if ! wait -n; then fail=1; fi
    running=$((running-1))
  fi
done

# Attendre le reste
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
exit $fail
