#!/bin/bash
# Payload: exécute 1 calcul ADVANTG -> MCNP pour un .i
set -eE -o pipefail

# --- Protéger le source contre -u ---
set +u
export PYTHONPATH="${PYTHONPATH:-}"
source /soft_snc/advantg/3.2.1/advantg.rc
set -u

# ---- Threads (pris sur l'allocation Slurm) ----
NTHREADS="${SLURM_CPUS_PER_TASK:-4}"

# ---- Entrée ----
if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/input.i" >&2
  exit 2
fi
filename="$1"

# ---- Extraction des sections ADVANTG / MCNP ----
getsection() {
  local infile=$1
  if grep -q DEBUT_ADVANTG "$infile" && grep -q FIN_ADVANTG "$infile"; then
    csplit -s -z "$infile" /DEBUT_ADVANTG/+1 '/FIN_ADVANTG/' '{*}'
    if [ -f xx01 ]; then
      sed 's/\r$//' xx01 > temp_adv
      sed -i '/./,$!d' temp_adv
      mv temp_adv input_adv.adv
      rm -f xx01
    fi
    if [ -f xx02 ]; then
      sed '/FIN_ADVANTG/d' xx02 > temp_mcnp
      sed -i 's/\r$//' temp_mcnp
      sed -i '/./,$!d' temp_mcnp
      mv temp_mcnp i.mcnp
      rm -f xx02
    fi
    rm -f xx00 2>/dev/null || true
  else
    echo "✖ Le fichier ne contient pas DEBUT_ADVANTG/FIN_ADVANTG." >&2
    exit 1
  fi
}
getsection "$filename"

# ============================ PHASE 1 : ADVANTG (MCNP5 interne) ============================
# Utilise le xsdir "classique" sous xdata (contient ENDF71SaB/…, lwtr.20t, etc.)
XSDIR5_ROOT="/soft_snc/MCNP/MCNP6/MCNP_DATA/xdata"
XSDIR5_FILE="$XSDIR5_ROOT/xsdir"
[ -f "$XSDIR5_FILE" ] || { echo "✖ xsdir introuvable: $XSDIR5_FILE"; exit 2; }

# ADVANTG (et MCNP5) liront $DATAPATH/xsdir avec des chemins relatifs à $DATAPATH
DATAPATH="$XSDIR5_ROOT" MCNP_DATA="$XSDIR5_ROOT" XSDIR="$XSDIR5_FILE" \
  advantg --threads="${NTHREADS}" input_adv.adv

# ---- Contrôle sortie ADVANTG ----
if [ ! -f ./output/inp ] || [ ! -f ./output/wwinp ]; then
  echo "✖ ADVANTG n'a pas produit ./output/inp ou ./output/wwinp" >&2
  [ -f model/outp ] && tail -n 100 model/outp >&2 || true
  exit 3
fi

# ============================ PHASE 2 : MCNP6 final =======================================
# Préparer l'entrée MCNP finale
cp ./output/inp ./i.mcnp
cp ./output/wwinp ./

# Retirer toute carte 'message' éventuelle (prudence)
sed -i -E '/^[[:space:]]*[Mm][Ee][Ss][Ss][Aa][Gg][Ee]([[:space:]]|:)/d' i.mcnp

# Cette build MCNP6 attend xsdir_mcnp6.3 (confirmé par find)
XSDIR6_FILE="/soft_snc/MCNP/MCNPDATA/MCNP63/xsdir_mcnp6.3"
XSDIR6_ROOT="$(dirname "$XSDIR6_FILE")"
[ -f "$XSDIR6_FILE" ] || { echo "✖ xsdir_mcnp6.3 introuvable: $XSDIR6_FILE"; exit 2; }

# Insérer un bloc MESSAGE: xsdir=… en tête du deck (comme dans ton cas qui marchait)
tmpfile="$(mktemp)"
{
  printf 'MESSAGE: xsdir=%s\n\n' "$XSDIR6_FILE"
  cat i.mcnp
} > "$tmpfile"
mv "$tmpfile" i.mcnp

# Env pour MCNP6 : chemins relatifs résolus depuis le dossier de xsdir_mcnp6.3
export DATAPATH="$XSDIR6_ROOT"
export MCNP_DATA="$XSDIR6_ROOT"
export XSDIR="$XSDIR6_FILE"

# Préfixe "safe" pour les fichiers MCNP (éviter '=' et '/')
BASE_NAME="$(basename "$filename")"
BASE_NOEXT="${BASE_NAME%.*}"        # ex: SR-U-UN-G1-C1-P_lead_40cm_D10m
SAFE_BASE="$(echo "$BASE_NOEXT" | sed 's/[^A-Za-z0-9._-]/_/g')"
N_PREFIX="${SAFE_BASE}."             # ex: SR-U-UN-G1-C1-P_lead_40cm_D10m.

# Fichier de sortie MCNP avec la même base et extension .o
OUTFILE="${BASE_NOEXT}.o"            # ex: SR-U-UN-G1-C1-P_lead_40cm_D10m.o

MCNP_BIN="/soft_snc/MCNP/MCNP63/mcnp-6.3.0-Linux/bin/mcnp6"
unset MESSAGE MCNP_MESSAGE MCNP_MSG || true

"$MCNP_BIN" i=i.mcnp o="$OUTFILE" n="$N_PREFIX" tasks "${NTHREADS}"

# ← ICI : s’assurer que le fichier de sortie existe et n’est pas vide
[ -s "$OUTFILE" ] || { echo "✖ MCNP n'a pas produit $OUTFILE"; exit 4; }

# Vérif du contenu du log (erreurs MCNP)
if grep -qiE 'bad trouble|fatal error|unrecognized symbol|cannot open xsdir|missing from xsdir' "$OUTFILE"; then
  echo "✖ MCNP a signalé une erreur (voir $OUTFILE)" >&2
  exit 4
fi

exit 0
