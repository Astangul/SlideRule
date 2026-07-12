#!/bin/bash
#-------------------------------------------------------------
# script to launch MCNP62 (or MCNP5 or MCNPX, -> exe to be modified) 
# from /soft_snc
# on the farux server with SLURM
# user need to be in the group snc-mcnp61 
# need a RSICC licence, then ask Mariya Brovchenko
# 2021
# ------------------------------------------------------------
# script to be COPIED in your personal directory (where is INPUT.mcnp) 
# and to be adjusted to your needs (parallelisation)!
# ------------------------------------------------------------
# to be used from your personal directory:
# sbatch mcnp61.sh INPUT.mcnp
# ------------------------------------------------------------
#
#  SLURM parameters
# -----------------------------------------------------------
#SBATCH -t 1-18:00:00      # timelimit 
#SBATCH -N 1               # nombre de nœuds
#SBATCH -n 1               # nombre de tâches (=1)
#SBATCH --cpus-per-task=4  # nombre de cœurs alloués à cette tâche
#SBATCH -p seq          # partition
###SBATCH --mem-per-cpu 2500

# ---------------------------------------------
# split file for adv and mcnp calculations
# ---------------------------------------------
source /soft_snc/advantg/3.2.1/advantg.rc
export DATAPATH="/soft_snc/MCNP/MCNPDATA/MCNP63/"

function getsection()
{
    local infile=$1   # input file
    
    # V�rifier si le fichier contient les sections DEBUT_ADVANTG et FIN_ADVANTG
    if grep -q DEBUT_ADVANTG "$infile" && grep -q FIN_ADVANTG "$infile"; then
        # Diviser le fichier autour des marqueurs ADVANTG
        csplit -s -z "$infile" /DEBUT_ADVANTG/+1 '/FIN_ADVANTG/' '{*}'

        # Manipuler le fichier pour ADVANTG
        if [ -f xx01 ]; then
            # Convertir les fins de ligne Windows (CR LF) en fins de ligne Unix (LF) avant de supprimer les lignes vides
            sed 's/\r$//' xx01 > temp_adv
            # Supprimer les lignes vides au d�but du fichier temporaire
            sed -i '/./,$!d' temp_adv
            # D�placer le fichier nettoy� vers input_adv.adv
            mv temp_adv input_adv.adv
            # Supprimer le fichier xx01 apr�s l'avoir trait�
            rm xx01
        fi
        
        # Manipuler le fichier pour MCNP, en enlevant la ligne FIN_ADVANTG si elle existe
        if [ -f xx02 ]; then
            # Enlever la ligne contenant 'FIN_ADVANTG' et pr�parer le fichier MCNP
            sed '/FIN_ADVANTG/d' xx02 > temp_mcnp
            # Convertir les fins de ligne Windows (CR LF) en fins de ligne Unix (LF) avant de supprimer les lignes vides
            sed -i 's/\r$//' temp_mcnp  # Convertit CR LF en LF (pour les fichiers Windows)
            sed -i '/./,$!d' temp_mcnp  # Supprime les lignes vides au d�but
            # D�placer le fichier nettoy� vers i.mcnp
            mv temp_mcnp i.mcnp
            # Ajouter un en-t�te sp�cifique avec un saut de ligne au fichier i.mcnp
            echo -e "MESSAGE: xsdir=/soft_snc/MCNP/MCNP6/MCNP_DATA/xdata/xsdir\n" | cat - i.mcnp > temp && mv temp i.mcnp
            #echo -e "MESSAGE: xsdir=/soft_snc/lib/mcnp_data/MCNP63_DATA/MCNP_DATA/xsdir_mcnp6.3\n" | cat - i.mcnp > temp && mv temp i.mcnp
            #echo -e "MESSAGE: xsdir=/soft_snc/lib/mcnp_data/MCNP63_DATA/MCNP_DATA/xsdir_all\n" | cat - i.mcnp > temp && mv temp i.mcnp
            #echo -e "MESSAGE: xsdir=/soft_snc/lib/mcnp_data/MCNP63_DATA/MCNP_DATA/xsdir\n" | cat - i.mcnp > temp && mv temp i.mcnp
            #echo -e "MESSAGE: xsdir=/soft_snc/lib/mcnp_data/MCNP61_DATA/xdata/xsdir\n" | cat - i.mcnp > temp && mv temp i.mcnp
        fi
        
        # Nettoyage: supprimer les fichiers temporaires inutilis�s
        rm xx00 xx02 2>/dev/null  # Supprime les fichiers s'ils existent, ignore les erreurs sinon
    else
        echo "Le fichier ne contient pas les sections requises ADVANTG."
    fi
}

# Utiliser le premier argument de la ligne de commande comme entr�e
filename=$1
shortname="${filename%.*}"
extension="${filename##*.}"
shortnamedot="$shortname."


getsection "$filename"
#advantg input_adv.adv
advantg --threads=4 input_adv.adv
wait
cp ./output/inp ./i.mcnp
cp ./output/wwinp ./
/soft_snc/MCNP/MCNP63/mcnp-6.3.0-Linux/bin/mcnp6 i=i.mcnp n=$shortnamedot tasks 4 xsdir=xsdir_mcnp6.3