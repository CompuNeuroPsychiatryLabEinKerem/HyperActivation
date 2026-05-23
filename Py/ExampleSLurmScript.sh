#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=fmriprepCifti

. /etc/profile.d/huji-lmod.sh
MEITAR_LAB="/sci/nosnap/shahar.arzy/meitar2706"
MEITAR_MAIN=$MEITAR_LAB"/Assuta"
URI_LAB="/sci/labs/shahar.arzy/uri.elias"
#FS_LICENSE=$MEITAR_LAB"/FS_License/license.txt"
SIF_FILE=$MEITAR_LAB"/Singularity/ciftify_v1.3.2-2.3.3_081025.sif"

OUT_DIR=$URI_LAB"/Projects_Data/Assuta_Meitar/ciftify"
FS_LICENSE_DIR=$URI_LAB"/FS_License"

module load spack
module load apptainer

SUBJECTS=(${SUBJECTS})
SBJ=${SUBJECTS[${SLURM_ARRAY_TASK_ID}]}

echo "DEBUG: SBJ=[${SBJ}] TASK_ID=[${SLURM_ARRAY_TASK_ID}]"

apptainer run --cleanenv \
    -B $MEITAR_MAIN:/data \
    -B $OUT_DIR:/out \
    -B $FS_LICENSE_DIR:/fs-license \
     $SIF_FILE \
    /data/BIDS /out participant  \
    --participant_label=${SBJ} \
    --read-from-derivatives /data/derivatives \
    --fs-license="/fs-license/license.txt" \
    --anat_only \
    --debug
    # --rerun-if-incomplete

# --anat_only
# --session_label=001  \
# --read-from-derivatives /data \

# sbatch --array=0-1 Singularity/to_ciftify.sh
 
# ln -s /my/data/partB123 /my/data/partB
# SUBJECTS = AvShA AvShB DaBeA DaBeB DoGrA DoGrB DvKaA DvKaB EfScA EfScB ElOvA ElOvB HaHaA HaHaB HaLaA HaLaB HaLiA HaLiB HiArA HiArB IlHoA IlHoB ItGuA ItGuB MaPeA MaPeB OlRaA OlRaB RiBeA RiBeB SaKaA SaKaB SaKrA SaKrB ShAzA ShAzB ShBeA ShBeB SmGuA SmGuB TzDoA TzDoB UzBiA UzBiB WaTiA WaTiB YaSh YeBeA YeBeB YoBoA YoBoB YoGuA YoGuB YoMeA YoMeB
# 0-54
#--------------- USAGE: sbatch --array=0-1 --export=SUBJECTS="AvShA AvShB" Post_fMRIPrep.sh

# USAGE: sbatch --array=0-1 --export=SUBJECTS="AvShA AvShB DaBeA DaBeB DoGrA DoGrB DvKaA DvKaB EfScA EfScB ElOvA ElOvB HaHaA HaHaB HaLaA HaLaB HaLiA HaLiB HiArA HiArB IlHoA IlHoB ItGuA ItGuB MaPeA MaPeB OlRaA OlRaB RiBeA RiBeB SaKaA SaKaB SaKrA SaKrB ShAzA ShAzB ShBeA ShBeB SmGuA SmGuB TzDoA TzDoB UzBiA UzBiB WaTiA WaTiB YaSh YeBeA YeBeB YoBoA YoBoB YoGuA YoGuB YoMeA YoMeB" CiftifyMeitar.sh
