#!/bin/bash
#SBATCH --job-name=ciftifyBold
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

. /etc/profile.d/huji-lmod.sh

export PATH=$PATH:/sci/labs/shahar.arzy/uri.elias/Installs/connectomewb/workbench/bin_linux64
export LD_PRELOAD="/lib/x86_64-linux-gnu/libfreetype.so.6 /lib/x86_64-linux-gnu/libfontconfig.so.1"

source /sci/labs/shahar.arzy/uri.elias/PyVirtualNew/bin/activate
python Ciftify_Meitar_DATA.py --array-index $SLURM_ARRAY_TASK_ID

# Number of subjects: 55
# Usage: sbatch --array=0-54 Ciftify_Meitar_DATA.sh


