#!/bin/sh
#
# Based on the simple "Hello World" submit script for Slurm.
# Runs the job to generate a dataset and push to HF
#
#SBATCH --account=edu
#SBATCH --job-name=PushData
#SBATCH -t 0-1:00                # Increase time limit (12 hours)
#SBATCH --mem=16gb                # Total memory for the job (better than per-cpu)

module load anaconda

export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/abr2184/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HUB_CACHE"

source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate caienv

python merge_and_push.py

#End of script
