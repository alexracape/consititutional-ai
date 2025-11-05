#!/bin/sh
#
# Based on the simple "Hello World" submit script for Slurm.
# Runs the job to generate a dataset and push to HF
#
#SBATCH --account=edu
#SBATCH --job-name=CAIPushData
#SBATCH -t 0-1:00   
#SBATCH --mem=16gb

export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/abr2184/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HUB_CACHE"

# Activate environment
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate caienv

# Manually fix PATH to prioritize conda env
export PATH=/insomnia001/depts/edu/COMS-E6998-012/abr2184/envs/caienv/bin:$PATH

python src/merge_and_push.py

#End of script
