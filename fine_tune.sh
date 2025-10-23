#!/bin/sh
#
# Based on the simple "Hello World" submit script for Slurm.
# Runs the job to generate a dataset and push to HF
#
#SBATCH --account=edu
#SBATCH --job-name=CAIFineTuning
#SBATCH -c 2                      # Increase CPU cores for better performance
#SBATCH -t 0-12:00                # Increase time limit (12 hours)
#SBATCH --mem=32gb                # Total memory for the job (better than per-cpu)
#SBATCH --gres=gpu:1              # Specify 1 GPU explicitly


module load anaconda

export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/abr2184/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HUB_CACHE"

export WANDB_API_KEY=$(cat ~/.wandb_token)
export WANDB_PROJECT="cai-fine-tuning"

source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate caienv

nvidia-smi

python fine_tune.py

#End of script
