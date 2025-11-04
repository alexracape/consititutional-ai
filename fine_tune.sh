#!/bin/bash
#
# Based on the simple "Hello World" submit script for Slurm.
# Runs the job to generate a dataset and push to HF
#
#SBATCH --account=edu
#SBATCH --job-name=CAIFineTuning
#SBATCH -t 0-12:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins082

module load anaconda

export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/abr2184/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HUB_CACHE"

export WANDB_API_KEY=$(cat ~/.wandb_token)
export WANDB_PROJECT="cai-fine-tuning"

# Activate environment
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate caienv

# Manually set PATH
export PATH=/insomnia001/depts/edu/COMS-E6998-012/abr2184/envs/caienv/bin:$PATH

# Python debugging info
echo "Python location: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

debug="$HOME/job_history.txt"
touch $debug
echo "------------------------------" >> $debug
date >> $debug
echo $SLURM_JOB_ID >> $debug
echo $HOSTNAME >> $debug
echo "Cuda visible devices output is $CUDA_VISIBLE_DEVICES" >> $debug
echo "Here is nvidi-smi:" >> $debug
nvidia-smi >> $debug

python fine_tune.py

#End of script