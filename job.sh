#!/bin/sh
#
# Based on the simple "Hello World" submit script for Slurm.
# Runs the job to generate a dataset and push to HF
#
#SBATCH --account=edu
#SBATCH --job-name=CAIDataGeneration
#SBATCH -c 4                      # Increase CPU cores for better performance
#SBATCH -t 0-12:00                # Increase time limit (12 hours)
#SBATCH --mem=128gb                # Total memory for the job (better than per-cpu)
#SBATCH --gres=gpu:1              # Specify 1 GPU explicitly
#SBATCH --array=1-1               # Start with 2 jobs 


module load anaconda

export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/abr2184/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HUB_CACHE"

source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate caienv

nvidia-smi

echo "Running task $SLURM_ARRAY_TASK_ID on GPU $CUDA_VISIBLE_DEVICES"

python generate_data.py --job_id $SLURM_ARRAY_TASK_ID --num_jobs $SLURM_ARRAY_TASK_COUNT

#End of script
