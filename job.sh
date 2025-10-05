#!/bin/sh
#
# Based on the simple "Hello World" submit script for Slurm.
# Runs the job to generate a dataset and push to HF
#
#SBATCH --account=edu
#SBATCH --job-name=CAIDataGeneration
#SBATCH -c 4                      # Increase CPU cores for better performance
#SBATCH -t 0-2:00                 # Increase time limit (2 hours)
#SBATCH --mem=32gb                # Total memory for the job (better than per-cpu)
#SBATCH --gres=gpu:1              # Specify 1 GPU explicitly

module load anaconda

#Command to execute Python program
python generate_data.py

#End of script
