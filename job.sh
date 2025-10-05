#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=edu
#SBATCH --job-name=CAIDataGeneration
#SBATCH -c 1                      # The number of cpu cores to use
#SBATCH -t 0-0:30                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5gb         # The memory the job will use per cpu core
#SBATCH --gres=gpu     

module load anaconda

source activate myenv

#Command to execute Python program
python generate_data.py
