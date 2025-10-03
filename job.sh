#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=ACCOUNT         # Replace ACCOUNT with your group account name
#SBATCH --job-name=HelloWorld     # The job name.
#SBATCH -c 1                      # The number of cpu cores to use
#SBATCH -t 0-0:30                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5gb         # The memory the job will use per cpu core
module load anaconda

#Command to execute Python program
python generate_data.py
