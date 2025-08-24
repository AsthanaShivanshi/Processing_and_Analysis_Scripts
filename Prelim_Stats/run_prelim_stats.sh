#!/bin/bash
#SBATCH --job-name=Plot_Mean_Annual_Cycle
#SBATCH --output=logs/plot_mean%j.out
#SBATCH --error=logs/plot_mean%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=16:00:00


module load python
source environment.sh

python Prelim_Stats/Mean_Annual_Cycle_Comparison.py #--var $SLURM_ARRAY_TASK_ID