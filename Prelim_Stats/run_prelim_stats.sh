#!/bin/bash
#SBATCH --job-name=Plot_Distributions
#SBATCH --output=logs/plot_distributions%j.out
#SBATCH --error=logs/plot_distributions%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

module load python
source environment.sh

python Prelim_Stats/plot_distributions_dataset_comparisons.py --var $SLURM_ARRAY_TASK_ID