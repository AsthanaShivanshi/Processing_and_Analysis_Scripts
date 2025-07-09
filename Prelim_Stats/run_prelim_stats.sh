#!/bin/bash
#SBATCH --job-name=Prelim_Stats
#SBATCH --output=logs/prelim_stats%j.out
#SBATCH --error=logs/prelim_stats%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3

module load python 
source environment.sh 


python Prelim_Stats/thresholded_mse.py --var $SLURM_ARRAY_TASK_ID