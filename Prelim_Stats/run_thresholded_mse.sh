#!/bin/bash
#SBATCH --job-name=thresholded_mse
#SBATCH --output=logs/thresholded_mse%j.out
#SBATCH --error=logs/thresholded_mse%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3

module load python 
source environment.sh 


python Prelim_Stats/thresholded_mse.py --var $SLURM_ARRAY_TASK_ID