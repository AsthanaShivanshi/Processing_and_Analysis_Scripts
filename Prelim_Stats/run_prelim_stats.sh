#!/bin/bash
#SBATCH --job-name=Thresholded_RMSE
#SBATCH --output=logs/thresh_rmse%j.out
#SBATCH --error=logs/thresh_rmse%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3

module load python 
source environment.sh 


python Prelim_Stats/thresholded_rmse.py --var $SLURM_ARRAY_TASK_ID