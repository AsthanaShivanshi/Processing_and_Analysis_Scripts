#!/bin/bash
#SBATCH --job-name=Zurich_Thresh_RMSE
#SBATCH --output=logs/thresholded_rmse%j.out
#SBATCH --error=logs/thresholded_rmse%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

module load python 
source environment.sh 

python Prelim_Stats/thresholded_rmse_gridwise.py --var $SLURM_ARRAY_TASK_ID --city "Zurich" --lat 47.3769 --lon 8.5417
#python Prelim_Stats/thresholded_rmse_gridwise.py --var $SLURM_ARRAY_TASK_ID --city "Locarno" --lat 46.6042 --lon 8.7969
#python Prelim_Stats/thresholded_rmse_gridwise.py --var $SLURM_ARRAY_TASK_ID --city "Geneva" --lat 46.2044 --lon 6.1432