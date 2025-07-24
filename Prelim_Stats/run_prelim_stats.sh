#!/bin/bash
#SBATCH --job-name=Spatial_Threshold_RMSE
#SBATCH --output=logs/spatial_threshold_rmse%j.out
#SBATCH --error=logs/spatial_threshold_rmse%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

module load python
source environment.sh

python Prelim_Stats/thresholded_rmse_spatial.py --var $SLURM_ARRAY_TASK_ID