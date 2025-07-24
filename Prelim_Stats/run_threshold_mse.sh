#!/bin/bash
#SBATCH --job-name=Spatial_Quantile_Bias
#SBATCH --output=logs/spatial_quantile_bias%j.out
#SBATCH --error=logs/spatial_quantile_bias%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

module load python 
source environment.sh 


python Prelim_Stats/thresholded_mse.py --var $SLURM_ARRAY_TASK_ID --city "Zurich" --lat 47.3769 --lon 8.5417