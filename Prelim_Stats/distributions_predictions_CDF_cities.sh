#!/bin/bash
#SBATCH --job-name=plot_distributions_Geneva
#SBATCH --output=logs/distributions_Geneva_%A_%a.out
#SBATCH --error=logs/distributions_Geneva_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3

module load python 
source environment.sh
cd $BASE_DIR/sasthana/Downscaling/Processing_and_Analysis_Scripts/Prelim_Stats

python distributions_predictions_CDF_cities.py