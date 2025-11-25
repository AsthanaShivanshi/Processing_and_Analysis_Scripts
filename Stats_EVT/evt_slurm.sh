#!/bin/bash
#SBATCH --job-name=Multiplicative_Return_Levels_Bias_Baselines
#SBATCH --output=logs/Multiplicative_Return_Levels_Bias_Baselines_%j.out
#SBATCH --error=logs/Multiplicative_Return_Levels_Bias_Baselines_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=10:00:00
##SBATCH --array=1-5

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
source environment.sh

cd Stats_EVT

#python spatial_quantile_BC_comparison.py 
python Multiplicative_Return_Levels_Bias_Baselines.py
#python Spatial_Quantile_Bias_Comparison_Cities.py