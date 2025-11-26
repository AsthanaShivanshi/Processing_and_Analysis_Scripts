#!/bin/bash
#SBATCH --job-name=spatial_quantile_BC_comparison
#SBATCH --output=logs/spatial_quantile_BC_comparison_%j.out
#SBATCH --error=logs/spatial_quantile_BC_comparison_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=10:00:00
##SBATCH --array=1-5

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
source environment.sh

cd Stats_EVT

python spatial_quantile_BC_comparison.py 
#python Multiplicative_99th_cities.py
#python Spatial_Quantile_Bias_Comparison_Cities.py