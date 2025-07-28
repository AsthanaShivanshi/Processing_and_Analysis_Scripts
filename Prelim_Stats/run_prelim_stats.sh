#!/bin/bash
#SBATCH --job-name=QB_Comparison
#SBATCH --output=logs/qb_comparison%j.out
#SBATCH --error=logs/qb_comparison%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=08:00:00

module load python
source environment.sh

python Prelim_Stats/spatial_quantile_bias_comparison.py