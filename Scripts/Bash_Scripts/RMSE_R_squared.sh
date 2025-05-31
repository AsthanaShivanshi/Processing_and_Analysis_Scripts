#!/bin/bash
#SBATCH --job-name=RMSE_R_squared
#SBATCH --output=logs/RMSE_Rsquared_%j.out
#SBATCH --error=logs/RMSE_Rsquared_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
source environment.sh

# Run your evaluation script
python Python_Pipeline_Scripts/eval_metrics.py config_files/evaluation_metrics.yaml
