#!/bin/bash
#SBATCH --job-name=EVT_runner
#SBATCH --output=logs/EVT_runner_%j.out
#SBATCH --error=logs/EVT_runner_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=10:00:00

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
source environment.sh

python Stats_EVT/Return_Levels_gridded_BM.py