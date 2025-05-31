#!/bin/bash
#SBATCH --job-name=Data_Processing
#SBATCH --output=logs/Data_Processing_%j.out
#SBATCH --error=logs/Data_Processing_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
source environment.sh

module load cdo

which python

python Python_Pipeline_Scripts/data_processing.py
