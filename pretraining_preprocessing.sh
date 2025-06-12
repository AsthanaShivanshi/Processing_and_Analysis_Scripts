#!/bin/bash
#SBATCH --job-name=pretraining_preprocessing
#SBATCH --output=pretraining_preprocessing_%j.out
#SBATCH --error=proetraining_proeprocessing_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
module load micromamba
eval "$(micromamba shell hook --shell=bash)"
source environment.sh
python regridding_pretraining_dataset.py

