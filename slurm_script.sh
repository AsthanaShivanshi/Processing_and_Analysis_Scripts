#!/bin/bash
#SBATCH --job-name=coarsening
#SBATCH --output=coarsen_%j.out
#SBATCH --error=coarsen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

source ../Variables_Config_Scripts/Slurm_Env_Path.sh

module load micromamba

# micromamba shell hook
eval "$(micromamba shell hook --shell=bash)"

micromamba activate "$ENVIRONMENT"

python f"${BASE_DIR}/regridding.py"

