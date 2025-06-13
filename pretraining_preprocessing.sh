#!/bin/bash
#SBATCH --job-name=pretraining_preprocessing
#SBATCH --output=logs/pretraining_preprocessing_%j.out
#SBATCH --error=logs/pretraining_preprocessing_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00

module load micromamba
module load cdo
eval "$(micromamba shell hook --shell=bash)"
source environment.sh

cd "$BASE_DIR" || exit 1

for VAR in precip temp tmin tmax; do
    echo "Preprocessing var: $VAR"
    python "$BASE_DIR/sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/regridding_pretraining_dataset.py" --var "$VAR"
    echo "Completed var: $VAR"
done
echo "All preprocessing tasks completed."

