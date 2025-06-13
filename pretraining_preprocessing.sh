#!/bin/bash
#SBATCH --job-name=pretrain_prep
#SBATCH --output=logs/pretraining_preprocessing_%A_%a.out
#SBATCH --error=logs/pretraining_preprocessing_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3%1 #One task at a time 

# Serial processing
VARS=(precip temp tmin tmax)
VAR=${VARS[$SLURM_ARRAY_TASK_ID]}

module load micromamba
module load cdo
eval "$(micromamba shell hook --shell=bash)"
source environment.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR" || exit 1
echo "Processing variable: $VAR"
python "$BASE_DIR/sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/regridding_pretraining_dataset.py" --var "$VAR"
echo "Finished variable: $VAR"
