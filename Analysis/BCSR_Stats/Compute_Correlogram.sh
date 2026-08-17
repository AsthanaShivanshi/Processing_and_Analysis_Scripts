#!/bin/bash
#SBATCH --job-name=Correlogram
#SBATCH --array=0-10
#SBATCH --output=logs/Correlogram_%A_%a.log
#SBATCH --error=logs/Correlogram_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=01-00:00:00
#SBATCH --partition=cpu

module load python

source ../Downscaling_Models/diffscaler.sh

cd ../Processing_and_Analysis_Scripts


BASELINES=(
  "OBS"
  "EQM + Bilinear"
  "CDF-t + Bilinear"
  "dOTC + Bilinear"
  "EQM + Bilinear + U-Net"
  "CDF-t + Bilinear + U-Net"
  "dOTC + Bilinear + U-Net"
  "EQM + Bilinear + U-Net + DDIM"
  "CDF-t + Bilinear + U-Net + DDIM"
  "dOTC + Bilinear + U-Net + DDIM"
  "Fine-scale EQM"
)



TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"

if (( TASK_ID < 0 || TASK_ID >= ${#BASELINES[@]} )); then
    echo "Invalid array task ID: $TASK_ID" >&2
    exit 1
fi

LABEL="${BASELINES[$TASK_ID]}"
CSV_DIR="Analysis/BCSR_Stats/Correlogram_bcsr_csv"

mkdir -p "$CSV_DIR"

echo "Task: $TASK_ID"
echo "Baseline: $LABEL"

python Analysis/BCSR_Stats/Correlogram.py \
    --label "$LABEL" \
    --csv_dir "$CSV_DIR"

echo "Finished: $LABEL"
