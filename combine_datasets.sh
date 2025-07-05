#!/bin/bash
#SBATCH --job-name=combine_datasets_processing
#SBATCH --chdir=/work/FAC/FGSE/IDYST/tbeucler/downscaling
#SBATCH --output=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/combine_datasets_processing_%A_%a.log
#SBATCH --error=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/combine_datasets_processing_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --array 0-3 #depending on vars processed

source sasthana/Downscaling/Processing_and_Analysis_Scripts/environment.sh
export PROJ_LIB="$ENVIRONMENT/share/proj"

python sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/combining_datasets.py $SLURM_ARRAY_TASK_ID 
