#!/bin/bash
#SBATCH --job-name=model_OP_preprocessing 
#SBATCH --chdir=/work/FAC/FGSE/IDYST/tbeucler/downscaling
#SBATCH --output=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/model_OP_preprocessing%j.log
#SBATCH --error=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/model_OP_preprocessing%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1      
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

source sasthana/Downscaling/Processing_and_Analysis_Scripts/environment.sh
export PROJ_LIB="$ENVIRONMENT/share/proj"

python /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/model_outputs_preprocessing.py
