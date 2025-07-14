#!/bin/bash
#SBATCH --job-name=tmin_tmax_combination
#SBATCH --chdir=/work/FAC/FGSE/IDYST/tbeucler/downscaling
#SBATCH --output=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/combine_tmin_tmax_%j.log
#SBATCH --error=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/combine_tmin_tmax_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00

source sasthana/Downscaling/Processing_and_Analysis_Scripts/environment.sh
export PROJ_LIB="$ENVIRONMENT/share/proj"

for idx in 2; do 
    echo "Processing variable index $idx"
    python /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/combining_datasets.py $idx
done