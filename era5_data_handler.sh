#!/bin/bash
#SBATCH --job-name=era5_downloader
#SBATCH --chdir=/work/FAC/FGSE/IDYST/tbeucler/downscaling
#SBATCH --output=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/era5_downloader%j.log
#SBATCH --error=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/era5_downloader%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1      
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00


source sasthana/Downscaling/Processing_and_Analysis_Scripts/environment.sh
export PROJ_LIB="$ENVIRONMENT/share/proj"
python sasthana/Downscaling/Processing_and_Analysis_Scripts/era5_data_handler.py