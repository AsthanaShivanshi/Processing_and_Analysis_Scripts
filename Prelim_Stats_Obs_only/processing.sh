#!/bin/bash
#SBATCH --job-name=Quantile_Bias
#SBATCH --chdir=/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/Prelim_Stats_Obs_only
#SBATCH --output=sasthana/Downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/Quantile_Bias%j.log
#SBATCH --error=sasthana/Downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/Quantile_Bias%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3      
#SBATCH --mem=128G
#SBATCH --time=05:00:00

source ../environment.sh
module load cdo
#python psnr_ssim_rmse.py 
python quantile_bias.py
#python QQ_plots.py
#python correlogram.py