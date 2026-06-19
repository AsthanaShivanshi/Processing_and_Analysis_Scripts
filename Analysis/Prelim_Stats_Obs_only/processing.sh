#!/bin/bash
#SBATCH --job-name=SR_metrics_temp
#SBATCH --chdir=/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/Prelim_Stats_Obs_only
#SBATCH --output=sasthana/Downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/SR_metrics_temp%j.log
#SBATCH --error=sasthana/Downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/SR_metrics_temp%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1   
#SBATCH --mem=128G
#SBATCH --time=03:00:00

source ../environment.sh
module load cdo


python psnr_ssim_rmse.py 
#python quantile_bias.py
#python QQ_plots.py
#python correlogram_precip.py