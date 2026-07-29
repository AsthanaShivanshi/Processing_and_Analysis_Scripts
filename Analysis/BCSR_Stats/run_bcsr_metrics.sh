#!/bin/bash
#SBATCH --job-name=LHD_RMSE_RALSD_bcsr_metrics
#SBATCH --output=logs/LHD_RMSE_RALSD_bcsr_metrics_%j.log
#SBATCH --error=logs/LHD_RMSE_RALSD_bcsr_metrics_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --partition=cpu


#Running from PAS

module load python

source ../Downscaling_Models/diffscaler.sh

cd ../Processing_and_Analysis_Scripts

python Analysis/BCSR_Stats/metrics_table_4.py \
  --mask_hr_file ../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc \
  --mask_lr_file ../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_LR.nc 

