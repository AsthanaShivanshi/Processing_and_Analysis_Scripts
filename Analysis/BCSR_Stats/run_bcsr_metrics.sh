#!/bin/bash
#SBATCH --job-name=Table_4_bcsr_metrics
#SBATCH --output=logs/Table_4_bcsr_metrics_%j.log
#SBATCH --error=logs/Table_4_bcsr_metrics_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01-00:00:00
#SBATCH --partition=cpu


#Running from PAS

module load python

source ../Downscaling_Models/diffscaler.sh

cd ../Processing_and_Analysis_Scripts

python Analysis/BCSR_Stats/metrics_table_4.py \
  #--mask_hr_file ../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc



#python Analysis/BCSR_Stats/spearman.py \
  #--mask_hr_file ../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc


#python Analysis/BCSR_Stats/Autocorrelation.py \
  #--lags 1,2,3 \
  #--out_csv Analysis/BCSR_Stats/Tables/autocorrelation_tas_pr_2015_2023.csv



#python Analysis/BCSR_Stats/Autocorrelation_plots.py --plot_lags 1
