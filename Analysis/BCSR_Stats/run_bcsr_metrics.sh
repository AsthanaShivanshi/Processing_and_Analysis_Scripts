#!/bin/bash
#SBATCH --job-name=Table4_bcsr_metrics
#SBATCH --output=logs/Table4_bcsr_metrics_%j.log
#SBATCH --error=logs/Table4_bcsr_metrics_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=01-00:00:00
#SBATCH --partition=cpu


#Running from PAS

module load python

source ../Downscaling_Models/diffscaler.sh

cd ../Processing_and_Analysis_Scripts



python Analysis/BCSR_Stats/metrics_table_4.py

#python Analysis/BCSR_Stats/spearman.py \
  #--mask_hr_file ../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc


#python Analysis/BCSR_Stats/Autocorrelation.py --var tas


#python Analysis/BCSR_Stats/logpdf.py 
