#!/bin/bash
#SBATCH --job-name=Ensmeans_metrics
#SBATCH --output=logs/Ensmeans_metrics_%j.log
#SBATCH --error=logs/Ensmeans_metrics_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --partition=cpu


#Running from PAS

module load python

source ../Downscaling_Models/diffscaler.sh

cd ../Processing_and_Analysis_Scripts



#with Ensmeans 


python Analysis/BCSR_Stats/Autocorrelation.py \
  --var tas \
  --mask_hr_file ../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc \
  --eval_start 2015 \
  --eval_end 2023



python Analysis/BCSR_Stats/spearman.py \
  --mask_hr_file ../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc \
  --eval_start 2015 \
  --eval_end 2023



python Analysis/BCSR_Stats/trends.py \
  --mask_hr_file ../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc




#with Enspools

#python Analysis/BCSR_Stats/logpdf.py  #Requires pooling



#python Analysis/BCSR_Stats/metrics_table_4.py \
  #--eval_start 2015 --eval_end 2023 \
  #--verbose_loader





