#!/bin/bash
#SBATCH --job-name=gridwise_winner
#SBATCH --output=logs/gridwise_winner%j.out
#SBATCH --error=logs/gridwise_winner%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=16:00:00
##SBATCH --array=0-3


module load python
source environment.sh


#python Prelim_Stats/KGE_comparison_baselines.py

#python Prelim_Stats/PSS_citywise_all_baselines.py
python Prelim_Stats/daily_climatology_comparison_BC_methods.py 
#python Prelim_Stats/Indices_future_scenarios.py --city "Zurich" --lat 47.3769 --lon 8.5417 #--var $SLURM_ARRAY_TASK_ID
#python Prelim_Stats/Indices_future_scenarios.py --city "Locarno" --lat 46.1670 --lon 8.7943 #--var $SLURM_ARRAY_TASK_ID
#python Prelim_Stats/Indices_future_scenarios.py --city "Geneva" --lat 46.2044 --lon 6.1432 #--var $SLURM_ARRAY_TASK_ID

#python Prelim_Stats/plot_distributions_dataset_comparisons.py
#python Prelim_Stats/spatial_quantile_bias_comparison.py --var $SLURM_ARRAY_TASK_ID
#python Prelim_Stats/thresholded_rmse_spatial_comparison.py --var $SLURM_ARRAY_TASK_ID
#python Prelim_Stats/gridwise_PSS.py --var $SLURM_ARRAY_TASK_ID
#python Prelim_Stats/Mean_Annual_Cycle_Comparison.py --city "Zürich" --lat 47.3769 --lon 8.5417 --var $SLURM_ARRAY_TASK_ID
#python Prelim_Stats/Mean_Annual_Cycle_Comparison.py --city "Locarno" --lat 46.1670 --lon 8.7943 --var $SLURM_ARRAY_TASK_ID