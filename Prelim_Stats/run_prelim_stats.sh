#!/bin/bash
#SBATCH --job-name=Plot_distributions
#SBATCH --output=logs/plot_distributions%j.out
#SBATCH --error=logs/plot_distributions%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=16:00:00
##SBATCH --array=0-3


module load python
source environment.sh

#python Prelim_Stats/Indices_future_scenarios.py --city "Zurich" --lat 47.3769 --lon 8.5417 #--var $SLURM_ARRAY_TASK_ID
#python Prelim_Stats/Indices_future_scenarios.py --city "Locarno" --lat 46.1670 --lon 8.7943 #--var $SLURM_ARRAY_TASK_ID
#python Prelim_Stats/Indices_future_scenarios.py --city "Geneva" --lat 46.2044 --lon 6.1432 #--var $SLURM_ARRAY_TASK_ID

python Prelim_Stats/plot_distributions_dataset_comparisons.py