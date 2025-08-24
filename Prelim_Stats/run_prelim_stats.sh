#!/bin/bash
#SBATCH --job-name=WDF
#SBATCH --output=logs/plot_wdf%j.out
#SBATCH --error=logs/plot_wdf%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=16:00:00
#SBATCH --array=0-3


module load python
source environment.sh

python Prelim_Stats/Mean_Annual_Cycle_WDF.py --var $SLURM_ARRAY_TASK_ID --city "Zurich" --lat 47.3769 --lon 8.5417
python Prelim_Stats/Mean_Annual_Cycle_WDF.py --var $SLURM_ARRAY_TASK_ID --city "Locarno" --lat 46.1670 --lon 8.7943
python Prelim_Stats/Mean_Annual_Cycle_WDF.py --var $SLURM_ARRAY_TASK_ID --city "Geneva" --lat 46.2044 --lon 6.1432
