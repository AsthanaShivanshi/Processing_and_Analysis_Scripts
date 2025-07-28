#!/bin/bash
#SBATCH --job-name=Baselines_Comparison
#SBATCH --output=logs/baselines_comparison%j.out
#SBATCH --error=logs/baselines_comparison%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

module load python
source environment.sh

python Prelim_Stats/Gridwise_CVM_comparison.py --var $SLURM_ARRAY_TASK_ID