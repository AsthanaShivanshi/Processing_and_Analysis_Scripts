#!/bin/bash
#SBATCH --job-name=POOLED_CVM
#SBATCH --output=logs/pooled_cvm%j.out
#SBATCH --error=logs/pooled_cvm%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

module load python
source environment.sh

python Prelim_Stats/Cramer_Von_Mises_Gridded_allvars.py --var $SLURM_ARRAY_TASK_ID