#!/bin/bash
#SBATCH --job-name=EVT_runner
#SBATCH --output=logs/EVT_return_bm%j.out
#SBATCH --error=logs/EVT_return_bm%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=20:00:00


source environment.sh
module load python

python Stats_EVT/Return_Levels_gridded_BM.py