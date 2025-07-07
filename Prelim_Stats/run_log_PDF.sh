#!/bin/bash
#SBATCH --job-name=log_PDF
#SBATCH --output=logs/log_PDF%j.out
#SBATCH --error=logs/log_PDF%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3

module load python 
source environment.sh 

python Prelim_Stats/log_PDF.py