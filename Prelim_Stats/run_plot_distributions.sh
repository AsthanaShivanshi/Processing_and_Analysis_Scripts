#!/bin/bash
#SBATCH --job-name=plot_distributions
#SBATCH --output=logs/distributions_%j.out
#SBATCH --error=logs/distributions_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

module load python 
source environment.sh 

python Prelim_Stats/plot_distributions.py