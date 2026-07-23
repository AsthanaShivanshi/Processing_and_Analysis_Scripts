#!/bin/bash
#SBATCH --job-name=bcsr_metrics
#SBATCH --output=logs/bcsr_metrics_%j.log
#SBATCH --error=logs/bcsr_metrics_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=01-00:00:00


#This used to run the bcsr metrics on the overlapping test set (2015-2023) for all model chains wrt. observations. 

module load python
source diffscaler.sh

#python #####file.py