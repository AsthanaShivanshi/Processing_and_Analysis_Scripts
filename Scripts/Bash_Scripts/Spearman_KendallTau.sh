#!/bin/bash
#SBATCH --job-name=Spearman_KendallTau_CPU    
#SBATCH --output=job_output-%j.txt 
#SBATCH --error=job_error-%j.txt  
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=4     
#SBATCH --time=3-00:00:00        
#SBATCH --mem=64G  
#SBATCH --partition=cpu  
# (NO --gres=gpu:1)

source ../../environment.sh

module load python

cd ../Scripts/Functions

python correlation_Kendall_Spearman.py

