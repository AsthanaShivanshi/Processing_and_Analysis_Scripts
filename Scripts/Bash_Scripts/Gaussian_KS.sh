#!/bin/bash
#SBATCH --job-name=Gaussian_tests     
#SBATCH --output=job_output-%j.txt 
#SBATCH --error=job_error-%j.txt  
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=4         # 4 CPU cores (Dask + blocks)
#SBATCH --time=3-00:00:00        
#SBATCH --mem=64G                
#SBATCH --partition=cpu         
# (NO --gres=gpu:1)

module load python

source ../../environment.sh

cd ../Scripts/Functions

python Run_KS_Tests.py
