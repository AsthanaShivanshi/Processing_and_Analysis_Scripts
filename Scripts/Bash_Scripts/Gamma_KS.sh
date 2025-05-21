#!/bin/bash
#SBATCH --job-name=Gamma_tests     
#SBATCH --array=0-3 #For all four seasons , three jobs shall run in parallel on different cpu nodes
#SBATCH --output=job_output-%j.txt 
#SBATCH --error=job_error-%j.txt  
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=4         # 4 CPU cores (Dask + blocks)
#SBATCH --time=3-00:00:00        
#SBATCH --mem=64G                
#SBATCH --partition=cpu         
# (NO --gres=gpu:1 )
              
source ../../environment.sh

module load python

#Directory containing the functions
cd ../Scripts/Functions

SEASON_LIST=("JJA" "SON" "DJF" "MAM")

SEASON=${SEASON_LIST[$SLURM_ARRAY_TASK_ID]}

python Run_Gamma_Tests.py $SEASON

