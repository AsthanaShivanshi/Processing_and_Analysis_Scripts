#!/bin/bash
#SBATCH --job-name=Gamma_test_gridded_precip    
#SBATCH --array=0-3
#SBATCH --output=job_output-%j.txt 
#SBATCH --error=job_error-%j.txt  
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00        
#SBATCH --mem=64G                
#SBATCH --partition=cpu         

module load micromamba

source environment.sh

micromamba run -p "$ENVIRONMENT" which python

cd Scripts/Functions

SEASON_LIST=("JJA" "SON" "DJF" "MAM")
SEASON=${SEASON_LIST[$SLURM_ARRAY_TASK_ID]}

micromamba run -p "$ENVIRONMENT" python Run_Gamma_Tests.py $SEASON

