#!/bin/bash
#SBATCH --job-name=Gamma_test_gridded_rhiresd_90_pc_confidence   
#SBATCH --array=0-3
#SBATCH --output=logs/job_output-%A_%a.txt 
#SBATCH --error=logs/job_error-%A_%a.txt  
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00        
#SBATCH --mem=64G                
#SBATCH --partition=cpu         

module load micromamba
source environment.sh

cd "$BASE_DIR/Scripts/Functions"

SEASON_LIST=("JJA" "SON" "DJF" "MAM")
SEASON=${SEASON_LIST[$SLURM_ARRAY_TASK_ID]}

python Run_Gamma_Tests.py "$SEASON"
