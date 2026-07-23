#!/bin/bash
#SBATCH --job-name=SAL_SR_Metrics
#SBATCH --output=logs/SAL_SR_Metrics_job_%j.log
#SBATCH --error=logs/SAL_SR_Metrics_job_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=cpu

source ../Downscaling_Models/diffscaler.sh

export PYTHONPATH="$PROJECT_DIR"


module load cdo


export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


cd ../Processing_and_Analysis_Scripts

which python

python -c "import wandb; print(wandb.__version__)"


#python Analysis/Paper_Stats/cobweb_metrics.py

#python Analysis/Paper_Stats/plot_cobweb.py

python Analysis/Paper_Stats/sal.py



