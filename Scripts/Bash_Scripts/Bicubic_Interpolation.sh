#!/bin/bash
#SBATCH --job-name=bicubic_baseline_for_UNet
#SBATCH --output=bicubic_%j.out
#SBATCH --error=bicubic_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
source environment.sh
module load cdo

cdo remapbic,"Wetday_files/Targets/RhiresD_wet_1971_2023.nc" "Wetday_files/Features/rhiresd_wet_coarsened.nc" "Wetday_files/Features/bicubic_rhiresd.nc"
cdo remapbic,"Wetday_files/Targets/TabsD_wet_1971_2023.nc" "Wetday_files/Features/tabsd_wet_coarsened.nc" "Wetday_files/Features/bicubic_tabsd.nc"
cdo remapbic,"Wetday_files/Targets/TmaxD_wet_1971_2023.nc" "Wetday_files/Features/tmaxd_wet_coarsened.nc" "Wetday_files/Features/bicubic_tmax.nc"
cdo remapbic,"Wetday_files/Targets/TminD_wet_1971_2023.nc" "Wetday_files/Features/tmind_wet_coarsened.nc" "Wetday_files/Features/bicubic_tmind.nc"
