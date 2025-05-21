#!/bin/bash
#SBATCH --job-name=bilinear
#SBATCH --output=coarsen_%j.out
#SBATCH --error=coarsen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
source environment.sh
module load cdo

cdo remapbil,"Wetday_files/Targets/RhiresD_wet_1971_2023.nc" "Wetday_files/Features/rhiresd_wet_coarsened.nc" "Wetday_files/Features_Bilinear/bilinear_rhiresd.nc"
cdo remapbil,"Wetday_files/Targets/TabsD_wet_1971_2023.nc" "Wetday_files/Features/tabsd_wet_coarsened.nc" "Wetday_files_Bilinear/Features_Bilinear/bilinear_tabsd.nc"
cdo remapbil,"Wetday_files/Targets/TmaxD_wet_1971_2023.nc" "Wetday_files/Features/tmaxd_wet_coarsened.nc" "Wetday_files_Bilinear/Features_Bilinear/bilinear_tmax.nc"
cdo remapbil,"Wetday_files/Targets/TminD_wet_1971_2023.nc" "Wetday_files/Features/tmind_wet_coarsened.nc" "Wetday_files_Bilinear/Features_Bilinear/bilinear_tmind.nc"


