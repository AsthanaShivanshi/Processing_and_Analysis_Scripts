#!/bin/bash
#SBATCH --job-name=bicubic
#SBATCH --output=coarsen_%j.out
#SBATCH --error=coarsen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

# Load paths from ignored config
source ../Variable_Config_Scripts/env_filepaths_bicubic.sh

cd "$BASE_DIR"
module load cdo

cdo remapbic,"$GRID_DIR/RhiresD_wet_1971_2023.nc" "$FEATURE_DIR/rhiresd_wet_coarsened.nc" "$OUTPUT_DIR/bicubic_rhiresd.nc"
cdo remapbic,"$GRID_DIR/TabsD_wet_1971_2023.nc" "$FEATURE_DIR/tabsd_wet_coarsened.nc" "$OUTPUT_DIR/bicubic_tabsd.nc"
cdo remapbic,"$GRID_DIR/TmaxD_wet_1971_2023.nc" "$FEATURE_DIR/tmaxd_wet_coarsened.nc" "$OUTPUT_DIR/bicubic_tmax.nc"
cdo remapbic,"$GRID_DIR/TminD_wet_1971_2023.nc" "$FEATURE_DIR/tmind_wet_coarsened.nc" "$OUTPUT_DIR/bicubic_tmind.nc"
