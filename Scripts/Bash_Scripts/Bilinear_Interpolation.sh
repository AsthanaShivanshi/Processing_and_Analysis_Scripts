#!/bin/bash
#SBATCH --job-name=bilinear
#SBATCH --output=coarsen_%j.out
#SBATCH --error=coarsen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

source ../Variable_Config_Scripts/env_filepaths_bilinear.sh
cd "$BASE_DIR"
module load cdo


# Perform bicubic interpolation using the saved grid files
cdo remapbil,"$GRID_DIR/RhiresD_wet_1971_2023.nc" "$FEATURE_DIR/rhiresd_wet_coarsened.nc" "$OUTPUT_DIR/bilinear_rhiresd.nc"
cdo remapbil,"$GRID_DIR/TabsD_wet_1971_2023.nc" "$FEATURE_DIR/tabsd_wet_coarsened.nc" "$OUTPUT_DIR/bilinear_tabsd.nc"
cdo remapbil,"$GRID_DIR/TmaxD_wet_1971_2023.nc" "$FEATURE_DIR/tmaxd_wet_coarsened.nc" "$OUTPUT_DIR/bilinear_tmax.nc"
cdo remapbil,"$GRID_DIR/TminD_wet_1971_2023.nc" "$FEATURE_DIR/tmind_wet_coarsened.nc" "$OUTPUT_DIR/bilinear_tmind.nc"


