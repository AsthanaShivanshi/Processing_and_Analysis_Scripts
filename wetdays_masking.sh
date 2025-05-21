#!/bin/bash
#SBATCH --job-name=coarsening_rhiresd_tabsd_HR_LR_files
#SBATCH --output=coarsen_%j.out
#SBATCH --error=coarsen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
module load micromamba
eval "$(micromamba shell hook --shell=bash)"
source environment.sh
module load CDO

RAW_DIR="Raw_files"
COARSE_DIR="Coarsened_Files"
OUT_DIR="Wetday_files"

HIGH_MASK="rhiresd_high_res_wetday_mask.nc"
LOW_MASK="rhiresd_low_res_wetday_mask.nc"

mkdir -p "$OUT_DIR"

echo "HR files masking"
for file in ${RAW_DIR}/*.nc; do
    filename=$(basename "$file")
    temp_file="${OUT_DIR}/temp_${filename}"
    outfile="${OUT_DIR}/${filename%.nc}_wetdays.nc"

    echo "Masking $filename → $outfile"
    
    #Multiplying data by mask
    cdo mul "$file" "$HIGH_MASK" "$temp_file"
    
    # Setting zeros to NaN
    cdo setrtomiss,0,0 "$temp_file" "$outfile"

    rm -f "$temp_file"
done

echo "LR files masking"
for file in ${COARSE_DIR}/*.nc; do
    filename=$(basename "$file")
    temp_file="${OUT_DIR}/temp_${filename}"
    outfile="${OUT_DIR}/${filename%.nc}_wetdays_coarse.nc"

    echo "Masking $filename → $outfile"

    # Multiplying by mask
    cdo mul "$file" "$LOW_MASK" "$temp_file"

    # Zeroes to NaN
    cdo setrtomiss,0,0 "$temp_file" "$outfile"

    rm -f "$temp_file"
done

echo "COMPLETE."