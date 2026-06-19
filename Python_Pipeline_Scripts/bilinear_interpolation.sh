#!/bin/bash
# bilinear_interp.sh

set -e  
COARSE_FILE="$1"
TARGET_FILE="$2"
OUTPUT_FILE="$3"




echo "[INFO] Performing bilinear interpolation using CDO..."
cdo remapbil,"$TARGET_FILE" "$COARSE_FILE" "$OUTPUT_FILE"
