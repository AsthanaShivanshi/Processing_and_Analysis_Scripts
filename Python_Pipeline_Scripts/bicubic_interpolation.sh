#!/bin/bash
# bicubic_interp.sh

set -e  
COARSE_FILE="$1"
TARGET_FILE="$2"
OUTPUT_FILE="$3"

echo "[INFO] Performing bicubic interpolation using CDO..."
cdo remapbic,"$TARGET_FILE" "$COARSE_FILE" "$OUTPUT_FILE"
