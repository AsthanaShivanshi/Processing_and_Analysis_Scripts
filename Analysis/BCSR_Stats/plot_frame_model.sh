#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/plot_frame_model_run.py"

VAR="pr"                      # pr or tas
SCENARIO="ssp370"
MEMBER="r1i1p1f1"
RCM="RegCM5-0"
VERSION="v1-r1"

METHODS=(
  "EQM_C"
  "dOTC"
  "CDFT-t"
)

MODELS=(
  "MPI-ESM1-2-HR"
  "EC-Earth3-Veg"
  "NorESM2-MM"
)

DATES=(
  "2045-05-15"
  "2096-01-15"
  "2099-01-15"
  "2020-10-15"
)

OUTDIR="${SCRIPT_DIR}/Figures/multi_model_runs"
mkdir -p "${OUTDIR}"

for METHOD in "${METHODS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for DATE in "${DATES[@]}"; do
      echo "Running method=${METHOD}, model=${MODEL}, date=${DATE}"
      python "${PY_SCRIPT}" \
        --date "${DATE}" \
        --var "${VAR}" \
        --method "${METHOD}" \
        --model "${MODEL}" \
        --scenario "${SCENARIO}" \
        --member "${MEMBER}" \
        --rcm "${RCM}" \
        --version "${VERSION}" \
        --out "${OUTDIR}/frames_2col_${METHOD}_${MODEL}_${SCENARIO}_${VAR}_${DATE}.png"
    done
  done
done

echo "Done: ${OUTDIR}"