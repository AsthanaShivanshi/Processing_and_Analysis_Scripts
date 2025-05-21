#!/bin/bash
# environment.sh

export BASE_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling"
export ENVIRONMENT="${BASE_DIR}/sasthana/MyPythonEnvNew"

module load micromamba
eval "$(micromamba shell hook --shell=bash)"

micromamba activate "$ENVIRONMENT"
