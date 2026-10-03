#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

: "${CUDA_VISIBLE_DEVICES:=0,1}"
export CUDA_VISIBLE_DEVICES

if ! command -v accelerate > /dev/null; then
    echo "accelerate is not available. Activate the PAD environment first." >&2
    exit 1
fi

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed-zero2-2gpus-p25601.yaml scripts/run_pad.py training_configs/gemma-2-2b-it-pd.yaml
