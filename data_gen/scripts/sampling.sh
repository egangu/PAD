#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${PROMPT_DATASET:?Set PROMPT_DATASET to a local Hugging Face dataset with a train split and prompt column.}"
: "${STUDENT_MODEL:=google/gemma-2-2b-it}"
: "${STUDENT_DIR:=data/generated/ultrafeedback/gemma-2b-it}"

export CUDA_VISIBLE_DEVICES

if [[ ! -d "${PROMPT_DATASET}" ]]; then
    echo "PROMPT_DATASET does not exist: ${PROMPT_DATASET}" >&2
    exit 1
fi

for seed in 0 1 2 3 4; do
    python data_gen/gen/sampling.py --model_name "${STUDENT_MODEL}" \
        --dataset "${PROMPT_DATASET}" \
        --dataset_split train \
        --local \
        --max_tokens 4096 \
        --temperature 1.0 \
        --top_p 0.95 \
        --seed "${seed}" \
        --output_dir "${STUDENT_DIR}"
done
