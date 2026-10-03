#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${SPLIT_COUNT:=5}"
: "${GEMMA_INPUT:=data/generated/ultrafeedback/gemma-2b-it/agg_outputs_n5.json}"
: "${GEMMA_OUTPUT:=data/generated/ultrafeedback/gemma-2b-it/split_n5}"
: "${LLAMA_INPUT:=data/generated/ultrafeedback/llama-3b-it/agg_outputs_n5.json}"
: "${LLAMA_OUTPUT:=data/generated/ultrafeedback/llama-3b-it/split_n5}"

for input_file in "${GEMMA_INPUT}" "${LLAMA_INPUT}"; do
    if [[ ! -f "${input_file}" ]]; then
        echo "Input file does not exist: ${input_file}" >&2
        exit 1
    fi
done

python data_gen/gen/split.py -k "${SPLIT_COUNT}" \
    --input_file "${GEMMA_INPUT}" \
    --output_dir "${GEMMA_OUTPUT}"

python data_gen/gen/split.py -k "${SPLIT_COUNT}" \
    --input_file "${LLAMA_INPUT}" \
    --output_dir "${LLAMA_OUTPUT}"
