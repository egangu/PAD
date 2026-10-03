#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${N:=4}"
: "${TEMPERATURE:=1}"
: "${TEACHER_MODEL:=google/gemma-2-9b-it}"
: "${TEACHER_ID:=gemma}"
: "${STUDENT_DIR:=data/generated/ultrafeedback/gemma-2b-it}"

export CUDA_VISIBLE_DEVICES

shopt -s nullglob
sampled_outputs=("${STUDENT_DIR}"/output_*.json)
if (( ${#sampled_outputs[@]} < N )); then
    echo "Need at least ${N} sampled outputs in ${STUDENT_DIR}; found ${#sampled_outputs[@]}. Run sampling.sh first." >&2
    exit 1
fi

python data_gen/gen/agg.py --generation_file_dir "${STUDENT_DIR}" -n "${N}"

python data_gen/gen/prob_sl.py --model_name "${TEACHER_MODEL}" \
    --temperature "${TEMPERATURE}" \
    --input_file "${STUDENT_DIR}/agg_outputs_n${N}.json" \
    --output_file "${STUDENT_DIR}/agg_outputs_n${N}.prob.${TEACHER_ID}.sl.json"

python data_gen/gen/prob.py --model_name "${TEACHER_MODEL}" \
    --temperature "${TEMPERATURE}" \
    --num_options "${N}" \
    --input_file "${STUDENT_DIR}/agg_outputs_n${N}.json" \
    --output_file "${STUDENT_DIR}/agg_outputs_n${N}.prob.${TEACHER_ID}.json"

python data_gen/gen/generate_dataset.py \
    --prob_file "${STUDENT_DIR}/agg_outputs_n${N}.prob.${TEACHER_ID}.json" \
    --prob_sl_file "${STUDENT_DIR}/agg_outputs_n${N}.prob.${TEACHER_ID}.sl.json" \
    --output_dir "${STUDENT_DIR}/pkd-dataset-teacher-${TEACHER_ID}-n${N}"
