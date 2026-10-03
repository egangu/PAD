# On-Policy Preference Data Generation

The scripts in this directory generate the local preference dataset consumed by
`training_configs/gemma-2-2b-it-pd.yaml`. They must be run from the
repository root.

## Prerequisites

Install [vLLM](https://github.com/vllm-project/vllm); Gemma-2 runs also need
`flashinfer`. You must provide:

- a local Hugging Face dataset saved with `datasets.save_to_disk`, with a
  `train` split and a `prompt` column;
- access to the student and teacher model checkpoints; and
- a CUDA environment suitable for vLLM.

The public repository does not include the prepared prompt dataset, generated
outputs, or model weights.

## Build the Gemma dataset

Set the prompt dataset once, then run the two stages:

```sh
export PROMPT_DATASET=/absolute/path/to/ultrafeedback-split
bash data_gen/scripts/sampling.sh
bash data_gen/scripts/pipeline_n4_gemma.sh
```

`sampling.sh` writes `output_<seed>.json` files to
`data/generated/ultrafeedback/gemma-2b-it/` by default. The pipeline then
aggregates four responses per prompt, scores them with the teacher model, and
writes a Hugging Face dataset to:

```
data/generated/ultrafeedback/gemma-2b-it/pkd-dataset-teacher-gemma-n4
```

Override `STUDENT_MODEL`, `TEACHER_MODEL`, `STUDENT_DIR`, `N`, or
`CUDA_VISIBLE_DEVICES` when using different checkpoints or output locations.
