# Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models

This repository contains the PAD training code and the scripts used to build
the local preference dataset for the reported Gemma experiment.

## Reproduction scope

The released training configuration uses two A800 80G GPUs, as reported below.
It also requires access to the gated Gemma checkpoints and to a prompt dataset
saved locally with Hugging Face `datasets`; neither model weights nor prepared
data are included in this repository.

`environment.yml` defines the base Python environment but is not a complete
dependency lockfile. The archived run log records Transformers 4.50.1,
DeepSpeed 0.16.4, and Weights & Biases 0.19.8. The custom trainers also depend
on a compatible TRL release. A fresh environment therefore needs a dependency
lockfile before it can be presented as a fully reproducible run.

## Setup

Create and activate the base environment:

```sh
mamba env create -f environment.yml
mamba activate pad
```

## Data Generation

Run these commands from the repository root. Set `PROMPT_DATASET` to a local
dataset directory whose `train` split contains a `prompt` column, and make
sure that the environment provides `vllm` (and `flashinfer` for Gemma-2).
The scripts accept `STUDENT_MODEL`, `TEACHER_MODEL`, and `STUDENT_DIR`
overrides for local checkpoints or other Hugging Face model identifiers.

```sh
export PROMPT_DATASET=/absolute/path/to/ultrafeedback-split
bash data_gen/scripts/sampling.sh
bash data_gen/scripts/pipeline_n4_gemma.sh
```

See [data_gen/README.md](data_gen/README.md) for the files each stage produces.

## Training

After the data pipeline has written
`data/generated/ultrafeedback/gemma-2b-it/pkd-dataset-teacher-gemma-n4`,
launch the two-GPU training configuration:

```sh
bash run_ppd.sh
```

Outputs are written below `outputs/`. Set the model path and any non-default
data path in `training_configs/gemma-2-2b-it-pd.yaml` before launching.

## Evaluation

We follow the official implementation for evaluation on AlpacaEval 2, Arena-Hard, MT-Bench and GSM8K.

* AlpacaEval 2: Please refer to the [AlpacaEval repo](https://github.com/tatsu-lab/alpaca_eval) for evaluation.

* Arena-Hard: Please refer to the [Arena-Hard-Auto repo](https://github.com/lm-sys/arena-hard-auto) for evaluation.

* MT-Bench: Please refer to the [FastChat repo](https://github.com/lm-sys/FastChat) for evaluation.

* GSM8K: Please refer to the [ZeroEval repo](https://github.com/WildEval/ZeroEval) for evaluation.


## Training Report

### Overview
This part contains training logs and comparative analysis of three preference alignment methods: SimPO, DPO, and PAD. We document the training process, implementation details, and performance metrics for each approach.

### Implementations
- **DPO**: Based on the implementation from [TRL](https://github.com/huggingface/trl)
- **SimPO**: Based on the implementation from [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO)

### Training Configuration

#### Models
- **Student Model**: Gemma-2-2B-It
- **Teacher Model**: Gemma-2-9B-It

#### Hardware
- **GPUs**: 2 × A800 (80G)

#### Training Parameters
- **Training Type**: Full parameter fine-tuning
- **Memory Optimization**: ZeRO Stage 2
- **Epochs**: 1
- **Precision**: BFloat16
- **Dataset Size**:
  - Training samples: 55,321
  - Test samples: 1,130
- **Batch Size**: 128
- **Total Training Steps**: 432
- **Maximum Sequence Length**: 2048
- **Per Device Train Batch Size**: 2
- **Per Device Evaluation Batch Size**: 2
- **Gradient Accumulation Steps**: 32
- **Evaluation Frequency**: Every 100 training steps
- **Gradient Checkpointing**: Enabled

For additional parameters, please refer to the paper or the configuration files.

### Results

| Method | GPU Hours | Alpaca-Eval 2.0 LC (%) |
|--------|-----------|---------------------------------|
| DPO    | 8.7856    | 43.77                           |
| SimPO  | 7.2672    | 44.94                           |
| PAD    | 7.2884    | 45.73                           |

You can find the training log under `gemma-log/*`.

#### Analysis
- **Training Efficiency**: PAD and SimPO require similar computational resources, while DPO demands notably more. This efficiency difference is primarily because DPO requires loading an additional reference model during training, whereas PAD and SimPO do not.
- **Performance**: PAD outperforms both SimPO and DPO in terms of win rate, which aligns with the findings reported in the submission paper.
