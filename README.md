# Natural GaLore

This repo contains the pre-release version of Natural GaLore algorithm

## Installation

### Install Natural GaLore optimizer

In Python 3.10 environment, install using pip inside the code folder:

```bash
pip install .
```

## Usage

### Save optimizer memory using Natural GaLore optimizers

```python
from natural_galore import SubSpaceAdamW
# define param groups as galore_params and non_galore_params
param_groups = [{'params': non_galore_params}, 
                {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}]
optimizer = SubSpaceAdamW(param_groups, lr=0.01)
```

## Benchmark 1: Pre-Training LLaMA on C4 dataset

`pytorchrun_main.py` is the main script for training LLaMA models on C4 with GaLore. Our benchmark scripts for various sizes of models are in `scripts/benchmark_c4` folder.
For example, to train a 60m model on C4, do the following:

```bash
# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node
python pytorchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer natural_galore_adamw
```

## Benchmark 2: Fine-Tuning RoBERTa on GLUE tasks

`run_glue.py` is the main script for fine-tuning RoBERTa models on GLUE tasks with Natural GaLore. An example script is shown below:

```bash
python run_glue.py \
    --model_name_or_path roberta-base \
    --task_name mrpc \
    --enable_galore \
    --lora_all_modules \
    --max_length 512 \
    --seed=1234 \
    --lora_r 4 \
    --galore_scale 4 \
    --per_device_train_batch_size 16 \
    --update_proj_gap 500 \
    --learning_rate 3e-5 \
    --num_train_epochs 30 \
    --output_dir results/ft/roberta_base/mrpc
```
