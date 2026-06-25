#!/usr/bin/env bash
set -e

uv run cs336_basics/train_vibe.py \
  --wandb \
  --profile \
  --no_wandb_watch \
  --wandb_run_name lr_3e-3-small-transformer \
  --device 'mps' \
  --steps 1000 \
  --batch_size 32 \
  --with_gradient_clipping \
  --lr 3e-3 \
  --lr_min 1e-4 \
  --lr_max 1e-3 \
  --T_warmup 10 \
  --T_c 1000 \
  --with_torch_compile \
#  --with_cosine_lr \
#  --mixed_precision \