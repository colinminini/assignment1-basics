#!/usr/bin/env bash
set -e

uv run cs336_basics/train_vibe.py \
  --wandb \
  --profile \
  --no_wandb_watch \
  --wandb_run_name torch_compile-small-transformer \
  --device 'mps' \
  --steps 100 \
  --batch_size 32 \
  --with_gradient_clipping \
  --with_cosine_lr \
  --lr 1e-3 \
  --lr_min 1e-3 \
  --lr_max 0.5e-2 \
  --T_warmup 20 \
  --T_c 100 \
  --torch_compile \
#  --mixed_precision \