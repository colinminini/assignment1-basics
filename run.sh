#!/usr/bin/env bash
set -e

uv run cs336_basics/train_vibe.py \
  --wandb \
  --profile \
  --no_wandb_watch \
  --wandb_run_name lr_e-4-small-transformer \
  --steps 100 \
  --batch_size 32 \
  --with_gradient_clipping \
  --lr 1e-4
#  --with_cosine_lr \