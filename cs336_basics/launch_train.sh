#!/usr/bin/env bash
set -e

uv run cs336_basics/train_vibe.py \
  --wandb \
  --wandb_project cs336-basics \
  --wandb_run_name small-transformer \
  --steps 100 \
  --batch_size 64 \
  --profile