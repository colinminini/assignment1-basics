#!/usr/bin/env bash
set -e

# uv run cs336_basics/train_vibe.py \

# Each line is one flag; comment any out. The array is joined into a single
# --args="..." string because main(args: str) takes one option, not many.
ARGS=(
  --train_file_path "/data/owt-train.npy"
  --val_file_path "/data/owt-valid.npy"
  --out_checkpoint_path "/ckpt/owt-run_"
  --wandb
  --wandb_project "cs336-basics"
  --no_wandb_watch
  --wandb_run_name "owt-B200-full_run-transformer-22M"
  --device "cuda"
  --steps 40000
  --val_and_log_every 1000
  --checkpoint_every 5000
  --batch_size 32
  --lr_min 1e-4
  --lr_max 1e-3
  --T_warmup 800
  --T_c 40000
  --with_gradient_clipping
  --with_torch_compile
  --with_cosine_lr

#  --profile
#  --lr 3e-3
#  --with_mixed_precision
)

modal run cs336_basics/train_vibe_modal.py::main --args="${ARGS[*]}"

# 22M parameter run on 327.68M tokens (40k steps at batch_size = 32 & context_len = 256)
# T_warmup should be around 1-2% of total steps
# T_c should be the final step
# lr/bacth size come from grid search (20 experiments)
# lr_max = lr from grid search
# lr_min = 1/10 * lr_max