#!/usr/bin/env bash
set -euo pipefail

# ===== edit these =====
DEVICE="cuda"                     # cuda on Modal (was mps locally)
STEPS=100
BATCH_SIZE=32
CONTEXT_LENGTH=""                 # empty = use config default
LR=1e-3                           # no-op while COSINE_LR=1 (schedule overrides it)
WEIGHT_DECAY=""                   # empty = use config default

# cosine schedule (used when COSINE_LR=1)
LR_MIN=1e-3                       # floor; was 1e-3 > LR_MAX, which looked swapped
LR_MAX=0.5e-2                       # peak
T_WARMUP=20
T_C=100

TRAIN_FILE="/data/TinyStoriesV2-GPT4-train.npy"      # path inside the Modal volume mount
VAL_FILE="/data/TinyStoriesV2-GPT4-valid.npy"
CKPT_PREFIX="/ckpt/run_"          # save_checkpoint appends "{step}.pt"

WANDB_PROJECT="cs336-basics"
WANDB_RUN_NAME="modal_L4-torch_compile-small-transformer"   # empty = wandb auto-name

# toggles: 1 = on, 0 = off
WANDB=1
NO_WANDB_WATCH=1
PROFILE=1
COSINE_LR=1
GRAD_CLIP=1
TORCH_COMPILE=1
MIXED_PRECISION=0                 # leave off until AdamW fp32-master / logits.float() is confirmed

DETACH=""                         # set to "--detach" for long runs (survives disconnect)
# ======================

ARGS=(
  --device "$DEVICE"
  --steps "$STEPS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --train_file_path "$TRAIN_FILE"
  --val_file_path "$VAL_FILE"
  --out_checkpoint_path "$CKPT_PREFIX"
)

if [ -n "$CONTEXT_LENGTH" ]; then ARGS+=( --context_length "$CONTEXT_LENGTH" ); fi
if [ -n "$WEIGHT_DECAY" ];   then ARGS+=( --weight_decay "$WEIGHT_DECAY" ); fi

if [ "$WANDB" = 1 ]; then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  if [ -n "$WANDB_RUN_NAME" ]; then ARGS+=( --wandb_run_name "$WANDB_RUN_NAME" ); fi
  if [ "$NO_WANDB_WATCH" = 1 ]; then ARGS+=( --no_wandb_watch ); fi
fi

if [ "$PROFILE" = 1 ]; then ARGS+=( --profile ); fi
if [ "$GRAD_CLIP" = 1 ]; then ARGS+=( --with_gradient_clipping ); fi

if [ "$COSINE_LR" = 1 ]; then
  ARGS+=( --with_cosine_lr --lr_min "$LR_MIN" --lr_max "$LR_MAX" --T_warmup "$T_WARMUP" --T_c "$T_C" )
fi

if [ "$MIXED_PRECISION" = 1 ]; then ARGS+=( --mixed_precision ); fi
if [ "$TORCH_COMPILE" = 1 ];   then ARGS+=( --torch_compile ); fi

echo "modal run cs336_basics/train_vibe_modal.py --args=\"${ARGS[*]}\""
modal run $DETACH cs336_basics/train_vibe_modal.py --args="${ARGS[*]}"