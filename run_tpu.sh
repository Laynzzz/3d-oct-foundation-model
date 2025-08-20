#!/usr/bin/env bash
set -euo pipefail
export XLA_USE_BF16=1
export TF_CPP_MIN_LOG_LEVEL=1
export PJRT_DEVICE=TPU

# PyTorch 2.7 uses torchrun for distributed training
WANDB_MODE=online \
torchrun --nproc_per_node=4 \
  pretraining/train.py --config $1