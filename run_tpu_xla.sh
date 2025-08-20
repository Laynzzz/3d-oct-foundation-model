#!/usr/bin/env bash
set -euo pipefail
export XLA_USE_BF16=1
export TF_CPP_MIN_LOG_LEVEL=1
export PJRT_DEVICE=TPU

# Use XLA spawn for TPU distributed training (PyTorch 2.7 compatible)
WANDB_MODE=online \
python -m torch_xla.distributed.xla_spawn \
  --num_workers=4 \
  pretraining/train.py --config $1