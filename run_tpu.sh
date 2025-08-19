#!/usr/bin/env bash
set -euo pipefail
export XLA_USE_BF16=1
export TF_CPP_MIN_LOG_LEVEL=1
WANDB_MODE=online \
python -m torch_xla.distributed.xla_spawn --num_workers=8 \
  pretraining/train.py --config $1