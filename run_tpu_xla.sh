#!/usr/bin/env bash
set -euo pipefail
export XLA_USE_BF16=1
export TF_CPP_MIN_LOG_LEVEL=1
export PJRT_DEVICE=TPU

# Use direct Python execution for PyTorch 2.7 XLA multiprocessing
WANDB_MODE=disabled \
python pretraining/train.py --config $1