# TPU Smoke Test Instructions

## 🚨 CRITICAL REQUIREMENTS

### MANDATORY: Use worker=all
**⚠️ ALL commands must use `--worker=all` for TPU distributed training to work properly!**

- TPU v4 has **4 workers × 4 cores = 16 total cores**
- Distributed training requires **coordination across all workers**
- **NEVER** use `--worker=0` or single worker for training operations

### PyTorch 2.7.1 / XLA 2.7.0 Requirements
- Uses `torchrun` instead of `xla_spawn`
- Requires specific environment variables
- All workers must have identical code and dependencies

## Pre-flight Checklist

### 1. Connect to TPU VM
```bash
# Connect to your TPU VM
gcloud compute ssh oct-jepa2-v4-32 --zone=us-central2-b
```

### 2. Verify Environment (MUST use worker=all)
```bash
# Check TPU is available on ALL workers (PyTorch 2.7 compatible)
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 \
    --zone=us-central2-b \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && python -c 'import torch_xla.runtime as xr; print(\"TPU cores:\", xr.local_device_count())'"

# Verify Python environment on all workers
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 \
    --zone=us-central2-b \
    --worker=all \
    --command="which python"
# Should be: /home/layne/miniconda/envs/torch-xla/bin/python on all workers
```

### 3. Verify GCS Access (MUST use worker=all)
```bash
# Test GCS access from all workers
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 \
    --zone=us-central2-b \
    --worker=all \
    --command="gsutil ls gs://layne-tpu-code-sync/OCTdata/OCTdata/ | head -5"

# Check manifest file exists from all workers  
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 \
    --zone=us-central2-b \
    --worker=all \
    --command="gsutil ls gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest.tsv"
```

### 4. Set Environment Variables (PyTorch 2.7 Compatible)
```bash
export XLA_USE_BF16=1
export TF_CPP_MIN_LOG_LEVEL=1
export DATA_CACHE_DIR=/tmp/oct_cache
export WANDB_MODE=online

# PyTorch 2.7 specific optimizations
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"
export PJRT_DEVICE=TPU
```

### 5. Navigate to Project Directory
```bash
cd /path/to/your/project  # Update this path
# Or if code is in GCS, sync first:
# gsutil -m rsync -r gs://layne-tpu-code-sync/code/ ./oct-foundation/
```

## Run Smoke Test

### Step 1: PyTorch 2.7 Compatibility Check
Test PyTorch 2.7.1 / XLA 2.7.0 compatibility:
```bash
python tpu_pytorch27_test.py
```

### Step 1b: Quick Validation (if compatibility test passes)
Test imports and basic functionality:
```bash
python -c "
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from models.vjepa_3d import VJEPA3D
from data_setup.datasets import OCTDICOMDataset
print('✅ All imports successful')
print(f'PyTorch: {torch.__version__}')
print(f'XLA: {torch_xla.__version__}')
print(f'TPU device: {xm.xla_device()}')
print(f'TPU cores: {xr.local_device_count()}')
"
```

### Step 2: Launch Smoke Test (CRITICAL: Use worker=all)
```bash
# REMOTE execution (recommended - run from local machine)
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 \
    --zone=us-central2-b \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && bash run_tpu.sh configs/smoke_test.yaml"

# LOCAL execution (if logged into TPU VM directly)
# bash run_tpu.sh configs/smoke_test.yaml
```

### Expected Output
The smoke test should show:
```
Starting V-JEPA2 pretraining...
Config: vjepa2_smoke_test
Device count: 8
Loading data...
Creating model...
Model parameters: ~23M
Training started...
Step 1/10 - Loss: X.XXXX
Step 2/10 - Loss: X.XXXX
...
Step 10/10 - Loss: X.XXXX
✅ Smoke test completed successfully!
```

## What to Monitor

### 1. Successful Initialization
- ✅ 8 TPU cores detected
- ✅ Model created (~23M parameters)
- ✅ GCS data loading works
- ✅ Transform pipeline functional

### 2. Training Progress
- ✅ Loss computation working
- ✅ Gradients flowing
- ✅ Memory usage stable
- ✅ No OOM errors

### 3. W&B Logging
- ✅ Experiment appears in wandb
- ✅ Loss curves updating
- ✅ System metrics logged

## Troubleshooting

### 🚨 Most Common Issue: Wrong Worker Usage
**Error**: Training fails, imports work on single worker but not training
**Solution**: ALWAYS use `--worker=all` for:
- Training commands
- Dependency installation  
- Code synchronization
- Any distributed operation

### TPU-Specific Issues

**Issue: "TPU initialization failed: Operation not permitted"**
```bash
# Restart TPU (REQUIRED)
gcloud compute tpus stop oct-jepa2-v4-32 --zone=us-central2-b
gcloud compute tpus start oct-jepa2-v4-32 --zone=us-central2-b

# Wait 2-3 minutes, then test
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 --zone=us-central2-b --worker=all --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && python -c 'import torch_xla.runtime as xr; print(\"TPU cores:\", xr.local_device_count())'"
```

**Issue: "No TPU devices found"**
```bash
# Check TPU status
gcloud compute tpus describe oct-jepa2-v4-32 --zone=us-central2-b
```

**Issue: "torch_xla.distributed.xla_spawn not found"**
This is normal for PyTorch 2.7! We use `torchrun` instead.
```bash
# Verify you're using the updated run_tpu.sh
cat run_tpu.sh  # Should show "torchrun --nproc_per_node=4"
```

### Dependency & Environment Issues

**Issue: "ModuleNotFoundError" on some workers**
```bash
# Install dependencies on ALL workers
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 --zone=us-central2-b --worker=all --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && pip install -r requirements.txt"
```

**Issue: "Code not found" or "git repository missing"**
```bash
# Clone repository to ALL workers  
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 --zone=us-central2-b --worker=all --command="cd /home/layne && git clone https://github.com/Laynzzz/3d-oct-foundation-model.git"
```

**Issue: "Import errors after code changes"**
```bash
# Always pull to ALL workers after code changes
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 --zone=us-central2-b --worker=all --command="cd ~/3d-oct-foundation-model && git pull"
```

### GCS & Data Issues

**Issue: "GCS access denied"**
```bash
# Check authentication on all workers
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 --zone=us-central2-b --worker=all --command="gcloud auth list"

# Re-authenticate if needed
gcloud auth application-default login
```

**Issue: "Data loading errors"**
```bash
# Verify DICOM files exist
gsutil ls gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/*/
```

### Performance Issues

**Issue: "OOM errors"**
The smoke test config is already minimal, but if OOM occurs:
1. Reduce `per_core_batch_size` to 1 (already set)
2. Reduce `image_size` to [24, 128, 128]

### Environment Issues

**Issue: "Wrong Python path"**
```bash
# Always set PATH in commands
export PATH=/home/layne/miniconda/envs/torch-xla/bin:$PATH
```

**Issue: "Batch size warnings"**
```bash
# Check config matches TPU setup:
# global_batch_size should be divisible by (4 workers × 4 processes = 16)
# Example: global_batch_size=16, per_core_batch_size=1, grad_accum_steps=1
```

## Success Criteria

The smoke test passes if:
1. ✅ All 10 training steps complete
2. ✅ Loss is computed (value doesn't matter)
3. ✅ No crashes or OOM errors
4. ✅ W&B logging works
5. ✅ Process completes cleanly

## Next Steps After Success

If smoke test passes:
1. Run short validation (2 epochs): modify config and rerun
2. Proceed to full single-domain training
3. Scale to multi-domain training

Time estimate: 5-10 minutes for smoke test