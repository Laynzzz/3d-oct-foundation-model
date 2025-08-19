# TPU Smoke Test Instructions

## Pre-flight Checklist

### 1. Connect to TPU VM
```bash
# Connect to your TPU VM
gcloud compute ssh oct-jepa2-v4-32 --zone=us-central2-b
```

### 2. Verify Environment
```bash
# Check TPU is available
python -c "import torch_xla; print('TPU cores:', torch_xla.xrt.device_count())"

# Verify Python environment
which python
# Should be: /home/layne/miniconda/envs/torch-xla/bin/python

# Activate environment if needed
conda activate torch-xla
```

### 3. Verify GCS Access
```bash
# Test GCS access
gsutil ls gs://layne-tpu-code-sync/OCTdata/OCTdata/

# Check manifest file exists
gsutil ls gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest.tsv

# Verify DICOM files are expanded
gsutil ls gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/topcon_triton/ | head -5
```

### 4. Set Environment Variables
```bash
export XLA_USE_BF16=1
export TF_CPP_MIN_LOG_LEVEL=1
export DATA_CACHE_DIR=/tmp/oct_cache
export WANDB_MODE=online
```

### 5. Navigate to Project Directory
```bash
cd /path/to/your/project  # Update this path
# Or if code is in GCS, sync first:
# gsutil -m rsync -r gs://layne-tpu-code-sync/code/ ./oct-foundation/
```

## Run Smoke Test

### Step 1: Quick Validation
Test imports and basic functionality:
```bash
python -c "
import torch_xla.core.xla_model as xm
from models.vjepa_3d import VJEPA3D
from data_setup.datasets import OCTDICOMDataset
print('✅ All imports successful')
print(f'TPU device: {xm.xla_device()}')
print(f'TPU cores: {xm.xrt.device_count()}')
"
```

### Step 2: Launch Smoke Test
```bash
# Run the TPU smoke test
bash run_tpu.sh configs/smoke_test.yaml
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

### Common Issues & Solutions

**Issue: "No TPU devices found"**
```bash
# Check TPU status
gcloud compute tpus describe oct-jepa2-v4-32 --zone=us-central2-b
# Restart if needed
gcloud compute tpus stop oct-jepa2-v4-32 --zone=us-central2-b
gcloud compute tpus start oct-jepa2-v4-32 --zone=us-central2-b
```

**Issue: "GCS access denied"**
```bash
# Check authentication
gcloud auth list
# Re-authenticate if needed
gcloud auth application-default login
```

**Issue: "OOM errors"**
The smoke test config is already minimal, but if OOM occurs:
1. Reduce `per_core_batch_size` to 1 (already set)
2. Reduce `image_size` to [24, 128, 128]

**Issue: "Import errors"**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Issue: "Data loading errors"**
Check if dataset expansion completed:
```bash
# Verify DICOM files exist
gsutil ls gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/*/
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