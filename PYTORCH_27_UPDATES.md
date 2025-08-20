# PyTorch 2.7.1 / XLA 2.7.0 Compatibility Updates

## Summary of Changes Made

The codebase has been updated to ensure compatibility with PyTorch 2.7.1 and XLA 2.7.0. Here are the key changes:

### 1. XLA API Updates

#### Import Changes
```python
# OLD (PyTorch 2.5/2.6)
import torch_xla.test.test_utils as test_utils

# NEW (PyTorch 2.7)
import torch_xla.runtime as xr
```

#### Device Count API
```python
# OLD
device_count = xm.xrt.device_count()

# NEW
device_count = xr.device_count()
```

#### World Size API
```python
# OLD
num_replicas = xm.xrt_world_size()

# NEW
num_replicas = xr.world_size()
```

#### Runtime Initialization Check
```python
# OLD
torch_xla._XLAC._xla_runtime_is_initialized()

# NEW
torch_xla._XLAC.is_runtime_initialized()
```

### 2. Transform Function Names
```python
# OLD
from data_setup.transforms import get_train_transforms, get_validation_transforms

# NEW
from data_setup.transforms import create_pretraining_transforms, create_validation_transforms
```

### 3. Dataset API Parameter Names
```python
# OLD
create_file_lists(..., strategy=config.list_strategy, ...)

# NEW
create_file_lists(..., list_strategy=config.list_strategy, ...)
```

### 4. Environment Variables
Added PyTorch 2.7 specific optimizations:
```bash
export PJRT_DEVICE=TPU
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"
```

## Files Modified

### Core Training Files
1. **`pretraining/train.py`**
   - Updated XLA imports
   - Fixed transform function calls
   - Updated distributed sampler APIs
   - Fixed dataset parameter names

2. **`tpu_quick_test.py`**
   - Updated XLA runtime APIs
   - Added PyTorch 2.7 compatibility checks

### New Files Added
1. **`tpu_pytorch27_test.py`**
   - Comprehensive PyTorch 2.7 / XLA 2.7 compatibility test
   - Tests all major APIs used in training
   - Validates model, data loading, and optimization

2. **`PYTORCH_27_UPDATES.md`** (this file)
   - Documentation of all changes made

### Updated Documentation
1. **`run_tpu_smoke_test.md`**
   - Added PyTorch 2.7 compatibility check step
   - Updated environment variables
   - Added version-specific validation

## Testing Sequence for PyTorch 2.7

### Step 1: Compatibility Test
```bash
python tpu_pytorch27_test.py
```
Expected output: All 5 tests should pass

### Step 2: Quick Import Test
```bash
python -c "
import torch
import torch_xla
import torch_xla.runtime as xr
print(f'PyTorch: {torch.__version__}')
print(f'XLA: {torch_xla.__version__}')
print(f'TPU cores: {xr.device_count()}')
"
```

### Step 3: Smoke Test
```bash
bash run_tpu.sh configs/smoke_test.yaml
```

## Key Compatibility Notes

### What Works Differently in PyTorch 2.7
1. **Runtime APIs**: New `torch_xla.runtime` module centralizes device/world management
2. **Autocast**: Better BF16 support with XLA device type specification
3. **Distributed Training**: Improved synchronization and communication
4. **Memory Management**: Enhanced garbage collection and memory optimization

### Backward Compatibility
- Most XLA APIs maintain backward compatibility
- `xm.xla_device()` and `xm.get_ordinal()` still work as before
- `xm.optimizer_step()` and `xm.mark_step()` unchanged

### Performance Improvements in 2.7
- Better BF16 handling on TPU
- Improved compilation caching
- Enhanced distributed training performance
- Better memory usage patterns

## Verification Checklist

Before running the smoke test, ensure:
- [ ] PyTorch version is 2.7.1 (`python -c "import torch; print(torch.__version__)"`)
- [ ] XLA version is 2.7.0 (`python -c "import torch_xla; print(torch_xla.__version__)"`)
- [ ] TPU compatibility test passes (`python tpu_pytorch27_test.py`)
- [ ] Environment variables are set correctly
- [ ] All imports work without errors

## Expected Performance
With PyTorch 2.7.1 and XLA 2.7.0, you should see:
- Faster compilation times
- Better memory utilization
- More stable training on TPU
- Improved gradient accumulation handling

The smoke test should complete in 5-10 minutes with proper loss computation and W&B logging.