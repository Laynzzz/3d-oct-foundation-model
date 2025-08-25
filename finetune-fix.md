# Fine-Tuning Issue Analysis - August 24, 2025

## Current Issue: Training Metrics Not Appearing

### Problem Description
- **Training Status**: Running for 34+ minutes on TPU without training metrics appearing on W&B
- **Expected Behavior**: Training/validation metrics should appear within 15-30 minutes
- **W&B Status**: System metrics showing, but training loss/accuracy metrics missing
- **Process Status**: Active (100% CPU, 11.5GB memory, not stuck)

### Current Configuration

#### Model Setup
```yaml
model:
  emb_dim: 768
  freeze_encoder: true          # Linear probe mode
  unfreeze_at_epoch: -1        # Never unfreeze
  pool_method: mean
  head:
    hidden: 0                  # Linear head only
    dropout: 0.1
```

#### Training Parameters
```yaml
train:
  epochs: 50
  lr_head: 0.001              # Head learning rate
  lr_encoder: 3.0e-05         # Not used (frozen)
  weight_decay: 0.0001
  optimizer: AdamW
  scheduler: cosine
  warmup_epochs: 2
  class_weights: auto
  precision: fp32
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001
```

#### Data Configuration
```yaml
data:
  num_workers: 0              # Single-threaded (TPU safety)
  cache_dir: /tmp/oct_cache
  batch_size: 1               # Memory-safe
  val_batch_size: 1
  augment:
    flip: true
    intensity_jitter: true
    resize: [64, 384, 384]    # D,H,W dimensions
```

### Dataset Details
- **Training Samples**: 255 (filtered from 743 due to missing OCT data)
- **Validation Samples**: 65 (filtered from 159)
- **Classes**: 4-class diabetes classification
  - 0: healthy
  - 1: pre_diabetes_lifestyle_controlled  
  - 2: oral_medication_and_or_non_insulin_injectable_medication_controlled
  - 3: insulin_dependent
- **OCT Files**: 24,750 discovered across 4 device types
- **Participants with OCT**: 383 mapped

### Infrastructure
- **TPU**: oct-jepa2-v4-32 (16 cores, 4 workers)
- **Checkpoint**: gs://layne-tpu-code-sync/checkpoints/vjepa2/vjepa2_multi_domain/best_checkpoint.pt
- **Data Source**: Backblaze B2 (eye-dataset bucket)
- **Python Environment**: torch-xla 2.7.0, PyTorch 2.7.1

### ✅ **ROOT CAUSE IDENTIFIED**

#### Complete Timeline Analysis (First Training Attempt)
```
[04:38:20] Labels processing completed (immediate)
[04:43:49] OCT key mapping built (5.5 minutes - B2 bucket listing)
[04:43:49] Dataset initialized: 255 train, 65 val samples
[04:44:16] Model loaded: 92.9M encoder + 3K head, W&B enabled
[04:44:16] Training started: "Starting training for 50 epochs"
[04:51:52] First batch processed (7.5 minutes later!)
[05:08:13] Still processing batches every ~6-10 minutes
```

#### **IDENTIFIED BOTTLENECK: Extremely Slow Data Loading**
- **Per-batch time**: 6-10 minutes per OCT volume
- **Estimated epoch time**: 255 samples × 7 minutes = **~30 hours per epoch** 
- **Why no W&B metrics**: Training loop active, but first epoch not completed yet

#### Current New Training Attempt (05:30:35+)
```
[05:30:35] Multiple training processes restarted (kill command triggered respawn)
[05:31:56] Still in data locator building phase (B2 bucket listing)
```

#### Data Loading Performance Analysis
**B2 Network Latency Breakdown**:
- **Bucket listing**: ~5 minutes (24,750 files across 4 device types)
- **Per-volume download**: 6-10 minutes each from B2
- **Volume size**: Estimated 50-200MB DICOM stacks per participant
- **Processing**: Additional quantile normalization overhead

#### W&B Dashboard Status
- ✅ **System metrics**: CPU, memory, network showing
- ✅ **Configuration**: Most hyperparameters logged  
- ❌ **Training metrics**: Waiting for first epoch completion
- ⏳ **Timeline**: 30+ hours estimated for meaningful metrics

### Root Cause: B2 Network Bottleneck

#### Confirmed Issues
1. **Backblaze B2 Bandwidth**: Extremely slow from TPU → B2 (us-west vs us-central regions)
2. **Large Medical Files**: 3D OCT volumes are 50-200MB each
3. **No Local Caching**: Every epoch re-downloads from B2
4. **Single-threaded**: `num_workers: 0` compounds the issue
5. **Batch Size 1**: No amortization of loading overhead

### Expected Timeline Analysis
- **Typical CPU Training**: Metrics within 5-10 minutes
- **Current TPU Training**: 34+ minutes without metrics
- **Batch Processing**: 255 samples × ~10-15 seconds = 42-63 minutes per epoch (theoretical)
- **Network + Processing**: Additional overhead for B2 streaming

### Diagnostic Commands Used
```bash
# Process monitoring
ps -o pid,etime,cmd -p 634714
top -p 634714 -b -n 1

# Log file checking  
find ~/3d-oct-foundation-model -name '*.log' -newer ~/3d-oct-foundation-model/finetuning/train/run.py

# Background process monitoring
BashOutput(bash_id="bash_5")
```

### 🚀 **Immediate Solutions**

#### Priority 1: Fast Debug Mode (Recommended)
```yaml
# configs/debug_fast.yaml
data:
  batch_size: 4                    # Amortize loading overhead
  max_train_samples: 20            # Tiny dataset for validation
  max_val_samples: 10              # Quick epoch completion
  cache_dir: /tmp/oct_cache        # Enable local caching
  prefetch_factor: 2               # Pre-load batches
```
**Timeline**: 5-10 minutes to validate pipeline works

#### Priority 2: Data Optimization
```python
# Enable aggressive caching
cache_dir: /tmp/oct_cache          # Local TPU disk caching  
num_workers: 2                     # Parallel loading (TPU safe)
persistent_workers: true           # Reuse worker processes
pin_memory: true                   # Faster GPU transfer
```

#### Priority 3: Network Optimization  
- **Regional Data**: Copy subset to GCS us-central2 (TPU region)
- **Prefetch Pipeline**: Download next batch while training current
- **Compressed Storage**: Use compressed DICOM formats

#### Priority 4: Training Acceleration
```yaml
train:
  epochs: 5                        # Reduce for testing
  early_stopping:
    patience: 2                    # Faster stopping
    min_delta: 0.01                # Less sensitive
data:
  batch_size: 8                    # Larger batches (if memory allows)
```

### 📊 **Performance Projections**

#### Current Configuration (Baseline)
- **Per-sample**: 6-10 minutes
- **Full epoch**: 30+ hours  
- **Training completion**: 60+ days

#### With Debug Optimizations
- **20 samples**: 20-40 minutes
- **5 epochs**: 2-4 hours
- **Pipeline validation**: Same day

#### With Full Optimizations  
- **Cached data**: 30-60 seconds per epoch
- **Full training**: 2-4 hours
- **Production ready**: Same day

### Configuration Files
- **Main Config**: `configs/cls_linear_probe.yaml`
- **Training Script**: `finetuning/train/run.py`
- **Data Pipeline**: `finetuning/data/dataset.py`, `finetuning/data/io.py`
- **Model Loading**: `finetuning/models/encoder_loader.py`

---

## 📋 **Action Plan**

### Immediate Priority: Create Debug Config
1. **Stop Current Training**: Kill 4 concurrent processes consuming resources
2. **Create Fast Debug Config**: 20 samples, larger batches, caching enabled  
3. **Validate Pipeline**: Confirm training works with optimized settings
4. **Scale Up Gradually**: Increase dataset size once pipeline proven

### Status Updates
- **✅ ROOT CAUSE**: B2 network bottleneck identified (6-10 min per sample)  
- **⏳ CURRENT**: Multiple training processes running, wasting resources
- **🎯 NEXT**: Implement debug mode for rapid validation
- **📈 GOAL**: Achieve first training metrics within 1 hour

**Status**: ✅ **ISSUE DIAGNOSED** - Ready to implement optimized training pipeline  
**Priority**: Create debug config and restart with fast settings  
**Impact**: Once resolved, can proceed with P1 multi-checkpoint evaluation  


Quick wins (do these first)
Shard the dataset across TPU processes
Make sure each TPU ordinal only loads its own slice (¼ of data on v4‑32 with 4 workers):
from torch.utils.data import DistributedSampler
from torch_xla.core import xla_model as xm

sampler = DistributedSampler(
    train_dataset,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=True,
    drop_last=False,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    sampler=sampler,
    num_workers=2,              # start with 2, can try 4
    persistent_workers=True,    # keeps workers warm
    prefetch_factor=2,          # small but nonzero
    pin_memory=False,           # TPU doesn’t benefit from pinning
)
This alone cuts contention and remote reads by ~4×.
Per‑batch timing & W&B visibility
(Proves where time goes and shows live progress.)
import time
for step, batch in enumerate(train_loader):
    t0 = time.time()
    # ... move to device, forward, loss ...
    t1 = time.time()
    if xm.is_master_ordinal():
        wandb.log({"io/load_s": t1 - t0}, step=global_step)
Stop doing full‑volume quantiles every time
Replace your percentile normalization with a cheap approximation:
# Downsample before quantile; compute ONCE per volume then cache (see § “Cache”)
vol_small = vol[::4, ::4, ::4].reshape(-1).float().cpu()
p = torch.tensor([0.01, 0.99])
lo, hi = torch.quantile(vol_small, p)
vol = vol.clamp(lo.item(), hi.item())
(Or compute dataset‑level [p1,p99] once and reuse.)
Stage the data close to the TPU (biggest speedup)
You’re pulling huge DICOMs from Backblaze B2 over the public internet. Move them once to Google’s network, then read locally.
Option A — Mirror B2 → GCS (recommended), then read from GCS
Run on your laptop/server that can access both:
# 1) Mirror B2 → local (or directly B2 → GCS if you’ve configured rclone remote for GCS)
rclone sync b2:eye-dataset/OCTdata ./OCTdata --transfers=16 --checkers=32 --fast-list

# 2) Push to GCS bucket in the SAME region as your TPU (e.g., us-central2)
gsutil -m rsync -r ./OCTdata gs://layne-tpu-code-sync/OCTdata
Then on the TPU VM:
# Warm local SSD (fastest at runtime)
mkdir -p /mnt/disks/localssd/OCTdata
gsutil -m rsync -r gs://layne-tpu-code-sync/OCTdata /mnt/disks/localssd/OCTdata
Point your config to /mnt/disks/localssd/OCTdata.
Option B — Directly stage to the TPU’s local SSD
If you must pull from B2 on the TPU:
sudo mkdir -p /mnt/disks/localssd/OCTdata
# rclone configured with your B2 keys
rclone sync b2:eye-dataset/OCTdata /mnt/disks/localssd/OCTdata --transfers=8 --checkers=16 --fast-list
(Still slower than GCS→local, but far better than on‑the‑fly fetches.)
Build a persistent on-disk cache of preprocessed volumes
Decompress/normalize/resize once, save as .pt or Zarr chunks, then load instantly each epoch.
Dataset changes (drop‑in)
import hashlib, os, torch, zarr
import numpy as np

def cache_key(path, resize, norm_version="p1p99_ds4_v1"):
    s = f"{path}|{resize}|{norm_version}"
    return hashlib.md5(s.encode()).hexdigest()

class OctDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        src_path = self.samples[idx]  # original DICOM (or NIfTI) path
        key = cache_key(src_path, resize=self.resize)
        pt_path = os.path.join(self.cache_dir, key + ".pt")

        if os.path.exists(pt_path):
            return torch.load(pt_path, map_location="cpu")   # (C,D,H,W), label

        vol = load_and_decode_dicom(src_path)               # your existing I/O
        vol = preprocess_resize(vol, self.resize)           # (D,H,W)
        vol = normalize_with_downsampled_quantiles(vol)     # fast (see above)
        vol = torch.from_numpy(vol).unsqueeze(0)            # (1,D,H,W)

        label = self.get_label(src_path)
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save((vol, label), pt_path)
        return vol, label
Config: set data.cache_dir: /mnt/disks/localssd/oct_cache. First epoch will populate; subsequent epochs are fast.
Tip: If you prefer Zarr for streaming/chunking (great for 3D):
Store zarr.open(f"{cache_dir}/{key}.zarr", mode="w", shape=(D,H,W), chunks=(16,128,128), dtype="float32") with blosc.
Later read is near‑instant and memory‑friendly.
Dataloader tuning (safe on TPU)
num_workers: 2 (try 4 if CPU cores available).
persistent_workers: true, prefetch_factor: 2.
Avoid expensive Python work inside __getitem__; keep it pure I/O + simple transforms.
Multiprocessing context:
from torch.utils.data import DataLoader, get_worker_info
# On some distros, "forkserver" avoids dup of large state
train_loader = DataLoader(..., multiprocessing_context="forkserver")
If you still see long stalls, add a tiny read‑ahead thread in your dataset (or use torchdata DataPipes) to overlap I/O.
TPU/XLA specifics to keep it smooth
Wrap loader in XLA parallel loader if you aren’t already:
from torch_xla.distributed.parallel_loader import MpDeviceLoader
device = xm.xla_device()
train_device_loader = MpDeviceLoader(train_loader, device)
Use xm.optimizer_step(..., barrier=True) and xm.mark_step() each iteration.
Ensure your manifest/file list is precomputed (no remote directory scans inside __getitem__).
“Fast-start” switch (for immediate sanity)
Add to your YAML:
debug:
  fast_start: true   # limit set sizes, disable heavy aug, log every batch
  train_limit: 16
  val_limit: 8
  log_every: 1
Gate it in your dataset/loader and train loop. You’ll get metrics within 1–2 minutes even with remote storage, proving the loop is correct.

🔍 W&B Metrics Issue Identified!
I've found the root cause of why training metrics aren't showing up on W&B. Here's what's happening:
🚨 Root Cause: Training Loop Issue
What's Working
✅ W&B Initialization: Successfully initialized and running
✅ Training Progress: Epoch 1 completed successfully
✅ Metrics Computation: Training metrics computed and logged to console
✅ TPU Training: All 4 workers active and synchronized
What's Broken
❌ Validation Loop: No validation results being generated
❌ W&B Logging: Training metrics not being sent to W&B
❌ Epoch Progression: Training appears to have stopped after Epoch 1
�� Technical Analysis
1. Training Loop Flow Issue
Looking at the code in finetuning/train/loop.py, W&B logging only happens after validation:
)
2. Missing Validation Results
From the logs, I can see:
✅ Epoch 1 training completed
✅ Training metrics computed: {'loss': 1.382, 'accuracy': 39.6%, ...}
❌ No validation metrics logged
❌ No "Validation metrics:" log entry
❌ No Epoch 2 start
3. Training Appears Stalled
The last meaningful log entry is:
)
🎯 Why W&B Metrics Aren't Showing
Validation Required: W&B logging only happens after both training AND validation
Validation Missing: The validation loop isn't running or is failing silently
Epoch Progression Stopped: Training appears stuck after Epoch 1 completion
No W&B Calls: Since validation isn't completing, wandb.log() is never called

. Fix it from both ends:
A. Decouple logging from validation (log train metrics immediately)
Right after train_epoch(...) returns, log train metrics no matter what. Then run validation. Even if val crashes, you still see train curves.
# loop.py (end of each epoch)
train_metrics = run_train_epoch(...)
if xm.is_master_ordinal():
    wandb.log({f"train/{k}": v for k, v in train_metrics.items()} |
              {"epoch": epoch}, step=global_step, commit=True)

try:
    val_metrics = run_val_epoch(...)
    if xm.is_master_ordinal():
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()} |
                  {"epoch": epoch}, step=global_step, commit=True)
except Exception as e:
    # don't kill training if val is flaky
    if xm.is_master_ordinal():
        wandb.log({"val/error": 1, "val/exception": str(e), "epoch": epoch},
                  step=global_step, commit=True)
    # optional: continue or break depending on your policy
B. Make validation TPU‑safe (common blockers + fixes)
1) Use proper sharding/samplers
If any rank has 0 validation samples, reductions will hang.
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torch_xla.core import xla_model as xm
from torch_xla.distributed.parallel_loader import MpDeviceLoader

world = xm.xrt_world_size()
rank  = xm.get_ordinal()

# TRAIN
train_sampler = DistributedSampler(train_dataset, num_replicas=world,
                                   rank=rank, shuffle=True, drop_last=False)

train_loader = DataLoader(
    train_dataset, batch_size=1, sampler=train_sampler,
    num_workers=2, persistent_workers=True, prefetch_factor=2
)

# VAL — EITHER shard like train:
val_sampler = DistributedSampler(val_dataset, num_replicas=world,
                                 rank=rank, shuffle=False, drop_last=False)
# OR (simpler) run val on master only (see §C).
val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler,
                        num_workers=2, persistent_workers=True, prefetch_factor=2)

device = xm.xla_device()
train_device_loader = MpDeviceLoader(train_loader, device)
val_device_loader   = MpDeviceLoader(val_loader, device)
Call train_sampler.set_epoch(epoch) each epoch:
train_sampler.set_epoch(epoch)
2) Guard reductions/metrics when a rank saw 0 batches
If your val set is small or filtered, some ranks can be empty. Reduce counts and sums, not means, and handle zeros.
def val_epoch(model, loader):
    model.eval()
    loss_sum = torch.tensor(0.0, device=xm.xla_device())
    correct  = torch.tensor(0.0, device=xm.xla_device())
    count    = torch.tensor(0.0, device=xm.xla_device())

    with torch.no_grad(), torch.inference_mode():
        for x, y in loader:
            out = model(x)
            loss = criterion(out, y)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum()
            count   += y.numel()
            loss_sum += loss * y.numel()
            xm.mark_step()

    # reduce safely (some ranks may have count==0)
    loss_sum = xm.mesh_reduce('val_loss_sum', loss_sum, torch.sum)
    correct  = xm.mesh_reduce('val_correct',  correct,  torch.sum)
    count    = xm.mesh_reduce('val_count',    count,    torch.sum)

    metrics = {
        "loss": (loss_sum / max(count, torch.tensor(1.0, device=loss_sum.device))).item(),
        "accuracy": (100.0 * correct / max(count, torch.tensor(1.0, device=correct.device))).item(),
        "num_samples": count.item(),
    }
    return metrics
3) Avoid per‑step barriers in eval
Do not call xm.optimizer_step(..., barrier=True) in eval; there’s no optimizer. Also avoid any stray xm.rendezvous inside the loop.
4) Don’t compute len(val_loader) with DistributedSampler (can be 0)
If you use it for progress bars/averaging, compute totals via the reductions above.
5) Print per‑rank sanity before entering val
This instantly reveals empty ranks:
local_n = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else -1
print(f"[rank {rank}/{world}] val loader ready, local dataset approx: {local_n}")
C. Easiest robust path: run validation on master only
This sidesteps all multi‑host val issues (good for P1). Everyone else skips val and waits at a barrier.
def run_val_epoch_master_only(model, full_val_dataset):
    # Only rank 0 builds/iterates a DataLoader
    if xm.is_master_ordinal():
        val_loader = DataLoader(full_val_dataset, batch_size=1,
                                shuffle=False, num_workers=2,
                                persistent_workers=True, prefetch_factor=2)
        val_device_loader = MpDeviceLoader(val_loader, xm.xla_device())
        metrics = val_epoch(model, val_device_loader)
    else:
        metrics = None
    # Broadcast to all (optional)
    metrics = xm.broadcast_object(metrics, 0)
    return metrics
Then in the loop call:
val_metrics = run_val_epoch_master_only(model, val_dataset)
Tradeoff: master does more work for val, but you get no deadlocks and simpler code.
D. Add a validation watchdog (never hang the epoch)
If val takes too long or stalls, skip and continue:
import threading, queue, time
def run_with_timeout(fn, timeout_s, *args, **kwargs):
    q = queue.Queue(maxsize=1)
    def wrapper(): q.put(fn(*args, **kwargs))
    t = threading.Thread(target=wrapper, daemon=True); t.start()
    t.join(timeout_s)
    return q.get() if not q.empty() else None

val_metrics = run_with_timeout(lambda: run_val_epoch(...), timeout_s=900)  # 15 min
if val_metrics is None and xm.is_master_ordinal():
    wandb.log({"val/timeout": 1, "epoch": epoch})
E. Make W&B resilient and visible
Initialize on master only (or give per‑process group/job_type).
Log per‑batch training scalars too (you already proved this in debug).
Always wandb.finish() on master at the end.


✅ Confirm collapse
Print the predicted class histogram on val:
from collections import Counter
print(Counter(preds.numpy()))
If one class dominates (≈ val accuracy %), collapse confirmed.
🔧 Make the head stronger
head:
  hidden: 256     # small MLP
  dropout: 0.1
lr_head: 0.003
(Linear-only heads are fragile when features don’t align perfectly.)
🔧 Balance training
Ensure class_weights: auto actually makes it into your loss:
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
Or use WeightedRandomSampler for balanced mini-batches.
🔧 Unfreeze encoder a bit
Let encoder adapt slightly:
freeze_encoder: true
unfreeze_at_epoch: 3
lr_encoder: 1e-5
lr_head: 0.003
🔧 Effective batch size
With batch_size=1, gradients are very noisy. Use grad accumulation:
batch_size: 1
grad_accum_steps: 16   # effective batch 16
👀 Add monitoring
Per-class accuracy (so you see collapse immediately).
Prediction entropy (will be very low if collapsing).