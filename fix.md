# 3D OCT Foundation Model - Stabilization Progress

## Status: ✅ STEP 3 COMPLETE - SCHEDULER TESTING IN PROGRESS

### Problem Summary
Training was experiencing NaN losses at step ~514 and LR=0 issues that caused instability and crashes.

### Solution: A-prime Fix + Graduated Stabilization Plan

---

## ✅ COMPLETED STEPS

### Step 1: A-prime Minimal Fix (COMPLETED ✅)
**Status**: Successfully passed step 514 and reached 1113+ steps stable

**Implementation**:
- ✅ Disabled TorchInductor (`TORCH_COMPILE_DISABLE=1`)  
- ✅ Constant LR without touching each step
- ✅ XLA-optimized gradient handling (no loops, no all_reduce)
- ✅ Master-only W&B logging
- ✅ Minimal dataloader settings (`workers: 0`, `persistent_workers: false`, `pin_memory: false`)

**Results**: Training proceeded past step 514 with no NaN losses, reached 1113+ steps stable.

### Step 2: Verify XLA-friendly Metrics (COMPLETED ✅)
**Status**: Current implementation verified as XLA-optimized

**Verification Results**:
- ✅ Grad norm: Single fused `clip_grad_norm_` call
- ✅ No cross-replica ops: No `xm.all_reduce` or `xm.mesh_reduce`
- ✅ Master-only logging: W&B only on `xm.is_master_ordinal()`
- ✅ No parameter loops: No manual iteration over model parameters
- ⚠️ Minor: `.item()` calls present but on scalars (likely safe)

### Step 3: Monotonic Cosine Scheduler (COMPLETED ✅)
**Status**: Implementation complete, currently testing

**Implementation**:
```python
# Step-based cosine scheduler (not epoch-based)
world = xr.world_size()  # PyTorch 2.7 compatible
eff_batch = config.per_core_batch_size * world * config.grad_accum_steps
steps_per_epoch = max(1, (len(train_loader.dataset) // eff_batch))
num_train_steps = max(1, steps_per_epoch * config.epochs)

warmup = max(1, int(0.03 * num_train_steps))  # 3% warmup
base_lr = float(config.base_lr)  # 1e-4
min_lr = base_lr * 0.1  # 1e-5

def lr_lambda(step):
    if step < warmup:
        return step / warmup
    t = (step - warmup) / max(1, num_train_steps - warmup)
    return (min_lr/base_lr) + 0.5*(1 - (min_lr/base_lr))*(1 + math.cos(math.pi*t))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# In training loop - scheduler.step() only on UPDATE steps
if (batch_idx + 1) % config.grad_accum_steps == 0:
    scheduler.step()
```

**Config**: `use_scheduler: true` (enabled)

**Success Criteria** (currently testing):
- ✅ train/lr is smooth/monotonic (no jumps)
- 🔄 Loss stays finite through +1k updates
- 🔄 No NaNs at previous trouble zones

---

## 📋 PENDING STEPS

### Step 4: Optimize Throughput Carefully
**Status**: Pending - after Step 3 verification

**Plan**: One change at a time, keep global batch constant
- Option A: Increase `per_core_batch_size: 2`, keep `grad_accum_steps`
- Option B: Keep `per_core_batch_size: 1`, reduce `grad_accum_steps: 1`
- Re-measure: throughput, loss curve, no NaNs/spikes

### Step 5: DataLoader Performance  
**Status**: Pending - after Step 4

**Plan**: Only after stability confirmed
- Set `workers: 2` and `persistent_workers: true`
- Keep `pin_memory: false` on TPU
- Confirm batch structure avoids mappingproxy crashes

### Step 6: EMA Momentum Schedule
**Status**: Pending - conditional on teacher usage

**Plan**: Re-enable with cosine momentum from `m_base → ~1.0`
- Update EMA only on update steps
- Log `train/ema_m` (master only) for trend verification

### Step 7: Safe Checkpointing Cadence
**Status**: Pending

**Plan**: 
- Master only, local atomic save, then optional GCS copy
- Use interval that won't coincide with accumulation cadence (e.g., every 997 update steps)
- Skip save on failure - don't crash training

### Step 8: Re-enable Compiler (Optional)
**Status**: Optional - keep `TORCH_COMPILE_DISABLE=1` for stability

---

## 🔧 CURRENT CONFIGURATION

### Training Parameters
```yaml
base_lr: 1e-4
weight_decay: 0.05
global_batch_size: 32
per_core_batch_size: 1
grad_accum_steps: 2
epochs: 150
use_scheduler: true  # Step 3 enabled
use_bf16: true
```

### DataLoader Settings (Minimal for stability)
```yaml
workers: 0
persistent_workers: false
pin_memory: false
```

### Environment
```bash
export TORCH_COMPILE_DISABLE=1
export PATH=/home/layne/miniconda/envs/torch-xla/bin:$PATH
```

---

## 📊 SUCCESS METRICS

### Baseline (A-prime fix)
- ✅ Passed critical step 514 without NaN
- ✅ Reached 1113+ steps stable
- ✅ Regular checkpointing (every 5 epochs)
- ✅ No crashes or BrokenProcessPool errors

### Current Test (Step 3 - Scheduler)
**Watch for**:
- 🔄 `train/lr`: Smooth warmup (0 → 1e-4) then cosine decay (1e-4 → 1e-5)
- 🔄 `train/grad_norm`: Finite and >0
- 🔄 `train/loss`: No NaN/N/A through 1k+ updates
- 🔄 No jumps or restarts in LR curve

---

## 🚨 CRITICAL WORKFLOW

**Every code change requires**:
```bash
# 1. Local: commit and push
git add . && git commit -m "message" && git push

# 2. TPU: pull to ALL workers
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 --zone=us-central2-b --project=d-oct-foundational-model --worker=all --command="cd ~/3d-oct-foundation-model && git pull"
```

---

*Last updated: Step 3 scheduler implementation complete, testing in progress*