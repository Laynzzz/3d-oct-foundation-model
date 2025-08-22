Stabilization & Verification Plan (TPU, 16 cores)
0) Preconditions (quick)
Create a debug branch:
git checkout -b fix/lr0-nan514
Ensure logs + checkpoints go to local disk (not gcsfuse) and only rank 0 writes (you already have this from earlier fix).
1) Instrumentation Patch (single place to measure LR, grad norm, NaNs)
File: pretraining/train.py
Apply the following helpers (top of file or utils):
import math, contextlib
import torch
import torch_xla.core.xla_model as xm

def global_grad_norm_sq(params_iter):
    sq = torch.tensor(0.0, device=xm.xla_device())
    for p in params_iter:
        if p.grad is not None:
            g = p.grad.detach()
            sq = sq + (g * g).sum()
    # cross-replica sum across all TPU cores
    sq = xm.all_reduce(xm.REDUCE_SUM, sq)
    return sq

def grads_all_finite(params_iter):
    ok = torch.tensor(1, device=xm.xla_device())
    for p in params_iter:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            ok = torch.tensor(0, device=xm.xla_device())
            break
    ok = xm.all_reduce(xm.REDUCE_MIN, ok)
    return bool(ok.item())

def is_update_step(step_idx, grad_accum_steps):
    return (step_idx + 1) % grad_accum_steps == 0
In your training loop (just after computing loss and before any optimizer step):
# 1) Immediate NaN/inf loss guard
if not torch.isfinite(loss):
    # Dump minimal diagnostics
    if xm.is_master_ordinal():
        wandb.log({"train/loss_isfinite": 0, "train/step": global_step})
        print(f"[NaN loss] step={global_step} batch_idx={batch_idx}")
    optimizer.zero_grad()
    continue

# Backward
loss.backward()

# 2) On update step only:
if is_update_step(batch_idx, config.grad_accum_steps):
    # True global grad-norm BEFORE clipping
    gn2 = global_grad_norm_sq(iter(model.parameters()))
    pre_clip_gn = gn2.sqrt().item()

    # If non-finite grads, skip this update
    if not grads_all_finite(iter(model.parameters())):
        if xm.is_master_ordinal():
            wandb.log({"train/grad_nonfinite": 1, "train/step": global_step})
            print(f"[Nonfinite grads] step={global_step} -> skipping update")
        optimizer.zero_grad()
        continue

    # Clip then step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)

    xm.optimizer_step(optimizer)
    optimizer.zero_grad()

    # Force a constant LR (until scheduler is reintroduced)
    for pg in optimizer.param_groups:
        pg["lr"] = float(config.base_lr)

    # Log from master only
    if xm.is_master_ordinal():
        wandb.log({
            "train/grad_norm": pre_clip_gn,
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/step": global_step
        })
else:
    # optional: log micro-step to see accumulation cadence
    if xm.is_master_ordinal():
        wandb.log({"train/microstep": 1, "train/step": global_step})
Why this fixes things
LR is explicitly set each update → can’t drift to 0.
Grad-norm is measured before clipping/step and globally across replicas.
NaN loss or non-finite grads are skipped cleanly (no crash cascade).
Logging is on update steps only (no more “LR=0” on micro-steps).
2) Pin down the step-514 trigger (data vs schedule vs eval)
Add a batch provenance log around the suspicious range. In your dataloader, make sure each sample returns its file path (e.g., "path"). Then, around the update step window:
# After you build the batch dict, on master replica only:
if xm.is_master_ordinal() and (512 <= global_step <= 520):
    try:
        paths = batch.get("path", None)
        if paths is not None:
            # paths could be list[str] or tensorized strings; handle simply:
            print(f"[DEBUG paths @ step {global_step}] {paths}")
            wandb.log({f"debug/paths_step_{global_step}": 1})
    except Exception:
        pass
If step 514 always pulls the same files, it’s a bad batch. If it’s different files, it’s a periodic code path (e.g., evaluation/EMA/augmentation/ckpt interval) firing at that cadence. Confirm no eval/ckpt code runs exactly every ~512 steps; if it does, shift the interval temporarily (e.g., to 997) and see if NaNs shift too.
3) Fix LR=0 at the source (belt & suspenders)
Ensure the optimizer is created with the intended LR and never overwritten to zero elsewhere:
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.base_lr), weight_decay=0.05, betas=(0.9, 0.95))
Disable any scheduler wiring for now:
scheduler = None  # intentionally disabled during stabilization
Assert LR > 0 on first update:
if is_update_step(batch_idx, config.grad_accum_steps) and xm.is_master_ordinal() and global_step < 5:
    print("[LR check] step", global_step, optimizer.param_groups[0]["lr"])
If LR still logs as 0 after this patch, grep the repo for any place setting pg['lr']=0 or scaling by a zero factor:
grep -R "param_groups" -n pretraining | cat
grep -R "lr" -n pretraining | cat
4) Reintroduce a monotonic step-based cosine schedule (optional, after stable)
Once stable (see Tests below), you can add:
# Build once after dataloader known:
steps_per_epoch = max(1, (num_train_samples // (xm.xrt_world_size() * config.per_core_batch_size)) // config.grad_accum_steps)
num_train_steps = max(1, steps_per_epoch * config.epochs)
warmup = max(1, int(0.03 * num_train_steps))
min_lr = float(config.base_lr) * 0.1

def lr_lambda(step):
    if step < warmup:
        return step / warmup
    t = (step - warmup) / max(1, num_train_steps - warmup)
    return (min_lr / config.base_lr) + 0.5 * (1 - (min_lr / config.base_lr)) * (1 + math.cos(math.pi * t))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# IMPORTANT: call only on update steps
if is_update_step(batch_idx, config.grad_accum_steps):
    scheduler.step()
Guardrail: Log the first 1k scheduled LRs offline before training:
if xm.is_master_ordinal():
    sim = [optimizer.param_groups[0]["lr"] * lr_lambda(s) for s in range(min(1000, num_train_steps))]
    print("[LR sim first 20]", sim[:20])
If this isn’t strictly smooth/monotonic (no restarts), do not enable it.
5) Test Matrix (automated runs)
Run A — Constant LR sanity
Config: base_lr=1e-4, grad_clip=0.01, grad_accum_steps=2, per_core_batch_size=1, workers=0, persistent_workers=False
Code: instrumentation patch from §1, disable scheduler (§3)
Command (adjust names):
export TPU_NAME=oct-jepa2-v4-32
export ZONE=us-central2-b
export PROJECT_ID=d-oct-foundational-model
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --project=${PROJECT_ID} --worker=all \
  --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && bash run_tpu_xla.sh configs/pretrain_vjepa_multi_domain.yaml"
Success criteria (within 700 update steps):
train/lr is exactly 1e-4 at each update step
train/grad_norm is finite and non-zero
train/loss stays finite past steps 512–520
No train/grad_nonfinite logs
Run B — Shift suspected periodic trigger
Temporarily disable/shift any periodic routines (eval/ckpt/EMA momentum change) so they do not fire near 512–520; e.g., if ckpt every 512 steps, set to 997.
Goal: if NaN moves to the new periodic point → it’s that routine.
Run C — Data batch isolation
If NaN still at ~514, enable the path logging window (§2) to capture the batch contents and try a CPU mini-repro on that batch only (10 steps). If CPU also NaNs, it’s the data or loss function; inspect the inputs (min/max/NaNs) and loss inputs (e.g., log, sqrt, division).
6) If NaN persists (common causes & quick fixes)
Loss uses log/sqrt/division: add eps clamps
x = torch.clamp(x, min=1e-6)
denom = torch.clamp(denom, min=1e-6)
Normalization (LayerNorm/BatchNorm) on all-masked tokens or empty sets → check mask ratios & ensure at least one valid token.
BF16 edge: set problematic ops to FP32 cast:
with torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=False):
    stable_part = unstable_op(input.float())
Augmentations: turn off one by one in the 500–520 window to see if a specific op blows up.
7) Success Criteria (end of plan)
W&B shows:
train/lr flat at 1e-4 (Run A) and monotonic if scheduler reintroduced (Run optional).
train/grad_norm finite throughout >1000 updates.
train/loss finite beyond step 700 with no periodic N/A.
No crashes; no BrokenProcessPool.
8) Artifacts to keep (for regression safety)
Patch diff in PR fix/lr0-nan514.
Logs around steps 500–520 (stdout and W&B).
If data-related, a manifest of the offending file paths.

Quick Fix Plan (Run A-prime)
1) Disable TorchInductor / torch.compile for this run
(That “compile_worker (1 of 32)” is TorchInductor; we don’t want it here.)
# in your launcher env for this run only
export TORCH_COMPILE_DISABLE=1
# optional visibility:
export TORCH_LOGS=  # (empty)   # avoid Dynamo/Inductor logs
Also ensure you are not wrapping the model with torch.compile(...). If you are, remove it for this run.
2) Make LR constant without touching it each step
Set at optimizer construction only; do not reassign every step.
# before training loop
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(config.base_lr),        # e.g., 1e-4
    weight_decay=0.05,
    betas=(0.9, 0.95),
)
scheduler = None  # keep disabled for Run A-prime
3) Remove custom global-grad checks (no loops, no all_reduce)
Use the built-in, XLA-optimized helper once per update step:
# DELETE: global_grad_norm_sq, grads_all_finite, any xm.all_reduce in logging

# inside training loop, after loss.backward(), on UPDATE steps only:
if (batch_idx + 1) % config.grad_accum_steps == 0:
    # 1) get pre-clip norm in a single fused op (XLA-friendly)
    pre_clip_gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf')))

    # 2) guard: skip if non-finite (cheap check, no cross-replica)
    if not math.isfinite(pre_clip_gn):
        optimizer.zero_grad()
        continue

    # 3) actual clip
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)

    # 4) step + zero
    xm.optimizer_step(optimizer)
    optimizer.zero_grad()

    # 5) log from master only (no .item() except what we already cast above)
    if xm.is_master_ordinal():
        wandb.log({
            "train/grad_norm": pre_clip_gn,
            "train/lr": optimizer.param_groups[0]["lr"],  # should be 1e-4
            "train/step": global_step
        })
Notes
• This avoids parameter iteration in the graph and avoids any cross-replica ops.
• It also removes per-step Python loops over param groups on device tensors.
• Loss logging: use if xm.is_master_ordinal(): wandb.log({"train/loss": float(loss)}) (cast once).
4) Ensure single W&B run
Initialize W&B only on master; non-masters should not call wandb.init() and should never log.
if xm.is_master_ordinal():
    import wandb
    wandb.init(project="oct-3d", config=dict(...))
else:
    wandb = None  # sentinel; never call on non-master
5) Minimal dataloader settings (keep these for A-prime)
workers: 0
persistent_workers: false
pin_memory: false
6) Re-run “A-prime”
Use the same launcher as Run A, with the TORCH_COMPILE_DISABLE=1 env var set.
What to look for
W&B starts logging shortly after the first optimizer step.
train/lr is constant at your base_lr (e.g., 1e-4).
train/grad_norm is finite and > 0.
No compile-worker swarm; CPU usage should move from compile to steady train.
If it still stalls before logging: add a single heartbeat print right before the first xm.optimizer_step(optimizer):
if (batch_idx + 1) % config.grad_accum_steps == 0 and xm.is_master_ordinal() and global_step < 3:
    print("[heartbeat] about to optimizer_step at update step", global_step)
Seeing that line means the compile completed and you’ve reached the first update.
If it hangs again (fast triage)
Accidental .item() / host syncs in the hot path
Search and move any .item(), len(tensor), or tensor.cpu().numpy() out of the update path (keep them master-only and outside compiled regions).
Any remaining cross-replica ops?
Search for xm.all_reduce / xm.mesh_reduce in training loop; remove for A-prime.
Model wrapped with torch.compile?
Double-check no compile wrappers remain and env TORCH_COMPILE_DISABLE=1 is in effect on all workers.
Success criteria for A-prime
Training proceeds past the first optimizer step and starts logging.
No NaNs at step ~514; if they reappear, we’ll investigate data/periodic hooks (ckpt/eval) next with a tiny, XLA-safe probe.

Next steps (do these in order)
1) Snapshot the stable baseline
Save a checkpoint (master-only, local disk, atomic rename).
Record throughput and key stats over the last ~200 updates:
train/lr (should be flat at 1e-4)
train/grad_norm (finite, >0)
train/loss trend
Keep the minimal instrumentation you’re using now.
2) Re-enable lightweight metrics safely
Keep everything XLA-friendly:
Grad norm: pre_clip_gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf')))
(compute once per update step, before clipping/step)
Loss: float(loss) on master only
Master-only W&B logging; no .item() on device tensors elsewhere.
3) Bring back a monotonic, step-based LR schedule (no restarts)
Only after step 2 runs clean for ~1k updates.
Code (drop-in):
# Build once after you know dataset length:
world = xm.xrt_world_size()
eff_batch = config.per_core_batch_size * world * config.grad_accum_steps
steps_per_epoch = max(1, (len(train_set) // eff_batch))
num_train_steps = max(1, steps_per_epoch * config.epochs)

warmup = max(1, int(0.03 * num_train_steps))
base_lr = float(config.base_lr)       # e.g., 1e-4
min_lr  = base_lr * 0.1

def lr_lambda(step):
    if step < warmup:
        return step / warmup
    t = (step - warmup) / max(1, num_train_steps - warmup)
    return (min_lr/base_lr) + 0.5*(1 - (min_lr/base_lr))*(1 + math.cos(math.pi*t))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# in the loop, on UPDATE steps only:
if (batch_idx + 1) % config.grad_accum_steps == 0:
    # backward done, pre_clip_gn computed, clipped, optimizer stepped
    scheduler.step()
    if xm.is_master_ordinal():
        wandb.log({"train/lr": optimizer.param_groups[0]["lr"], "train/step": global_step})
Success check: train/lr is smooth/monotonic (no jumps), loss stays finite through +1k updates.
4) Carefully improve throughput
Do one change at a time; keep global batch constant.
Option A — increase per-core batch:
Try per_core_batch_size: 2, keep grad_accum_steps the same.
If memory is tight, revert and use Option B.
Option B — reduce grad accumulation:
Keep per_core_batch_size: 1, set grad_accum_steps: 1 (same global batch only if you doubled per-core previously).
Re-measure:
Throughput (samples/sec/core)
Loss curve unchanged in character
No NaNs/spikes
5) DataLoader performance (only after stability)
Set workers: 2 and persistent_workers: true.
Keep pin_memory: false on TPU.
Confirm batch structure is plain dict/list/tensor (no custom objects) to avoid the earlier mappingproxy crash.
6) EMA momentum schedule (if you’re using a teacher)
Re-enable with a cosine momentum from m_base → ~1.0 over training.
Update EMA only on update steps.
Log train/ema_m (master only) to sanity-check it trends upward slowly.
7) Checkpoint cadence & safety
Master only, local atomic save, then optional gsutil cp to GCS.
Use an interval that won’t coincide with your accumulation cadence (e.g., every 997 update steps).
On failure, skip save—don’t crash training.
8) (Optional) Re-enable compiler
You previously hit Inductor compile storms. Keep TORCH_COMPILE_DISABLE=1 for this phase. If you do try torch.compile later, do it after the above is stable and measure compile time vs throughput.
What to expect as you proceed
After step 3: LR will gently warm up then decay; no restarts, and loss remains finite past where it used to blow up.
After step 4/5: Throughput should improve; loss/grad_norm behavior unchanged qualitatively.
No more N/A near previous trouble zones; if anything resurfaces, the last change identifies the cause.
Deliverables for each phase (so your agent can auto-check)
Baseline snapshot: checkpoint + W&B panel export (lr, loss, grad_norm) over last 200 updates.
Scheduler run: plot of train/lr (monotonic), loss stable for +1k updates.
Throughput run: table with (per_core_bs, grad_accum, throughput, stability_ok) comparing before/after.
Dataloder run: same table after changing workers/persistent_workers.
EMA run (if used): show train/ema_m trend and unchanged stability.