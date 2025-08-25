# Downstream Classification Fine-Tuning Plan (Multi-Checkpoint)

**Objective:** Fine-tune (or linear-probe) your V-JEPA2 3D OCT foundation encoder on a 4-class diabetes status classification task and **compare three pretrained checkpoints** side-by-side:

- `/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_multi_domain.pt`
- `/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_single_domain_01.pt`
- `/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_single_domain_02.pt`

**Labels TSV:** `/Users/layne/Mac/Acdamic/UCInspire/3d_oct_fundation_model/fine-tuneing-data/participants.tsv`  
**Data Store (S3-compatible Backblaze B2):** `s3.us-west-004.backblazeb2.com` → bucket `eye-dataset`  

---

## 0) Tech Stack & Conventions

- **PyTorch** (core training), **(Optional) PyTorch/XLA** for TPU later (mirror API).
- **MONAI** or light custom 3D transforms (normalize, crop/resize, flips, jitter).
- **pandas** (labels), **numpy**, **scikit-learn** (metrics).
- **fsspec / s3fs / boto3** for Backblaze B2 via S3 interface.
- **hydra**/**YAML** configs for reproducibility and sweeps.
- **Weights & Biases (W&B)** optional for logging.
- **Seeded, deterministic** (where feasible).

> 🔐 **Credentials**: Use environment variables or a secrets manager (do **not** hard-code).  
> Required env vars (example):
> ```bash
> export AWS_ACCESS_KEY_ID="<YOUR_B2_KEY_ID>"
> export AWS_SECRET_ACCESS_KEY="<YOUR_B2_APPLICATION_KEY>"
> export AWS_DEFAULT_REGION="us-west-004"
> export S3_ENDPOINT_URL="https://s3.us-west-004.backblazeb2.com"
> ```

---

## 1) Task Overview

- **Input**: 3D OCT volume per participant (`retinal_oct == TRUE`), shape expected by the encoder (e.g., `[1, D, H, W]`).  
- **Target**: `study_group` mapped to 4 classes:
  - `healthy` → 0  
  - `pre_diabetes_lifestyle_controlled` → 1  
  - `oral_medication_and_or_non_insulin_injectable_medication_controlled` → 2  
  - `insulin_dependent` → 3
- **Splits**: Respect `recommended_split` in `participants.tsv` (`train`, `val`, `test` all available).
- **Outputs**: Best checkpoint(s), per-class metrics, confusion matrix, a **leaderboard** comparing the three pretrained checkpoints under identical data/splits.

---

## 2) Priority-Ordered Implementation

### P0 — Skeleton, Config, and Smoke Tests (Highest Priority)

**Goals:**
1. Load any checkpoint → build encoder → produce embeddings for a dummy 3D tensor.
2. Parse labels TSV → construct train/val splits for `retinal_oct == TRUE` rows.
3. Locate and stream **N=5** volumes from B2 (confirm prefix & file format).
4. Overfit a tiny subset (e.g., 8 samples) using **linear probe** (encoder frozen).

**Modules to implement:**
- `configs/`  
  - `cls_linear_probe.yaml` (baseline)  
  - `cls_finetune.yaml` (unfreeze schedule)  
  - `sweep_checkpoints.yaml` (Hydra multirun for the 3 checkpoints)
- `finetuning/storage/b2.py` (extend existing structure)
  - `get_s3fs()` using `S3_ENDPOINT_URL`; retries; optional local cache dir.
- `finetuning/data/labels.py`  
  - `load_labels(tsv_path)` → `pd.DataFrame`  
  - `filter_oct_available(df)` → `df[df.retinal_oct == True]`  
  - `map_classes(df)` → `(df, class_to_idx)`  
  - `split_by_recommendation(df)` → `df_train, df_val, df_test`
- `finetuning/data/locator.py`  
  - `resolve_oct_key(participant_id: str) -> str`  
  - Uses a cached prefix list (JSON) to avoid repeated bucket listings. **No device hint needed** - fine-tuning dataset has different structure than pretraining.
- `finetuning/data/io.py`  
  - `read_volume(s3fs, key) -> torch.Tensor[1, D, H, W]`  
  - DICOM/NIfTI/NPY adapters as needed; normalization matching V-JEPA2 pretraining.
- `finetuning/data/dataset.py`  
  - `OCTVolumeDataset(df, transforms, locator, s3fs)` returning `(x, y, pid)`.
- `finetuning/data/transforms.py`  
  - **Match V-JEPA2 preprocessing**: resize/crop to `[64, 384, 384]`; same normalization as pretraining; optional flips/jitter.
- `finetuning/models/encoder_loader.py`  
  - `load_vjepa2_encoder(checkpoint_path, freeze=True) -> VisionTransformer3D` - extract `context_encoder` from V-JEPA2 checkpoint.
- `finetuning/models/classifier.py`  
  - `ClassificationHead(embed_dim=768, num_classes=4, hidden=0, dropout=0.1)` (linear or MLP head).
- `finetuning/models/model.py`  
  - Combines V-JEPA2 encoder + classification head; forward returns logits.
- `finetuning/train/loop.py`  
  - Linear probe & fine-tune modes; optimizer (AdamW), cosine w/ warmup; class weights if imbalance.
- `finetuning/utils/checks.py`  
  - Bucket list test, sample volume load, shape checks, NaN guards.

**Smoke-test flow (must pass before P1):**
- Can list `eye-dataset/` and find the expected OCT prefix.  
- First volume loads & shapes correctly; forward pass on encoder produces `[B, emb_dim]`.  
- Tiny overfit with frozen encoder reaches near-zero loss.

---

### P1 — Multi-Checkpoint Sweep & Leaderboard (High)

**Goals:** Evaluate all three checkpoints under the **same pipeline & splits**.

**What to implement:**
- `finetuning/experiments/sweep.py`  
  - Loops over:
    - `best_checkpoint_multi_domain.pt`
    - `best_checkpoint_single_domain_01.pt`
    - `best_checkpoint_single_domain_02.pt`
  - For each: run **linear probe** first; optionally run **fine-tune** (unfreeze at epoch K).  
  - Collect metrics: accuracy, **balanced accuracy**, macro-F1, per-class F1, AUROC (OvR), confusion matrix image path.
  - Save `results/leaderboard.csv` and `results/leaderboard.md` (sorted by balanced accuracy).

**Hydra multirun example:**
```bash
# Linear probe sweep
python -m finetuning.train.run -m \
  model.freeze_encoder=true model.unfreeze_at_epoch=-1 \
  paths.checkpoint_path="/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_multi_domain.pt","/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_single_domain_01.pt","/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_single_domain_02.pt"
```

**Deliverables:**
- `runs/<ckpt_name>/best.ckpt`
- `runs/<ckpt_name>/val_preds.parquet` (pid, y_true, y_pred, logits)
- `results/leaderboard.*`

---

### P3 — Data Pipeline Hardening (Medium - Post TPU)

- Add retry/backoff & file-level exception capture; record failures to `data_errors.csv` with `participant_id`, `key`, `exception`.
- Optional **local staging cache** (`/tmp/oct_cache`) to reduce repeated S3 reads.
- Validate D/H/W stats and clip outliers; log dataset summary (counts per class & per split).

---

### P4 — Training Modes & Configs (High)

- **Configs** (Hydra or plain YAML):
  - `freeze_encoder: true|false`
  - `unfreeze_at_epoch: -1|5` (−1 means linear probe only)
  - `lr_head`, `lr_encoder`, `weight_decay`, `optim: AdamW`
  - `sched: cosine`, `warmup_epochs`
  - `batch_size`, `num_workers`, `epochs`
  - `augment` toggles
  - `class_weights: auto|[w0,w1,w2,w3]`
  - `precision: fp32|bf16`
- Two defaults:
  - `cls_linear_probe.yaml` (frozen)
  - `cls_finetune.yaml` (unfreeze at epoch 3–5 with small `lr_encoder`)

---

### P5 — Metrics, Logging & Reports (High)

- Metrics: accuracy, **balanced accuracy (primary)**, macro-F1, per-class F1, AUROC.  
- Confusion matrix per checkpoint.  
- Save: best model (head & full), `class_to_idx.json`, `config.yaml`, `val_preds.parquet`.  
- Optional W&B logging (`project`, `entity`, `group=checkpoint_name`).

---

### P6 — Regularization & Robustness (Medium)

- Early stopping on **val balanced-acc** (patience=10).  
- Weight decay sweep (1e-6…1e-3), dropout (0.1–0.3 for MLP head).  
- Mild 3D augments (flips, random crop, intensity jitter).  
- Deterministic flag for ablations; track seed in artifacts.

---

### P2 — TPU Migration (High Priority - After P1)

- **TPU deployment after P1 local validation complete**
- Mirror trainer for **TPU (PyTorch/XLA)** with proven local flow
- Upload trained checkpoints and code to TPU VM
- Scale training with larger batches and distributed processing
- Mixed precision (bf16) on TPU; ensure numerics stable (no NaNs)

### P7 — Performance & Scale (Medium)

- Dataloader perf: `persistent_workers`, `prefetch_factor`, pin memory, larger batches if memory allows
- Advanced TPU optimizations and memory management

---

### P8 — Reproducibility & Packaging (Medium)

- CLI entry points:
  - `python -m oct_cls.train.run --config configs/cls_linear_probe.yaml`
  - `python -m oct_cls.eval.report --ckpt runs/<ckpt_name>/best.ckpt`
  - `python -m oct_cls.experiments.sweep --mode linear_probe --all-checkpoints`
- Save git SHA, full env dump, seeds, and data manifest snapshot (list of keys used for splits).  
- `README_cls.md` for quick start.

---

## 3) Config Templates

### `configs/cls_linear_probe.yaml`
```yaml
project_name: oct_cls_v1
seed: 42

paths:
  labels_tsv: "/Users/layne/Mac/Acdamic/UCInspire/3d_oct_fundation_model/fine-tuneing-data/participants.tsv"
  # Set one of these per run; or use Hydra multirun with a list
  checkpoint_path: "/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_multi_domain.pt"
  s3_bucket: "eye-dataset"
  s3_prefix: ""         # TODO: confirm exact B2 bucket structure for fine-tuning data
s3:
  endpoint_env: "S3_ENDPOINT_URL"

data:
  num_workers: 4
  cache_dir: "/tmp/oct_cache"
  batch_size: 2
  val_batch_size: 2
  augment:
    flip: true
    intensity_jitter: true
    resize: [64, 384, 384]         # D,H,W — match V-JEPA2 pretraining

classes:
  mapping:
    healthy: 0
    pre_diabetes_lifestyle_controlled: 1
    oral_medication_and_or_non_insulin_injectable_medication_controlled: 2
    insulin_dependent: 3

model:
  emb_dim: 768
  freeze_encoder: true
  unfreeze_at_epoch: -1
  head:
    hidden: 0
    dropout: 0.1

train:
  epochs: 50
  lr_head: 1.0e-3
  lr_encoder: 3.0e-5
  weight_decay: 1.0e-4
  warmup_epochs: 2
  class_weights: "auto"
  precision: "fp32"

log:
  wandb: false
  ckpt_dir: "./runs/cls_lp_v1"
```

### `configs/cls_finetune.yaml`
```yaml
defaults:
  - cls_linear_probe

model:
  freeze_encoder: false
  unfreeze_at_epoch: 3

train:
  lr_head: 5.0e-4
  lr_encoder: 1.0e-5
  epochs: 60
```

### `configs/sweep_checkpoints.yaml`
```yaml
# Used with: python -m oct_cls.train.run -m +sweep=true \
#   paths.checkpoint_path=...</multi_domain.pt>,...</single_01.pt>,...</single_02.pt
sweep: true
```

---

## 4) First-Run Checklist (for the coding agent)

1. **Bucket reachability**: list `eye-dataset/` using `s3fs` with `S3_ENDPOINT_URL`.
2. **Labels**: read 5 rows from TSV; verify class mapping counts & split distribution.
3. **Key resolution**: confirm exact OCT prefix; `resolve_oct_key(pid)` returns existing keys for 5 sample PIDs.
4. **I/O**: load one volume → tensor `[1, D, H, W]`; run through encoder → `[B, emb_dim]`.
5. **Smoke train**: linear probe on 8 samples; loss → ~0.
6. **Run sweep**: execute linear-probe for all 3 checkpoints; generate `results/leaderboard.csv`.
7. **Optional fine-tune**: unfreeze at epoch 3; compare to linear probe in leaderboard.

---

## 5) CLI Examples

```bash
# Linear probe for a single checkpoint
python -m finetuning.train.run --config configs/cls_linear_probe.yaml \
  paths.checkpoint_path="/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_single_domain_01.pt"

# Hydra multirun over all three (linear probe)
python -m finetuning.train.run -m --config configs/cls_linear_probe.yaml \
  paths.checkpoint_path="/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_multi_domain.pt","/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_single_domain_01.pt","/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_single_domain_02.pt"

# Fine-tune one checkpoint (unfreeze at epoch 3)
python -m finetuning.train.run --config configs/cls_finetune.yaml \
  paths.checkpoint_path="/Users/layne/Mac/Acdamic/UCInspire/checkpoints/best_checkpoint_multi_domain.pt"

# Produce a metrics report / confusion matrix for a saved run
python -m finetuning.eval.report --ckpt runs/cls_lp_v1/best.ckpt
```

---

## 6) Notes & Open Items

- **Exact OCT key layout** under `eye-dataset` bucket must be confirmed for the fine-tuning dataset structure. Update `s3_prefix` and the locator accordingly.
- **Encoder input spec** (expected size, normalization) **MUST match V-JEPA2 pretraining**: `[64, 384, 384]` size, same normalization to avoid distribution shift.
- **V-JEPA2 Integration**: Load `context_encoder` from checkpoint, freeze for linear probe, unfreeze for fine-tuning.
- Start with **linear probe** to compare the three encoders fairly; then **fine-tune** the top-1 for maximum performance.
- Consider **stratified K-fold** later to reduce split variance.
- Keep `participants.tsv` the **source of truth** for splits; never *peek* into val during sweep tuning.
- Avoid committing secrets; rotate your keys if they were shared broadly.
