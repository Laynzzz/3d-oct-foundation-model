# Implementation Plan: 3D Retinal OCT Foundation Model (V‑JEPA2 Focus)

## 1) Project Overview
Our mission is to build and evaluate self‑supervised **3D foundation models** for retinal OCT. We focus on a **Video Joint‑Embedding Predictive Architecture (V‑JEPA2)** variant that learns semantic representations by **predicting latent features** of masked 3D regions rather than reconstructing pixels.

**Pretraining Experiments**
- **Single‑Domain**: train only on **`topcon_triton`** data.
- **Multi‑Domain**: train on **all four devices** (`heidelberg_spectralis`, `topcon_triton`, `topcon_maestro2`, `zeiss_cirrus`).

**Evaluation Plan** (tracked but implemented later): downstream fine‑tuning on classification, retinal layer segmentation, and OCT→OCTA translation (out of scope for this version; see placeholders in §12).

---

## 2) Tech Stack & Environment
- **Compute**: Google Cloud **TPU v4** (v5 OK). Default: 8 cores, BF16.
  - **TPU VM**: `oct-jepa2-v4-32` (zone: `us-central2-b`).
- **Python / Env**: `/home/layne/miniconda/envs/torch-xla/bin/python` (Conda).  
- **Frameworks**: PyTorch + **torch_xla** (primary). JAX optional.
- **Core libs**: **MONAI**, **timm**, **einops**, **pydicom**, **pylibjpeg** family (`pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`), **gcsfs/fsspec**, **wandb**, **omegaconf** (or yacs), **numpy**.
- **Data**: Google Cloud Storage (GCS) streaming; optional on‑VM cache.
- **Version control**: Git + GitHub.

**`requirements.txt` (high‑level)**
```
torch==2.*
torchvision==0.*
torch-xla==2.*
monai
pydicom
pylibjpeg
pylibjpeg-libjpeg
pylibjpeg-openjpeg
gcsfs
fsspec
timm
einops
omegaconf
wandb

# optional
nibabel
```

---

## 3) Data Handling (GCS ↔ TPU)

**Source bucket**: `gs://layne-tpu-code-sync/OCTdata/OCTdata`

### 3.1 Manifest schema (TSV)
Columns (observed):
```
participant_id	manufacturer	manufacturers_model_name	anatomic_region	imaging	laterality	height	width	number_of_frames	pixel_spacing	slice_thickness	sop_instance_uid	filepath	reference_instance_uid	reference_filepath
```
- Example `filepath`: `/retinal_oct/structural_oct/heidelberg_spectralis/1001/1001_spectralis_ppol_mac_hr_oct_l_...dcm`
- `number_of_frames`: integer (e.g., 27, 60, 61). Indicates **multi‑frame DICOM volume**.
- `pixel_spacing`: may be array like `[0.003872, 0.01206]` (mm/pixel **[row, col]**) or strings like `Varies by frame`. We will **prefer actual DICOM tags** (see below).

### 3.2 On‑disk layout after unzip
- Top‑level participant folders: `1001/`, `1002/`, …  
- Each folder contains **one or more `.dcm` files** (count varies). Each `.dcm` is typically a multi‑frame OCT volume for a region/eye.

### 3.3 Device detection
- Parse the device **from `filepath`** under `/retinal_oct/structural_oct/<DEVICE>/…`.
- Map to `{heidelberg_spectralis, topcon_triton, topcon_maestro2, zeiss_cirrus}`.

### 3.4 Reading DICOM from GCS (streaming)
- Use `gcsfs` → `fsspec.open('gs://...dcm', 'rb')` → `pydicom.dcmread(BytesIO, force=True, defer_size='512 KB')`.
- Pixel data: `ds.pixel_array` (requires `pylibjpeg` backends for JPEG2000 etc.).
- **Per‑frame spacing**: Prefer `PerFrameFunctionalGroupsSequence` → `PixelMeasuresSequence` `(0028,0030)` **PixelSpacing** and `(0018,0050)` **SliceThickness**. Fallback to manifest values if tags missing.
- **Normalization**: Convert to float32, rescale by RescaleSlope/Intercept if present, then **z‑score per volume**.

### 3.5 Local caching (optional)
- If `DATA_CACHE_DIR` is set, wrap with `fsspec.open_local_cache` to cache first read onto VM ephemeral SSD; TTL configurable.

### 3.6 Missing/invalid data policy
- If `.dcm` is unreadable or corrupt: **log warning + skip** (do not crash). Count and surface a metric `skipped_files`.
- If spacing is missing: default to `[1.0, 1.0, 1.0]` mm and flag `assumed_spacing=1` in logs.

### 3.7 One‑time GCS dataset expansion (ZIP → DICOM matching manifest)
**Current state**: GCS contains one ZIP per device under `gs://layne-tpu-code-sync/OCTdata/OCTdata/<device>/*.zip`.
**Goal**: Expand ZIPs so that individual `.dcm` objects exist at paths matching the `manifest.tsv` `filepath`, e.g.:
`gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/<device>/<participant_id>/<file>.dcm`

> Run **once** on the TPU VM, then treat the dataset as read‑only.

**stream‑unzip directly from/to GCS (no local disk)**
```bash
python - <<'PY'
import gcsfs, zipfile, io
fs = gcsfs.GCSFileSystem()
SRC = 'gs://layne-tpu-code-sync/OCTdata/OCTdata'
DST = SRC + '/retinal_oct/structural_oct'
for dev in ['heidelberg_spectralis','topcon_triton','topcon_maestro2','zeiss_cirrus']:
    for zpath in fs.glob(f"{SRC}/{dev}/*.zip"):
        with fs.open(zpath, 'rb') as f:
            data = f.read()  # for very large zips, prefer Option A
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if not name.lower().endswith('.dcm'): continue
                dst = f"{DST}/{dev}/" + name.lstrip('./')
                fs.mkdirs(dst.rsplit('/',1)[0], exist_ok=True)
                with zf.open(name) as zfh, fs.open(dst, 'wb') as out:
                    out.write(zfh.read())
print('Done')
PY
```

**Post‑condition**: The Dataset now reads `.dcm` directly via `gcsfs` and paths align with `manifest.tsv`.

---

## 4) Datasets, Splits & Transforms

### 4.1 Splits
- **Single‑Domain list**: filter **`device == 'topcon_triton'`** by parsing `filepath`.
- **Multi‑Domain list**: include all devices.
- Optionally stratify by `anatomic_region` & `laterality` to balance contexts.

### 4.2 Dataset API (`data_setup/datasets.py`)
- `OCTDICOMDataset(manifest_path: str, gcs_root: str, file_list: List[str], transforms: Compose, use_cache: bool)`
- Returns dict with keys: `{ 'image': Tensor[C=1,D,H,W], 'spacing': (dz,dy,dx), 'meta': {...} }`.
- **Image shape policy**: resample to **fixed voxel spacing** (e.g., `(dz, dy, dx) = (0.05, 0.02, 0.02)` mm) then **resize/crop** to **`D×H×W = 64×384×384`**.

### 4.3 MONAI 3D transforms (pretraining)
- `LoadImaged` (custom via pydicom stream) → tensor  
- `Spacingd` (target spacing as above)  
- `NormalizeIntensityd`  
- `RandSpatialCropd` to sample 3D patches  
- `RandFlipd` (spatial axes)  
- `RandAffined` (small translations/rotations)  
- `RandGaussianNoised` (low σ)  
- **Mask generator**: produces binary mask for **JEPA targets** over patch grid (mask ratio ~ **0.6**)

---

## 5) V‑JEPA2 3D Model (Decisions)

- **Backbone**: 3D ViT (either timm 3D ViT if available, or a lightweight custom 3D ViT).  
  Default: `embed_dim=768`, `depth=12`, `num_heads=12`, `patch_size=(4,16,16)`, `in_chans=1`.
- **Views**: For each volume, sample 2 augmented **3D views**: `context` and `target`. The **target view** is masked (cube masks over patch grid). The **context encoder** sees unmasked regions; **predictor** maps context latents to target latent space.
- **Target encoder**: Momentum/EMA copy of context encoder.
- **Predictor**: 2‑layer MLP with BN/GELU, output dim = backbone embed dim.

**Loss (chosen)**: **Normalized MSE (cosine‑style regression)** between **`L2‑normalized`** predictor output and target encoder latents **on masked patches only**.  
**EMA schedule (chosen)**: base **`m0 = 0.996`** with **cosine ramp** → 1.0 across training steps:  
`m_t = 1 - (1 - m0) * (cos(π * t/T) + 1)/2`.

---

## 6) XLA / TPU Training Configuration
- **Launcher**:  
  ```
  python -m torch_xla.distributed.xla_spawn --num_workers=8 pretraining/train.py --config configs/pretrain_*.yaml
  ```
- **Device**: `xm.xla_device()`
- **Dataloader**: `DistributedSampler` (drop_last=True)  
- **Parallel IO**: start `num_workers=4` per process; tune.  
- **ParallelLoader**: wrap DataLoader with `pl.ParallelLoader` for feed to device.  
- **Optimizer step**: `xm.optimizer_step(optimizer)`  
- **bf16**: set `XLA_USE_BF16=1` (or amp autocast for XLA).  
- **Master‑only** logging/saving: `if xm.is_master_ordinal(): ...`

**`run_tpu.sh` (example)**
```
#!/usr/bin/env bash
set -euo pipefail
export XLA_USE_BF16=1
export TF_CPP_MIN_LOG_LEVEL=1
WANDB_MODE=online \
python -m torch_xla.distributed.xla_spawn --num_workers=8 \
  pretraining/train.py --config $1
```

---

## 7) Project Structure
```
oct_foundation_model/
├── configs/
│   ├── pretrain_vjepa_single_domain.yaml
│   └── pretrain_vjepa_multi_domain.yaml
├── data_setup/
│   └── datasets.py
├── models/
│   ├── vjepa_3d.py
│   └── heads.py               # (placeholder for later fine‑tuning)
├── pretraining/
│   └── train.py
├── finetuning/                # (scaffold only for now)
│   ├── classify.py
│   └── segment.py
├── utils/
│   ├── config_parser.py
│   └── logging_setup.py
├── run_tpu.sh
└── requirements.txt
```

---

## 8) Configs & Default Hyperparameters

**Shared keys (YAML)**
```yaml
experiment_name: vjepa2_single_domain
seed: 1337
wandb:
  project: oct-foundation
  entity: layne
  log_artifacts: true
  ckpt_artifact_name: "${experiment_name}-ckpt"

# data
gcs_root: gs://layne-tpu-code-sync/OCTdata/OCTdata
manifest_path: ${gcs_root}/manifest.tsv
list_strategy: single_domain    # [single_domain|multi_domain]
cache_local: true
cache_dir: /tmp/oct_cache

# spatial
target_spacing: [0.05, 0.02, 0.02]  # [dz, dy, dx] mm
image_size: [64, 384, 384]          # [D, H, W]
patch_size: [4, 16, 16]
mask_ratio: 0.6

# train
global_batch_size: 128
per_core_batch_size: 2
grad_accum_steps: 8                 # adjust to satisfy global batch
epochs: 120
base_lr: 1.5e-3
weight_decay: 0.05
warmup_epochs: 10
ema_base: 0.996

# logging/ckpt
ckpt_dir: gs://layne-tpu-code-sync/checkpoints/vjepa2/${experiment_name}
ckpt_every_epochs: 5
log_every_steps: 50

# loader
workers: 4
prefetch_factor: 2
pin_memory: false
```

**Differences by config**
- `pretrain_vjepa_single_domain.yaml`: `list_strategy: single_domain`.
- `pretrain_vjepa_multi_domain.yaml`: `list_strategy: multi_domain` and possibly **increase epochs** (e.g., 150) or slightly lower mask ratio if heterogeneity hurts convergence.

---

## 9) Step‑by‑Step Implementation Plan

### Phase 1 — Data Pipeline (Highest 🥇)
1. Implement `OCTDICOMDataset` with GCS streaming + pydicom reading.
2. Implement device parsing + two file lists:
   - `topcon_triton_files` (single‑domain)
   - `all_devices_files` (multi‑domain)
3. Implement MONAI 3D transforms and mask generator for JEPA.
4. **Verification notebook**: connect to GCS, parse manifest, visualize random volume & patches, log a grid to W&B.

**Outputs**: a) verified dataset; b) counts per device; c) sample visualizations in W&B; d) `skipped_files` metric.

### Phase 2 — Model & Trainer (High 🥈)
1. Implement 3D ViT backbone + JEPA modules (context/target encoders + predictor).
2. Loss = normalized MSE on masked patches; EMA momentum schedule as in §5.
3. Training script `pretraining/train.py` with XLA best practices, YAML configs, and artifact saving.

**Outputs**: first dry run on 1 process; then xla_spawn 8‑core smoke run.

### Phase 3 — Pretraining Runs (High 🥉)
**A) Single‑Domain (topcon_triton)**
- **Smoke test**: randomly sample **16–32 volumes**; `epochs=2–3`; confirm loss↓ and W&B logs.
- **Full run**: use all `topcon_triton`; `epochs≈120`; monitor LR, loss, throughput.

**B) Multi‑Domain (all devices)**
- **Smoke test**: 16–32 mixed volumes; `epochs=2–3`.
- **Full run**: all devices; `epochs≈150`.

**Outputs**: checkpoints in GCS + W&B Artifacts; run summaries (loss curves, throughput).

### Phase 4 — Fine‑Tuning Scaffolding (Medium)
- Create `models/heads.py`, `finetuning/` stubs with config hooks. **(Deferred)**

---

## 10) Logging, Reproducibility & Storage
- **W&B**: log training loss, learning rate, grad‑norm, EMA momentum, **throughput (vol/s/core)**, `skipped_files`, and **sample 2D grids** (random slices across D) from input views.
- **Artifacts**: save checkpoints every `ckpt_every_epochs`; keep last + best (by proxy metric like moving‑avg loss).
- **Seeds**: set `torch.manual_seed`, `numpy.random_seed`, Python `random.seed`; XLA may be non‑deterministic, but we still log seeds and env hash.
- **Storage**: primary GCS path in config; W&B artifacts enabled.

---

## 11) Automation Notes (for Coding Agent)
- **If multi‑process launch fails**: automatically retry with `--num_workers=1` (single process, 1 core) to narrow issues; then scale back up.
- **OOM handling (XLA)**: on `RuntimeError` containing `OOM` or `resource exhausted`:
  1) halve `per_core_batch_size`; 2) if <1, set to 1 and **increase `grad_accum_steps`** to keep `global_batch_size` constant; 3) if still OOM, reduce `image_size` to `64×320×320` and retry once.
- **Bad files**: catch read errors, log warning (`bad_file=1` with filepath), skip.
- **Pixel spacing missing**: substitute `[1.0,1.0,1.0]` and mark `assumed_spacing=1`.
- **Checkpoints**: always save to `ckpt_dir`; also push to W&B artifacts when `xm.is_master_ordinal()`.
- **Run naming**: `${experiment_name}-<date>-<git_short_sha>`.

---

## 12) Open Items & Future Work
- Downstream heads/losses (classification/segmentation/OCT→OCTA) to be finalized later.
- Possible **curriculum**: start mask ratio at 0.5 → 0.7 via cosine schedule.
- Consider **domain‑balanced sampling** for multi‑domain training.

---

## 13) Quick Start
1. **(Only once if your GCS has ZIPs)** run §3.7 to expand ZIPs → DICOM in GCS.
2. `pip install -r requirements.txt`
3. `bash run_tpu.sh configs/pretrain_vjepa_single_domain.yaml`
4. Monitor W&B project `oct-foundation` for live metrics.
