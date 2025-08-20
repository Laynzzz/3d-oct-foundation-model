# OCT Foundation Model - Project Status & Configuration

## Project Overview
3D Retinal OCT Foundation Model using Video Joint-Embedding Predictive Architecture (V-JEPA2) for self-supervised learning on retinal OCT volumes.

## Current Implementation Status

### ✅ Completed (Sections 1-5)

#### Section 1-2: Project Setup & Tech Stack
- **Project structure** created with all directories (`configs/`, `data_setup/`, `models/`, `pretraining/`, `finetuning/`, `utils/`)
- **Requirements.txt** with complete dependency list (PyTorch, torch-xla, MONAI, pydicom, gcsfs, etc.)
- **TPU training script** (`run_tpu.sh`) configured for 8-core TPU execution
- **Configuration files**:
  - `configs/pretrain_vjepa_single_domain.yaml` - Single domain (topcon_triton) training
  - `configs/pretrain_vjepa_multi_domain.yaml` - Multi-domain (all 4 devices) training
- **Utility modules**:
  - `utils/config_parser.py` - YAML config loading and validation
  - `utils/logging_setup.py` - Logging, W&B setup, metrics tracking

#### Section 3: Data Handling (GCS ↔ TPU)
- **GCS DICOM Reader** (`data_setup/gcs_dicom_reader.py`):
  - Stream DICOM reading from GCS with `gcsfs` + `pydicom`
  - Support for `pylibjpeg` backends (JPEG2000)
  - Per-frame spacing extraction from DICOM functional groups
  - Z-score normalization per volume
  - Local caching with `fsspec` (optional)
  - Error handling for corrupt/missing files
  
- **Manifest Parser** (`data_setup/manifest_parser.py`):
  - TSV manifest parsing with device detection
  - File list generation for single-domain (`topcon_triton`) and multi-domain training
  - Stratified splitting by device/anatomic region/laterality
  - Statistics and filtering capabilities
  
- **Dataset Expansion** (`data_setup/expand_gcs_dataset.py`):
  - ✅ **COMPLETED**: ZIP files extracted to individual DICOM files
  - Stream-unzip directly from/to GCS (no local disk usage)
  - Structured paths: `gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/<device>/<participant_id>/<file>.dcm`
  - All 4 devices processed: `heidelberg_spectralis`, `topcon_triton`, `topcon_maestro2`, `zeiss_cirrus`

#### Section 4: Datasets, Splits & Transforms
- **OCTDICOMDataset** (`data_setup/datasets.py`):
  - ✅ **API**: `OCTDICOMDataset(manifest_path, gcs_root, file_list, transforms, use_cache)`
  - ✅ **Returns**: `{'image': Tensor[C=1,D,H,W], 'spacing': (dz,dy,dx), 'meta': {...}}`
  - ✅ **Image shape policy**: resample to fixed voxel spacing `(0.05, 0.02, 0.02)` mm → resize/crop to `64×384×384`
  - ✅ **File lists**: `create_file_lists()` for single-domain/multi-domain strategies
  - ✅ **Stratified splits**: `stratified_split_by_device()` by anatomic region & laterality
  - ✅ **Custom collate**: `collate_fn()` handles None samples gracefully

- **MONAI 3D Transforms** (`data_setup/transforms.py`):
  - ✅ **LoadDICOMd**: Custom loader for GCS-streamed DICOM data
  - ✅ **Spacingd**: Target spacing `(0.05, 0.02, 0.02)` mm
  - ✅ **NormalizeIntensityd**: Intensity normalization
  - ✅ **RandSpatialCropd**: Sample 3D patches
  - ✅ **RandFlipd**: Spatial axes flipping
  - ✅ **RandAffined**: Small translations/rotations
  - ✅ **RandGaussianNoised**: Low σ = 0.05
  - ✅ **JEPAMaskGeneratord**: Binary mask for JEPA targets, mask ratio = 0.6
  - ✅ **TwoViewTransform**: Creates context and target views for JEPA
  - ✅ **Validation transforms**: No-augmentation pipeline

### ✅ Completed (Sections 1-6)

#### Section 5: V-JEPA2 3D Model
- [x] **3D ViT Backbone** (`models/vjepa_3d.py`):
  - ✅ **VisionTransformer3D**: embed_dim=768, depth=12, patch_size=(4,16,16)
  - ✅ **PatchEmbed3D**: 3D patch embedding for OCT volumes
  - ✅ **Attention3D**: Multi-head self-attention for 3D patches
  - ✅ **Block3D**: Transformer blocks with DropPath regularization
  - ✅ **Position embeddings**: Learnable 3D position encoding
- [x] **Context and Target Encoders**:
  - ✅ **EMAEncoder**: Target encoder with exponential moving average
  - ✅ **cosine_ema_schedule**: EMA momentum schedule from 0.996 → 1.0
  - ✅ **Gradient isolation**: Target encoder parameters frozen
- [x] **Predictor Network**:
  - ✅ **Predictor**: 2-layer MLP with BatchNorm + GELU activation
  - ✅ **Hidden dimension**: Configurable (default = embed_dim)
- [x] **Loss Function**:
  - ✅ **NormalizedMSELoss**: L2-normalized cosine-style regression
  - ✅ **Masked prediction**: Loss computed only on masked patches
- [x] **Complete VJEPA3D Model**:
  - ✅ **Integrated architecture**: Context encoder + Target encoder + Predictor
  - ✅ **Forward pass**: Returns loss, predictions, targets
  - ✅ **EMA updates**: Automatic target encoder momentum updates
  - ✅ **Inference methods**: encode_context() and encode_target()

#### Section 6: XLA/TPU Training ✅ COMPLETE
- [x] **Complete Training Script** (`pretraining/train.py`):
  - ✅ **XLA best practices**: Device placement, parallel loader, optimizer step
  - ✅ **Distributed training**: 8-core TPU with `xla_spawn` launcher
  - ✅ **Mixed precision**: BF16 support with autocast
  - ✅ **Gradient accumulation**: Configurable steps for memory efficiency
  - ✅ **Learning rate scheduling**: Cosine annealing with warmup
  - ✅ **Error handling**: Automatic OOM recovery with batch size reduction
- [x] **Config Integration**: YAML loading, validation, and interpolation
- [x] **W&B Logging**: Metrics, artifacts, and system info tracking
- [x] **Checkpoint System**: GCS saving/loading with W&B artifacts
- [x] **Robust Error Handling**: OOM detection and automatic recovery

## Environment Configuration

### TPU VM Details
- **Instance**: `oct-jepa2-v4-32` (zone: `us-central2-b`)
- **Python Environment**: `/home/layne/miniconda/envs/torch-xla/bin/python`
- **Cores**: 8 TPU v4 cores
- **PyTorch Version**: 2.7.1
- **XLA Version**: 2.7.0

### GCS Configuration
- **Bucket**: `gs://layne-tpu-code-sync/OCTdata/OCTdata`
- **Manifest**: `gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest.tsv`
- **Checkpoints**: `gs://layne-tpu-code-sync/checkpoints/vjepa2/`
- **Data Structure**: Individual DICOM files now available at structured paths

### Key Environment Variables
```bash
export XLA_USE_BF16=1
export TF_CPP_MIN_LOG_LEVEL=1
export DATA_CACHE_DIR=/tmp/oct_cache  # Optional local caching

# PyTorch 2.7 specific optimizations
export PJRT_DEVICE=TPU
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"
```

### W&B Configuration
- **Project**: `oct-foundation`
- **Entity**: `layne`
- **Artifacts**: Checkpoint saving enabled

## Training Configuration

### Spatial Parameters
- **Target spacing**: [0.05, 0.02, 0.02] mm (dz, dy, dx)
- **Image size**: [64, 384, 384] (D, H, W)
- **Patch size**: [4, 16, 16]
- **Mask ratio**: 0.6

### Training Parameters
- **Global batch size**: 128
- **Per-core batch size**: 2
- **Gradient accumulation**: 8 steps
- **Learning rate**: 1.5e-3
- **Weight decay**: 0.05
- **Epochs**: 120 (single-domain), 150 (multi-domain)
- **EMA base**: 0.996

### Device Statistics (Post-Expansion)
Based on manifest analysis:
- **heidelberg_spectralis**: [number] files
- **topcon_triton**: [number] files  
- **topcon_maestro2**: [number] files
- **zeiss_cirrus**: [number] files

## Commands Reference

### Dataset Operations
```bash
# Validate dataset expansion
python run_dataset_expansion.py --validate-only

# Test data pipeline
python -m data_setup.test_data_pipeline
```

### Training Commands

#### Local Execution (on TPU VM)
```bash
# Single-domain pretraining
bash run_tpu.sh configs/pretrain_vjepa_single_domain.yaml

# Multi-domain pretraining
bash run_tpu.sh configs/pretrain_vjepa_multi_domain.yaml

# Smoke test
bash run_tpu.sh configs/smoke_test.yaml
```

#### Remote Execution (from local machine)

**⚠️ CRITICAL: Always use `--worker=all` for training commands. TPU distributed training requires coordination across all workers.**

```bash
# Set environment variables
export TPU_NAME=oct-jepa2-v4-32
export ZONE=us-central2-b
export PROJECT_ID=your-project-id

# Single-domain pretraining (MUST use worker=all)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && bash run_tpu.sh configs/pretrain_vjepa_single_domain.yaml"

# Smoke test (MUST use worker=all)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && bash run_tpu.sh configs/smoke_test.yaml"

# Multi-domain pretraining (MUST use worker=all)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && bash run_tpu.sh configs/pretrain_vjepa_multi_domain.yaml"

# Check TPU status (can use worker=all or single worker)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && python -c 'import torch_xla.runtime as xr; print(\"TPU cores:\", xr.local_device_count())'"

# Clone repository to all workers (MUST use worker=all)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="cd /home/layne && git clone https://github.com/Laynzzz/3d-oct-foundation-model.git"

# Install dependencies on all workers (MUST use worker=all)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && pip install -r requirements.txt"

# Pull code updates to all workers (MUST use worker=all)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="cd ~/3d-oct-foundation-model && git pull"
```

### Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (when implemented)
python -m pytest tests/
```

## 🔥 CRITICAL TPU Rules & Requirements

### 🚨 MANDATORY: worker=all Usage
**NEVER use single worker for any training or setup operations!**

- ✅ **ALWAYS**: `--worker=all` for training, dependency installation, git operations
- ❌ **NEVER**: `--worker=0` or `--worker=1` for distributed operations
- **Why**: TPU v4 has 4 workers × 4 cores = 16 total cores. All workers must coordinate for distributed training.

### 🔧 PyTorch 2.7.1 / XLA 2.7.0 Specific Rules

#### Environment Setup (Required)
```bash
export PATH=/home/layne/miniconda/envs/torch-xla/bin:$PATH  # ALWAYS set this
export XLA_USE_BF16=1
export TF_CPP_MIN_LOG_LEVEL=1
export PJRT_DEVICE=TPU
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"
```

#### Code Synchronization (Required)
```bash
# ALWAYS sync code to all workers before training
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="cd ~/3d-oct-foundation-model && git pull"
```

#### Dependency Installation (Required)
```bash
# ALWAYS install on all workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && pip install -r requirements.txt"
```

### 🐛 Common Issues & Solutions

#### TPU Device Permission Errors
```
RuntimeError: TPU initialization failed: open(/dev/accel1): Operation not permitted
```
**Solution**: Restart TPU
```bash
gcloud compute tpus stop ${TPU_NAME} --zone=${ZONE}
gcloud compute tpus start ${TPU_NAME} --zone=${ZONE}
```

#### Missing Dependencies
**Always check all workers have dependencies**:
```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && python -c 'import torch_xla, gcsfs, omegaconf; print(\"✅ Dependencies OK\")'"
```

#### Import Errors
**Common fix**: Ensure `torch` is imported in all utility modules
```python
import torch  # Required for type hints like torch.nn.Module
```

### 🎯 Training Launch Rules

#### Correct Launcher (PyTorch 2.7)
```bash
# OLD (doesn't work in 2.7)
python -m torch_xla.distributed.xla_spawn --num_workers=8

# NEW (PyTorch 2.7 compatible)
torchrun --nproc_per_node=4
```

#### Batch Size Configuration
- **Global batch size**: Must be divisible by (num_workers × nproc_per_node)
- **TPU v4**: 4 workers × 4 processes = 16 total processes
- **Example**: global_batch_size=128, per_core_batch_size=8, grad_accum_steps=1

### OOM Handling Strategy
1. Halve `per_core_batch_size`
2. If <1, set to 1 and increase `grad_accum_steps`
3. If still OOM, reduce `image_size` to 64×320×320

### Error Handling
- **Bad files**: Log warning, skip, continue processing
- **Missing spacing**: Default to [1.0, 1.0, 1.0] mm, log assumption
- **Multi-process failures**: Retry with `--num_workers=1`
- **TPU initialization failures**: Restart TPU and retry

## Data Pipeline Verification

### Pre-Flight Checklist
- [x] GCS bucket accessible
- [x] Individual DICOM files extracted from ZIPs
- [x] Manifest TSV parsed successfully
- [x] Device detection working
- [x] DICOM reading with metadata extraction
- [x] `OCTDICOMDataset` class implemented
- [x] MONAI 3D transform pipeline created
- [x] JEPA mask generation implemented
- [x] Two-view transform for context/target views
- [ ] Sample volumes loaded and visualized (verification notebook needed)
- [ ] End-to-end data pipeline tested
- [ ] Mask generation validated with visualizations

## Next Immediate Tasks
1. Create verification notebook for data pipeline testing
2. Implement 3D ViT + V-JEPA2 architecture (Section 5)
3. Set up training loop with XLA optimization (Section 6)
4. Run smoke test with 16-32 volumes
5. Phase 1 verification: load sample data with transforms

### Implementation Status: Sections 1-6 ✅ COMPLETE

#### ✅ **Phase 1-2**: Data Pipeline + V-JEPA2 Model
- **Dataset class**: Ready for training with GCS streaming
- **Transform pipeline**: Full MONAI 3D implementation with JEPA masking
- **File management**: Single/multi-domain splits ready
- **Mask generation**: JEPA targets with 0.6 ratio
- **GCS integration**: Stream processing optimized
- **V-JEPA2 Model**: Complete 3D ViT architecture with EMA target encoder
- **Loss function**: Normalized MSE on masked patches
- **Training components**: Context/target encoders + predictor ready

#### ✅ **Phase 3**: XLA/TPU Training Infrastructure
- **Training script**: Full `pretraining/train.py` with XLA optimization
- **Distributed training**: 8-core TPU support with proper synchronization
- **Memory management**: Gradient accumulation + automatic OOM recovery
- **Mixed precision**: BF16 support for TPU efficiency
- **Monitoring**: W&B integration with metrics and artifact saving
- **Checkpointing**: GCS-based saving with resume capability
- **Error handling**: Robust recovery from training failures

## 🚀 **Ready for Training**

The complete V-JEPA2 3D OCT foundation model is now ready for pretraining on TPU infrastructure.

### Next Phase: Training Execution
Run training using the provided commands in the Commands Reference section.

---

*Last updated: After completing Section 6 - XLA/TPU Training Implementation*