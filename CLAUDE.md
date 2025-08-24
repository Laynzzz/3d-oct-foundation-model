# OCT Foundation Model - Project Status & Configuration

## Project Overview
3D Retinal OCT Foundation Model using Video Joint-Embedding Predictive Architecture (V-JEPA2) for self-supervised learning on retinal OCT volumes.

## ✅ **PRODUCTION READY - TRAINING COMPLETE**

The complete V-JEPA2 3D OCT foundation model is **fully operational** with successful pretraining completed. Three trained checkpoints are available for downstream fine-tuning.

### 🎯 **Current Phase: Fine-Tuning Setup**

**Pretraining Status**: ✅ **COMPLETE**
- Three V-JEPA2 checkpoints successfully trained and saved locally
- Ready for downstream classification tasks

**Next Steps**: Setting up fine-tuning pipeline for 4-class diabetes classification

### 🚨 **CRITICAL REMINDER: Code Update Workflow**
```bash
# Every time you change local files:
git add . && git commit -m "message" && git push
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 --zone=us-central2-b --project=d-oct-foundational-model --worker=all --command="cd ~/3d-oct-foundation-model && git pull"
```

### Current Implementation Status

**Pretraining Components Complete:**
- ✅ **Data Pipeline**: GCS DICOM streaming with robust validation
- ✅ **V-JEPA2 Model**: 29.4M parameter 3D ViT with EMA target encoder  
- ✅ **Training Infrastructure**: XLA distributed training on 16 TPU cores
- ✅ **Monitoring**: W&B integration with metrics and checkpointing
- ✅ **Error Handling**: Robust DICOM validation and empty batch handling
- ✅ **Trained Checkpoints**: 3 foundation models ready for fine-tuning

**Fine-Tuning Components (In Development):**
- 🔄 **Classification Pipeline**: Setting up diabetes status classification
- 🔄 **B2 Data Integration**: Backblaze B2 storage for fine-tuning dataset
- 🔄 **Multi-Checkpoint Evaluation**: Compare 3 pretrained models

## Environment Configuration

### TPU VM Details
- **Instance**: `oct-jepa2-v4-32` (zone: `us-central2-b`)
- **Python Environment**: `/home/layne/miniconda/envs/torch-xla/bin/python`
- **Cores**: 16 TPU v4 cores (4 workers × 4 cores each)
- **PyTorch Version**: 2.7.1
- **XLA Version**: 2.7.0

### Data Configuration

**Pretraining Data (GCS):**
- **Bucket**: `gs://layne-tpu-code-sync/OCTdata/OCTdata`
- **Manifest**: `gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest.tsv`
- **Remote Checkpoints**: `gs://layne-tpu-code-sync/checkpoints/vjepa2/`
- **Data Structure**: Individual DICOM files at structured paths

**Local Trained Checkpoints:**
- **Directory**: `/Users/layne/Mac/Acdamic/UCInspire/checkpoints/`
- **Multi-domain**: `best_checkpoint_multi_domain.pt` (1.5GB)
- **Single-domain 01**: `best_checkpoint_single_domain_01.pt` (1.5GB)
- **Single-domain 02**: `best_checkpoint_single_domain_02.pt` (1.5GB)

**Fine-Tuning Data (Backblaze B2):**
- **Endpoint**: `s3.us-west-004.backblazeb2.com`
- **Bucket**: `eye-dataset`
- **Labels**: `/fine-tuneing-data/participants.tsv` (4-class diabetes classification)
- **Participants**: 1001-1100+ with train/val/test splits

### Environment Variables
```bash
export PATH=/home/layne/miniconda/envs/torch-xla/bin:$PATH  # ALWAYS set this
export TF_CPP_MIN_LOG_LEVEL=1
export PJRT_DEVICE=TPU
export DATA_CACHE_DIR=/tmp/oct_cache  # Optional local caching

# Optional: Deprecated XLA_USE_BF16 (use config use_bf16 instead)
export XLA_USE_BF16=1
```

### W&B Configuration
- **Project**: `3d-oct-foundation-model`
- **Entity**: `laynzzz-university-at-buffalo`
- **Artifacts**: Checkpoint saving enabled

## Training Configuration

### Dataset
- **Available data**: 601 OCT volumes from participants 1001-1100
- **Devices**: heidelberg_spectralis, topcon_triton, topcon_maestro2, zeiss_cirrus
- **Data loading**: Single-threaded (`workers: 0`) for stability
- **Validation**: Enhanced DICOM pixel data checking

### Model Parameters
- **Architecture**: V-JEPA2 3D ViT
- **Parameters**: 29.4M
- **Embed dim**: 768, depth: 12
- **Patch size**: [4, 16, 16]
- **Target spacing**: [0.05, 0.02, 0.02] mm (dz, dy, dx)
- **Image size**: [64, 384, 384] (D, H, W)
- **Mask ratio**: 0.6

### Training Parameters
**Production Training:**
- **Global batch size**: 128
- **Per-core batch size**: 2  
- **Gradient accumulation**: 4 steps
- **Learning rate**: 1.5e-3
- **Weight decay**: 0.05
- **Epochs**: 120 (single-domain), 150 (multi-domain)
- **Mixed precision**: `use_bf16: true`

**Smoke Test:**
- **Global batch size**: 8
- **Per-core batch size**: 1
- **Gradient accumulation**: 1 step
- **Max steps**: 10
- **Mixed precision**: `use_bf16: true`

## Commands Reference

### 🚨 **CRITICAL: Always use `--worker=all`**
TPU distributed training requires coordination across all workers. Use `--worker=all` for training, dependency installation, and git operations.

```bash
# Environment variables for remote execution
export TPU_NAME=oct-jepa2-v4-32
export ZONE=us-central2-b
export PROJECT_ID=d-oct-foundational-model
```

### Training Commands

#### Local Execution (on TPU VM)
```bash
# Single-domain pretraining
bash run_tpu_xla.sh configs/pretrain_vjepa_single_domain.yaml

# Multi-domain pretraining
bash run_tpu_xla.sh configs/pretrain_vjepa_multi_domain.yaml

# Smoke test
bash run_tpu_xla.sh configs/smoke_test.yaml
```

#### Remote Execution (from local machine)
```bash
# Smoke test
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && bash run_tpu_xla.sh configs/smoke_test.yaml"

# Single-domain pretraining
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && bash run_tpu_xla.sh configs/pretrain_vjepa_single_domain.yaml"

# Multi-domain pretraining
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && bash run_tpu_xla.sh configs/pretrain_vjepa_multi_domain.yaml"
```

### Development Workflow

#### 🚨 **CRITICAL WORKFLOW: Code Changes → TPU Deployment**

**EVERY time you make changes to local files, you MUST follow this complete workflow:**

```bash
# 1. Local: Stage, commit, and push changes
git add .
git commit -m "Your commit message"
git push

# 2. TPU: Pull changes to ALL workers (MANDATORY)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="cd ~/3d-oct-foundation-model && git pull"
```

**⚠️ Why this is CRITICAL:**
- TPU workers operate independently and don't auto-sync
- Code changes only exist locally until git push + TPU git pull
- Running training with stale code leads to confusing results
- ALL 4 workers must have the same code version for distributed training

#### Additional Development Commands
```bash
# Pull code updates to all workers (after local git push)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="cd ~/3d-oct-foundation-model && git pull"

# Install dependencies on all workers (MUST use worker=all)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && pip install -r requirements.txt"

# Check TPU status
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && python -c 'import torch_xla.runtime as xr; print(\"TPU cores:\", xr.local_device_count())'"
```

## Technical Implementation Details

### Data Pipeline
- **GCS DICOM Reader**: Streaming with enhanced pixel data validation
- **Manifest Parser**: TSV parsing with device detection and participant filtering
- **Transforms**: MONAI 3D pipeline with memory-efficient processing
- **Error Handling**: Graceful handling of corrupted/missing DICOM files

### V-JEPA2 Architecture
- **Context Encoder**: 3D ViT backbone with learnable position embeddings
- **Target Encoder**: EMA-updated encoder with momentum scheduling (0.996 → 1.0)
- **Predictor**: 2-layer MLP with BatchNorm + GELU activation
- **Loss Function**: Normalized MSE loss on masked patches

### XLA/TPU Training
- **Distributed**: 16 TPU cores with XLA multiprocessing
- **Mixed Precision**: BF16 support via config (`use_bf16: true`)
- **Memory Management**: Gradient accumulation + automatic OOM recovery
- **Checkpointing**: GCS-based saving with W&B artifact integration
- **Multiprocessing**: `forkserver` start method for Python 3.11 + XLA 2.7.0 compatibility

## PyTorch 2.7.1 / XLA 2.7.0 Compatibility

### Key Changes
- **API Updates**: Use `torch_xla.runtime.world_size()` instead of deprecated `xm.xrt_world_size()`
- **Launcher**: Use `xmp.spawn(_mp_fn, nprocs=None)` - let XLA auto-detect devices
- **Environment**: Set `PJRT_DEVICE=TPU` for device coordination
- **Multiprocessing**: Use `forkserver` start method for compatibility

### Verified Working Configuration
- ✅ **16 TPU workers** spawned successfully across 4 nodes
- ✅ **V-JEPA3D model** (29.4M parameters) running on all workers
- ✅ **W&B monitoring** with concurrent runs
- ✅ **Data pipeline** with robust DICOM validation
- ✅ **No crashes** from empty batches or corrupted files

## Recent Improvements (August 2025)

### DICOM Validation Enhancement
- **Enhanced validation**: Check for PixelData tag `(7FE0,0010)` before accessing pixel_array
- **Graceful error handling**: Skip corrupted files instead of crashing
- **Empty batch handling**: Return `None` from collate function instead of raising `RuntimeError`

### Warning Fixes
- **Gradient accumulation**: Fixed batch size consistency warnings
- **BF16 modernization**: Added `use_bf16` config option to replace deprecated `XLA_USE_BF16`
- **Training pipeline**: Now runs warning-free while maintaining functionality

### Current Status

**Pretraining Pipeline:**
```
✅ XLA distributed training: 16 TPU workers operational
✅ W&B monitoring: Multiple concurrent runs completed
✅ Data pipeline: 601 DICOM files processed successfully
✅ Error handling: Robust validation prevents crashes
✅ Mixed precision: Modern BF16 configuration working
✅ Warning-free: All configuration issues resolved
✅ Model checkpoints: 3 trained foundation models saved locally
```

**Fine-Tuning Pipeline (Next Phase):**
```
🔄 Fine-tuning plan: Updated and aligned with V-JEPA2 architecture
🔄 B2 data integration: Pending bucket structure confirmation
🔄 Classification head: Linear probe + full fine-tuning modes
🔄 Multi-checkpoint comparison: Evaluate all 3 pretrained models
```

## Troubleshooting

### Common Issues
- **TPU Permission Errors**: Usually resolve automatically after retry
- **Empty Batches**: Now handled gracefully - training continues with next batch
- **DICOM Corruption**: Files without pixel data are automatically skipped
- **Memory Issues**: Automatic OOM recovery with batch size reduction

### OOM Handling Strategy
1. Halve `per_core_batch_size`
2. If <1, set to 1 and increase `grad_accum_steps`  
3. If still OOM, reduce `image_size` to [64, 320, 320]

---

## Fine-Tuning Setup

### Available Checkpoints
Three V-JEPA2 foundation models trained and ready:
1. **Multi-domain**: Trained on all device types (heidelberg, topcon, zeiss)
2. **Single-domain 01**: Trained on specific device subset  
3. **Single-domain 02**: Trained on specific device subset

### Fine-Tuning Task
- **Objective**: 4-class diabetes status classification
- **Classes**: healthy, pre_diabetes_lifestyle_controlled, oral_medication_controlled, insulin_dependent
- **Data**: Backblaze B2 storage with train/val/test splits
- **Approach**: Compare all 3 checkpoints via linear probe + full fine-tuning

### Next Actions
1. Confirm B2 bucket structure for fine-tuning data
2. Implement classification pipeline in `finetuning/` directory
3. Set up multi-checkpoint evaluation framework
4. Execute comparative analysis of foundation models

---

*Last updated: After pretraining completion and fine-tuning plan setup - Ready for downstream evaluation phase*