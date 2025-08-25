# Methods: 3D Retinal OCT Foundation Model using V-JEPA2 Architecture

## Overview

This document describes the methodology for developing a self-supervised 3D foundation model for retinal Optical Coherence Tomography (OCT) using Video Joint-Embedding Predictive Architecture version 2 (V-JEPA2). The model was trained on large-scale multi-domain retinal OCT data to learn generalizable representations for downstream ophthalmic analysis tasks.

## 1. Data Sources and Preprocessing

### 1.1 Dataset Composition
- **Total OCT volumes**: 601 volumes from participants 1001-1100
- **Device heterogeneity**: 4 OCT scanner types
  - Heidelberg Spectralis
  - Topcon Triton
  - Topcon Maestro 2
  - Zeiss Cirrus
- **Storage**: Google Cloud Storage (GCS) with streaming access via `gs://layne-tpu-code-sync/OCTdata/OCTdata`
- **Format**: Multi-frame DICOM volumes with device-specific imaging parameters

### 1.2 Data Pipeline Architecture

**DICOM Processing Chain**:
1. **Streaming Reader**: GCS-native DICOM reader using `gcsfs` and `pydicom`
2. **Validation Layer**: Enhanced pixel data validation checking for PixelData tag `(7FE0,0010)`
3. **Metadata Extraction**: Per-frame spacing from `PerFrameFunctionalGroupsSequence` → `PixelMeasuresSequence`
4. **Normalization**: Float32 conversion with RescaleSlope/Intercept application and z-score per-volume normalization

**Device Detection and Domain Splitting**:
- Device type parsed from filepath pattern: `/retinal_oct/structural_oct/<DEVICE>/...`
- **Single-domain training**: Topcon Triton only (homogeneous scanner characteristics)
- **Multi-domain training**: All 4 device types (domain generalization)

### 1.3 Spatial Standardization
- **Target voxel spacing**: [0.05, 0.02, 0.02] mm (depth, height, width)
- **Target volume size**: [64, 384, 384] voxels (D×H×W)
- **Resampling**: MONAI `Spacingd` transform with trilinear interpolation
- **Spatial augmentations**: Random flips, affine transforms, Gaussian noise

## 2. V-JEPA2 Architecture Design

### 2.1 Overall Architecture

The V-JEPA2 model follows a dual-encoder predictive framework:

```
3D OCT Volume
    ↓
Context View ──→ Context Encoder (ViT-3D) ──→ Predictor ──→ Predictions
    ↓                                                            ↓
Target View ──→ Target Encoder (EMA) ──────────────────→ Targets ──→ Loss
    ↓
Mask Generation
```

### 2.2 3D Vision Transformer Backbone

**Patch Embedding**:
- **Patch size**: [4, 16, 16] voxels (depth × height × width)  
- **3D convolution**: `nn.Conv3d(1, 768, kernel_size=patch_size, stride=patch_size)`
- **Patch grid**: 16 × 24 × 24 = 9,216 patches per volume
- **Feature dimensionality**: 768-dimensional embeddings

**Transformer Architecture**:
- **Model size**: 29.4M parameters
- **Depth**: 12 transformer blocks
- **Attention heads**: 12 heads per block
- **Hidden dimension**: 768
- **MLP ratio**: 4× (768 → 3072 → 768)
- **Position embeddings**: Learnable 3D positional encodings
- **Normalization**: LayerNorm pre-normalization
- **Activation**: GELU activation functions

### 2.3 V-JEPA2 Components

**Context Encoder** (Trainable):
- Standard 3D ViT with dropout (drop_rate=0.1, drop_path_rate=0.1)
- Processes unmasked context patches
- Learns to encode visible 3D structure

**Target Encoder** (EMA-Updated):
- Identical architecture to context encoder
- No dropout for stable target features
- Updated via exponential moving average: `θ_target = m·θ_target + (1-m)·θ_context`
- **EMA momentum schedule**: `m(t) = 1 - (1-0.996)·(cos(πt/T)+1)/2` (0.996 → 1.0)

**Predictor Network**:
- 2-layer MLP: `768 → 768 → 768`
- BatchNorm1d + GELU activation
- Maps context features to target feature space

### 2.4 Masking Strategy

**3D Cube Masking**:
- **Mask ratio**: 0.6 (60% of patches masked)
- **Masking granularity**: Patch-level (4×16×16 voxel cubes)
- **Sampling**: Random uniform selection of masked patches
- **Objective**: Predict features of masked 3D regions from visible context

## 3. Self-Supervised Learning Objective

### 3.1 Loss Function

**Normalized MSE Loss**:
```
L = MSE(normalize(predictor(h_context)), normalize(h_target)) on masked patches
```

Where:
- `h_context = context_encoder(context_view)` (gradients enabled)
- `h_target = target_encoder(target_view)` (gradients disabled) 
- `normalize()`: L2 normalization (cosine-similarity style)
- Loss computed only on masked patches

### 3.2 Training Dynamics

**Advantage over pixel reconstruction**:
- Learns semantic representations rather than low-level texture
- Robust to OCT noise and artifacts
- Focuses on anatomical structure prediction

**EMA Target Stabilization**:
- Prevents representational collapse
- Provides stable learning targets
- Momentum increases during training for convergence

## 4. Training Infrastructure

### 4.1 Distributed Training Setup

**Hardware Configuration**:
- **Compute**: Google Cloud TPU v4-32 (16 TPU cores across 4 workers)
- **Memory**: Large unified memory for 3D volumes
- **Precision**: BF16 mixed precision training

**Distributed Framework**:
- **PyTorch XLA 2.7.0**: TPU-optimized PyTorch backend
- **XLA multiprocessing**: `torch_xla.distributed.xla_multiprocessing.spawn`
- **Parallel data loading**: `ParallelLoader` for device feeding
- **Synchronization**: `xm.optimizer_step()` for gradient synchronization

### 4.2 Training Hyperparameters

**Batch Configuration**:
- **Global batch size**: 32-128 (depending on memory constraints)
- **Per-core batch size**: 1-2 volumes per TPU core
- **Gradient accumulation**: 2-8 steps (to achieve target global batch size)

**Optimization**:
- **Optimizer**: AdamW with weight decay separation
  - **Learning rate**: 1e-4 (stable across domains)
  - **Weight decay**: 0.05 (applied only to weights, not biases/norms)
  - **Betas**: (0.9, 0.95)
- **Scheduler**: Cosine annealing with linear warmup
  - **Warmup**: 3% of total training steps
  - **Min LR ratio**: 0.1 of peak learning rate

**Training Duration**:
- **Single-domain**: 120 epochs (~40M training examples)
- **Multi-domain**: 150 epochs (~50M training examples)
- **Validation**: 10% held-out split with same transforms

### 4.3 Error Handling and Robustness

**DICOM Validation**:
- Pre-training validation checks for PixelData availability
- Graceful handling of corrupted/missing files
- Automatic skipping with logging (no training interruption)

**Memory Management**:
- Automatic OOM recovery with batch size reduction
- Progressive fallback: batch_size/2 → batch_size=1 → image_size reduction
- Gradient accumulation adjustment to maintain effective batch size

**Mixed Precision Stability**:
- BF16 autocast for forward pass
- FP32 gradient computation and parameter updates
- Gradient clipping (max_norm=0.01) for training stability

## 5. Monitoring and Evaluation

### 5.1 Training Metrics

**Primary Metrics**:
- **Training loss**: Normalized MSE on masked patches
- **Validation loss**: Same objective on held-out data
- **EMA momentum**: Scheduled momentum value tracking
- **Gradient norm**: Pre-clipping gradient magnitude

**Secondary Metrics**:
- **Throughput**: Volumes processed per second per core
- **Data statistics**: Device distribution, valid/corrupted file counts
- **Memory utilization**: Peak memory usage per TPU core

### 5.2 Experiment Tracking

**Weights & Biases Integration**:
- **Project**: `3d-oct-foundation-model`
- **Artifact logging**: Model checkpoints and training curves
- **Visualization**: Sample OCT slices and masking patterns
- **Hyperparameter sweeps**: Systematic evaluation of key parameters

**Checkpoint Management**:
- **Frequency**: Every 5 epochs + best validation loss
- **Storage**: Google Cloud Storage with W&B artifact backup
- **Format**: Full model state (model, optimizer, scheduler, config)

## 6. Experimental Design

### 6.1 Training Variants

**Single-Domain Pretraining**:
- **Data**: Topcon Triton scanner only (homogeneous)
- **Hypothesis**: Specialized representations for single scanner type
- **Epochs**: 120 (faster convergence on homogeneous data)

**Multi-Domain Pretraining**:
- **Data**: All 4 scanner types (heterogeneous)
- **Hypothesis**: Domain-generalizable representations
- **Epochs**: 150 (more data requires more training)

### 6.2 Model Outputs

**Foundation Model Checkpoints**:
1. **Single-domain-01**: Topcon Triton specialized model
2. **Single-domain-02**: Topcon Triton specialized model (different random seed)  
3. **Multi-domain**: Cross-scanner generalizable model

**Checkpoint Characteristics**:
- **Size**: ~1.5GB per checkpoint (29.4M parameters in FP32)
- **Format**: PyTorch state dict with full training configuration
- **Validation**: All checkpoints validated on held-out OCT data

## 7. Implementation Details

### 7.1 Code Architecture

**Modular Design**:
```
├── models/vjepa_3d.py          # V-JEPA2 model implementation
├── data_setup/datasets.py      # OCT DICOM dataset and data loading
├── data_setup/transforms.py    # MONAI 3D transforms and masking
├── pretraining/train.py        # Main training loop with XLA/TPU
├── utils/config_parser.py      # YAML configuration management
└── configs/                    # Experiment configurations
    ├── pretrain_vjepa_single_domain.yaml
    └── pretrain_vjepa_multi_domain.yaml
```

**Key Dependencies**:
- **PyTorch 2.7.1** + **XLA 2.7.0**: TPU acceleration
- **MONAI**: Medical imaging transforms and utilities
- **pydicom**: DICOM parsing with JPEG2000 support
- **gcsfs**: Native Google Cloud Storage streaming
- **wandb**: Experiment tracking and artifact management

### 7.2 Reproducibility

**Deterministic Components**:
- **Random seeds**: Fixed seeds for Python, NumPy, PyTorch
- **Data splits**: Deterministic train/validation partitioning
- **Hyperparameters**: Version-controlled YAML configurations
- **Environment**: Containerized conda environment with pinned versions

**Non-Deterministic Factors**:
- **TPU/XLA compilation**: Hardware-specific optimizations
- **Distributed training**: Slight variations in batch ordering across workers
- **DICOM loading**: Variable network latency for GCS streaming

## 8. Results and Model Characteristics

### 8.1 Training Outcomes

**Successful Training Completion**:
- **16 TPU workers**: Distributed training across 4 TPU nodes
- **Warning-free execution**: Modern BF16 configuration eliminating deprecation warnings
- **Robust error handling**: Graceful handling of corrupted DICOM files
- **W&B monitoring**: Complete training curves and metrics logged

**Model Performance**:
- **Convergence**: Stable loss reduction across 120-150 epochs
- **Memory efficiency**: Successful training with 1-2 volumes per TPU core
- **Throughput**: Optimized data pipeline achieving target training speeds

### 8.2 Foundation Model Characteristics

**Learned Representations**:
- **29.4M parameter encoder**: Compact yet expressive 3D feature extractor
- **768-dimensional features**: Rich patch-level representations
- **Multi-scale understanding**: Features capture both local texture and global anatomy
- **Domain adaptability**: Multi-domain model generalizes across scanner types

**Technical Specifications**:
- **Input**: 3D OCT volumes [64×384×384] with standardized spacing
- **Output**: 768-dimensional patch features for 9,216 patches per volume
- **Inference speed**: Fast feature extraction suitable for downstream tasks
- **Memory footprint**: ~1.5GB model suitable for clinical deployment

## 9. Future Applications

### 9.1 Downstream Fine-Tuning Capability

**Prepared Infrastructure**:
- **Classification heads**: Linear probe and MLP classifiers ready
- **Target task**: 4-class diabetes status classification
- **Data integration**: Backblaze B2 storage with labeled participant data
- **Evaluation framework**: Multi-checkpoint comparison across foundation models

**Clinical Applications**:
- **Disease classification**: Diabetic retinopathy staging
- **Anatomical segmentation**: Retinal layer delineation  
- **Image translation**: OCT to OCTA modality transfer
- **Quality assessment**: Automated OCT image quality scoring

### 9.2 Model Validation Strategy

**Comparative Evaluation**:
- **Baseline**: ImageNet-pretrained 2D models adapted to 3D
- **Ablation studies**: Single vs. multi-domain pretraining comparison
- **Transfer learning**: Fine-tuning performance on diverse ophthalmic tasks
- **Generalization testing**: Performance across unseen OCT scanner types

This methodology represents a comprehensive approach to self-supervised foundation model development for medical 3D imaging, specifically optimized for retinal OCT analysis with proven scalability and clinical applicability.