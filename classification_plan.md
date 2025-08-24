# Implementation Plan: Downstream Classification for V-JEPA2 3D OCT Foundation Model

## Overview of the Task

You have a pre-trained V-JEPA2 3D OCT foundation model (29.4M parameters) that has learned rich representations from retinal OCT volumes through self-supervised learning. The goal is to leverage this pre-trained model for downstream classification tasks using your clinical dataset with diabetes-related labels.

**Key Components:**
- **Pre-trained Model**: V-JEPA2 3D ViT encoder (768-dim embeddings)
- **Data Source**: 2TB Backblaze B2 bucket with OCT volumes + clinical metadata
- **Classification Target**: Diabetes status classification (healthy, pre-diabetes, controlled, insulin-dependent)
- **Infrastructure**: TPU training capability with XLA distributed training

## Step-by-Step Implementation Breakdown

### Phase 1: Data Pipeline & Infrastructure Setup (Priority: HIGH)
**Estimated Time: 2-3 days**

#### 1.1 Backblaze B2 Integration
- **Module**: `data_setup/backblaze_reader.py`
- **Technology**: `boto3` for S3-compatible API
- **Implementation**: 
  - Configure B2 credentials and endpoint
  - Implement streaming DICOM reader for B2 bucket
  - Add authentication and error handling
  - Cache frequently accessed data locally

#### 1.2 Clinical Data Integration
- **Module**: `data_setup/clinical_metadata.py`
- **Technology**: Pandas for TSV parsing, SQLite for local caching
- **Implementation**:
  - Parse `participants.tsv` with clinical labels
  - Map participant IDs to OCT volume paths
  - Create train/val/test splits based on `recommended_split`
  - Handle missing data and class imbalance

#### 1.3 Classification Dataset
- **Module**: `data_setup/classification_dataset.py`
- **Technology**: PyTorch Dataset, MONAI transforms
- **Implementation**:
  - Load pre-processed OCT volumes (64×384×384)
  - Apply classification-specific transforms
  - Return (volume, label) pairs
  - Handle variable sequence lengths

### Phase 2: Model Architecture & Fine-tuning (Priority: HIGH)
**Estimated Time: 3-4 days**

#### 2.1 Classification Head
- **Module**: `models/classification_head.py`
- **Technology**: PyTorch nn.Module
- **Implementation**:
  - Global average pooling over spatial dimensions
  - Multi-layer classification head (768 → 512 → 256 → num_classes)
  - Dropout and batch normalization
  - Support for different classification tasks

#### 2.2 Fine-tuned Model
- **Module**: `models/finetuned_vjepa.py`
- **Technology**: PyTorch, inherits from VJEPA3D
- **Implementation**:
  - Load pre-trained VJEPA2 encoder
  - Freeze/unfreeze encoder layers selectively
  - Add classification head
  - Support for different fine-tuning strategies

#### 2.3 Training Configuration
- **Module**: `configs/finetune_classification.yaml`
- **Technology**: YAML configuration
- **Implementation**:
  - Learning rate schedules (lower than pre-training)
  - Batch size and gradient accumulation
  - Data augmentation parameters
  - Evaluation metrics

### Phase 3: Training Pipeline (Priority: HIGH)
**Estimated Time: 2-3 days**

#### 3.1 Fine-tuning Script
- **Module**: `finetuning/train_classification.py`
- **Technology**: PyTorch XLA, TPU training
- **Implementation**:
  - Load pre-trained checkpoint from local path
  - Initialize classification head
  - Implement fine-tuning loop with validation
  - Support for TPU distributed training

#### 3.2 Evaluation & Metrics
- **Module**: `finetuning/evaluate.py`
- **Technology**: Scikit-learn, PyTorch
- **Implementation**:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix visualization
  - ROC curves and AUC
  - Per-class performance analysis

### Phase 4: Advanced Features & Optimization (Priority: MEDIUM)
**Estimated Time: 2-3 days**

#### 4.1 Data Augmentation
- **Module**: `data_setup/classification_transforms.py`
- **Technology**: MONAI, Albumentations
- **Implementation**:
  - 3D rotation, scaling, intensity variations
  - Mixup and CutMix for regularization
  - Domain-specific augmentations for OCT

#### 4.2 Model Interpretability
- **Module**: `finetuning/interpretability.py`
- **Technology**: Grad-CAM, attention visualization
- **Implementation**:
  - Attention weight visualization
  - Feature importance analysis
  - Clinical region highlighting

#### 4.3 Hyperparameter Optimization
- **Module**: `finetuning/hyperopt.py`
- **Technology**: Optuna, Ray Tune
- **Implementation**:
  - Learning rate, batch size optimization
  - Architecture search for classification head
  - Cross-validation strategies

## Technology Stack & Implementation Details

### Core Technologies
- **PyTorch 2.7.1**: Main deep learning framework
- **PyTorch XLA 2.7.0**: TPU training support
- **MONAI**: Medical imaging transforms and utilities
- **Boto3**: Backblaze B2 S3-compatible API
- **WandB**: Experiment tracking and model versioning

### Data Pipeline Architecture
```
Backblaze B2 → B2 Reader → DICOM Parser → Clinical Metadata → Classification Dataset
     ↓              ↓           ↓              ↓                ↓
  S3 API      Streaming    Volume Load    Label Mapping    PyTorch Dataset
```

### Model Architecture
```
Pre-trained VJEPA2 Encoder (frozen) → Global Pooling → Classification Head → Output
    768-dim embeddings              →   768-dim     →    MLP layers    → num_classes
```

### Training Strategy
1. **Stage 1**: Freeze encoder, train only classification head
2. **Stage 2**: Unfreeze last few encoder layers, fine-tune together
3. **Stage 3**: Full fine-tuning with very low learning rate

## Priority Order & Dependencies

### Immediate (Week 1)
1. **Backblaze B2 integration** - Critical for data access
2. **Clinical metadata parsing** - Required for labels
3. **Basic classification dataset** - Foundation for training

### High Priority (Week 2)
1. **Classification head implementation** - Core model component
2. **Fine-tuning script** - Training pipeline
3. **Basic evaluation metrics** - Performance assessment

### Medium Priority (Week 3)
1. **Data augmentation** - Improve generalization
2. **Model interpretability** - Clinical insights
3. **Hyperparameter optimization** - Performance tuning

## Implementation Notes

### TPU Training Considerations
- **Memory**: Classification requires less memory than pre-training
- **Batch Size**: Can use larger batches (8-16 per core)
- **Mixed Precision**: BF16 for efficiency
- **Worker Coordination**: Always use `--worker=all` for distributed training

### Data Management
- **Local Caching**: Cache frequently accessed volumes locally
- **Streaming**: Use streaming for large datasets to avoid memory issues
- **Validation**: Robust DICOM validation from existing pipeline

### Model Loading
- **Checkpoint Path**: `/Users/layne/Mac/Acdamic/UCInspire/checkpoints`
- **Compatibility**: Ensure PyTorch version compatibility
- **State Dict**: Handle potential architecture mismatches

## Success Metrics

1. **Data Pipeline**: Successfully load 1000+ OCT volumes from B2
2. **Training**: Achieve >80% validation accuracy on diabetes classification
3. **Performance**: Faster training than pre-training (larger batches)
4. **Clinical Relevance**: Interpretable results for medical professionals

## File Structure

```
finetuning/
├── __init__.py
├── train_classification.py      # Main training script
├── evaluate.py                  # Evaluation and metrics
├── interpretability.py          # Model interpretability tools
└── hyperopt.py                  # Hyperparameter optimization

data_setup/
├── backblaze_reader.py          # B2 bucket integration
├── clinical_metadata.py         # Clinical data parsing
├── classification_dataset.py    # Classification dataset
└── classification_transforms.py # Classification-specific transforms

models/
├── classification_head.py       # Classification head architecture
└── finetuned_vjepa.py          # Complete fine-tuned model

configs/
└── finetune_classification.yaml # Fine-tuning configuration
```

This plan provides a structured approach to implementing downstream classification while leveraging your existing infrastructure and pre-trained model. The modular design allows for incremental development and testing at each phase.
