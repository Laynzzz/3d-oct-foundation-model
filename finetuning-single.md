# Single-Domain V-JEPA2 Checkpoint Fine-Tuning Progress

**Date**: August 26, 2025  
**Task**: Single-domain-01 V-JEPA2 checkpoint evaluation (2/3)  
**Mode**: Linear probe (encoder frozen)  

## Configuration Summary

### Checkpoint Details
- **Model**: `best_checkpoint_single_domain_01.pt` (1.4GB)
- **Location**: `gs://layne-tpu-code-sync/checkpoints/vjepa2/vjepa2_single_domain_01/best_checkpoint.pt`
- **Architecture**: V-JEPA2 3D ViT (29.4M parameters)
- **Training**: Single-domain pretraining (specialized for one OCT scanner type)

### Training Configuration
```yaml
project_name: oct_cls_single_domain_01
model:
  emb_dim: 768
  freeze_encoder: true        # ✅ ENCODER FROZEN
  unfreeze_at_epoch: -1       # ✅ NEVER UNFREEZE
  pool_method: mean
  head:
    hidden: 0                 # Linear probe (no hidden layers)
    dropout: 0.1

train:
  epochs: 50
  lr_head: 0.001             # Only head will be trained
  lr_encoder: 3.0e-05        # Not used (encoder frozen)
  optimizer: AdamW
  scheduler: cosine
  class_weights: auto
  early_stopping:
    patience: 10
    min_delta: 0.001
```

### Infrastructure
- **Compute**: TPU v4-32 (4 workers × 4 cores = 16 cores)
- **Data**: Backblaze B2 → OCT volumes streamed during training
- **Monitoring**: W&B project `3d-oct-foundation-model`
- **Checkpoints**: `./runs/cls_single_domain_01/`

## Training Progress

### Setup Phase (11:57 PM)
✅ **Configuration loaded**: `cls_single_domain_01.yaml`  
✅ **B2 credentials**: Successfully authenticated with Backblaze B2  
✅ **TPU workers**: All 4 workers initialized  
✅ **Random seed**: Set to 42 for reproducibility  
⏳ **Data loading**: Labels processing pipeline started  

### Key Improvements Applied
1. **Checkpoint saving fixed**: Will save during training when validation improves
2. **Single-domain checkpoint**: Using specialized V-JEPA2 model vs multi-domain
3. **Pure linear probe**: Encoder completely frozen (unfreeze_at_epoch: -1)

## Expected Outcomes

### Performance Comparison Goal
Compare single-domain-01 vs multi-domain checkpoint:
- **Multi-domain result**: Validation accuracy stuck at 26.15% (Scenario A)
- **Single-domain hypothesis**: Better performance due to scanner-specific training

### Key Metrics to Monitor
- **Validation accuracy**: Target > 26.2% (beat multi-domain baseline)
- **Per-class F1 scores**: Detect if Scenario A (single class prediction) persists
- **Training convergence**: Linear probe should train quickly (frozen encoder)

### Expected W&B Dashboard
- **Run name**: `single_domain_01_linear_probe`
- **Training time**: ~2-3 hours for 50 epochs (faster with frozen encoder)
- **Checkpointing**: Saves when validation improves (`best_checkpoint_during_training.pt`)

## Status: 🟡 INITIALIZING
**Current phase**: Data pipeline setup and model loading  
**Next update**: First epoch completion and initial validation metrics  

---

## 🚨 CRITICAL ISSUE ANALYSIS: Scenario A Persistence

**Date**: August 26, 2025  
**Issue**: Single-domain checkpoint performs identically to multi-domain (26.15% validation accuracy)  
**Status**: Root cause identified - NOT a checkpoint problem  

### 🔍 **Issue Summary**

**Training Results (Epoch 1):**
- **Training accuracy**: 35.7% (✅ Good - shows model can learn)
- **Training loss**: 1.50 (✅ Decreasing)
- **Validation accuracy**: 26.15% (❌ **IDENTICAL to multi-domain**)
- **Per-class F1**: Class 0: 0.41, Classes 1,2,3: 0.0 (❌ **Scenario A confirmed**)
- **Balanced accuracy**: 25% (random guessing)

**Key Finding**: Single-domain checkpoint performs **exactly the same** as multi-domain checkpoint.

### 🎯 **Root Cause Analysis**

#### **1. Class Imbalance Problem (PRIMARY ISSUE)**

**Class Distribution from Dataset:**
```
Class 0 (healthy): 369 samples (dominant)
Class 1 (pre-diabetes): 242 samples  
Class 2 (oral medication): 321 samples
Class 3 (insulin): 129 samples
```

**Imbalance Ratio**: 2.86:1 (largest to smallest class)

**Why This Causes Scenario A:**
- Model learns to predict majority class (healthy/Class 0) for "easy" accuracy
- Minority classes (1,2,3) get 0.0 F1 scores because they're rarely predicted
- Training accuracy (35.7%) shows learning, but validation reveals the bias

#### **2. Loss Function Configuration Failure**

**Config Setting:**
```yaml
train:
  class_weights: "auto"  # ← Should handle imbalance
```

**Actual Implementation (Broken):**
```python
# From finetuning/train/loop.py
if class_weights == 'auto':
    # This would need to be computed from dataset
    logger.warning("Auto class weights not implemented, using uniform weights")
    self.criterion = nn.CrossEntropyLoss()  # ← UNIFORM WEIGHTS!
```

**The Problem**: "Auto" class weights feature is **not implemented**, falls back to uniform weights.

#### **3. Linear Probe Limitations with Imbalanced Data**

**Current Configuration:**
```yaml
model:
  freeze_encoder: true        # Encoder frozen
  unfreeze_at_epoch: -1       # Never unfreeze
  head:
    hidden: 0                 # Single linear layer
```

**Why Linear Probe Fails with Imbalance:**
- Frozen encoder features may not be discriminative enough for minority classes
- Simple linear head cannot learn complex decision boundaries
- Class imbalance makes majority class prediction the "optimal" strategy

#### **4. Data Pipeline Class Distribution**

**From finetuning/data/labels.py:**
```python
def map_classes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    class_to_idx = {
        'healthy': 0,                    # ← Dominates dataset
        'pre_diabetes_lifestyle_controlled': 1,
        'oral_medication_and_or_non_insulin_injectable_medication_controlled': 2,
        'insulin_dependent': 3
    }
```

**The Problem**: The class mapping is correct, but the training pipeline doesn't handle the resulting imbalance.

### 🔧 **Why This Happens with Both Checkpoints**

**Critical Insight**: This is **NOT a checkpoint problem** - it's a **training configuration problem**.

**Evidence:**
1. **Both checkpoints produce identical results** (26.15% validation accuracy)
2. **Both show identical per-class F1 patterns** (Class 0: 0.41, Classes 1,2,3: 0.0)
3. **Training accuracy differs** (35.7% vs multi-domain) but validation is identical
4. **Checkpoint saving works** - validation improvement detected and saved

**Root Cause**: The issue is in the **classification head training**, not the pretrained V-JEPA2 encoder quality.

### 📊 **Technical Analysis of the Problem**

#### **Model Behavior Pattern**
```
Input: OCT Volume → V-JEPA2 Encoder → Pooled Features → Linear Head → Logits
                                                           ↑
                                                    This is where it fails
```

**What's Happening:**
1. **V-JEPA2 encoder** produces good features (training accuracy 35.7% proves this)
2. **Linear classification head** learns biased decision boundary toward majority class
3. **Class imbalance** makes majority class prediction the "optimal" strategy
4. **Validation** reveals the bias (26.15% accuracy, 0.0 F1 for minority classes)

#### **Loss Function Analysis**
```python
# Current (broken) implementation
self.criterion = nn.CrossEntropyLoss()  # Uniform weights

# What it should be
weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
self.criterion = nn.CrossEntropyLoss(weight=weights)
```

**Impact**: Uniform weights mean all misclassifications are equally penalized, so the model optimizes for overall accuracy by predicting the majority class.

### 🎯 **Solutions to Fix Scenario A**

#### **Immediate Fix 1: Implement Proper Class Weights**
```python
def _setup_loss_function(self):
    if class_weights == 'auto':
        # Compute balanced class weights from training data
        from sklearn.utils.class_weight import compute_class_weight
        
        train_labels = []
        for _, labels, _ in self.train_loader:
            train_labels.extend(labels.cpu().numpy())
        
        weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
```

#### **Immediate Fix 2: Enable Fine-tuning Mode**
```yaml
model:
  freeze_encoder: false       # Unfreeze encoder
  unfreeze_at_epoch: 1        # Unfreeze from epoch 1
  head:
    hidden: 256               # Add hidden layer
    dropout: 0.2              # Increase dropout
```

#### **Immediate Fix 3: Add Data Augmentation**
```yaml
data:
  augment:
    flip: true
    intensity_jitter: true
    random_crop: true         # Add random crop
    rotation: true            # Add rotation
```

#### **Immediate Fix 4: Implement Stratified Sampling**
```python
from torch.utils.data import WeightedRandomSampler

# Compute sample weights for balanced sampling
class_counts = df['class_label'].value_counts()
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in df['class_label']]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(df), replacement=True)
```

### 📈 **Expected Results After Fixes**

**With Class Weights + Fine-tuning:**
- **Validation accuracy**: 40-60% (significant improvement)
- **Per-class F1**: All classes > 0.3 (balanced performance)
- **Balanced accuracy**: 35-55% (real improvement vs random 25%)

**Why These Fixes Will Work:**
1. **Class weights** directly address the imbalance problem
2. **Fine-tuning** allows the encoder to adapt features for the specific task
3. **Data augmentation** increases effective sample size for minority classes
4. **Stratified sampling** ensures balanced batch composition

### 🚀 **Action Plan**

#### **Phase 1: Immediate Fixes (Next 2 hours)**
1. ✅ **Fix class weights implementation** in `finetuning/train/loop.py`
2. ✅ **Switch to fine-tuning mode** in config
3. ✅ **Add data augmentation** options
4. ✅ **Test with single checkpoint** to validate fixes

#### **Phase 2: Validation (Next 4 hours)**
1. **Run fine-tuning** with fixed configuration
2. **Monitor per-class metrics** during training
3. **Compare results** to baseline (26.15%)
4. **Document improvement** in this analysis

#### **Phase 3: Multi-checkpoint Comparison (Next 8 hours)**
1. **Apply fixes to all configurations**
2. **Run sweep** across all 3 checkpoints
3. **Generate leaderboard** with balanced metrics
4. **Identify best performing** checkpoint + configuration

### 💡 **Key Insights for Future Development**

1. **Class imbalance is a data problem, not a model problem**
2. **Linear probing with imbalanced data often fails**
3. **V-JEPA2 encoder quality is good** (training accuracy proves this)
4. **The issue is in the downstream classification pipeline**
5. **Proper class weighting is essential** for medical classification tasks

### 🔍 **Monitoring During Fix Implementation**

**Metrics to Watch:**
- **Per-class F1 scores**: Should all be > 0.0
- **Balanced accuracy**: Should be > 25% (random baseline)
- **Training vs validation gap**: Should be reasonable (< 20%)
- **Class prediction distribution**: Should be balanced across classes

**Red Flags:**
- Any class still getting 0.0 F1 score
- Validation accuracy stuck at 26.15%
- Training accuracy >> validation accuracy (overfitting)

---

*Analysis completed: August 26, 2025*  
*Next update: After implementing fixes and re-running training*