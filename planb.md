# Plan B: V-JEPA2 Pretrained Foundation Model for 3D OCT

## Overview

Alternative approach to leverage existing V-JEPA2 pretrained checkpoints by mapping the **time dimension** in video to the **depth dimension** in 3D OCT volumes. This strategy enables transfer of large-scale video pretraining to medical 3D imaging with minimal architectural modifications.

## Core Concept: Time-as-Depth Mapping

### Dimensional Correspondence
```
V-JEPA2 Video:     [B, T, C, H, W] → typically [B, 16..32, 3, 224, 224]
3D OCT Volume:     [B, C, D, H, W] → e.g., [B, 1, 64, 384, 384]

Strategic Mapping:  T (time) ↔ D (depth)
```
### Conceptual Justification

**Temporal Coherence → Spatial Coherence**:
- **Video**: Smooth object motion and scene changes across time
- **OCT**: Continuous anatomical structures across retinal depth layers

**Sequential Structure Preservation**:
- **Video frames**: Ordered temporal sequence with inter-frame dependencies
- **OCT slices**: Ordered depth sequence with inter-slice anatomical continuity

**Pattern Recognition Transfer**:
- **Motion dynamics** in video → **Structural patterns** across OCT depth
- **Object persistence** across frames → **Anatomical continuity** across slices
- **Multi-scale temporal features** → **Multi-scale depth features**

---

## Implementation Strategy

### Phase 1: Direct Adaptation with Minimal Changes

#### 1.1 Input Data Preprocessing

**OCT Volume Reformatting** (`data_setup/datasets.py`):
```python
def prepare_oct_for_vjepa2(oct_volume):
    """Convert OCT volume format for V-JEPA2 compatibility.
    
    Args:
        oct_volume: [B, C, D, H, W] - OCT volume in standard format
        
    Returns:
        video_format: [B, D, C, H, W] - Video-like format for V-JEPA2
    """
    B, C, D, H, W = oct_volume.shape
    # Treat each depth slice as a temporal "frame"
    return oct_volume.permute(0, 2, 1, 3, 4)  # [B, D, C, H, W]

def resize_spatial_dimensions(oct_volume, target_size=(224, 224)):
    """Resize OCT spatial dimensions to match V-JEPA2 input size.
    
    Args:
        oct_volume: [B, D, C, H, W] - OCT in video format
        target_size: (H_new, W_new) - Target spatial resolution
        
    Returns:
        resized_volume: [B, D, C, H_new, W_new]
    """
    B, D, C, H, W = oct_volume.shape
    # Reshape for batch resizing
    volume_flat = oct_volume.view(B * D, C, H, W)
    # Resize using interpolation
    resized_flat = F.interpolate(volume_flat, size=target_size, mode='bilinear', align_corners=False)
    # Reshape back to video format
    _, C, H_new, W_new = resized_flat.shape
    return resized_flat.view(B, D, C, H_new, W_new)
```

#### 1.2 Model Architecture Adaptation

**Pretrained Checkpoint Loading** (`models/vjepa_3d.py`):
```python
def load_pretrained_vjepa2(model, checkpoint_path, config):
    """Load pretrained V-JEPA2 weights with OCT adaptations.
    
    Args:
        model: VJEPA3D model instance
        checkpoint_path: Path to pretrained V-JEPA2 checkpoint
        config: Training configuration
    """
    logger = logging.getLogger('oct_foundation')
    
    # Load pretrained checkpoint
    try:
        if checkpoint_path.startswith('gs://'):
            # Handle GCS paths
            fs = gcsfs.GCSFileSystem()
            with fs.open(checkpoint_path, 'rb') as f:
                checkpoint = torch.load(f, map_location='cpu')
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        pretrained_state = checkpoint.get('model_state_dict', checkpoint)
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        return
    
    # Handle input channel adaptation (RGB → Grayscale)
    if 'patch_embed.proj.weight' in pretrained_state:
        rgb_weights = pretrained_state['patch_embed.proj.weight']  # [embed_dim, 3, patch_h, patch_w]
        
        if model.context_encoder.patch_embed.proj.in_channels == 1:  # OCT grayscale
            # Average RGB channels to create grayscale initialization
            grayscale_weights = rgb_weights.mean(dim=1, keepdim=True)  # [embed_dim, 1, patch_h, patch_w]
            pretrained_state['patch_embed.proj.weight'] = grayscale_weights
            logger.info("Adapted RGB patch embedding weights to grayscale")
    
    # Handle positional embedding size mismatch
    if 'pos_embed' in pretrained_state:
        pretrained_pos_embed = pretrained_state['pos_embed']  # [1, N_video, embed_dim]
        model_pos_embed = model.context_encoder.pos_embed      # [1, N_oct, embed_dim]
        
        if pretrained_pos_embed.shape != model_pos_embed.shape:
            # Interpolate positional embeddings for different sequence lengths
            adapted_pos_embed = interpolate_positional_embeddings(
                pretrained_pos_embed, model_pos_embed.shape[1]
            )
            pretrained_state['pos_embed'] = adapted_pos_embed
            logger.info(f"Interpolated positional embeddings: {pretrained_pos_embed.shape} → {adapted_pos_embed.shape}")
    
    # Load weights with partial matching
    missing_keys, unexpected_keys = model.context_encoder.load_state_dict(
        pretrained_state, strict=False
    )
    
    # Initialize target encoder from adapted context encoder
    model.target_encoder.update(model.context_encoder, momentum=0.0)
    
    logger.info(f"Loaded pretrained V-JEPA2 weights")
    logger.info(f"Missing keys: {len(missing_keys)}")
    logger.info(f"Unexpected keys: {len(unexpected_keys)}")
    
    return model

def interpolate_positional_embeddings(pretrained_pos_embed, target_length):
    """Interpolate positional embeddings for different sequence lengths.
    
    Args:
        pretrained_pos_embed: [1, N_pretrained, embed_dim]
        target_length: N_target (target sequence length)
        
    Returns:
        interpolated_pos_embed: [1, N_target, embed_dim]
    """
    _, N_pretrained, embed_dim = pretrained_pos_embed.shape
    
    if N_pretrained == target_length:
        return pretrained_pos_embed
    
    # Reshape for interpolation: [1, embed_dim, N_pretrained]
    pos_embed_transposed = pretrained_pos_embed.transpose(1, 2)
    
    # Interpolate along sequence dimension
    interpolated = F.interpolate(
        pos_embed_transposed, 
        size=target_length, 
        mode='linear', 
        align_corners=False
    )
    
    # Reshape back: [1, N_target, embed_dim]
    return interpolated.transpose(1, 2)
```

#### 1.3 Training Script Integration

**Training Script Updates** (`pretraining/train.py`):
```python
def create_model_with_pretraining(config: DictConfig) -> nn.Module:
    """Create V-JEPA2 model with optional pretrained initialization."""
    
    # Create model architecture
    model = VJEPA3D(
        img_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.get('embed_dim', 768),
        depth=config.get('depth', 12),
        num_heads=config.get('num_heads', 12),
        ema_momentum=config.ema_base
    )
    
    # Load pretrained weights if specified
    if config.get('pretrained_vjepa2_path'):
        model = load_pretrained_vjepa2(
            model, 
            config.pretrained_vjepa2_path, 
            config
        )
        
        # Optional: Freeze certain layers for gradual adaptation
        if config.get('freeze_backbone_layers', 0) > 0:
            freeze_backbone_layers(model, config.freeze_backbone_layers)
    
    return model

def freeze_backbone_layers(model, num_layers_to_freeze):
    """Freeze early transformer layers for gradual adaptation."""
    logger = logging.getLogger('oct_foundation')
    
    # Freeze patch embedding
    for param in model.context_encoder.patch_embed.parameters():
        param.requires_grad = False
    
    # Freeze first N transformer blocks
    for i in range(min(num_layers_to_freeze, len(model.context_encoder.blocks))):
        for param in model.context_encoder.blocks[i].parameters():
            param.requires_grad = False
        logger.info(f"Frozen transformer block {i}")
    
    logger.info(f"Frozen {num_layers_to_freeze} backbone layers for gradual adaptation")

# Update main_worker function
def main_worker(config: DictConfig):
    """Main training worker with pretrained model support."""
    # ... existing setup code ...
    
    # Create model with optional pretraining
    model = create_model_with_pretraining(config).to(device)
    
    # Adjust learning rate for pretrained models
    if config.get('pretrained_vjepa2_path'):
        # Lower learning rate for pretrained model fine-tuning
        base_lr = config.base_lr * config.get('pretrained_lr_multiplier', 0.1)
        logger.info(f"Using reduced learning rate for pretrained model: {base_lr}")
    else:
        base_lr = config.base_lr
    
    # ... rest of training code ...
```

---

### Phase 2: Configuration and Experimental Setup

#### 2.1 Configuration Updates

**New Configuration Options** (`configs/pretrain_vjepa2_adapted.yaml`):
```yaml
experiment_name: vjepa2_pretrained_adapted
seed: 1337

# Pretrained model settings
pretrained_vjepa2_path: "path/to/vjepa2/checkpoint.pth"  # Path to pretrained V-JEPA2
adaptation_method: "time_as_depth"
pretrained_lr_multiplier: 0.1  # Reduced LR for pretrained model
freeze_backbone_layers: 4  # Freeze first 4 transformer blocks initially

# Gradual unfreezing schedule
unfreezing_schedule:
  enabled: true
  unfreeze_every_epochs: 10  # Unfreeze one layer every 10 epochs
  
# Depth subsampling (Ablation support)
depth_subsampling:
  enabled: true            # toggle for ablation
  targets: [16, 32, 64]    # number of depth slices to keep
  strategy: "uniform"      # ["uniform", "contiguous", "random"]
  random_offset: true      # randomize start index for uniform/contiguous
  contiguous_span: 16      # used when strategy == contiguous
  preserve_first_last: true
  eval_full_depth: true    # always evaluate on full depth D, regardless of training

# Spatial resolution adaptation
spatial_resize_target: [224, 224]  # Match V-JEPA2 input size initially
progressive_resize:
  enabled: true
  target_final: [384, 384]  # OCT native-ish resolution
  resize_schedule_epochs: [20, 40, 60]  # Progressive upsampling

# Data settings (modified for pretrained adaptation)
target_spacing: [0.05, 0.02, 0.02]  # Keep OCT spacing
image_size: [64, 224, 224]          # Start with V-JEPA2 compatible size
patch_size: [4, 16, 16]             # Temporal patches → depth patches

# Training (conservative for adaptation)
global_batch_size: 64               # Smaller batch for stability
per_core_batch_size: 1
grad_accum_steps: 4
epochs: 80                          # Fewer epochs needed with pretraining
base_lr: 1e-5                       # Much lower base LR
weight_decay: 0.01                  # Reduced weight decay
warmup_epochs: 5                    # Shorter warmup

# Progressive training strategy
progressive_training:
  enabled: true
  phase1_epochs: 20                 # Frozen backbone, small resolution
  phase2_epochs: 30                 # Gradual unfreezing
  phase3_epochs: 30                 # Full fine-tuning, full resolution
```

#### 2.2 Progressive Training Strategy

**Multi-Phase Training** (`pretraining/progressive_trainer.py`):
```python
class ProgressiveTrainer:
    """Progressive training strategy for V-JEPA2 adaptation."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.current_phase = 1
        self.frozen_layers = config.get('freeze_backbone_layers', 0)
        
    def update_training_phase(self, epoch):
        """Update training phase based on epoch."""
        phase_config = self.config.get('progressive_training', {})
        
        if not phase_config.get('enabled', False):
            return
            
        phase1_end = phase_config.get('phase1_epochs', 20)
        phase2_end = phase1_end + phase_config.get('phase2_epochs', 30)
        
        if epoch < phase1_end:
            self.current_phase = 1
            # Phase 1: Frozen backbone, small resolution
            self._set_phase1_config()
        elif epoch < phase2_end:
            self.current_phase = 2
            # Phase 2: Gradual unfreezing
            self._set_phase2_config(epoch - phase1_end)
        else:
            self.current_phase = 3
            # Phase 3: Full fine-tuning
            self._set_phase3_config()
    
    def _set_phase1_config(self):
        """Phase 1: Conservative adaptation."""
        # Keep backbone frozen; use smaller spatial resolution
        pass
    
    def _set_phase2_config(self, phase2_epoch):
        """Phase 2: Gradual unfreezing."""
        unfreeze_schedule = self.config.get('unfreezing_schedule', {})
        
        if unfreeze_schedule.get('enabled', False):
            unfreeze_every = unfreeze_schedule.get('unfreeze_every_epochs', 10)
            layers_to_unfreeze = phase2_epoch // unfreeze_every
            
            if layers_to_unfreeze > self.frozen_layers:
                self._unfreeze_next_layer()
    
    def _set_phase3_config(self):
        """Phase 3: Full fine-tuning."""
        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Update to full resolution if configured
        self._update_spatial_resolution()
    
    def _unfreeze_next_layer(self):
        """Unfreeze the next transformer layer."""
        if self.frozen_layers > 0:
            layer_to_unfreeze = len(self.model.context_encoder.blocks) - self.frozen_layers
            for param in self.model.context_encoder.blocks[layer_to_unfreeze].parameters():
                param.requires_grad = True
            self.frozen_layers -= 1
            logging.getLogger('oct_foundation').info(f"Unfroze layer {layer_to_unfreeze}")
```

---

### Phase 3: Evaluation and Comparison

#### 3.1 Comparative Analysis

**Model Comparison Framework**:
```python
def compare_pretraining_approaches():
    """Compare from-scratch vs. pretrained V-JEPA2 approaches."""
    
    models_to_evaluate = {
        'from_scratch_single': 'checkpoints/vjepa2_single_domain/best_checkpoint.pt',
        'from_scratch_multi': 'checkpoints/vjepa2_multi_domain/best_checkpoint.pt', 
        'pretrained_adapted': 'checkpoints/vjepa2_pretrained_adapted/best_checkpoint.pt'
    }
    
    evaluation_metrics = [
        'downstream_classification_accuracy',
        'segmentation_dice',
        'feature_quality_score',
        'training_efficiency',
        'convergence_speed'
    ]
    
    return evaluation_results
```

#### 3.2 Expected Advantages of Pretrained Approach

**Training Efficiency**:
- **Faster convergence**: Pretrained features accelerate learning
- **Lower computational cost**: Fewer epochs needed
- **Better initialization**: Rich temporal dynamics transfer to depth understanding

**Feature Quality**:
- **Multi-scale representations**: Learned from large-scale video data
- **Temporal reasoning**: Transfers to anatomical layer understanding
- **Robustness**: Pretrained on diverse video content

**Clinical Relevance**:
- **Cross-domain knowledge**: Video motion understanding → OCT structure analysis
- **Generalization**: Better performance on unseen OCT scanner types
- **Data efficiency**: Less OCT data needed for good performance

#### 3.3 **Ablation: Depth Subsampling (Time-as-Depth Length Sensitivity)**

**Goal.** Quantify how well V-JEPA2 temporal priors transfer when OCT depth length **D** differs from the video clip length used in pretraining, and find the best practical **D_target** for compute vs. accuracy.

**Why this matters.**
- V-JEPA2 checkpoints are usually trained with **T ≈ 16–32**. OCT scans often have **D ≈ 64–128**. Mismatched sequence lengths stress positional embeddings and attention range.
- Subsampling may improve stability and reduce memory, but could discard fine layers needed for segmentation.

**Experimental Matrix.**
- **D_target ∈ {8, 16, 24, 32, 48, 64}** (extend if device memory allows).
- **Sampling strategy**:
  - `uniform`: pick evenly spaced indices across D (optionally with `random_offset` each epoch).
  - `contiguous`: pick one contiguous span of length `contiguous_span` (moves with random offset).
  - `random`: sample without replacement (fixed seed per-epoch for reproducibility).
- **Fairness controls**:
  - Keep **tokens-per-step** similar by adjusting `grad_accum_steps` or batch size so the total depth tokens processed per optimizer step is comparable across settings.
  - Maintain constant **masking ratio** over depth tokens by adjusting the mask generator to the effective sequence length.
  - Always **evaluate on full depth** (if `eval_full_depth: true`) to measure representation generalization beyond the training window.

**Implementation (data layer).**
```python
def subsample_depth(x, D_target, strategy='uniform', random_offset=True, contiguous_span=None, preserve_first_last=True):
    """x: [B, D, C, H, W] -> subsampled [B, D_target, C, H, W]"""
    B, D, C, H, W = x.shape
    if D_target >= D:
        return x  # no-op
    
    if strategy == 'uniform':
        # even spacing with optional random phase
        stride = D / D_target
        start = np.random.uniform(0, stride) if random_offset else 0.0
        idx = np.clip(np.floor(start + np.arange(D_target) * stride).astype(int), 0, D-1)
    elif strategy == 'contiguous':
        span = contiguous_span or D_target
        start_max = max(0, D - span)
        start = np.random.randint(0, start_max + 1) if random_offset else (D - span)//2
        idx = np.arange(start, start + span)
        if len(idx) > D_target:
            idx = np.linspace(start, start + span - 1, D_target).astype(int)
    else:  # random
        idx = np.sort(np.random.choice(D, D_target, replace=False))
    
    if preserve_first_last:
        idx[0], idx[-1] = 0, D-1  # ensure global anchors
        idx = np.unique(idx)       # dedupe if collisions
        # pad if uniqueness reduced length
        while len(idx) < D_target:
            idx = np.sort(np.unique(np.r_[idx, np.random.randint(0, D)]))
            idx = idx[:D_target]
    
    return x[:, idx, ...]
```

**Config knobs (YAML).**
```yaml
depth_subsampling:
  enabled: true
  targets: [8, 16, 24, 32, 48, 64]
  strategy: "uniform"         # ["uniform", "contiguous", "random"]
  random_offset: true
  contiguous_span: 16
  preserve_first_last: true
  eval_full_depth: true
tokens_per_step_equalization:
  enabled: true
  method: "adjust_grad_accum" # or "adjust_batch_size"
masking:
  keep_ratio: 0.8
  rescale_for_D_target: true
logging:
  record_slice_indices: true
  wandb_hist_slices: true
```

**Positional embeddings.**
- Reuse the existing `interpolate_positional_embeddings` to resize to **N_target** after subsampling.
- For **contiguous** sampling, also try relative 3D-RoPE (if available) to reduce absolute length dependence.

**Metrics & readouts.**
- **Downstream classification**: accuracy, macro-F1, AUROC (per-device & pooled).
- **Segmentation**: Dice/IoU for retinal layers (to stress depth continuity).
- **Linear probe vs. full fine-tune** on frozen encoder.
- **Convergence**: epochs/steps to a fixed validation threshold.
- **Compute**: tokens/sec/core, memory usage, wall-clock to target metric.
- **Feature quality**: kNN top-1 on held-out features; CKA vs. from-scratch.

**Result table template.**
| D_target | Strategy    | Val Acc (clf) | AUROC | Dice (seg) | Steps→Thresh | Tokens/Step | Notes |
|---------:|-------------|---------------|-------|------------|--------------|-------------|-------|
| 16       | uniform     |               |       |            |              |             |       |
| 32       | uniform     |               |       |            |              |             |       |
| 64       | full depth  |               |       |            |              |             |       |

**Hypotheses.**
- `D_target = 16–32` with **uniform** sampling will match or slightly lag full-depth training but **converge faster** with lower memory.
- **Segmentation** benefits more from **higher D_target** or **contiguous spans**.
- If negative transfer appears at high D, subsampling mitigates positional mismatch.

---

## Implementation Timeline

### Week 1: Core Adaptation
- [ ] Implement input preprocessing for time-as-depth mapping
- [ ] Create pretrained checkpoint loading with channel adaptation
- [ ] Test basic forward pass compatibility

### Week 2: Training Integration
- [ ] Integrate pretrained loading into training pipeline
- [ ] Implement progressive training strategy
- [ ] Create adapted configuration files

### Week 3: Experimental Validation **(+ Depth Subsampling Ablation)**
- [ ] Run comparative experiments (pretrained vs. from-scratch)
- [ ] **Run ablation grid over D_target ∈ {8,16,24,32,48,64} with strategies ∈ {uniform, contiguous}**
- [ ] Evaluate **classification + segmentation** and compute trade-offs
- [ ] Document results and optimal configuration

### Week 4: Optimization and Documentation
- [ ] Fine-tune hyperparameters and training strategy
- [ ] Complete comparative analysis (include ablation findings)
- [ ] Update methodology documentation

---

## Risk Assessment and Mitigation

### Technical Risks:
**1. Domain Gap Issues**  
- **Risk**: Video features may not transfer well to medical imaging  
- **Mitigation**: Progressive training and conservative LR; **subsampling** to match pretraining sequence lengths

**2. Resolution Mismatch**  
- **Risk**: V-JEPA2 optimized for 224×224, OCT uses 384×384  
- **Mitigation**: Progressive spatial resolution increase during training

**3. Sequence Length Differences**  
- **Risk**: V-JEPA2 trained on shorter sequences than OCT depth  
- **Mitigation**: **Depth subsampling ablation** + positional embedding interpolation; evaluate full-depth generalization

### Performance Risks:
**1. Negative Transfer**  
- **Risk**: Pretrained features may hurt rather than help  
- **Mitigation**: Baseline comparison; try `D_target` close to pretrained **T**

**2. Training Instability**  
- **Risk**: Fine-tuning pretrained models can be unstable  
- **Mitigation**: Conservative LR, gradual unfreezing; depth subsampling to reduce context length

---

## Success Metrics

**Technical Success**
- [ ] Successful loading and adaptation of V-JEPA2 checkpoint
- [ ] Stable training without catastrophic forgetting
- [ ] Comparable or better convergence than from-scratch training

**Performance Success**
- [ ] Equal or superior downstream task performance
- [ ] Reduced training time (fewer epochs to convergence)
- [ ] Better generalization across OCT scanner types
- [ ] **Clear trade-off curve from depth subsampling ablation**

**Scientific Success**
- [ ] Demonstrates effective transfer learning from video to 3D medical imaging
- [ ] Validates time-as-depth conceptual mapping (including sequence length sensitivity)
- [ ] Provides insights for future cross-domain foundation model development
