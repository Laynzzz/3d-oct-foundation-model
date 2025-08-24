"""
Combined model module that combines V-JEPA2 encoder with classification head.
Supports both linear probing and full fine-tuning modes.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging

from .encoder_loader import load_vjepa2_encoder, VisionTransformer3D
from .classifier import ClassificationHead, create_classification_head

logger = logging.getLogger(__name__)


class OCTClassificationModel(nn.Module):
    """OCT classification model combining V-JEPA2 encoder with classification head."""
    
    def __init__(
        self,
        encoder: VisionTransformer3D,
        head: ClassificationHead,
        pool_method: str = 'mean',
        freeze_encoder: bool = True
    ):
        """
        Initialize classification model.
        
        Args:
            encoder: Pre-trained V-JEPA2 encoder
            head: Classification head
            pool_method: Global pooling method ('mean', 'max', 'cls')
            freeze_encoder: Whether to freeze encoder parameters
        """
        super().__init__()
        
        self.encoder = encoder
        self.head = head
        self.pool_method = pool_method
        self.freeze_encoder = freeze_encoder
        
        # Set up pooling
        if pool_method == 'cls':
            # For CLS token, we need to modify the encoder
            logger.warning("CLS pooling requires encoder modification - using mean pooling instead")
            self.pool_method = 'mean'
        
        # Validate dimensions
        self._validate_dimensions()
        
        logger.info(f"Created OCT classification model:")
        logger.info(f"  Encoder frozen: {freeze_encoder}")
        logger.info(f"  Pooling method: {self.pool_method}")
        logger.info(f"  Input shape: {encoder.patch_embed.img_size}")
        logger.info(f"  Output classes: {head.num_classes}")
    
    def _validate_dimensions(self):
        """Validate that encoder and head dimensions match."""
        encoder_dim = self.encoder.embed_dim
        head_input_dim = self.head.embed_dim
        
        if encoder_dim != head_input_dim:
            raise ValueError(f"Dimension mismatch: encoder {encoder_dim} != head {head_input_dim}")
    
    def set_freeze_encoder(self, freeze: bool):
        """Set encoder freeze mode."""
        self.freeze_encoder = freeze
        
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        
        if freeze:
            self.encoder.eval()
            logger.info("Encoder frozen for linear probing")
        else:
            self.encoder.train()
            logger.info("Encoder unfrozen for fine-tuning")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model.
        
        Args:
            x: Input OCT volume [B, C, D, H, W]
            
        Returns:
            Class logits [B, num_classes]
        """
        # Encoder forward pass
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)  # [B, N, D]
        else:
            features = self.encoder(x)  # [B, N, D]
        
        # Global pooling
        if self.pool_method == 'mean':
            pooled = features.mean(dim=1)  # [B, D]
        elif self.pool_method == 'max':
            pooled = features.max(dim=1)[0]  # [B, D]
        else:
            raise ValueError(f"Unsupported pooling method: {self.pool_method}")
        
        # Classification head
        logits = self.head(pooled)  # [B, num_classes]
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification head.
        
        Args:
            x: Input OCT volume [B, C, D, H, W]
            
        Returns:
            Pooled features [B, embed_dim]
        """
        with torch.no_grad():
            features = self.encoder(x)  # [B, N, D]
            
            if self.pool_method == 'mean':
                pooled = features.mean(dim=1)
            elif self.pool_method == 'max':
                pooled = features.max(dim=1)[0]
            else:
                raise ValueError(f"Unsupported pooling method: {self.pool_method}")
        
        return pooled
    
    def get_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from encoder.
        
        Args:
            x: Input OCT volume [B, C, D, H, W]
            
        Returns:
            Patch features [B, num_patches, embed_dim]
        """
        with torch.no_grad():
            features = self.encoder(x)  # [B, N, D]
        
        return features
    
    def train(self, mode: bool = True):
        """Set training mode, respecting encoder freeze state."""
        super().train(mode)
        
        # Keep encoder in eval mode if frozen
        if self.freeze_encoder:
            self.encoder.eval()
        
        return self
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        encoder_total = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        
        head_total = sum(p.numel() for p in self.head.parameters())
        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        
        return {
            'encoder_total': encoder_total,
            'encoder_trainable': encoder_trainable,
            'head_total': head_total,
            'head_trainable': head_trainable,
            'total': encoder_total + head_total,
            'trainable': encoder_trainable + head_trainable
        }


def create_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    freeze_encoder: bool = True,
    device: Optional[torch.device] = None
) -> OCTClassificationModel:
    """
    Create classification model from V-JEPA2 checkpoint and config.
    
    Args:
        checkpoint_path: Path to V-JEPA2 checkpoint
        config: Model configuration
        freeze_encoder: Whether to freeze encoder
        device: Device to load model on
        
    Returns:
        OCT classification model
    """
    logger.info(f"Creating model from checkpoint: {checkpoint_path}")
    
    # Load encoder
    encoder = load_vjepa2_encoder(
        checkpoint_path=checkpoint_path,
        freeze=freeze_encoder,
        device=device
    )
    
    # Create classification head
    head = create_classification_head(config)
    
    # Move head to device
    if device is not None:
        head = head.to(device)
    
    # Create combined model
    model = OCTClassificationModel(
        encoder=encoder,
        head=head,
        pool_method=config.get('pool_method', 'mean'),
        freeze_encoder=freeze_encoder
    )
    
    # Log model info
    param_counts = model.count_parameters()
    logger.info(f"Model parameters:")
    for key, count in param_counts.items():
        logger.info(f"  {key}: {count:,}")
    
    return model


def create_multi_checkpoint_models(
    checkpoint_paths: list,
    config: Dict[str, Any],
    freeze_encoder: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, OCTClassificationModel]:
    """
    Create models from multiple checkpoints for comparison.
    
    Args:
        checkpoint_paths: List of checkpoint paths
        config: Model configuration
        freeze_encoder: Whether to freeze encoders
        device: Device to load models on
        
    Returns:
        Dictionary mapping checkpoint names to models
    """
    models = {}
    
    for checkpoint_path in checkpoint_paths:
        # Extract checkpoint name from path
        checkpoint_name = checkpoint_path.split('/')[-1].replace('.pt', '').replace('best_checkpoint_', '')
        
        try:
            model = create_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                config=config,
                freeze_encoder=freeze_encoder,
                device=device
            )
            models[checkpoint_name] = model
            logger.info(f"Successfully loaded model: {checkpoint_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {checkpoint_path}: {e}")
    
    logger.info(f"Created {len(models)} models from checkpoints")
    return models


class EnsembleModel(nn.Module):
    """Ensemble of multiple OCT classification models."""
    
    def __init__(self, models: Dict[str, OCTClassificationModel], ensemble_method: str = 'average'):
        """
        Initialize ensemble model.
        
        Args:
            models: Dictionary of models to ensemble
            ensemble_method: 'average', 'voting', or 'weighted'
        """
        super().__init__()
        
        self.models = nn.ModuleDict(models)
        self.ensemble_method = ensemble_method
        self.model_names = list(models.keys())
        
        # Validate all models have same output dimension
        num_classes = set(model.head.num_classes for model in models.values())
        if len(num_classes) > 1:
            raise ValueError(f"All models must have same number of classes, got: {num_classes}")
        
        self.num_classes = num_classes.pop()
        
        # For weighted ensemble, initialize weights
        if ensemble_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        
        logger.info(f"Created ensemble with {len(models)} models using {ensemble_method} method")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        # Get predictions from all models
        predictions = []
        for model in self.models.values():
            pred = model(x)  # [B, num_classes]
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, B, num_classes]
        
        # Ensemble predictions
        if self.ensemble_method == 'average':
            output = predictions.mean(dim=0)
        elif self.ensemble_method == 'voting':
            # Convert to probabilities and average
            probs = torch.softmax(predictions, dim=-1)
            output = torch.log(probs.mean(dim=0))
        elif self.ensemble_method == 'weighted':
            # Weighted average
            weights = torch.softmax(self.weights, dim=0)
            output = (predictions * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return output


def save_model_checkpoint(
    model: OCTClassificationModel,
    checkpoint_path: str,
    epoch: int,
    loss: float,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        epoch: Training epoch
        loss: Validation loss
        metrics: Additional metrics to save
        config: Model configuration
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'encoder_frozen': model.freeze_encoder,
        'pool_method': model.pool_method,
        'num_classes': model.head.num_classes,
        'embed_dim': model.encoder.embed_dim
    }
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved model checkpoint to {checkpoint_path}")


def load_model_checkpoint(
    checkpoint_path: str,
    encoder: VisionTransformer3D,
    head: ClassificationHead,
    device: Optional[torch.device] = None
) -> Tuple[OCTClassificationModel, Dict[str, Any]]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        encoder: Encoder instance
        head: Head instance  
        device: Device to load on
        
    Returns:
        Tuple of (model, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu')
    
    model = OCTClassificationModel(
        encoder=encoder,
        head=head,
        pool_method=checkpoint.get('pool_method', 'mean'),
        freeze_encoder=checkpoint.get('encoder_frozen', True)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if device is not None:
        model = model.to(device)
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {})
    }
    
    logger.info(f"Loaded model checkpoint from {checkpoint_path}")
    return model, checkpoint_info