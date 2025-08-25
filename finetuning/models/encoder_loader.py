"""
Encoder loader module for extracting and loading V-JEPA2 context encoder.
Handles checkpoint loading and encoder extraction for fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging
import os
import sys
import tempfile
import subprocess

# Add models directory to path for importing V-JEPA2
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.vjepa_3d import VJEPA3D, VisionTransformer3D

logger = logging.getLogger(__name__)


def _download_gcs_file(gcs_path: str) -> str:
    """
    Download GCS file to temporary location.
    
    Args:
        gcs_path: GCS path (gs://...)
        
    Returns:
        Path to downloaded temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Use gsutil to download file
        result = subprocess.run(
            ['gsutil', 'cp', gcs_path, temp_path],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Downloaded {gcs_path} to {temp_path}")
        return temp_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {gcs_path}: {e.stderr}")
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        raise FileNotFoundError(f"Could not download GCS file: {gcs_path}")
    except FileNotFoundError:
        logger.error("gsutil not found. Make sure Google Cloud SDK is installed.")
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        raise


def load_vjepa2_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load V-JEPA2 checkpoint from file or GCS.
    
    Args:
        checkpoint_path: Path to checkpoint file (local or gs://)
        
    Returns:
        Loaded checkpoint dictionary
    """
    temp_path = None
    actual_path = checkpoint_path
    
    try:
        # Handle GCS paths
        if checkpoint_path.startswith('gs://'):
            logger.info(f"Downloading GCS checkpoint: {checkpoint_path}")
            temp_path = _download_gcs_file(checkpoint_path)
            actual_path = temp_path
        elif not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint on CPU first - use weights_only=False for our trusted checkpoints
        checkpoint = torch.load(actual_path, map_location='cpu', weights_only=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        # Log checkpoint info
        if 'epoch' in checkpoint:
            logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_loss' in checkpoint:
            logger.info(f"Checkpoint best loss: {checkpoint['best_loss']:.4f}")
        
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise
    finally:
        # Clean up temporary file if we downloaded one
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")


def extract_encoder_config(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract encoder configuration from V-JEPA2 checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        
    Returns:
        Encoder configuration dictionary
    """
    # Look for config in common checkpoint locations
    config = None
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'model_config' in checkpoint:
        config = checkpoint['model_config']
    elif 'args' in checkpoint:
        config = checkpoint['args']
    
    if config is None:
        logger.warning("No config found in checkpoint, using defaults")
        # Use V-JEPA2 defaults from the fine-tuning plan
        config = {
            'img_size': [64, 384, 384],
            'patch_size': [4, 16, 16],
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'in_chans': 1
        }
    
    # Convert to expected format
    encoder_config = {
        'img_size': tuple(config.get('img_size', [64, 384, 384])),
        'patch_size': tuple(config.get('patch_size', [4, 16, 16])),
        'in_chans': config.get('in_chans', 1),
        'embed_dim': config.get('embed_dim', 768),
        'depth': config.get('depth', 12),
        'num_heads': config.get('num_heads', 12),
        'mlp_ratio': config.get('mlp_ratio', 4.0),
        'qkv_bias': config.get('qkv_bias', True),
        'drop_rate': 0.0,  # No dropout for fine-tuning
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.0
    }
    
    logger.info(f"Encoder config: {encoder_config}")
    return encoder_config


def load_vjepa2_encoder(
    checkpoint_path: str, 
    freeze: bool = True,
    device: Optional[torch.device] = None
) -> VisionTransformer3D:
    """
    Load V-JEPA2 context encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to V-JEPA2 checkpoint file
        freeze: Whether to freeze encoder parameters
        device: Device to load model on
        
    Returns:
        VisionTransformer3D encoder ready for fine-tuning
    """
    # Load checkpoint
    checkpoint = load_vjepa2_checkpoint(checkpoint_path)
    
    # Extract encoder configuration
    encoder_config = extract_encoder_config(checkpoint)
    
    # Create encoder model
    encoder = VisionTransformer3D(**encoder_config)
    
    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
    if isinstance(state_dict, dict):
        # Extract context encoder weights from V-JEPA2 checkpoint
        encoder_state_dict = {}
        
        for key, value in state_dict.items():
            # Remove 'context_encoder.' prefix if present
            if key.startswith('context_encoder.'):
                new_key = key.replace('context_encoder.', '')
                encoder_state_dict[new_key] = value
            # Also check for direct encoder keys (in case checkpoint was saved differently)
            elif any(key.startswith(prefix) for prefix in [
                'patch_embed.', 'pos_embed', 'blocks.', 'norm.'
            ]):
                encoder_state_dict[key] = value
        
        if not encoder_state_dict:
            logger.warning("No encoder weights found in checkpoint, using random initialization")
        else:
            # Load weights
            missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys in encoder: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
            logger.info(f"Loaded encoder weights from {checkpoint_path}")
    
    # Set freeze mode
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        logger.info("Encoder weights frozen for linear probing")
    else:
        for param in encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder weights unfrozen for fine-tuning")
    
    # Move to device
    if device is not None:
        encoder = encoder.to(device)
        logger.info(f"Encoder moved to device: {device}")
    
    return encoder


def get_encoder_embedding_dim(checkpoint_path: str) -> int:
    """
    Get embedding dimension from V-JEPA2 checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        Embedding dimension
    """
    checkpoint = load_vjepa2_checkpoint(checkpoint_path)
    config = extract_encoder_config(checkpoint)
    return config['embed_dim']


def validate_encoder_output(encoder: VisionTransformer3D, input_shape: Tuple[int, int, int, int, int]) -> bool:
    """
    Validate that encoder produces expected output shape.
    
    Args:
        encoder: Loaded encoder
        input_shape: Expected input shape [B, C, D, H, W]
        
    Returns:
        True if validation passes
    """
    try:
        encoder.eval()
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Forward pass
            output = encoder(dummy_input)
            
            # Check output shape
            batch_size = input_shape[0]
            expected_seq_len = encoder.patch_embed.num_patches
            expected_embed_dim = encoder.embed_dim
            
            expected_shape = (batch_size, expected_seq_len, expected_embed_dim)
            
            if output.shape == expected_shape:
                logger.info(f"Encoder validation passed: {input_shape} -> {output.shape}")
                return True
            else:
                logger.error(f"Encoder output shape mismatch: expected {expected_shape}, got {output.shape}")
                return False
                
    except Exception as e:
        logger.error(f"Encoder validation failed: {e}")
        return False


def create_pooled_encoder(encoder: VisionTransformer3D, pool_method: str = 'cls') -> nn.Module:
    """
    Create encoder with global pooling for classification.
    
    Args:
        encoder: Base encoder
        pool_method: Pooling method ('cls', 'mean', 'max')
        
    Returns:
        Encoder with pooling layer
    """
    class PooledEncoder(nn.Module):
        def __init__(self, encoder, pool_method):
            super().__init__()
            self.encoder = encoder
            self.pool_method = pool_method
            
            if pool_method == 'cls':
                # Add CLS token to encoder
                num_patches = encoder.patch_embed.num_patches
                embed_dim = encoder.embed_dim
                
                # Expand position embedding for CLS token
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                
                # Update position embedding
                old_pos_embed = encoder.pos_embed
                new_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                new_pos_embed.data[:, 1:] = old_pos_embed.data
                encoder.pos_embed = new_pos_embed
                
                # Initialize CLS token
                nn.init.trunc_normal_(self.cls_token, std=0.02)
                nn.init.trunc_normal_(new_pos_embed.data[:, 0:1], std=0.02)
        
        def forward(self, x):
            # Get encoder features [B, N, D]
            if self.pool_method == 'cls':
                # Add CLS token
                B = x.shape[0]
                cls_tokens = self.cls_token.expand(B, -1, -1)
                
                # Patch embedding
                x = self.encoder.patch_embed(x)
                
                # Concatenate CLS token
                x = torch.cat([cls_tokens, x], dim=1)
                
                # Add position embedding (now includes CLS)
                x = x + self.encoder.pos_embed
                x = self.encoder.pos_drop(x)
                
                # Apply transformer blocks
                for block in self.encoder.blocks:
                    x = block(x)
                
                x = self.encoder.norm(x)
                
                # Return CLS token
                return x[:, 0]  # [B, D]
            else:
                # Standard encoder forward
                features = self.encoder(x)  # [B, N, D]
                
                if self.pool_method == 'mean':
                    return features.mean(dim=1)  # [B, D]
                elif self.pool_method == 'max':
                    return features.max(dim=1)[0]  # [B, D]
                else:
                    raise ValueError(f"Unknown pool method: {self.pool_method}")
    
    return PooledEncoder(encoder, pool_method)


def print_encoder_info(encoder: VisionTransformer3D):
    """Print encoder architecture information."""
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    logger.info("Encoder Architecture:")
    logger.info(f"  Input size: {encoder.patch_embed.img_size}")
    logger.info(f"  Patch size: {encoder.patch_embed.patch_size}")
    logger.info(f"  Number of patches: {encoder.patch_embed.num_patches}")
    logger.info(f"  Embedding dimension: {encoder.embed_dim}")
    logger.info(f"  Number of layers: {len(encoder.blocks)}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")