"""OCT to Video Format Adaptation for V-JEPA2 Transfer Learning.

This module implements the core time-as-depth mapping strategy, converting
OCT volumes [B,C,D,H,W] to video-like format [B,D,C,H,W] for compatibility
with pretrained V-JEPA2 checkpoints.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def prepare_oct_for_vjepa2(oct_volume: torch.Tensor) -> torch.Tensor:
    """Convert OCT volume format for V-JEPA2 compatibility.
    
    Maps OCT depth dimension to video time dimension:
    [B, C, D, H, W] → [B, D, C, H, W]
    
    Args:
        oct_volume: [B, C, D, H, W] - OCT volume in standard format
        
    Returns:
        video_format: [B, D, C, H, W] - Video-like format for V-JEPA2
    """
    B, C, D, H, W = oct_volume.shape
    # Treat each depth slice as a temporal "frame"
    return oct_volume.permute(0, 2, 1, 3, 4)  # [B, D, C, H, W]


def resize_spatial_dimensions(
    oct_volume: torch.Tensor, 
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """Resize OCT spatial dimensions to match V-JEPA2 input size.
    
    Args:
        oct_volume: [B, D, C, H, W] - OCT in video format
        target_size: (H_new, W_new) - Target spatial resolution
        
    Returns:
        resized_volume: [B, D, C, H_new, W_new]
    """
    B, D, C, H, W = oct_volume.shape
    H_new, W_new = target_size
    
    if (H, W) == (H_new, W_new):
        return oct_volume
    
    # Reshape for batch resizing: [B*D, C, H, W]
    volume_flat = oct_volume.view(B * D, C, H, W)
    
    # Resize using bilinear interpolation
    resized_flat = F.interpolate(
        volume_flat, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    
    # Reshape back to video format: [B, D, C, H_new, W_new]
    return resized_flat.view(B, D, C, H_new, W_new)


def subsample_depth(
    x: torch.Tensor,
    D_target: int,
    strategy: str = 'uniform',
    random_offset: bool = True,
    contiguous_span: Optional[int] = None,
    preserve_first_last: bool = True,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, np.ndarray]:
    """Subsample depth dimension for ablation studies.
    
    Args:
        x: [B, D, C, H, W] - Input volume in video format
        D_target: Target number of depth slices to keep
        strategy: Sampling strategy ['uniform', 'contiguous', 'random']
        random_offset: Whether to add random offset for uniform/contiguous
        contiguous_span: Span length for contiguous sampling
        preserve_first_last: Whether to always keep first and last slices
        seed: Random seed for reproducibility
        
    Returns:
        subsampled_volume: [B, D_target, C, H, W]
        selected_indices: Array of selected depth indices
    """
    if seed is not None:
        np.random.seed(seed)
    
    B, D, C, H, W = x.shape
    
    if D_target >= D:
        return x, np.arange(D)  # No-op if target >= current
    
    # Generate sampling indices based on strategy
    if strategy == 'uniform':
        # Even spacing with optional random phase
        stride = D / D_target
        start = np.random.uniform(0, stride) if random_offset else 0.0
        idx = np.clip(
            np.floor(start + np.arange(D_target) * stride).astype(int), 
            0, D - 1
        )
        
    elif strategy == 'contiguous':
        # One contiguous span with optional random offset
        span = contiguous_span or D_target
        start_max = max(0, D - span)
        start = np.random.randint(0, start_max + 1) if random_offset else (D - span) // 2
        idx = np.arange(start, start + span)
        
        # If span > D_target, subsample within the span
        if len(idx) > D_target:
            idx = np.linspace(start, start + span - 1, D_target).astype(int)
            
    elif strategy == 'random':
        # Random sampling without replacement
        idx = np.sort(np.random.choice(D, D_target, replace=False))
        
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # Preserve first and last slices if requested
    if preserve_first_last and D_target >= 2:
        idx = idx.copy()
        idx[0] = 0
        idx[-1] = D - 1
        idx = np.unique(idx)  # Remove duplicates
        
        # Pad if uniqueness reduced length
        while len(idx) < D_target:
            missing_count = D_target - len(idx)
            # Sample from remaining indices
            available = np.setdiff1d(np.arange(D), idx)
            if len(available) == 0:
                break
            additional = np.random.choice(
                available, 
                min(missing_count, len(available)), 
                replace=False
            )
            idx = np.sort(np.unique(np.concatenate([idx, additional])))
        
        idx = idx[:D_target]  # Ensure exact length
    
    # Apply subsampling
    selected_volume = x[:, idx, ...]  # [B, D_target, C, H, W]
    
    logger.debug(f"Depth subsampling: {D} → {D_target} using {strategy}, indices: {idx}")
    
    return selected_volume, idx


def convert_rgb_to_grayscale_weights(rgb_weights: torch.Tensor) -> torch.Tensor:
    """Convert RGB patch embedding weights to grayscale.
    
    Args:
        rgb_weights: [embed_dim, 3, patch_h, patch_w] - RGB conv weights
        
    Returns:
        grayscale_weights: [embed_dim, 1, patch_h, patch_w] - Grayscale weights
    """
    # Average across RGB channels
    grayscale_weights = rgb_weights.mean(dim=1, keepdim=True)
    logger.info(f"Converted RGB weights {rgb_weights.shape} → grayscale {grayscale_weights.shape}")
    return grayscale_weights


def interpolate_positional_embeddings(
    pretrained_pos_embed: torch.Tensor, 
    target_length: int
) -> torch.Tensor:
    """Interpolate positional embeddings for different sequence lengths.
    
    Args:
        pretrained_pos_embed: [1, N_pretrained, embed_dim] - Original pos embed
        target_length: N_target - Target sequence length
        
    Returns:
        interpolated_pos_embed: [1, N_target, embed_dim] - Interpolated pos embed
    """
    _, N_pretrained, embed_dim = pretrained_pos_embed.shape
    
    if N_pretrained == target_length:
        return pretrained_pos_embed
    
    logger.info(f"Interpolating positional embeddings: {N_pretrained} → {target_length}")
    
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


class OCTVideoAdapter:
    """Adapter class for OCT-to-video format conversion and preprocessing."""
    
    def __init__(
        self,
        target_spatial_size: Tuple[int, int] = (224, 224),
        depth_subsampling_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize OCT-Video adapter.
        
        Args:
            target_spatial_size: Target (H, W) for spatial resizing
            depth_subsampling_config: Configuration for depth subsampling
        """
        self.target_spatial_size = target_spatial_size
        self.depth_config = depth_subsampling_config or {}
        self.subsampling_enabled = self.depth_config.get('enabled', False)
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply OCT-to-video adaptation to a data sample.
        
        Args:
            data: Data dictionary with 'image' key containing [C, D, H, W] tensor
            
        Returns:
            Adapted data with video-format image and metadata
        """
        d = dict(data)
        
        if 'image' not in d:
            return d
        
        oct_volume = d['image']  # [C, D, H, W]
        
        # Add batch dimension for processing
        if oct_volume.dim() == 4:
            oct_volume = oct_volume.unsqueeze(0)  # [1, C, D, H, W]
        
        # Step 1: Convert to video format [B, D, C, H, W]
        video_format = prepare_oct_for_vjepa2(oct_volume)
        
        # Step 2: Apply depth subsampling if enabled
        selected_indices = None
        if self.subsampling_enabled:
            targets = self.depth_config.get('targets', [32])
            # Use first target for now, could be randomized
            D_target = targets[0] if isinstance(targets, list) else targets
            
            video_format, selected_indices = subsample_depth(
                video_format,
                D_target=D_target,
                strategy=self.depth_config.get('strategy', 'uniform'),
                random_offset=self.depth_config.get('random_offset', True),
                contiguous_span=self.depth_config.get('contiguous_span'),
                preserve_first_last=self.depth_config.get('preserve_first_last', True)
            )
        
        # Step 3: Resize spatial dimensions
        video_format = resize_spatial_dimensions(video_format, self.target_spatial_size)
        
        # Remove batch dimension for compatibility
        if video_format.size(0) == 1:
            video_format = video_format.squeeze(0)  # [D, C, H, W]
        
        # Update data dictionary
        d['image'] = video_format
        d['video_format'] = True
        d['original_spatial_size'] = (oct_volume.shape[-2], oct_volume.shape[-1])
        d['adapted_spatial_size'] = self.target_spatial_size
        
        if selected_indices is not None:
            d['selected_depth_indices'] = selected_indices
            d['original_depth'] = oct_volume.shape[2]
            d['subsampled_depth'] = len(selected_indices)
        
        return d


def create_oct_video_transforms(
    target_spatial_size: Tuple[int, int] = (224, 224),
    depth_subsampling_config: Optional[Dict[str, Any]] = None
) -> OCTVideoAdapter:
    """Create OCT-to-video transform pipeline.
    
    Args:
        target_spatial_size: Target spatial resolution
        depth_subsampling_config: Depth subsampling configuration
        
    Returns:
        OCTVideoAdapter instance
    """
    return OCTVideoAdapter(
        target_spatial_size=target_spatial_size,
        depth_subsampling_config=depth_subsampling_config
    )


# Utility functions for debugging and analysis
def visualize_depth_sampling(original_depth: int, selected_indices: np.ndarray) -> str:
    """Create a visual representation of depth sampling.
    
    Args:
        original_depth: Original number of depth slices
        selected_indices: Selected depth indices
        
    Returns:
        Visual representation string
    """
    viz = ['-'] * original_depth
    for idx in selected_indices:
        if 0 <= idx < original_depth:
            viz[idx] = 'X'
    
    return ''.join(viz) + f' ({len(selected_indices)}/{original_depth})'


def compute_depth_sampling_stats(selected_indices: np.ndarray, original_depth: int) -> Dict[str, float]:
    """Compute statistics for depth sampling pattern.
    
    Args:
        selected_indices: Selected depth indices
        original_depth: Original number of depth slices
        
    Returns:
        Dictionary of sampling statistics
    """
    if len(selected_indices) <= 1:
        return {'coverage': 0.0, 'uniformity': 0.0, 'span_ratio': 0.0}
    
    coverage = len(selected_indices) / original_depth
    
    # Uniformity: how close to uniform spacing
    expected_spacing = (original_depth - 1) / (len(selected_indices) - 1)
    actual_spacings = np.diff(selected_indices)
    uniformity = 1.0 - np.std(actual_spacings) / expected_spacing if expected_spacing > 0 else 0.0
    
    # Span ratio: fraction of depth range covered
    span_ratio = (selected_indices[-1] - selected_indices[0]) / (original_depth - 1) if original_depth > 1 else 0.0
    
    return {
        'coverage': coverage,
        'uniformity': max(0.0, uniformity),
        'span_ratio': span_ratio,
        'mean_spacing': np.mean(actual_spacings) if len(actual_spacings) > 0 else 0.0,
        'std_spacing': np.std(actual_spacings) if len(actual_spacings) > 0 else 0.0
    }