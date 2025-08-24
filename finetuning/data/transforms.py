"""
Transforms module for OCT volumes.
Must match V-JEPA2 preprocessing: resize/crop to [64, 384, 384], same normalization.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ResizeVolume:
    """Resize 3D volume to target shape."""
    
    def __init__(self, target_shape: Tuple[int, int, int] = (64, 384, 384)):
        """
        Initialize resize transform.
        
        Args:
            target_shape: Target (D, H, W) shape
        """
        self.target_shape = target_shape
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Resize volume to target shape.
        
        Args:
            volume: Input volume [B, D, H, W] or [1, D, H, W]
            
        Returns:
            Resized volume with same batch dimension
        """
        if volume.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, D, H, W], got shape {volume.shape}")
        
        # Get current and target shapes
        current_shape = volume.shape[1:]  # (D, H, W)
        target_d, target_h, target_w = self.target_shape
        
        if current_shape == self.target_shape:
            return volume
        
        # Use trilinear interpolation for 3D volumes
        # F.interpolate expects [N, C, D, H, W] format, we have [N, D, H, W]
        # Add channel dimension: [N, D, H, W] -> [N, 1, D, H, W]
        volume_5d = volume.unsqueeze(1)
        
        resized = F.interpolate(
            volume_5d,
            size=(target_d, target_h, target_w),
            mode='trilinear',
            align_corners=False
        )
        
        # Remove channel dimension: [N, 1, D, H, W] -> [N, D, H, W]
        resized = resized.squeeze(1)
        
        logger.debug(f"Resized volume: {current_shape} -> {resized.shape[1:]}")
        return resized


class RandomFlip3D:
    """Random flips for 3D volumes."""
    
    def __init__(self, p: float = 0.5, axes: Tuple[int, ...] = (1, 2, 3)):
        """
        Initialize random flip transform.
        
        Args:
            p: Probability of applying flip
            axes: Axes to flip (1=D, 2=H, 3=W in [B, D, H, W] tensor)
        """
        self.p = p
        self.axes = axes
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply random flips."""
        if volume.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, D, H, W], got shape {volume.shape}")
        
        for axis in self.axes:
            if axis < volume.dim() and torch.rand(1).item() < self.p:  # Safety check
                volume = torch.flip(volume, dims=[axis])
        
        return volume


class IntensityJitter:
    """Random intensity jittering for augmentation."""
    
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1):
        """
        Initialize intensity jitter.
        
        Args:
            brightness: Brightness jitter range [-brightness, +brightness]
            contrast: Contrast jitter range [1-contrast, 1+contrast]
        """
        self.brightness = brightness
        self.contrast = contrast
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply intensity jittering."""
        if self.brightness > 0:
            brightness_factor = (torch.rand(1) * 2 - 1) * self.brightness  # Range: [-brightness, brightness]
            volume = volume + brightness_factor.item()
        
        if self.contrast > 0:
            contrast_factor = torch.rand(1) * (2 * self.contrast) + (1 - self.contrast)  # Range: [1-contrast, 1+contrast]
            volume = volume * contrast_factor.item()
        
        # Clamp to reasonable range
        volume = torch.clamp(volume, 0, 1)
        
        return volume


class GaussianNoise:
    """Add Gaussian noise for regularization."""
    
    def __init__(self, std: float = 0.01):
        """
        Initialize Gaussian noise.
        
        Args:
            std: Standard deviation of noise
        """
        self.std = std
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        if self.std > 0:
            noise = torch.randn_like(volume) * self.std
            volume = volume + noise
        
        return volume


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        """
        Initialize composed transforms.
        
        Args:
            transforms: List of transform callables
        """
        self.transforms = transforms
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            volume = transform(volume)
        return volume


def get_training_transforms(
    target_shape: Tuple[int, int, int] = (64, 384, 384),
    enable_augmentations: bool = True
) -> Compose:
    """
    Get training transforms pipeline.
    
    Args:
        target_shape: Target volume shape
        enable_augmentations: Whether to enable data augmentations
        
    Returns:
        Composed transforms
    """
    transforms = [
        ResizeVolume(target_shape=target_shape)
    ]
    
    if enable_augmentations:
        transforms.extend([
            RandomFlip3D(p=0.5, axes=(3, 4)),  # Flip H, W (not depth)
            IntensityJitter(brightness=0.05, contrast=0.1),
            GaussianNoise(std=0.005)
        ])
    
    return Compose(transforms)


def get_validation_transforms(
    target_shape: Tuple[int, int, int] = (64, 384, 384)
) -> Compose:
    """
    Get validation transforms pipeline (no augmentations).
    
    Args:
        target_shape: Target volume shape
        
    Returns:
        Composed transforms
    """
    return Compose([
        ResizeVolume(target_shape=target_shape)
    ])


class VJepa2Transforms:
    """Transforms pipeline matching V-JEPA2 pretraining exactly."""
    
    def __init__(
        self,
        target_shape: Tuple[int, int, int] = (64, 384, 384),
        augment: bool = False
    ):
        """
        Initialize V-JEPA2 compatible transforms.
        
        Args:
            target_shape: Target shape matching pretraining
            augment: Whether to apply augmentations
        """
        self.target_shape = target_shape
        self.augment = augment
        
        # Core transform (always applied)
        self.resize = ResizeVolume(target_shape=target_shape)
        
        # Augmentation transforms (optional)
        if augment:
            self.flip = RandomFlip3D(p=0.5, axes=(3, 4))  # H, W only
            self.intensity_jitter = IntensityJitter(brightness=0.05, contrast=0.05)
        
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Apply transforms pipeline.
        
        Args:
            volume: Input volume [1, D, H, W] or [B, D, H, W]
            
        Returns:
            Transformed volume with same batch structure
        """
        # Always resize to match pretraining
        volume = self.resize(volume)
        
        # Apply augmentations if enabled
        if self.augment:
            volume = self.flip(volume)
            volume = self.intensity_jitter(volume)
        
        # Final validation
        expected_shape = (1,) + self.target_shape if volume.shape[0] == 1 else (-1,) + self.target_shape
        if volume.shape[1:] != self.target_shape:
            logger.warning(f"Transform output shape {volume.shape} does not match expected {expected_shape}")
        
        return volume


def create_transforms(
    config: dict,
    is_training: bool = True
) -> VJepa2Transforms:
    """
    Create transforms from configuration.
    
    Args:
        config: Configuration dictionary with augmentation settings
        is_training: Whether this is for training (enables augmentations)
        
    Returns:
        Transforms pipeline
    """
    # Extract target shape from config
    target_shape = tuple(config.get('resize', [64, 384, 384]))
    
    # Determine if augmentations should be applied
    augment = is_training and config.get('augment', {}).get('flip', False)
    
    return VJepa2Transforms(
        target_shape=target_shape,
        augment=augment
    )


def validate_transform_output(volume: torch.Tensor, expected_shape: Tuple[int, int, int]) -> bool:
    """
    Validate that transform output has correct shape.
    
    Args:
        volume: Transformed volume
        expected_shape: Expected (D, H, W) shape
        
    Returns:
        True if valid
    """
    if volume.dim() not in [3, 4]:
        logger.error(f"Invalid volume dimensions: {volume.shape}")
        return False
    
    actual_shape = volume.shape[-3:] if volume.dim() == 4 else volume.shape
    
    if actual_shape != expected_shape:
        logger.error(f"Shape mismatch: {actual_shape} != {expected_shape}")
        return False
    
    if torch.isnan(volume).any() or torch.isinf(volume).any():
        logger.error("Volume contains NaN or infinite values after transforms")
        return False
    
    return True