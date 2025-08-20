"""MONAI 3D transforms for OCT pretraining."""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from monai.transforms import (
    Compose, MapTransform,
    Spacingd, NormalizeIntensityd, 
    RandSpatialCropd, RandFlipd, RandAffined, RandGaussianNoised,
    Resized, CenterSpatialCropd, SpatialPadd
)
from monai.data import MetaTensor
import logging


logger = logging.getLogger(__name__)


class LoadDICOMd(MapTransform):
    """Custom DICOM loader that works with our GCS reader output.
    
    This replaces LoadImaged since we already have the pixel data loaded.
    """
    
    def __init__(self, keys: List[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert image to MetaTensor format expected by MONAI.
        
        Args:
            data: Dictionary with 'image' key containing tensor [C, D, H, W]
            
        Returns:
            Data with image converted to MetaTensor
        """
        d = dict(data)
        
        for key in self.key_iterator(d):
            if key in d:
                # Image is already a tensor [C, D, H, W], convert to MetaTensor
                img = d[key]
                if not isinstance(img, MetaTensor):
                    # Add metadata for MONAI transforms
                    d[key] = MetaTensor(
                        img, 
                        meta={'spacing': d.get('spacing', (1.0, 1.0, 1.0))}
                    )
        
        return d


class JEPAMaskGeneratord(MapTransform):
    """Generate binary masks for JEPA targets over patch grid.
    
    As specified in section 4.3: mask ratio ~ 0.6
    """
    
    def __init__(
        self, 
        keys: List[str], 
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        mask_ratio: float = 0.6,
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mask for JEPA targets.
        
        Args:
            data: Dictionary containing image data
            
        Returns:
            Data with added 'mask' key containing binary mask
        """
        d = dict(data)
        
        for key in self.key_iterator(d):
            if key in d:
                img = d[key]
                # Image shape: [C, D, H, W]
                _, D, H, W = img.shape
                
                # Calculate patch grid dimensions
                patch_d, patch_h, patch_w = self.patch_size
                grid_d = D // patch_d
                grid_h = H // patch_h
                grid_w = W // patch_w
                
                total_patches = grid_d * grid_h * grid_w
                num_masked = int(total_patches * self.mask_ratio)
                
                # Create flat mask
                mask_flat = torch.zeros(total_patches, dtype=torch.bool)
                masked_indices = torch.randperm(total_patches)[:num_masked]
                mask_flat[masked_indices] = True
                
                # Reshape to patch grid
                mask_grid = mask_flat.reshape(grid_d, grid_h, grid_w)
                
                # Expand to full image resolution
                mask_full = torch.repeat_interleave(
                    torch.repeat_interleave(
                        torch.repeat_interleave(mask_grid, patch_d, dim=0),
                        patch_h, dim=1
                    ),
                    patch_w, dim=2
                )
                
                # Crop to exact image size if needed
                mask_full = mask_full[:D, :H, :W]
                
                # Add to data
                d['mask'] = mask_full
                d['mask_ratio'] = self.mask_ratio
                d['num_masked_patches'] = num_masked
                d['total_patches'] = total_patches
        
        return d


def create_pretraining_transforms(
    target_spacing: Tuple[float, float, float] = (0.05, 0.02, 0.02),
    image_size: Tuple[int, int, int] = (64, 384, 384),
    patch_size: Tuple[int, int, int] = (4, 16, 16),
    mask_ratio: float = 0.6
) -> Compose:
    """Create MONAI 3D transforms pipeline for pretraining.
    
    FIXED: Removed problematic Spacingd transform that was causing memory issues.
    Instead, we directly resize to target image size, which is more practical for training.
    
    Args:
        target_spacing: Target voxel spacing (dz, dy, dx) in mm (kept for config compatibility)
        image_size: Target image size (D, H, W)
        patch_size: Patch size for vision transformer
        mask_ratio: Mask ratio for JEPA targets
        
    Returns:
        MONAI Compose transform pipeline
    """
    transforms = [
        # Load image (custom loader for our DICOM data)
        LoadDICOMd(keys=['image']),
        
        # Resize directly to target image size (skip problematic spacing resample)
        Resized(
            keys=['image'],
            spatial_size=image_size,
            mode='trilinear'
        ),
        
        # Normalize intensity (z-score normalization already done in reader)
        NormalizeIntensityd(
            keys=['image'],
            nonzero=False,  # Use all voxels including zeros
            channel_wise=False
        ),
        
        # Random spatial crop to sample 3D patches
        RandSpatialCropd(
            keys=['image'],
            roi_size=image_size,
            random_center=True,
            random_size=False
        ),
        
        # Random flip (spatial axes)
        RandFlipd(
            keys=['image'],
            spatial_axis=[0, 1, 2],  # D, H, W axes
            prob=0.5
        ),
        
        # Random affine (small translations/rotations)
        RandAffined(
            keys=['image'],
            mode='bilinear',
            prob=0.5,
            rotate_range=[0.1, 0.1, 0.1],  # Small rotations in radians
            translate_range=[5, 5, 5],     # Small translations in voxels
            scale_range=[0.05, 0.05, 0.05], # Small scaling
            padding_mode='zeros'
        ),
        
        # Random Gaussian noise (low σ)
        RandGaussianNoised(
            keys=['image'],
            prob=0.3,
            mean=0.0,
            std=0.05  # Low standard deviation
        ),
        
        # Generate mask for JEPA targets
        JEPAMaskGeneratord(
            keys=['image'],
            patch_size=patch_size,
            mask_ratio=mask_ratio
        )
    ]
    
    return Compose(transforms)


def create_validation_transforms(
    target_spacing: Tuple[float, float, float] = (0.05, 0.02, 0.02),
    image_size: Tuple[int, int, int] = (64, 384, 384)
) -> Compose:
    """Create transforms for validation (no augmentation).
    
    FIXED: Removed problematic Spacingd transform.
    
    Args:
        target_spacing: Target voxel spacing (dz, dy, dx) in mm (kept for config compatibility)
        image_size: Target image size (D, H, W)
        
    Returns:
        MONAI Compose transform pipeline for validation
    """
    transforms = [
        # Load image
        LoadDICOMd(keys=['image']),
        
        # Resize directly to target image size (skip problematic spacing resample)
        Resized(
            keys=['image'],
            spatial_size=image_size,
            mode='trilinear'
        ),
        
        # Normalize intensity
        NormalizeIntensityd(
            keys=['image'],
            nonzero=False,
            channel_wise=False
        ),
        
        # Center crop to ensure exact size
        CenterSpatialCropd(
            keys=['image'],
            roi_size=image_size
        )
    ]
    
    return Compose(transforms)


class TwoViewTransform:
    """Transform that creates two augmented views for JEPA.
    
    For each volume, sample 2 augmented 3D views: context and target.
    The target view is masked (cube masks over patch grid).
    """
    
    def __init__(
        self,
        base_transforms: Compose,
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        mask_ratio: float = 0.6
    ):
        self.base_transforms = base_transforms
        self.mask_generator = JEPAMaskGeneratord(
            keys=['image'],
            patch_size=patch_size,
            mask_ratio=mask_ratio
        )
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Create two views: context and target.
        
        Args:
            sample: Input sample
            
        Returns:
            Sample with 'context_view' and 'target_view' keys
        """
        # Create context view (no mask)
        context_sample = dict(sample)
        context_view = self.base_transforms(context_sample)
        
        # Create target view (with mask)
        target_sample = dict(sample)
        target_view = self.base_transforms(target_sample)
        target_view = self.mask_generator(target_view)
        
        return {
            'context_view': context_view,
            'target_view': target_view,
            'meta': sample.get('meta', {})
        }