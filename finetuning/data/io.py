"""
I/O module for reading OCT volumes from various formats.
Supports DICOM, NIfTI, and NPY formats with normalization matching V-JEPA2 pretraining.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union
import logging
import io

logger = logging.getLogger(__name__)


def read_dicom_volume(data: bytes) -> torch.Tensor:
    """
    Read DICOM volume from bytes data.
    
    Args:
        data: Raw DICOM bytes
        
    Returns:
        3D tensor [D, H, W]
    """
    try:
        import pydicom
        from pydicom import dcmread
        
        # Read DICOM from bytes
        dataset = dcmread(io.BytesIO(data))
        
        # Check for pixel data
        if not hasattr(dataset, 'pixel_array') or (0x7FE0, 0x0010) not in dataset:
            raise ValueError("DICOM file has no pixel data")
        
        # Get pixel array
        pixel_array = dataset.pixel_array
        
        # Convert to float32 and ensure 3D
        if pixel_array.ndim == 2:
            # Single slice - add depth dimension
            pixel_array = pixel_array[np.newaxis, ...]
        elif pixel_array.ndim == 4:
            # Multi-frame with extra dimension - squeeze if needed
            pixel_array = np.squeeze(pixel_array)
            
        if pixel_array.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape: {pixel_array.shape}")
        
        # Convert to tensor
        volume = torch.from_numpy(pixel_array.astype(np.float32))
        
        # Ensure proper orientation (D, H, W)
        if volume.shape[0] > volume.shape[-1]:  # Likely (H, W, D) -> (D, H, W)
            volume = volume.permute(2, 0, 1)
        
        logger.debug(f"Loaded DICOM volume: {volume.shape}")
        return volume
        
    except Exception as e:
        logger.error(f"Failed to read DICOM data: {e}")
        raise


def read_nifti_volume(data: bytes) -> torch.Tensor:
    """
    Read NIfTI volume from bytes data.
    
    Args:
        data: Raw NIfTI bytes
        
    Returns:
        3D tensor [D, H, W]
    """
    try:
        import nibabel as nib
        
        # Read NIfTI from bytes
        nii_img = nib.Nifti1Image.from_bytes(data)
        pixel_array = nii_img.get_fdata()
        
        # Ensure 3D
        if pixel_array.ndim == 4:
            # Take first volume if 4D
            pixel_array = pixel_array[..., 0]
        
        if pixel_array.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape: {pixel_array.shape}")
        
        # Convert to tensor
        volume = torch.from_numpy(pixel_array.astype(np.float32))
        
        logger.debug(f"Loaded NIfTI volume: {volume.shape}")
        return volume
        
    except Exception as e:
        logger.error(f"Failed to read NIfTI data: {e}")
        raise


def read_numpy_volume(data: bytes) -> torch.Tensor:
    """
    Read NumPy volume from bytes data.
    
    Args:
        data: Raw NPY/NPZ bytes
        
    Returns:
        3D tensor [D, H, W]
    """
    try:
        # Try NPY format first
        try:
            pixel_array = np.load(io.BytesIO(data))
        except:
            # Try NPZ format
            npz_data = np.load(io.BytesIO(data))
            # Get first array from NPZ
            key = list(npz_data.keys())[0]
            pixel_array = npz_data[key]
        
        # Ensure 3D
        if pixel_array.ndim == 4:
            pixel_array = pixel_array[0]  # Take first batch
        elif pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]  # Add depth
        
        if pixel_array.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape: {pixel_array.shape}")
        
        # Convert to tensor
        volume = torch.from_numpy(pixel_array.astype(np.float32))
        
        logger.debug(f"Loaded NumPy volume: {volume.shape}")
        return volume
        
    except Exception as e:
        logger.error(f"Failed to read NumPy data: {e}")
        raise


def normalize_volume(volume: torch.Tensor, method: str = "vjepa2") -> torch.Tensor:
    """
    Normalize volume to match V-JEPA2 pretraining.
    
    Args:
        volume: Input volume tensor [D, H, W]
        method: Normalization method ("vjepa2", "zero_one", "z_score")
        
    Returns:
        Normalized volume tensor
    """
    if method == "vjepa2":
        # Match V-JEPA2 pretraining normalization
        # Based on typical OCT preprocessing: intensity normalization
        volume = volume.float()
        
        # FAST normalization: Skip expensive quantile computation (Performance Fix)
        # Use cheap approximation based on finetune-fix.md analysis
        try:
            # Downsample for VERY fast quantile approximation (4x4x4 subsampling)
            if volume.numel() > 1e6:  # Only downsample if large enough
                vol_small = volume[::4, ::4, ::4].flatten()
                if vol_small.numel() > 0:
                    q99 = torch.quantile(vol_small, 0.99)
                else:
                    q99 = 0.99 * volume.max()
            else:
                q99 = torch.quantile(volume, 0.99)
            volume = torch.clamp(volume, 0, q99)
        except RuntimeError as e:
            logger.warning(f"Quantile calculation failed, using max value clipping: {e}")
            # Fallback: use 99% of max value (much faster)
            volume = torch.clamp(volume, 0, 0.99 * volume.max())
        
        # Normalize to [0, 1] range
        volume_min = volume.min()
        volume_max = volume.max()
        if volume_max > volume_min:
            volume = (volume - volume_min) / (volume_max - volume_min)
        
        # Apply slight contrast enhancement (gamma correction)
        volume = torch.pow(volume, 0.8)
        
    elif method == "zero_one":
        # Simple [0, 1] normalization
        volume_min = volume.min()
        volume_max = volume.max()
        if volume_max > volume_min:
            volume = (volume - volume_min) / (volume_max - volume_min)
        
    elif method == "z_score":
        # Z-score normalization
        volume_mean = volume.mean()
        volume_std = volume.std()
        if volume_std > 0:
            volume = (volume - volume_mean) / volume_std
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return volume


def read_volume(s3fs, key: str, cache_dir: Optional[str] = None) -> torch.Tensor:
    """
    Read OCT volume from B2 storage and return as normalized tensor.
    
    Args:
        s3fs: S3 filesystem instance (from storage.b2.get_s3fs())
        key: B2 object key
        cache_dir: Optional local cache directory
        
    Returns:
        Normalized volume tensor [1, D, H, W] ready for model input
    """
    from ..storage.b2 import read_with_cache
    
    try:
        # Extract bucket name from key if present
        if key.startswith('eye-dataset/'):
            bucket_name = 'eye-dataset'
            clean_key = key.replace('eye-dataset/', '')
        else:
            bucket_name = 'eye-dataset'
            clean_key = key
        
        # Read raw data
        data = read_with_cache(bucket_name, clean_key, cache_dir)
        
        # Determine file format from key
        key_lower = clean_key.lower()
        
        if '.dcm' in key_lower or '.dicom' in key_lower:
            volume = read_dicom_volume(data)
        elif '.nii' in key_lower:
            volume = read_nifti_volume(data)
        elif '.npy' in key_lower or '.npz' in key_lower:
            volume = read_numpy_volume(data)
        else:
            # Try formats in order of preference
            for reader in [read_dicom_volume, read_nifti_volume, read_numpy_volume]:
                try:
                    volume = reader(data)
                    break
                except:
                    continue
            else:
                raise ValueError(f"Could not read volume from key: {key}")
        
        # Normalize volume
        volume = normalize_volume(volume, method="vjepa2")
        
        # Add batch dimension: [D, H, W] -> [1, D, H, W]
        volume = volume.unsqueeze(0)
        
        logger.debug(f"Successfully loaded volume {key}: {volume.shape}")
        return volume
        
    except Exception as e:
        logger.error(f"Failed to read volume from {key}: {e}")
        raise


def validate_volume_shape(volume: torch.Tensor, expected_shape: Tuple[int, int, int] = (64, 384, 384)) -> bool:
    """
    Validate that volume has expected shape for V-JEPA2 input.
    
    Args:
        volume: Volume tensor [1, D, H, W] or [D, H, W]
        expected_shape: Expected (D, H, W) shape
        
    Returns:
        True if shape is valid
    """
    if volume.dim() == 4:
        actual_shape = volume.shape[1:]  # Remove batch dimension
    elif volume.dim() == 3:
        actual_shape = volume.shape
    else:
        logger.warning(f"Unexpected volume dimensions: {volume.shape}")
        return False
    
    if actual_shape != expected_shape:
        logger.warning(f"Volume shape {actual_shape} != expected {expected_shape}")
        return False
    
    return True


def check_volume_integrity(volume: torch.Tensor) -> bool:
    """
    Check volume for common issues (NaN, infinite values, etc.).
    
    Args:
        volume: Volume tensor
        
    Returns:
        True if volume passes integrity checks
    """
    if torch.isnan(volume).any():
        logger.error("Volume contains NaN values")
        return False
    
    if torch.isinf(volume).any():
        logger.error("Volume contains infinite values")
        return False
    
    if volume.numel() == 0:
        logger.error("Volume is empty")
        return False
    
    # Check reasonable value range
    vol_min, vol_max = volume.min(), volume.max()
    if vol_min == vol_max:
        logger.warning("Volume has constant intensity")
        # Not necessarily an error, but suspicious
    
    logger.debug(f"Volume integrity check passed: shape={volume.shape}, range=[{vol_min:.3f}, {vol_max:.3f}]")
    return True