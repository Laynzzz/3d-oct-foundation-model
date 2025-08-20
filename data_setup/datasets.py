"""OCT DICOM datasets for pretraining and fine-tuning."""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Callable
import logging
from .gcs_dicom_reader import GCSDICOMReader
from .manifest_parser import ManifestParser


logger = logging.getLogger(__name__)


class OCTDICOMDataset(Dataset):
    """OCT DICOM dataset with GCS streaming and transforms.
    
    As specified in section 4.2 of plan.md:
    - Returns dict with keys: {'image': Tensor[C=1,D,H,W], 'spacing': (dz,dy,dx), 'meta': {...}}
    - Image shape policy: resample to fixed voxel spacing then resize/crop to D×H×W = 64×384×384
    """
    
    def __init__(
        self,
        manifest_path: str,
        gcs_root: str,
        file_list: List[str],
        transforms: Optional[Callable] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        target_spacing: tuple = (0.05, 0.02, 0.02),
        image_size: tuple = (64, 384, 384)
    ):
        """Initialize OCT DICOM dataset.
        
        Args:
            manifest_path: Path to manifest TSV file
            gcs_root: GCS root path
            file_list: List of file paths to include
            transforms: Optional transform pipeline
            use_cache: Whether to use local caching
            cache_dir: Cache directory path
            target_spacing: Target voxel spacing (dz, dy, dx) in mm
            image_size: Target image size (D, H, W)
        """
        self.manifest_path = manifest_path
        self.gcs_root = gcs_root
        self.file_list = file_list
        self.transforms = transforms
        self.target_spacing = target_spacing
        self.image_size = image_size
        
        # Initialize DICOM reader
        self.dicom_reader = GCSDICOMReader(
            use_cache=use_cache,
            cache_dir=cache_dir
        )
        
        # Filter out known problematic files (optional)
        self.filtered_file_list = self._filter_problematic_files(file_list)
        
        logger.info(f"Initialized OCTDICOMDataset with {len(self.filtered_file_list)} files (filtered from {len(file_list)})")
        logger.info(f"Target spacing: {target_spacing} mm")
        logger.info(f"Target image size: {image_size}")
    
    def _filter_problematic_files(self, file_list: List[str]) -> List[str]:
        """Filter out known problematic files.
        
        Args:
            file_list: Original list of files
            
        Returns:
            Filtered list with problematic files removed
        """
        # Known problematic patterns from the error logs
        problematic_patterns = [
            # Add patterns for files that consistently fail
            # These can be updated based on training experience
        ]
        
        filtered_list = []
        for file_path in file_list:
            # Skip files matching problematic patterns
            if any(pattern in file_path for pattern in problematic_patterns):
                logger.debug(f"Filtering out problematic file: {file_path}")
                continue
            filtered_list.append(file_path)
        
        logger.info(f"Filtered out {len(file_list) - len(filtered_list)} problematic files")
        return filtered_list
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.filtered_file_list)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get item by index with robust error handling.
        
        Args:
            idx: Dataset index
            
        Returns:
            Dictionary with keys:
            - 'image': Tensor[C=1,D,H,W] - normalized volume
            - 'spacing': tuple (dz,dy,dx) - voxel spacing in mm
            - 'meta': dict - metadata from DICOM
            
            None if file cannot be loaded
        """
        if idx >= len(self.filtered_file_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.filtered_file_list)}")
        
        gcs_path = self.filtered_file_list[idx]
        
        # Try to read DICOM volume with retry logic
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                dicom_data = self.dicom_reader.read_dicom_volume(gcs_path)
                if dicom_data is not None:
                    break
                elif attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed for index {idx}: {gcs_path}")
                    continue
                else:
                    logger.warning(f"Failed to load DICOM at index {idx} after {max_retries + 1} attempts: {gcs_path}")
                    return None
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed for index {idx}: {gcs_path} - {e}")
                    continue
                else:
                    logger.error(f"Failed to load DICOM at index {idx} after {max_retries + 1} attempts: {gcs_path} - {e}")
                    return None
        
        # Extract data
        pixel_array = dicom_data['pixel_array']  # [frames, height, width]
        spacing = dicom_data['spacing']          # (dz, dy, dx)
        metadata = dicom_data['metadata']
        
        # Convert to tensor and add channel dimension: [C=1, D, H, W]
        image = torch.from_numpy(pixel_array).unsqueeze(0).float()
        
        # Create sample dictionary
        sample = {
            'image': image,
            'spacing': spacing,
            'meta': {
                **metadata,
                'original_shape': pixel_array.shape,
                'filepath': gcs_path,
                'target_spacing': self.target_spacing,
                'target_size': self.image_size
            }
        }
        
        # Apply transforms if provided
        if self.transforms is not None:
            try:
                sample = self.transforms(sample)
            except Exception as e:
                logger.error(f"Transform failed for {gcs_path}: {e}")
                return None
        
        return sample
    
    def get_next_valid_sample(self, start_idx: int, max_lookahead: int = 10) -> Optional[Dict[str, Any]]:
        """Get the next valid sample starting from start_idx.
        
        This is useful when the DataLoader encounters None samples and needs
        to find a working sample to continue training.
        
        Args:
            start_idx: Starting index to search from
            max_lookahead: Maximum number of indices to look ahead
            
        Returns:
            Valid sample or None if none found within lookahead
        """
        for offset in range(max_lookahead):
            idx = (start_idx + offset) % len(self.filtered_file_list)
            try:
                sample = self.__getitem__(idx)
                if sample is not None:
                    logger.debug(f"Found valid sample at index {idx} (offset {offset})")
                    return sample
            except Exception as e:
                logger.debug(f"Error at index {idx}: {e}")
                continue
        
        logger.warning(f"No valid samples found within {max_lookahead} indices from {start_idx}")
        return None
    
    def get_file_at_index(self, idx: int) -> str:
        """Get file path at given index."""
        return self.filtered_file_list[idx]
    
    def get_original_file_list(self) -> List[str]:
        """Get the original unfiltered file list for debugging."""
        return self.file_list
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about dataset health and filtering.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'total_files': len(self.file_list),
            'filtered_files': len(self.filtered_file_list),
            'removed_files': len(self.file_list) - len(self.filtered_file_list),
            'filtering_ratio': (len(self.file_list) - len(self.filtered_file_list)) / len(self.file_list) if self.file_list else 0,
            'dicom_reader_stats': {
                'skipped_files': self.dicom_reader.skipped_files
            }
        }
    
    def get_device_for_index(self, idx: int) -> str:
        """Get device name for file at given index."""
        filepath = self.filtered_file_list[idx]
        # Extract device from GCS path
        for device in ['heidelberg_spectralis', 'topcon_triton', 'topcon_maestro2', 'zeiss_cirrus']:
            if device in filepath:
                return device
        return 'unknown'


def create_file_lists(
    manifest_path: str,
    gcs_root: str,
    list_strategy: str = 'single_domain'
) -> List[str]:
    """Create file lists based on strategy.
    
    Args:
        manifest_path: Path to manifest TSV
        gcs_root: GCS root path
        list_strategy: 'single_domain' or 'multi_domain'
        
    Returns:
        List of GCS file paths
    """
    parser = ManifestParser(manifest_path, gcs_root)
    parser.load_manifest()
    
    if list_strategy == 'single_domain':
        # Filter device == 'topcon_triton' by parsing filepath
        file_list = parser.get_single_domain_files('topcon_triton')
    elif list_strategy == 'multi_domain':
        # Include all devices
        file_list = parser.get_multi_domain_files()
    else:
        raise ValueError(f"Unknown list_strategy: {list_strategy}")
    
    logger.info(f"Created file list with strategy '{list_strategy}': {len(file_list)} files")
    return file_list


def stratified_split_by_device(
    manifest_path: str,
    gcs_root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Dict[str, List[str]]:
    """Create stratified splits by device.
    
    Optionally stratify by anatomic_region & laterality to balance contexts.
    
    Args:
        manifest_path: Path to manifest TSV
        gcs_root: GCS root path
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' file lists
    """
    parser = ManifestParser(manifest_path, gcs_root)
    parser.load_manifest()
    
    # Create stratified splits by device
    splits = parser.stratified_split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by='device',
        random_state=random_state
    )
    
    return splits


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Custom collate function that handles None samples with robust error handling.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data dictionary
    """
    # Filter out None samples
    valid_samples = [sample for sample in batch if sample is not None]
    
    if len(valid_samples) == 0:
        # Try to get a valid sample from the dataset if available
        logger.error("No valid samples in batch - this indicates a serious data loading issue")
        logger.error("Batch contents:")
        for i, sample in enumerate(batch):
            logger.error(f"  Sample {i}: {type(sample)} - {sample}")
        
        # Try to provide more context about what went wrong
        raise RuntimeError("No valid samples in batch - check DICOM file integrity and GCS permissions")
    
    # Log batch statistics for debugging
    total_samples = len(batch)
    valid_count = len(valid_samples)
    if valid_count < total_samples:
        logger.warning(f"Batch had {total_samples - valid_count}/{total_samples} failed samples")
    
    try:
        # Stack images
        images = torch.stack([sample['image'] for sample in valid_samples])
        
        # Collect spacings and metadata
        spacings = [sample['spacing'] for sample in valid_samples]
        metas = [sample['meta'] for sample in valid_samples]
        
        return {
            'image': images,
            'spacing': spacings,
            'meta': metas,
            'batch_size': len(valid_samples)
        }
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}")
        logger.error(f"Valid samples: {valid_count}, Total samples: {total_samples}")
        # Try to provide more debugging info
        for i, sample in enumerate(valid_samples):
            if sample is not None:
                logger.debug(f"Sample {i}: keys={list(sample.keys())}, image_shape={sample.get('image', 'NO_IMAGE').shape if hasattr(sample.get('image', 'NO_IMAGE'), 'shape') else 'NO_SHAPE'}")
        raise