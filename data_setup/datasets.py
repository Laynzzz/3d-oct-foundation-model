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
        
        logger.info(f"Initialized OCTDICOMDataset with {len(file_list)} files")
        logger.info(f"Target spacing: {target_spacing} mm")
        logger.info(f"Target image size: {image_size}")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get item by index.
        
        Args:
            idx: Dataset index
            
        Returns:
            Dictionary with keys:
            - 'image': Tensor[C=1,D,H,W] - normalized volume
            - 'spacing': tuple (dz,dy,dx) - voxel spacing in mm
            - 'meta': dict - metadata from DICOM
            
            None if file cannot be loaded
        """
        if idx >= len(self.file_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.file_list)}")
        
        gcs_path = self.file_list[idx]
        
        # Read DICOM volume
        dicom_data = self.dicom_reader.read_dicom_volume(gcs_path)
        
        if dicom_data is None:
            logger.warning(f"Failed to load DICOM at index {idx}: {gcs_path}")
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
    
    def get_file_at_index(self, idx: int) -> str:
        """Get file path at given index."""
        return self.file_list[idx]
    
    def get_device_for_index(self, idx: int) -> str:
        """Get device name for file at given index."""
        filepath = self.file_list[idx]
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
    """Custom collate function that handles None samples.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data dictionary
    """
    # Filter out None samples
    valid_samples = [sample for sample in batch if sample is not None]
    
    if len(valid_samples) == 0:
        raise RuntimeError("No valid samples in batch")
    
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