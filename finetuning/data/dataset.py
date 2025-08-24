"""
Dataset module for OCT classification fine-tuning.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Optional, Dict, Tuple, Any, Callable
import logging

from .locator import OCTLocator
from .io import read_volume, check_volume_integrity
from ..storage.b2 import get_s3fs

logger = logging.getLogger(__name__)


class OCTVolumeDataset(Dataset):
    """Dataset for OCT volume classification."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        transforms: Optional[Callable] = None,
        locator: Optional[OCTLocator] = None,
        cache_dir: Optional[str] = None,
        bucket_name: str = "eye-dataset"
    ):
        """
        Initialize OCT dataset.
        
        Args:
            df: DataFrame with participant_id and class_label columns
            transforms: Optional transform callable
            locator: OCTLocator instance for resolving participant IDs to keys
            cache_dir: Optional local cache directory
            bucket_name: B2 bucket name
        """
        self.df = df.copy()
        self.transforms = transforms
        self.locator = locator or OCTLocator(bucket_name=bucket_name)
        self.cache_dir = cache_dir
        self.bucket_name = bucket_name
        
        # Initialize S3 filesystem
        self.s3fs = get_s3fs()
        
        # Validate required columns
        required_cols = ['participant_id', 'class_label']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter out participants without OCT data
        self._filter_available_data()
        
        logger.info(f"Initialized dataset with {len(self)} samples")
    
    def _filter_available_data(self):
        """Filter DataFrame to only include participants with available OCT data."""
        available_participants = set(self.locator.get_available_participants())
        
        # Convert participant IDs to strings for comparison
        self.df['participant_id'] = self.df['participant_id'].astype(str)
        
        # Filter to available participants
        before_count = len(self.df)
        self.df = self.df[self.df['participant_id'].isin(available_participants)].copy()
        after_count = len(self.df)
        
        if after_count < before_count:
            logger.warning(f"Filtered dataset: {before_count} -> {after_count} samples "
                          f"({before_count - after_count} participants without OCT data)")
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (volume, class_label, participant_id)
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        row = self.df.iloc[idx]
        participant_id = str(row['participant_id'])
        class_label = int(row['class_label'])
        
        try:
            # Resolve participant ID to OCT key
            oct_key = self.locator.resolve_oct_key(participant_id)
            if oct_key is None:
                raise ValueError(f"No OCT data found for participant {participant_id}")
            
            # Read volume
            volume = read_volume(self.s3fs, oct_key, cache_dir=self.cache_dir)
            
            # Check volume integrity
            if not check_volume_integrity(volume):
                raise ValueError(f"Volume integrity check failed for {participant_id}")
            
            # Apply transforms if provided
            if self.transforms is not None:
                volume = self.transforms(volume)
            
            return volume, class_label, participant_id
            
        except Exception as e:
            logger.error(f"Failed to load sample {idx} (participant {participant_id}): {e}")
            # Return a placeholder or re-raise based on strategy
            raise
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution in the dataset."""
        return self.df['class_label'].value_counts().sort_index().to_dict()
    
    def get_participant_ids(self) -> list:
        """Get list of participant IDs in the dataset."""
        return self.df['participant_id'].tolist()
    
    def refresh_oct_cache(self):
        """Refresh the OCT locator cache."""
        self.locator.refresh_cache()
        self._filter_available_data()


def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Custom collate function that handles failed samples gracefully.
    
    Args:
        batch: List of (volume, class_label, participant_id) tuples
        
    Returns:
        Tuple of (volumes, labels, participant_ids)
    """
    # Filter out None values (failed samples)
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        # Return empty batch
        logger.warning("Empty batch after filtering failed samples")
        return torch.empty(0), torch.empty(0, dtype=torch.long), []
    
    volumes, labels, participant_ids = zip(*valid_batch)
    
    # Stack volumes and labels
    volumes = torch.stack(volumes, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return volumes, labels, list(participant_ids)


def create_dataloader(
    df: pd.DataFrame,
    batch_size: int = 2,
    transforms: Optional[Callable] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    use_distributed: bool = False,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for OCT classification.
    
    Args:
        df: DataFrame with participant data
        batch_size: Batch size
        transforms: Optional transforms
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        cache_dir: Optional cache directory
        use_distributed: Whether to use DistributedSampler for TPU training
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader instance
    """
    dataset = OCTVolumeDataset(
        df=df,
        transforms=transforms,
        cache_dir=cache_dir
    )
    
    # Handle distributed training
    sampler = None
    if use_distributed:
        try:
            # Try to import XLA for TPU distributed training
            from torch_xla.core import xla_model as xm
            from torch.utils.data import DistributedSampler
            
            sampler = DistributedSampler(
                dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle,
                drop_last=False
            )
            # Don't shuffle in DataLoader when using sampler
            shuffle = False
            logger.info(f"Using DistributedSampler with {xm.xrt_world_size()} replicas, rank {xm.get_ordinal()}")
        except ImportError:
            logger.warning("torch_xla not available, falling back to regular DataLoader")
    
    # Set safe defaults for multiprocessing
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'sampler': sampler,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': False,  # TPU doesn't benefit from pinning
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else None,
        **kwargs
    }
    
    # Remove None values
    dataloader_kwargs = {k: v for k, v in dataloader_kwargs.items() if v is not None}
    
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


class DebugDataset(OCTVolumeDataset):
    """Debug version of OCT dataset that returns dummy data instead of loading from B2."""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Return dummy data for debugging."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        row = self.df.iloc[idx]
        participant_id = str(row['participant_id'])
        class_label = int(row['class_label'])
        
        # Create dummy volume matching V-JEPA2 expected input: [1, 64, 384, 384]
        volume = torch.randn(1, 64, 384, 384)
        
        # Apply transforms if provided
        if self.transforms is not None:
            volume = self.transforms(volume)
        
        return volume, class_label, participant_id


def create_debug_dataloader(
    df: pd.DataFrame,
    batch_size: int = 2,
    transforms: Optional[Callable] = None,
    shuffle: bool = True,
    **kwargs
) -> torch.utils.data.DataLoader:
    """Create debug DataLoader with dummy data."""
    dataset = DebugDataset(df=df, transforms=transforms)
    
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': 0,  # No multiprocessing for debug
        'collate_fn': collate_fn,
        **kwargs
    }
    
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)