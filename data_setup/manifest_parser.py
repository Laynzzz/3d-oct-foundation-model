"""Manifest parser for OCT device detection and file listing."""

import pandas as pd
import logging
from typing import List, Dict, Set, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class ManifestParser:
    """Parser for OCT manifest TSV files with device detection."""
    
    # Device mapping from filepath patterns
    DEVICE_MAPPING = {
        'heidelberg_spectralis': 'heidelberg_spectralis',
        'topcon_triton': 'topcon_triton', 
        'topcon_maestro2': 'topcon_maestro2',
        'zeiss_cirrus': 'zeiss_cirrus'
    }
    
    def __init__(self, manifest_path: str, gcs_root: str):
        """Initialize manifest parser.
        
        Args:
            manifest_path: Path to manifest TSV file
            gcs_root: GCS root path for data
        """
        self.manifest_path = manifest_path
        self.gcs_root = gcs_root.rstrip('/')
        self.manifest_df = None
        
    def load_manifest(self) -> pd.DataFrame:
        """Load and parse manifest TSV file.
        
        Returns:
            Pandas DataFrame with manifest data
        """
        try:
            # Load TSV file
            self.manifest_df = pd.read_csv(self.manifest_path, sep='\t')
            
            # Log basic information
            logger.info(f"Loaded manifest with {len(self.manifest_df)} entries")
            logger.info(f"Manifest columns: {list(self.manifest_df.columns)}")
            
            # Add device column
            self.manifest_df['device'] = self.manifest_df['filepath'].apply(self._extract_device)
            
            # Add full GCS path
            self.manifest_df['gcs_path'] = self.manifest_df['filepath'].apply(
                lambda x: f"{self.gcs_root}{x}"
            )
            
            # Log device counts
            device_counts = self.manifest_df['device'].value_counts()
            logger.info(f"Device distribution: {device_counts.to_dict()}")
            
            return self.manifest_df
            
        except Exception as e:
            logger.error(f"Failed to load manifest {self.manifest_path}: {e}")
            raise
    
    def _extract_device(self, filepath: str) -> str:
        """Extract device name from filepath.
        
        Args:
            filepath: File path from manifest
            
        Returns:
            Device name or 'unknown'
        """
        # Expected pattern: /retinal_oct/structural_oct/<DEVICE>/...
        path_parts = filepath.split('/')
        
        for part in path_parts:
            if part in self.DEVICE_MAPPING:
                return self.DEVICE_MAPPING[part]
        
        # Try to match any known device in the path
        for device_key in self.DEVICE_MAPPING:
            if device_key in filepath:
                return self.DEVICE_MAPPING[device_key]
        
        logger.warning(f"Unknown device for filepath: {filepath}")
        return 'unknown'
    
    def get_single_domain_files(self, device: str = 'topcon_triton') -> List[str]:
        """Get file list for single domain (default: topcon_triton).
        
        Args:
            device: Device to filter for
            
        Returns:
            List of GCS paths for the specified device
        """
        if self.manifest_df is None:
            raise ValueError("Manifest not loaded. Call load_manifest() first.")
        
        device_files = self.manifest_df[self.manifest_df['device'] == device]
        file_list = device_files['gcs_path'].tolist()
        
        logger.info(f"Found {len(file_list)} files for device '{device}'")
        return file_list
    
    def get_multi_domain_files(self, exclude_unknown: bool = True) -> List[str]:
        """Get file list for multi-domain training (all devices).
        
        Args:
            exclude_unknown: Whether to exclude files with unknown devices
            
        Returns:
            List of GCS paths for all devices
        """
        if self.manifest_df is None:
            raise ValueError("Manifest not loaded. Call load_manifest() first.")
        
        if exclude_unknown:
            valid_files = self.manifest_df[self.manifest_df['device'] != 'unknown']
        else:
            valid_files = self.manifest_df
        
        file_list = valid_files['gcs_path'].tolist()
        
        logger.info(f"Found {len(file_list)} files for multi-domain training")
        return file_list
    
    def get_device_stats(self) -> Dict[str, int]:
        """Get statistics about devices in the manifest.
        
        Returns:
            Dictionary with device counts
        """
        if self.manifest_df is None:
            raise ValueError("Manifest not loaded. Call load_manifest() first.")
        
        stats = self.manifest_df['device'].value_counts().to_dict()
        return stats
    
    def get_anatomic_region_stats(self) -> Dict[str, int]:
        """Get statistics about anatomic regions.
        
        Returns:
            Dictionary with anatomic region counts
        """
        if self.manifest_df is None:
            raise ValueError("Manifest not loaded. Call load_manifest() first.")
        
        if 'anatomic_region' in self.manifest_df.columns:
            stats = self.manifest_df['anatomic_region'].value_counts().to_dict()
            return stats
        else:
            logger.warning("No 'anatomic_region' column found in manifest")
            return {}
    
    def get_laterality_stats(self) -> Dict[str, int]:
        """Get statistics about laterality (left/right eye).
        
        Returns:
            Dictionary with laterality counts
        """
        if self.manifest_df is None:
            raise ValueError("Manifest not loaded. Call load_manifest() first.")
        
        if 'laterality' in self.manifest_df.columns:
            stats = self.manifest_df['laterality'].value_counts().to_dict()
            return stats
        else:
            logger.warning("No 'laterality' column found in manifest")
            return {}
    
    def filter_by_criteria(
        self,
        devices: Optional[List[str]] = None,
        anatomic_regions: Optional[List[str]] = None,
        laterality: Optional[List[str]] = None,
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """Filter files by various criteria.
        
        Args:
            devices: List of devices to include
            anatomic_regions: List of anatomic regions to include
            laterality: List of laterality values to include
            min_frames: Minimum number of frames
            max_frames: Maximum number of frames
            
        Returns:
            List of filtered GCS paths
        """
        if self.manifest_df is None:
            raise ValueError("Manifest not loaded. Call load_manifest() first.")
        
        filtered_df = self.manifest_df.copy()
        
        # Filter by devices
        if devices:
            filtered_df = filtered_df[filtered_df['device'].isin(devices)]
        
        # Filter by anatomic regions
        if anatomic_regions and 'anatomic_region' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['anatomic_region'].isin(anatomic_regions)]
        
        # Filter by laterality
        if laterality and 'laterality' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['laterality'].isin(laterality)]
        
        # Filter by frame count
        if min_frames is not None and 'number_of_frames' in filtered_df.columns:
            filtered_df = filtered_df[
                pd.to_numeric(filtered_df['number_of_frames'], errors='coerce') >= min_frames
            ]
        
        if max_frames is not None and 'number_of_frames' in filtered_df.columns:
            filtered_df = filtered_df[
                pd.to_numeric(filtered_df['number_of_frames'], errors='coerce') <= max_frames
            ]
        
        file_list = filtered_df['gcs_path'].tolist()
        
        logger.info(f"Filtered to {len(file_list)} files based on criteria")
        return file_list
    
    def stratified_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_by: str = 'device',
        random_state: int = 42
    ) -> Dict[str, List[str]]:
        """Create stratified train/val/test splits.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            stratify_by: Column to stratify by ('device', 'anatomic_region', etc.)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' lists of GCS paths
        """
        if self.manifest_df is None:
            raise ValueError("Manifest not loaded. Call load_manifest() first.")
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if stratify_by not in self.manifest_df.columns:
            raise ValueError(f"Column '{stratify_by}' not found in manifest")
        
        splits = {'train': [], 'val': [], 'test': []}
        
        # Stratify by the specified column
        for group_value in self.manifest_df[stratify_by].unique():
            group_df = self.manifest_df[self.manifest_df[stratify_by] == group_value]
            group_df = group_df.sample(frac=1, random_state=random_state)  # Shuffle
            
            n_total = len(group_df)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            # Split indices
            train_df = group_df.iloc[:n_train]
            val_df = group_df.iloc[n_train:n_train + n_val]
            test_df = group_df.iloc[n_train + n_val:]
            
            # Add to splits
            splits['train'].extend(train_df['gcs_path'].tolist())
            splits['val'].extend(val_df['gcs_path'].tolist())
            splits['test'].extend(test_df['gcs_path'].tolist())
        
        logger.info(f"Created stratified splits by '{stratify_by}': "
                   f"train={len(splits['train'])}, val={len(splits['val'])}, "
                   f"test={len(splits['test'])}")
        
        return splits
    
    def save_file_lists(self, output_dir: str, splits: Optional[Dict[str, List[str]]] = None):
        """Save file lists to text files.
        
        Args:
            output_dir: Directory to save file lists
            splits: Optional splits dictionary to save
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save single and multi-domain lists
        single_domain = self.get_single_domain_files()
        multi_domain = self.get_multi_domain_files()
        
        with open(output_path / 'single_domain_files.txt', 'w') as f:
            f.write('\n'.join(single_domain))
        
        with open(output_path / 'multi_domain_files.txt', 'w') as f:
            f.write('\n'.join(multi_domain))
        
        # Save splits if provided
        if splits:
            for split_name, file_list in splits.items():
                with open(output_path / f'{split_name}_files.txt', 'w') as f:
                    f.write('\n'.join(file_list))
        
        logger.info(f"Saved file lists to {output_dir}")