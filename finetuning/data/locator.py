"""
OCT data locator module for resolving participant IDs to B2 object keys.
Fine-tuning dataset has different structure than pretraining data.
"""

import json
import os
from typing import Dict, List, Optional
import logging
from functools import lru_cache

from ..storage.b2 import get_s3fs, list_bucket_contents

logger = logging.getLogger(__name__)


class OCTLocator:
    """Locates OCT volumes for participants in the fine-tuning dataset."""
    
    def __init__(self, bucket_name: str = "eye-dataset", cache_file: Optional[str] = None):
        """
        Initialize OCT locator.
        
        Args:
            bucket_name: Name of the B2 bucket
            cache_file: Optional path to cache file for key mappings
        """
        self.bucket_name = bucket_name
        self.cache_file = cache_file or "oct_key_cache.json"
        self._key_cache: Optional[Dict[str, str]] = None
    
    def _load_cache(self) -> Dict[str, str]:
        """Load key cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded {len(cache)} cached keys from {self.cache_file}")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_file}: {e}")
        
        return {}
    
    def _save_cache(self, cache: Dict[str, str]):
        """Save key cache to file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Saved {len(cache)} keys to cache: {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_file}: {e}")
    
    def _build_key_mapping(self, force_rebuild: bool = False) -> Dict[str, str]:
        """
        Build mapping from participant_id to OCT file key.
        
        Args:
            force_rebuild: If True, rebuild cache from scratch
            
        Returns:
            Dictionary mapping participant_id to file key
        """
        if not force_rebuild and self._key_cache is not None:
            return self._key_cache
        
        # Load from cache first
        cache = self._load_cache() if not force_rebuild else {}
        
        if not cache or force_rebuild:
            logger.info("Building OCT key mapping from bucket contents...")
            
            # List all files in the fine-tuning data directory
            try:
                # Try common prefixes for fine-tuning data
                prefixes_to_try = [
                    "fine-tuneing-data/",  # As specified in plan (note: typo preserved)
                    "fine-tuning-data/",   # Common alternative
                    "participants/",       # Alternative structure
                    ""                     # Root level
                ]
                
                all_keys = []
                for prefix in prefixes_to_try:
                    try:
                        keys = list_bucket_contents(self.bucket_name, prefix, max_keys=10000)
                        # Filter for OCT-related files (common formats)
                        oct_keys = [k for k in keys if any(ext in k.lower() for ext in 
                                   ['.dcm', '.dicom', '.nii', '.nii.gz', '.npy', '.npz'])]
                        if oct_keys:
                            logger.info(f"Found {len(oct_keys)} OCT files under prefix '{prefix}'")
                            all_keys.extend(oct_keys)
                            break  # Use first prefix that has OCT files
                    except Exception as e:
                        logger.debug(f"No files found under prefix '{prefix}': {e}")
                        continue
                
                if not all_keys:
                    logger.warning("No OCT files found in bucket")
                    return {}
                
                # Extract participant IDs from keys
                # Assuming structure like: prefix/participant_id/... or prefix/participant_id.ext
                participant_keys = {}
                for key in all_keys:
                    # Extract participant ID from key path
                    participant_id = self._extract_participant_id(key)
                    if participant_id:
                        # If multiple files per participant, prefer certain formats
                        if participant_id not in participant_keys or self._is_preferred_format(key):
                            participant_keys[participant_id] = key
                
                cache = participant_keys
                self._save_cache(cache)
                logger.info(f"Built key mapping for {len(cache)} participants")
                
            except Exception as e:
                logger.error(f"Failed to build key mapping: {e}")
                return {}
        
        self._key_cache = cache
        return cache
    
    def _extract_participant_id(self, key: str) -> Optional[str]:
        """
        Extract participant ID from object key.
        
        Args:
            key: B2 object key
            
        Returns:
            Participant ID if extractable, None otherwise
        """
        # Remove bucket name if present
        key = key.replace(f"{self.bucket_name}/", "")
        
        # Common patterns for participant IDs (4-digit numbers like 1001-1100)
        import re
        
        # Pattern 1: participant_id anywhere in path (4-digit number)
        match = re.search(r'\b(1[0-9]{3})\b', key)
        if match:
            return match.group(1)
        
        # Pattern 2: Extract from filename
        filename = os.path.basename(key)
        match = re.search(r'\b(1[0-9]{3})\b', filename)
        if match:
            return match.group(1)
        
        # Pattern 3: First directory component
        parts = key.split('/')
        if len(parts) > 1:
            potential_id = parts[0] if parts[0] != 'fine-tuneing-data' else (parts[1] if len(parts) > 2 else None)
            if potential_id and re.match(r'^1[0-9]{3}$', potential_id):
                return potential_id
        
        return None
    
    def _is_preferred_format(self, key: str) -> bool:
        """Check if file format is preferred over others."""
        # Prefer DICOM, then NIfTI, then NPY
        key_lower = key.lower()
        if '.dcm' in key_lower or '.dicom' in key_lower:
            return True
        if '.nii' in key_lower:
            return True
        return False
    
    def resolve_oct_key(self, participant_id: str) -> Optional[str]:
        """
        Resolve participant ID to OCT file key.
        
        Args:
            participant_id: Participant ID (e.g., "1001")
            
        Returns:
            B2 object key for OCT file, or None if not found
        """
        key_mapping = self._build_key_mapping()
        
        # Direct lookup
        if participant_id in key_mapping:
            return key_mapping[participant_id]
        
        # Try with string conversion
        pid_str = str(participant_id)
        if pid_str in key_mapping:
            return key_mapping[pid_str]
        
        logger.warning(f"No OCT file found for participant {participant_id}")
        return None
    
    def get_available_participants(self) -> List[str]:
        """
        Get list of all participants with available OCT data.
        
        Returns:
            List of participant IDs
        """
        key_mapping = self._build_key_mapping()
        return list(key_mapping.keys())
    
    def refresh_cache(self):
        """Force refresh of the key cache."""
        logger.info("Refreshing OCT key cache...")
        self._key_cache = None
        self._build_key_mapping(force_rebuild=True)


@lru_cache(maxsize=1)
def get_default_locator() -> OCTLocator:
    """Get default OCT locator instance (cached)."""
    cache_dir = os.getenv('DATA_CACHE_DIR', '/tmp/oct_cache')
    cache_file = os.path.join(cache_dir, 'oct_key_cache.json')
    return OCTLocator(cache_file=cache_file)


def resolve_oct_key(participant_id: str) -> Optional[str]:
    """
    Convenience function to resolve participant ID to OCT key.
    
    Args:
        participant_id: Participant ID
        
    Returns:
        B2 object key for OCT file, or None if not found
    """
    locator = get_default_locator()
    return locator.resolve_oct_key(participant_id)


def get_available_participants() -> List[str]:
    """
    Convenience function to get available participants.
    
    Returns:
        List of participant IDs with OCT data
    """
    locator = get_default_locator()
    return locator.get_available_participants()