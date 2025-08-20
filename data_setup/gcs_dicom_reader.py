"""GCS DICOM reader with streaming support for OCT data."""

import os
import logging
from io import BytesIO
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gcsfs
import fsspec
import pydicom
from pydicom.errors import InvalidDicomError


logger = logging.getLogger(__name__)


class GCSDICOMReader:
    """DICOM reader with GCS streaming and optional local caching."""
    
    def __init__(self, use_cache: bool = True, cache_dir: Optional[str] = None):
        """Initialize GCS DICOM reader.
        
        Args:
            use_cache: Whether to use local caching
            cache_dir: Cache directory path (defaults to DATA_CACHE_DIR env var)
        """
        self.use_cache = use_cache
        self.cache_dir = cache_dir or os.environ.get('DATA_CACHE_DIR')
        self.skipped_files = 0
        
        # Initialize GCS filesystem
        self.gcs_fs = gcsfs.GCSFileSystem()
        
    def _get_file_opener(self, gcs_path: str):
        """Get file opener with optional caching.
        
        Args:
            gcs_path: GCS path to file
            
        Returns:
            File opener function
        """
        if self.use_cache and self.cache_dir:
            # Use local cache
            cached_fs = fsspec.filesystem(
                'filecache',
                target_protocol='gcs',
                cache_storage=self.cache_dir,
                target_options={'anon': False}
            )
            return lambda: cached_fs.open(gcs_path, 'rb')
        else:
            # Direct GCS access
            return lambda: fsspec.open(gcs_path, 'rb', anon=False).open()
    
    def read_dicom_volume(self, gcs_path: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """Read DICOM volume from GCS with metadata extraction and retry logic.
        
        Args:
            gcs_path: GCS path to DICOM file
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with keys:
            - 'pixel_array': numpy array [frames, height, width]
            - 'spacing': tuple (dz, dy, dx) in mm
            - 'metadata': dict with DICOM metadata
            - 'filepath': original file path
            
            None if file is corrupted or unreadable after retries
        """
        for attempt in range(max_retries + 1):
            try:
                file_opener = self._get_file_opener(gcs_path)
                
                with file_opener() as f:
                    # Read DICOM with deferred loading for large files
                    dataset = pydicom.dcmread(
                        BytesIO(f.read()),
                        force=True,
                        defer_size='512 KB'
                    )
                
                # Force decompression to handle compressed DICOM files
                try:
                    if hasattr(dataset, 'decompress'):
                        dataset.decompress()
                    logger.debug(f"Successfully decompressed DICOM: {gcs_path}")
                except Exception as e:
                    logger.warning(f"Decompression failed for {gcs_path}: {e}")
                    # Continue anyway - some files may not need decompression
                
                # Fallback: Convert transfer syntax if compression fails
                try:
                    if hasattr(dataset, 'file_meta') and hasattr(dataset.file_meta, 'TransferSyntaxUID'):
                        from pydicom.uid import ExplicitVRLittleEndian
                        if dataset.file_meta.TransferSyntaxUID != ExplicitVRLittleEndian:
                            logger.debug(f"Converting transfer syntax for {gcs_path}")
                            dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                except Exception as e:
                    logger.debug(f"Transfer syntax conversion failed for {gcs_path}: {e}")
                
                # Extract pixel data
                if not hasattr(dataset, 'pixel_array'):
                    logger.warning(f"No pixel data found in {gcs_path}")
                    self.skipped_files += 1
                    return None
                
                pixel_array = dataset.pixel_array
                
                # Apply rescale slope/intercept if present
                pixel_array = self._apply_rescaling(pixel_array, dataset)
                
                # Extract spacing information
                spacing = self._extract_spacing(dataset)
                
                # Extract metadata
                metadata = self._extract_metadata(dataset)
                
                # Normalize to float32 and apply z-score normalization
                pixel_array = self._normalize_volume(pixel_array)
                
                return {
                    'pixel_array': pixel_array,
                    'spacing': spacing,
                    'metadata': metadata,
                    'filepath': gcs_path
                }
                
            except (InvalidDicomError, FileNotFoundError, PermissionError) as e:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed for {gcs_path}: {type(e).__name__}: {e}")
                    continue
                else:
                    logger.warning(f"Failed to read DICOM {gcs_path} after {max_retries + 1} attempts: {type(e).__name__}: {e}")
                    self.skipped_files += 1
                    return None
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed for {gcs_path}: {type(e).__name__}: {e}")
                    continue
                else:
                    logger.error(f"Unexpected error reading {gcs_path} after {max_retries + 1} attempts: {type(e).__name__}: {e}")
                    # Log more details for debugging
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    self.skipped_files += 1
                    return None
                
        # If we get here, all retries failed
        logger.error(f"All {max_retries + 1} attempts failed for {gcs_path}")
        self.skipped_files += 1
        return None
    
    def _apply_rescaling(self, pixel_array: np.ndarray, dataset: pydicom.Dataset) -> np.ndarray:
        """Apply DICOM rescale slope and intercept.
        
        Args:
            pixel_array: Raw pixel data
            dataset: DICOM dataset
            
        Returns:
            Rescaled pixel array
        """
        # Convert to float32 first
        pixel_array = pixel_array.astype(np.float32)
        
        # Apply rescale slope and intercept if present
        slope = getattr(dataset, 'RescaleSlope', 1.0)
        intercept = getattr(dataset, 'RescaleIntercept', 0.0)
        
        if slope != 1.0 or intercept != 0.0:
            pixel_array = pixel_array * slope + intercept
            
        return pixel_array
    
    def _extract_spacing(self, dataset: pydicom.Dataset) -> Tuple[float, float, float]:
        """Extract voxel spacing from DICOM dataset.
        
        Args:
            dataset: DICOM dataset
            
        Returns:
            Spacing tuple (dz, dy, dx) in mm
        """
        # Try to get per-frame spacing from functional groups
        if hasattr(dataset, 'PerFrameFunctionalGroupsSequence'):
            try:
                first_frame = dataset.PerFrameFunctionalGroupsSequence[0]
                if hasattr(first_frame, 'PixelMeasuresSequence'):
                    pixel_measures = first_frame.PixelMeasuresSequence[0]
                    
                    # Get pixel spacing [row, column] in mm
                    if hasattr(pixel_measures, 'PixelSpacing'):
                        dy, dx = map(float, pixel_measures.PixelSpacing)
                    else:
                        dy, dx = 1.0, 1.0
                    
                    # Get slice thickness in mm
                    if hasattr(pixel_measures, 'SliceThickness'):
                        dz = float(pixel_measures.SliceThickness)
                    else:
                        dz = 1.0
                    
                    return (dz, dy, dx)
            except (AttributeError, IndexError, ValueError):
                pass
        
        # Fallback to standard DICOM tags
        try:
            if hasattr(dataset, 'PixelSpacing'):
                dy, dx = map(float, dataset.PixelSpacing)
            else:
                dy, dx = 1.0, 1.0
            
            if hasattr(dataset, 'SliceThickness'):
                dz = float(dataset.SliceThickness)
            else:
                dz = 1.0
                
            return (dz, dy, dx)
        except (AttributeError, ValueError):
            pass
        
        # Default spacing if no valid information found
        logger.warning(f"No valid spacing found, using default [1.0, 1.0, 1.0]")
        return (1.0, 1.0, 1.0)
    
    def _extract_metadata(self, dataset: pydicom.Dataset) -> Dict[str, Any]:
        """Extract relevant metadata from DICOM dataset.
        
        Args:
            dataset: DICOM dataset
            
        Returns:
            Dictionary with metadata
        """
        metadata = {}
        
        # Basic information
        metadata['sop_instance_uid'] = getattr(dataset, 'SOPInstanceUID', '')
        metadata['patient_id'] = getattr(dataset, 'PatientID', '')
        metadata['study_instance_uid'] = getattr(dataset, 'StudyInstanceUID', '')
        metadata['series_instance_uid'] = getattr(dataset, 'SeriesInstanceUID', '')
        
        # Image dimensions
        metadata['rows'] = getattr(dataset, 'Rows', 0)
        metadata['columns'] = getattr(dataset, 'Columns', 0)
        metadata['number_of_frames'] = getattr(dataset, 'NumberOfFrames', 1)
        
        # Acquisition parameters
        metadata['manufacturer'] = getattr(dataset, 'Manufacturer', '')
        metadata['manufacturer_model_name'] = getattr(dataset, 'ManufacturerModelName', '')
        metadata['photometric_interpretation'] = getattr(dataset, 'PhotometricInterpretation', '')
        
        # Transfer syntax
        if hasattr(dataset, 'file_meta') and hasattr(dataset.file_meta, 'TransferSyntaxUID'):
            metadata['transfer_syntax_uid'] = str(dataset.file_meta.TransferSyntaxUID)
        
        return metadata
    
    def _normalize_volume(self, pixel_array: np.ndarray) -> np.ndarray:
        """Apply z-score normalization per volume.
        
        Args:
            pixel_array: Input pixel array
            
        Returns:
            Z-score normalized array
        """
        # Convert to float32 if not already
        if pixel_array.dtype != np.float32:
            pixel_array = pixel_array.astype(np.float32)
        
        # Calculate mean and std across entire volume
        mean = np.mean(pixel_array)
        std = np.std(pixel_array)
        
        # Avoid division by zero
        if std == 0:
            logger.warning("Zero standard deviation in volume, skipping normalization")
            return pixel_array
        
        # Apply z-score normalization
        normalized = (pixel_array - mean) / std
        
        return normalized
    
    def get_skipped_files_count(self) -> int:
        """Get count of files that were skipped due to errors.
        
        Returns:
            Number of skipped files
        """
        return self.skipped_files
    
    def reset_skipped_count(self):
        """Reset the skipped files counter."""
        self.skipped_files = 0