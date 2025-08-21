#!/usr/bin/env python3
"""
Enhanced DICOM validation with improved error handling and recovery strategies.
"""

import logging
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from typing import Optional, Dict, Any, Tuple
from data_setup.gcs_dicom_reader import GCSDICOMReader

logger = logging.getLogger(__name__)

class EnhancedDICOMValidator:
    """Enhanced DICOM validator with multiple validation strategies."""
    
    def __init__(self):
        self.reader = GCSDICOMReader(use_cache=False)
        
    def validate_dicom_comprehensive(self, gcs_path: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Comprehensive DICOM validation with multiple recovery strategies.
        
        Returns:
            Tuple of (is_valid, volume_info, error_message)
        """
        
        # Strategy 1: Standard validation
        try:
            result = self._validate_standard(gcs_path)
            if result[0]:
                return result
        except Exception as e:
            logger.debug(f"Standard validation failed for {gcs_path}: {e}")
        
        # Strategy 2: Relaxed validation (ignore minor errors)
        try:
            result = self._validate_relaxed(gcs_path)
            if result[0]:
                return result
        except Exception as e:
            logger.debug(f"Relaxed validation failed for {gcs_path}: {e}")
        
        # Strategy 3: Force validation (very permissive)
        try:
            result = self._validate_force(gcs_path)
            if result[0]:
                return result
        except Exception as e:
            logger.debug(f"Force validation failed for {gcs_path}: {e}")
        
        # All strategies failed
        return False, None, "All validation strategies failed"
    
    def _validate_standard(self, gcs_path: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Standard validation using existing pipeline."""
        
        result = self.reader.read_dicom_volume(gcs_path)
        if result is None:
            return False, None, "Failed to load DICOM with standard method"
        
        pixel_array = result.get('pixel_array')
        if pixel_array is None:
            return False, None, "No pixel data available"
        
        # Validate pixel array properties
        if pixel_array.size == 0:
            return False, None, "Empty pixel array"
        
        if len(pixel_array.shape) < 3:
            return False, None, f"Invalid shape: {pixel_array.shape} (need 3D)"
        
        # Check for reasonable dimensions
        if any(dim < 10 for dim in pixel_array.shape):
            return False, None, f"Suspiciously small dimensions: {pixel_array.shape}"
        
        if any(dim > 2048 for dim in pixel_array.shape):
            return False, None, f"Excessively large dimensions: {pixel_array.shape}"
        
        volume_info = {
            'shape': pixel_array.shape,
            'dtype': str(pixel_array.dtype),
            'size_mb': pixel_array.nbytes / (1024*1024),
            'spacing': result.get('spacing', (1.0, 1.0, 1.0)),
            'metadata': result.get('metadata', {})
        }
        
        return True, volume_info, "Standard validation successful"
    
    def _validate_relaxed(self, gcs_path: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Relaxed validation that ignores minor DICOM errors."""
        
        try:
            file_opener = self.reader._get_file_opener(gcs_path)
            
            with file_opener() as f:
                # More permissive DICOM reading
                dataset = pydicom.dcmread(
                    f,
                    force=True,  # Ignore DICOM format violations
                    defer_size=None,  # Load everything
                    stop_before_pixels=False
                )
            
            # Try multiple ways to access pixel data
            pixel_array = None
            
            # Method 1: Direct pixel_array access
            try:
                pixel_array = dataset.pixel_array
            except Exception as e:
                logger.debug(f"Direct pixel access failed: {e}")
            
            # Method 2: Manual pixel data extraction
            if pixel_array is None:
                try:
                    if hasattr(dataset, 'PixelData'):
                        # Convert raw pixel data to numpy array
                        pixel_data = dataset.PixelData
                        rows = getattr(dataset, 'Rows', 512)
                        cols = getattr(dataset, 'Columns', 512)
                        frames = getattr(dataset, 'NumberOfFrames', 1)
                        
                        # Guess data type from bits allocated
                        bits_allocated = getattr(dataset, 'BitsAllocated', 16)
                        if bits_allocated <= 8:
                            dtype = np.uint8
                        elif bits_allocated <= 16:
                            dtype = np.uint16
                        else:
                            dtype = np.uint32
                        
                        # Reshape pixel data
                        expected_size = frames * rows * cols
                        pixel_array = np.frombuffer(pixel_data, dtype=dtype)
                        
                        if len(pixel_array) >= expected_size:
                            pixel_array = pixel_array[:expected_size].reshape(frames, rows, cols)
                        else:
                            return False, None, f"Insufficient pixel data: {len(pixel_array)} < {expected_size}"
                            
                except Exception as e:
                    logger.debug(f"Manual pixel extraction failed: {e}")
            
            if pixel_array is None:
                return False, None, "No pixel data accessible with relaxed validation"
            
            # Validate the extracted array
            if pixel_array.size == 0:
                return False, None, "Empty pixel array"
            
            # Ensure 3D
            if len(pixel_array.shape) == 2:
                pixel_array = pixel_array[np.newaxis, ...]  # Add frame dimension
            elif len(pixel_array.shape) < 2:
                return False, None, f"Invalid shape: {pixel_array.shape}"
            
            volume_info = {
                'shape': pixel_array.shape,
                'dtype': str(pixel_array.dtype),
                'size_mb': pixel_array.nbytes / (1024*1024),
                'spacing': (1.0, 1.0, 1.0),  # Default spacing
                'metadata': {'validation_method': 'relaxed'}
            }
            
            return True, volume_info, "Relaxed validation successful"
            
        except Exception as e:
            return False, None, f"Relaxed validation failed: {str(e)}"
    
    def _validate_force(self, gcs_path: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Force validation - very permissive approach."""
        
        try:
            file_opener = self.reader._get_file_opener(gcs_path)
            
            with file_opener() as f:
                # Read as binary and try to extract any structured data
                raw_data = f.read()
            
            # Try to read as DICOM even if severely corrupted
            try:
                dataset = pydicom.dcmread(
                    raw_data,
                    force=True,
                    stop_before_pixels=True  # Just read headers first
                )
                
                # Check if this looks like OCT data
                modality = getattr(dataset, 'Modality', '')
                if modality not in ['OCT', 'OPT', '']:
                    return False, None, f"Not OCT data: modality={modality}"
                
                # Now try to get pixel data with maximum permissiveness
                dataset_full = pydicom.dcmread(raw_data, force=True)
                
                if hasattr(dataset_full, 'pixel_array'):
                    pixel_array = dataset_full.pixel_array
                    
                    if pixel_array is not None and pixel_array.size > 0:
                        # Reshape to 3D if needed
                        if len(pixel_array.shape) == 2:
                            pixel_array = pixel_array[np.newaxis, ...]
                        
                        volume_info = {
                            'shape': pixel_array.shape,
                            'dtype': str(pixel_array.dtype),
                            'size_mb': pixel_array.nbytes / (1024*1024),
                            'spacing': (1.0, 1.0, 1.0),
                            'metadata': {'validation_method': 'force'}
                        }
                        
                        return True, volume_info, "Force validation successful"
                
            except Exception as e:
                logger.debug(f"Force DICOM reading failed: {e}")
            
            return False, None, "Force validation: no valid pixel data found"
            
        except Exception as e:
            return False, None, f"Force validation failed: {str(e)}"

def validate_file_batch(file_paths: list, max_workers: int = 8) -> Dict[str, Any]:
    """
    Validate a batch of DICOM files using enhanced validation.
    
    Returns:
        Dictionary with validation results
    """
    import concurrent.futures
    
    validator = EnhancedDICOMValidator()
    
    valid_files = []
    invalid_files = []
    validation_details = {}
    
    def validate_single_file(file_path):
        is_valid, volume_info, error_msg = validator.validate_dicom_comprehensive(file_path)
        return file_path, is_valid, volume_info, error_msg
    
    print(f"Validating {len(file_paths)} files with enhanced validation...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(validate_single_file, path) for path in file_paths]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            file_path, is_valid, volume_info, error_msg = future.result()
            
            if is_valid:
                valid_files.append(file_path)
                validation_details[file_path] = volume_info
            else:
                invalid_files.append((file_path, error_msg))
            
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{len(file_paths)} - Valid: {len(valid_files)}, Invalid: {len(invalid_files)}")
    
    return {
        'valid_files': valid_files,
        'invalid_files': invalid_files,
        'validation_details': validation_details,
        'success_rate': len(valid_files) / len(file_paths) if file_paths else 0
    }