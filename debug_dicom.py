#!/usr/bin/env python3
"""Debug script to diagnose DICOM reading issues."""

import os
import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from data_setup.gcs_dicom_reader import GCSDICOMReader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dicom_reading():
    """Test reading a few DICOM files to diagnose issues."""
    
    # Test files from the error message
    test_files = [
        "gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/topcon_triton/7399/7399_triton_3d_radial_oct_r_2.16.840.1.114517.10.5.1.4.94005920240724155817.1.1.dcm"
    ]
    
    reader = GCSDICOMReader(use_cache=False)
    
    for i, gcs_path in enumerate(test_files):
        logger.info(f"Testing file {i+1}: {gcs_path}")
        
        try:
            result = reader.read_dicom_volume(gcs_path)
            if result is None:
                logger.warning(f"File {i+1} returned None")
            else:
                logger.info(f"File {i+1} successful: shape={result['pixel_array'].shape}, spacing={result['spacing']}")
        except Exception as e:
            logger.error(f"File {i+1} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Reader stats: skipped_files={reader.skipped_files}")

def check_jpeg_support():
    """Check if JPEG decompression libraries are available."""
    logger.info("Checking JPEG support...")
    
    try:
        import pylibjpeg
        logger.info(f"pylibjpeg version: {pylibjpeg.__version__}")
    except ImportError as e:
        logger.error(f"pylibjpeg not available: {e}")
    
    try:
        import pylibjpeg.pydicom
        logger.info("pylibjpeg.pydicom plugin available")
    except ImportError as e:
        logger.error(f"pylibjpeg.pydicom plugin not available: {e}")
        
    try:
        import pydicom
        logger.info(f"pydicom version: {pydicom.__version__}")
        
        # Check available decoders
        from pydicom.pixel_data_handlers import get_available_pixel_handlers
        handlers = get_available_pixel_handlers()
        logger.info(f"Available pixel handlers: {[h.__name__ for h in handlers]}")
        
    except ImportError as e:
        logger.error(f"pydicom not available: {e}")

if __name__ == "__main__":
    logger.info("=== DICOM Debug Script ===")
    
    check_jpeg_support()
    print()
    test_dicom_reading()