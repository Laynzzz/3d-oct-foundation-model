"""Test script for data pipeline components."""

import logging
from typing import Optional
from .manifest_parser import ManifestParser
from .gcs_dicom_reader import GCSDICOMReader


def test_manifest_parsing(
    manifest_path: str,
    gcs_root: str,
    sample_size: Optional[int] = None
):
    """Test manifest parsing functionality.
    
    Args:
        manifest_path: Path to manifest TSV
        gcs_root: GCS root path
        sample_size: Optional number of files to sample for testing
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing manifest parsing...")
    
    # Initialize parser
    parser = ManifestParser(manifest_path, gcs_root)
    
    # Load manifest
    manifest_df = parser.load_manifest()
    logger.info(f"Loaded manifest with {len(manifest_df)} entries")
    
    # Get device statistics
    device_stats = parser.get_device_stats()
    logger.info(f"Device statistics: {device_stats}")
    
    # Get single and multi-domain file lists
    single_domain = parser.get_single_domain_files()
    multi_domain = parser.get_multi_domain_files()
    
    logger.info(f"Single domain (topcon_triton): {len(single_domain)} files")
    logger.info(f"Multi domain (all devices): {len(multi_domain)} files")
    
    # Sample files for testing if requested
    if sample_size and sample_size < len(single_domain):
        test_files = single_domain[:sample_size]
        logger.info(f"Selected {len(test_files)} files for testing")
        return test_files
    
    return single_domain[:10]  # Return first 10 for testing


def test_dicom_reading(
    test_files: list,
    use_cache: bool = True,
    cache_dir: Optional[str] = None
):
    """Test DICOM reading functionality.
    
    Args:
        test_files: List of GCS paths to test
        use_cache: Whether to use local caching
        cache_dir: Cache directory path
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing DICOM reading...")
    
    # Initialize reader
    reader = GCSDICOMReader(use_cache=use_cache, cache_dir=cache_dir)
    
    successful_reads = 0
    for i, gcs_path in enumerate(test_files):
        logger.info(f"Reading file {i+1}/{len(test_files)}: {gcs_path}")
        
        result = reader.read_dicom_volume(gcs_path)
        
        if result is not None:
            successful_reads += 1
            pixel_array = result['pixel_array']
            spacing = result['spacing']
            metadata = result['metadata']
            
            logger.info(f"  Shape: {pixel_array.shape}")
            logger.info(f"  Spacing: {spacing}")
            logger.info(f"  Device: {metadata.get('manufacturer_model_name', 'Unknown')}")
            logger.info(f"  Frames: {metadata.get('number_of_frames', 'Unknown')}")
            
        else:
            logger.warning(f"  Failed to read file")
    
    logger.info(f"Successfully read {successful_reads}/{len(test_files)} files")
    logger.info(f"Skipped files: {reader.get_skipped_files_count()}")


def main():
    """Main test function."""
    # Configuration
    GCS_ROOT = "gs://layne-tpu-code-sync/OCTdata/OCTdata"
    MANIFEST_PATH = f"{GCS_ROOT}/manifest.tsv"
    CACHE_DIR = "/tmp/oct_cache"
    SAMPLE_SIZE = 5
    
    # Test manifest parsing
    test_files = test_manifest_parsing(
        MANIFEST_PATH, 
        GCS_ROOT, 
        SAMPLE_SIZE
    )
    
    # Test DICOM reading
    test_dicom_reading(
        test_files,
        use_cache=True,
        cache_dir=CACHE_DIR
    )


if __name__ == "__main__":
    main()