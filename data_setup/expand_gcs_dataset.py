"""One-time GCS dataset expansion: ZIP → DICOM extraction.

This script expands ZIP files in GCS to individual DICOM files matching the manifest.tsv paths.
Run once on TPU VM, then treat dataset as read-only.
"""

import gcsfs
import zipfile
import io
import logging
import os
from typing import List, Set
from pathlib import Path


def setup_logging():
    """Setup logging for the expansion process."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s'
    )
    return logging.getLogger(__name__)


def expand_gcs_dataset(
    gcs_root: str = 'gs://layne-tpu-code-sync/OCTdata/OCTdata',
    devices: List[str] = None,
    dry_run: bool = False
):
    """Expand ZIP files in GCS to individual DICOM files.
    
    Args:
        gcs_root: GCS root path containing device ZIP files
        devices: List of devices to process (defaults to all 4 devices)
        dry_run: If True, only log what would be done without actual extraction
    """
    logger = setup_logging()
    
    if devices is None:
        devices = ['heidelberg_spectralis', 'topcon_triton', 'topcon_maestro2', 'zeiss_cirrus']
    
    # Initialize GCS filesystem
    logger.info("Initializing GCS filesystem...")
    fs = gcsfs.GCSFileSystem()
    
    SRC = gcs_root.rstrip('/')
    DST = f"{SRC}/retinal_oct/structural_oct"
    
    logger.info(f"Source: {SRC}")
    logger.info(f"Destination: {DST}")
    logger.info(f"Devices to process: {devices}")
    logger.info(f"Dry run: {dry_run}")
    
    total_files_extracted = 0
    total_zips_processed = 0
    
    for device in devices:
        logger.info(f"\n=== Processing device: {device} ===")
        
        # Find ZIP files for this device
        zip_pattern = f"{SRC}/{device}/*.zip"
        logger.info(f"Looking for ZIP files: {zip_pattern}")
        
        try:
            zip_paths = fs.glob(zip_pattern)
            logger.info(f"Found {len(zip_paths)} ZIP files for {device}")
            
            if not zip_paths:
                logger.warning(f"No ZIP files found for device {device}")
                continue
            
            for zip_path in zip_paths:
                logger.info(f"\nProcessing ZIP: {zip_path}")
                
                try:
                    # Read ZIP file from GCS
                    with fs.open(zip_path, 'rb') as f:
                        zip_data = f.read()
                    
                    logger.info(f"ZIP file size: {len(zip_data) / (1024*1024):.1f} MB")
                    
                    # Process ZIP contents
                    files_in_zip = 0
                    dcm_files_extracted = 0
                    
                    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                        zip_file_list = zf.namelist()
                        files_in_zip = len(zip_file_list)
                        
                        logger.info(f"ZIP contains {files_in_zip} files")
                        
                        for file_name in zip_file_list:
                            # Only process DICOM files
                            if not file_name.lower().endswith('.dcm'):
                                continue
                            
                            # Construct destination path
                            # Remove leading './' if present
                            clean_name = file_name.lstrip('./')
                            dst_path = f"{DST}/{device}/{clean_name}"
                            
                            if dry_run:
                                logger.info(f"Would extract: {file_name} → {dst_path}")
                                dcm_files_extracted += 1
                            else:
                                try:
                                    # Create parent directories
                                    parent_dir = dst_path.rsplit('/', 1)[0]
                                    fs.makedirs(parent_dir, exist_ok=True)
                                    
                                    # Extract and write file
                                    with zf.open(file_name) as zfh, fs.open(dst_path, 'wb') as out:
                                        out.write(zfh.read())
                                    
                                    dcm_files_extracted += 1
                                    
                                    if dcm_files_extracted % 100 == 0:
                                        logger.info(f"Extracted {dcm_files_extracted} DICOM files...")
                                
                                except Exception as e:
                                    logger.error(f"Failed to extract {file_name}: {e}")
                    
                    logger.info(f"Extracted {dcm_files_extracted} DICOM files from {zip_path}")
                    total_files_extracted += dcm_files_extracted
                    total_zips_processed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process ZIP {zip_path}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to process device {device}: {e}")
            continue
    
    logger.info(f"\n=== Expansion Complete ===")
    logger.info(f"Total ZIP files processed: {total_zips_processed}")
    logger.info(f"Total DICOM files extracted: {total_files_extracted}")
    
    if not dry_run:
        logger.info("Dataset expansion completed successfully!")
        logger.info("Individual DICOM files are now available in GCS at structured paths.")
    else:
        logger.info("Dry run completed. No files were actually extracted.")


def validate_expansion(
    gcs_root: str = 'gs://layne-tpu-code-sync/OCTdata/OCTdata',
    manifest_path: str = None,
    sample_size: int = 10
):
    """Validate that the expansion worked correctly.
    
    Args:
        gcs_root: GCS root path
        manifest_path: Path to manifest TSV file
        sample_size: Number of files to sample for validation
    """
    logger = setup_logging()
    logger.info("=== Validating dataset expansion ===")
    
    fs = gcsfs.GCSFileSystem()
    
    if manifest_path is None:
        manifest_path = f"{gcs_root}/manifest.tsv"
    
    try:
        # Load manifest to get expected file paths
        import pandas as pd
        
        logger.info(f"Loading manifest from: {manifest_path}")
        
        with fs.open(manifest_path, 'r') as f:
            manifest_df = pd.read_csv(f, sep='\t')
        
        logger.info(f"Manifest contains {len(manifest_df)} entries")
        
        # Sample files for validation
        sample_files = manifest_df.sample(n=min(sample_size, len(manifest_df)))
        
        successful_reads = 0
        for idx, row in sample_files.iterrows():
            filepath = row['filepath']
            full_gcs_path = f"{gcs_root}{filepath}"
            
            try:
                # Check if file exists
                if fs.exists(full_gcs_path):
                    # Try to get file info
                    file_info = fs.info(full_gcs_path)
                    file_size = file_info.get('size', 0)
                    logger.info(f"✓ {filepath} - Size: {file_size} bytes")
                    successful_reads += 1
                else:
                    logger.warning(f"✗ {filepath} - File not found")
            
            except Exception as e:
                logger.error(f"✗ {filepath} - Error: {e}")
        
        logger.info(f"\nValidation Results:")
        logger.info(f"Successfully found: {successful_reads}/{len(sample_files)} files")
        logger.info(f"Success rate: {successful_reads/len(sample_files)*100:.1f}%")
        
        if successful_reads == len(sample_files):
            logger.info("✓ Validation passed! Dataset expansion appears successful.")
        else:
            logger.warning("⚠ Some files missing. Check expansion process.")
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Expand GCS OCT dataset from ZIP to DICOM")
    parser.add_argument(
        '--gcs-root',
        default='gs://layne-tpu-code-sync/OCTdata/OCTdata',
        help='GCS root path containing device ZIP files'
    )
    parser.add_argument(
        '--devices',
        nargs='+',
        default=['heidelberg_spectralis', 'topcon_triton', 'topcon_maestro2', 'zeiss_cirrus'],
        help='Devices to process'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only log what would be done without actual extraction'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate the expansion by checking sample files'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation, skip expansion'
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        validate_expansion(args.gcs_root)
    else:
        # Run expansion
        expand_gcs_dataset(
            gcs_root=args.gcs_root,
            devices=args.devices,
            dry_run=args.dry_run
        )
        
        # Run validation if requested
        if args.validate:
            validate_expansion(args.gcs_root)


# Also provide the inline script version from the plan
def run_inline_expansion():
    """Run the inline expansion script from section 3.7 of the plan."""
    import gcsfs
    import zipfile
    import io
    
    fs = gcsfs.GCSFileSystem()
    SRC = 'gs://layne-tpu-code-sync/OCTdata/OCTdata'
    DST = SRC + '/retinal_oct/structural_oct'
    
    for dev in ['heidelberg_spectralis','topcon_triton','topcon_maestro2','zeiss_cirrus']:
        for zpath in fs.glob(f"{SRC}/{dev}/*.zip"):
            with fs.open(zpath, 'rb') as f:
                data = f.read()  # for very large zips, prefer Option A
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    if not name.lower().endswith('.dcm'): 
                        continue
                    dst = f"{DST}/{dev}/" + name.lstrip('./')
                    fs.makedirs(dst.rsplit('/',1)[0], exist_ok=True)
                    with zf.open(name) as zfh, fs.open(dst, 'wb') as out:
                        out.write(zfh.read())
    print('Done')


if __name__ == "__main__":
    main()