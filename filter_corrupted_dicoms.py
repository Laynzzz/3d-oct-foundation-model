#!/usr/bin/env python3
"""
Filter corrupted DICOM files from the manifest to create a clean training dataset.
This script validates DICOM files and creates a filtered manifest with only valid files.
"""

import os
import logging
import pandas as pd
from pathlib import Path
import concurrent.futures
from typing import List, Tuple, Optional
import time
from data_setup.gcs_dicom_reader import GCSDicomReader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dicom_filter')


def validate_dicom_file(file_path: str) -> Tuple[str, bool, Optional[str]]:
    """
    Validate a single DICOM file.
    
    Args:
        file_path: GCS path to DICOM file
        
    Returns:
        Tuple of (file_path, is_valid, error_message)
    """
    try:
        reader = GCSDicomReader()
        
        # Try to load and validate the DICOM
        dicom_data = reader.load_dicom(file_path)
        
        if dicom_data is None:
            return file_path, False, "Failed to load DICOM"
            
        # Try to access pixel array
        pixel_array = reader.get_pixel_array(dicom_data)
        
        if pixel_array is None:
            return file_path, False, "No pixel data available"
            
        # Check if pixel array has reasonable dimensions
        if len(pixel_array.shape) < 3:
            return file_path, False, f"Invalid pixel array shape: {pixel_array.shape}"
            
        # Check if pixel array is not empty
        if pixel_array.size == 0:
            return file_path, False, "Empty pixel array"
            
        logger.debug(f"✅ Valid: {file_path} - Shape: {pixel_array.shape}")
        return file_path, True, None
        
    except Exception as e:
        error_msg = str(e)
        logger.debug(f"❌ Invalid: {file_path} - Error: {error_msg}")
        return file_path, False, error_msg


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Load the manifest file."""
    logger.info(f"Loading manifest from: {manifest_path}")
    
    if manifest_path.startswith('gs://'):
        # For GCS paths, we'll need to download first
        import subprocess
        local_manifest = '/tmp/manifest.tsv'
        subprocess.run(['gsutil', 'cp', manifest_path, local_manifest], check=True)
        manifest_path = local_manifest
    
    df = pd.read_csv(manifest_path, sep='\t')
    logger.info(f"Loaded manifest with {len(df)} entries")
    return df


def filter_corrupted_files(manifest_df: pd.DataFrame, gcs_root: str, max_workers: int = 8) -> pd.DataFrame:
    """
    Filter corrupted DICOM files from the manifest.
    
    Args:
        manifest_df: DataFrame with DICOM file paths
        gcs_root: GCS root path
        max_workers: Number of parallel workers for validation
        
    Returns:
        Filtered DataFrame with only valid DICOM files
    """
    # Build full file paths
    file_paths = []
    for _, row in manifest_df.iterrows():
        file_path = f"{gcs_root}/{row['relative_path']}"
        file_paths.append(file_path)
    
    logger.info(f"Validating {len(file_paths)} DICOM files with {max_workers} workers...")
    
    valid_files = []
    invalid_files = []
    
    # Use ThreadPoolExecutor for I/O-bound DICOM validation
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all validation tasks
        future_to_path = {
            executor.submit(validate_dicom_file, path): path 
            for path in file_paths
        }
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_path)):
            file_path, is_valid, error_msg = future.result()
            
            if is_valid:
                valid_files.append(file_path)
            else:
                invalid_files.append((file_path, error_msg))
            
            # Progress reporting
            if (i + 1) % 50 == 0:
                valid_count = len(valid_files)
                invalid_count = len(invalid_files)
                total_processed = valid_count + invalid_count
                logger.info(f"Progress: {total_processed}/{len(file_paths)} processed, "
                          f"{valid_count} valid, {invalid_count} invalid")
    
    # Create filtered manifest
    valid_relative_paths = []
    for valid_path in valid_files:
        # Extract relative path
        relative_path = valid_path.replace(gcs_root + '/', '')
        valid_relative_paths.append(relative_path)
    
    # Filter the manifest to keep only valid files
    filtered_df = manifest_df[manifest_df['relative_path'].isin(valid_relative_paths)].copy()
    
    # Log results
    logger.info(f"Validation complete:")
    logger.info(f"  Total files: {len(file_paths)}")
    logger.info(f"  Valid files: {len(valid_files)} ({len(valid_files)/len(file_paths)*100:.1f}%)")
    logger.info(f"  Invalid files: {len(invalid_files)} ({len(invalid_files)/len(file_paths)*100:.1f}%)")
    
    # Log some examples of invalid files
    if invalid_files:
        logger.info("Examples of invalid files:")
        for file_path, error in invalid_files[:5]:
            logger.info(f"  ❌ {file_path}: {error}")
    
    return filtered_df


def save_filtered_manifest(filtered_df: pd.DataFrame, output_path: str):
    """Save the filtered manifest."""
    logger.info(f"Saving filtered manifest to: {output_path}")
    
    if output_path.startswith('gs://'):
        # Save locally first, then upload to GCS
        local_path = '/tmp/filtered_manifest.tsv'
        filtered_df.to_csv(local_path, sep='\t', index=False)
        
        import subprocess
        subprocess.run(['gsutil', 'cp', local_path, output_path], check=True)
        logger.info(f"Uploaded to GCS: {output_path}")
    else:
        filtered_df.to_csv(output_path, sep='\t', index=False)
    
    logger.info(f"Filtered manifest saved with {len(filtered_df)} valid entries")


def main():
    """Main function to filter corrupted DICOMs."""
    # Configuration
    manifest_path = "gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest.tsv"
    gcs_root = "gs://layne-tpu-code-sync/OCTdata/OCTdata"
    output_path = "gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest_filtered.tsv"
    max_workers = 16  # Increase for faster processing
    
    start_time = time.time()
    
    try:
        # Load manifest
        manifest_df = load_manifest(manifest_path)
        
        # Filter corrupted files
        filtered_df = filter_corrupted_files(manifest_df, gcs_root, max_workers)
        
        # Save filtered manifest
        save_filtered_manifest(filtered_df, output_path)
        
        # Summary
        elapsed_time = time.time() - start_time
        original_count = len(manifest_df)
        filtered_count = len(filtered_df)
        removed_count = original_count - filtered_count
        
        logger.info(f"\n{'='*50}")
        logger.info(f"DICOM FILTERING COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Time elapsed: {elapsed_time:.1f} seconds")
        logger.info(f"Original files: {original_count}")
        logger.info(f"Valid files: {filtered_count}")
        logger.info(f"Removed files: {removed_count}")
        logger.info(f"Success rate: {filtered_count/original_count*100:.1f}%")
        logger.info(f"Filtered manifest: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to filter DICOMs: {e}")
        raise


if __name__ == "__main__":
    main()